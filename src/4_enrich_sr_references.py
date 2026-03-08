"""
Enrich raw reference strings in JSONL into a structured representation.

Setup requirements (run these before using this script):
- Python 3.9+ and `pip`
- Python package: `requests`
- Python package: `tqdm`
- AnyStyle CLI (required for AnyStyle-first parsing)

Install commands:

1) Install Python dependency:
   pip install requests tqdm

2) Install Ruby + AnyStyle (Ubuntu/Debian):
   sudo apt-get update
   sudo apt-get install -y ruby-full build-essential
   gem install anystyle

3) Verify AnyStyle is available:
   anystyle --help

Run commands:

- Standard run (AnyStyle + Crossref):
  python3 src/enrich_sr_references.py --input data/refs/sr_references.jsonl --mailto pieer.achkar@imw.fraunhofer.de

- Disable Crossref (AnyStyle + heuristic only):
  python3 src/enrich_sr_references.py --input data/refs/sr_references.jsonl --disable-crossref

- Disable AnyStyle (Crossref + heuristic only):
  python3 src/enrich_sr_references.py --input data/refs/sr_references.jsonl --disable-anystyle --mailto pieer.achkar@imw.fraunhofer.de

Notes:
- If `anystyle` is not installed, the script prints a warning and falls back to Crossref/heuristic parsing.
- Output defaults to `<input>_enriched.jsonl` in the same directory.

Input format (per row):
{
  "id": "...",
  "references": [
    {"number": "1", "value": "raw reference text"},
    ...
  ]
}

Output format (per row):
{
  "id": "...",
  "references": [
    {
      "number": "1",
      "value": "raw reference text",
      "title": "...",
      "authors": ["..."],
      "year": 2020,
      "doi": "10....",
      "parse_source": "anystyle|anystyle+crossref_doi|anystyle+crossref_biblio|crossref_doi|crossref_biblio|heuristic",
      "confidence": 0.0
    },
    ...
  ]
}
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
try:
	from tqdm import tqdm
except ModuleNotFoundError:
	tqdm = None


DEFAULT_INPUT_PATH = Path("data/refs/sr_references.jsonl")
DEFAULT_LOG_PATH = Path("logs/enrich_sr_references.log")
DEFAULT_TIMEOUT_SECONDS = 20
DEFAULT_RETRIES = 3
DEFAULT_SLEEP_SECONDS = 0.12

DOI_RE = re.compile(r"(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


@dataclass
class ParsedReference:
	title: Optional[str]
	authors: List[str]
	year: Optional[int]
	doi: Optional[str]
	parse_source: str
	confidence: float


def setup_logger(log_path: Path) -> logging.Logger:
	log_path.parent.mkdir(parents=True, exist_ok=True)
	logger = logging.getLogger("enrich_sr_references")
	logger.setLevel(logging.INFO)
	logger.handlers.clear()
	file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
	file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
	logger.addHandler(file_handler)
	logger.propagate = False
	return logger


def compute_output_path(input_path: Path, explicit_output: Optional[Path]) -> Path:
	if explicit_output is not None:
		return explicit_output
	if input_path.suffix == ".jsonl":
		return input_path.with_name(f"{input_path.stem}_enriched.jsonl")
	return input_path.with_name(f"{input_path.name}_enriched")


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
	with path.open("r", encoding="utf-8") as handle:
		for line_no, line in enumerate(handle, start=1):
			line = line.strip()
			if not line:
				continue
			try:
				obj = json.loads(line)
			except json.JSONDecodeError:
				print(f"[warn] Skipping invalid JSON at line {line_no}")
				continue
			if isinstance(obj, dict):
				yield obj


def count_jsonl_objects(path: Path) -> int:
	count = 0
	with path.open("r", encoding="utf-8") as handle:
		for line in handle:
			if line.strip():
				count += 1
	return count


def normalize_space(text: str) -> str:
	return re.sub(r"\s+", " ", (text or "")).strip()


def clean_doi(doi: str) -> str:
	doi = doi.strip().strip(".,;:()[]{}")
	return doi.lower()


def extract_doi(text: str) -> Optional[str]:
	match = DOI_RE.search(text or "")
	if not match:
		return None
	return clean_doi(match.group(1))


def extract_year(text: str) -> Optional[int]:
	years = YEAR_RE.findall(text or "")
	if not years:
		return None
	# YEAR_RE returns tuples because of group in pattern prefix; prefer direct scan.
	matches = re.finditer(r"\b((?:19|20)\d{2})\b", text or "")
	values = [int(m.group(1)) for m in matches]
	if not values:
		return None
	# Most references mention publication year early; use first plausible one.
	return values[0]


def normalize_for_similarity(text: str) -> str:
	text = (text or "").lower()
	text = re.sub(r"https?://\S+", " ", text)
	text = re.sub(r"[^a-z0-9\s]", " ", text)
	return normalize_space(text)


def similarity(a: str, b: str) -> float:
	if not a or not b:
		return 0.0
	return SequenceMatcher(None, normalize_for_similarity(a), normalize_for_similarity(b)).ratio()


def parse_year_from_crossref_item(item: Dict[str, Any]) -> Optional[int]:
	for key in ("issued", "published-print", "published-online", "created"):
		obj = item.get(key)
		if not isinstance(obj, dict):
			continue
		date_parts = obj.get("date-parts")
		if isinstance(date_parts, list) and date_parts and isinstance(date_parts[0], list) and date_parts[0]:
			year = date_parts[0][0]
			if isinstance(year, int):
				return year
			if isinstance(year, str) and year.isdigit():
				return int(year)
	return None


def parse_authors_from_crossref_item(item: Dict[str, Any]) -> List[str]:
	authors: List[str] = []
	for author in item.get("author") or []:
		if not isinstance(author, dict):
			continue
		given = normalize_space(author.get("given") or "")
		family = normalize_space(author.get("family") or "")
		name = normalize_space(f"{given} {family}") or normalize_space(author.get("name") or "")
		if name:
			authors.append(name)
	return authors


def parse_from_crossref_item(item: Dict[str, Any], source: str, confidence: float) -> ParsedReference:
	title_list = item.get("title") or []
	title = None
	if isinstance(title_list, list) and title_list:
		title = normalize_space(str(title_list[0]))

	doi = item.get("DOI")
	if isinstance(doi, str) and doi.strip():
		doi = clean_doi(doi)
	else:
		doi = None

	return ParsedReference(
		title=title,
		authors=parse_authors_from_crossref_item(item),
		year=parse_year_from_crossref_item(item),
		doi=doi,
		parse_source=source,
		confidence=max(0.0, min(1.0, confidence)),
	)


def parse_year_from_date_parts(date_parts: Any) -> Optional[int]:
	if isinstance(date_parts, list) and date_parts and isinstance(date_parts[0], list) and date_parts[0]:
		year = date_parts[0][0]
		if isinstance(year, int):
			return year
		if isinstance(year, str) and year.isdigit():
			return int(year)
	return None


def request_json(
	session: requests.Session,
	url: str,
	params: Optional[Dict[str, str]],
	timeout_seconds: int,
	retries: int,
) -> Optional[Dict[str, Any]]:
	for attempt in range(1, retries + 1):
		try:
			resp = session.get(url, params=params, timeout=timeout_seconds)
			if resp.status_code == 200:
				return resp.json()
			if resp.status_code == 404:
				return None
			if resp.status_code in (429, 500, 502, 503, 504):
				time.sleep(min(2.0, 0.4 * attempt))
				continue
			return None
		except requests.RequestException:
			if attempt == retries:
				return None
			time.sleep(min(2.0, 0.4 * attempt))
	return None


def crossref_by_doi(
	session: requests.Session,
	doi: str,
	timeout_seconds: int,
	retries: int,
	cache: Dict[str, Optional[ParsedReference]],
) -> Optional[ParsedReference]:
	key = f"doi:{doi}"
	if key in cache:
		return cache[key]

	url = f"https://api.crossref.org/works/{doi}"
	payload = request_json(session, url, params=None, timeout_seconds=timeout_seconds, retries=retries)
	if not payload or not isinstance(payload.get("message"), dict):
		cache[key] = None
		return None

	parsed = parse_from_crossref_item(payload["message"], source="crossref_doi", confidence=0.98)
	cache[key] = parsed
	return parsed


def anystyle_parse_object(obj: Dict[str, Any]) -> ParsedReference:
	title: Optional[str] = None
	raw_title = obj.get("title")
	if isinstance(raw_title, str):
		title = normalize_space(raw_title)
	elif isinstance(raw_title, list) and raw_title:
		title = normalize_space(str(raw_title[0]))

	authors: List[str] = []
	raw_authors = obj.get("author")
	if isinstance(raw_authors, list):
		for a in raw_authors:
			if isinstance(a, str):
				name = normalize_space(a)
			elif isinstance(a, dict):
				given = normalize_space(str(a.get("given") or ""))
				family = normalize_space(str(a.get("family") or ""))
				literal = normalize_space(str(a.get("literal") or ""))
				name = literal or normalize_space(f"{given} {family}")
			else:
				name = ""
			if name:
				authors.append(name)

	year = None
	issued = obj.get("issued")
	if isinstance(issued, dict):
		year = parse_year_from_date_parts(issued.get("date-parts"))
	if year is None:
		raw_year = obj.get("year")
		if isinstance(raw_year, int):
			year = raw_year
		elif isinstance(raw_year, str):
			m = re.search(r"\b((?:19|20)\d{2})\b", raw_year)
			if m:
				year = int(m.group(1))

	doi = obj.get("DOI") or obj.get("doi")
	if isinstance(doi, str) and doi.strip():
		doi = clean_doi(doi)
	else:
		doi = None

	confidence = 0.45
	if title:
		confidence += 0.15
	if authors:
		confidence += 0.15
	if year is not None:
		confidence += 0.10
	if doi is not None:
		confidence += 0.10

	return ParsedReference(
		title=title,
		authors=authors,
		year=year,
		doi=doi,
		parse_source="anystyle",
		confidence=min(0.9, confidence),
	)


def run_anystyle_parse(
	raw_reference: str,
	timeout_seconds: int,
	anystyle_path: str,
) -> Optional[ParsedReference]:
	with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=True) as tmp:
		tmp.write(raw_reference + "\n")
		tmp.flush()

		commands = [
			[anystyle_path, "--stdout", "-f", "csl", "parse", tmp.name],
			[anystyle_path, "-f", "csl", "parse", tmp.name, "-"],
			[anystyle_path, "--stdout", "-f", "json", "parse", tmp.name],
		]

		for cmd in commands:
			try:
				res = subprocess.run(
					cmd,
					capture_output=True,
					text=True,
					check=False,
					timeout=timeout_seconds,
				)
			except (subprocess.SubprocessError, OSError):
				continue

			if res.returncode != 0 or not res.stdout.strip():
				continue

			try:
				parsed_json = json.loads(res.stdout)
			except json.JSONDecodeError:
				continue

			# CSL output should be a list with one record for one input line.
			if isinstance(parsed_json, list) and parsed_json:
				first = parsed_json[0]
				if isinstance(first, dict):
					return anystyle_parse_object(first)
			elif isinstance(parsed_json, dict):
				return anystyle_parse_object(parsed_json)

	return None


def parse_with_anystyle(
	raw_reference: str,
	timeout_seconds: int,
	anystyle_path: Optional[str],
	cache: Dict[str, Optional[ParsedReference]],
) -> Optional[ParsedReference]:
	if not anystyle_path:
		return None
	key = f"anystyle:{raw_reference}"
	if key in cache:
		return cache[key]
	parsed = run_anystyle_parse(raw_reference, timeout_seconds=timeout_seconds, anystyle_path=anystyle_path)
	cache[key] = parsed
	return parsed


def choose_crossref_from_anystyle(
	parsed_any: ParsedReference,
	raw_reference: str,
	session: requests.Session,
	mailto: Optional[str],
	timeout_seconds: int,
	retries: int,
	cache: Dict[str, Optional[ParsedReference]],
) -> Optional[ParsedReference]:
	doi_hint = parsed_any.doi or extract_doi(raw_reference)
	candidate: Optional[ParsedReference] = None

	if doi_hint:
		candidate = crossref_by_doi(
			session=session,
			doi=doi_hint,
			timeout_seconds=timeout_seconds,
			retries=retries,
			cache=cache,
		)

	if candidate is None:
		query_parts = [parsed_any.title or "", " ".join(parsed_any.authors[:3])]
		if parsed_any.year is not None:
			query_parts.append(str(parsed_any.year))
		query = normalize_space(" ".join(part for part in query_parts if part))
		if not query:
			query = raw_reference
		candidate = crossref_by_bibliographic(
			session=session,
			raw_reference=query,
			expected_year=parsed_any.year or extract_year(raw_reference),
			mailto=mailto,
			timeout_seconds=timeout_seconds,
			retries=retries,
			cache=cache,
		)

	if candidate is None:
		return None

	title_sim = similarity(parsed_any.title or raw_reference, candidate.title or "")
	year_ok = False
	if parsed_any.year is not None and candidate.year is not None:
		year_ok = abs(parsed_any.year - candidate.year) <= 1
	doi_ok = bool(parsed_any.doi and candidate.doi and parsed_any.doi == candidate.doi)

	if not (doi_ok or title_sim >= 0.60 or (title_sim >= 0.45 and year_ok)):
		return None

	merged = ParsedReference(
		title=candidate.title or parsed_any.title,
		authors=candidate.authors or parsed_any.authors,
		year=candidate.year or parsed_any.year,
		doi=candidate.doi or parsed_any.doi,
		parse_source=f"anystyle+{candidate.parse_source}",
		confidence=min(0.99, max(parsed_any.confidence, candidate.confidence) + 0.05),
	)
	return merged


def choose_crossref_candidate(
	raw_reference: str,
	expected_year: Optional[int],
	items: List[Dict[str, Any]],
) -> Optional[Tuple[Dict[str, Any], float]]:
	best_item: Optional[Dict[str, Any]] = None
	best_score = -1.0

	for item in items:
		title_list = item.get("title") or []
		title = str(title_list[0]) if isinstance(title_list, list) and title_list else ""
		title_score = similarity(raw_reference, title)

		score = title_score
		year = parse_year_from_crossref_item(item)
		if expected_year is not None and year is not None:
			if year == expected_year:
				score += 0.25
			elif abs(year - expected_year) == 1:
				score += 0.10

		if isinstance(item.get("DOI"), str):
			score += 0.05

		if score > best_score:
			best_score = score
			best_item = item

	if best_item is None:
		return None
	return best_item, best_score


def crossref_by_bibliographic(
	session: requests.Session,
	raw_reference: str,
	expected_year: Optional[int],
	mailto: Optional[str],
	timeout_seconds: int,
	retries: int,
	cache: Dict[str, Optional[ParsedReference]],
) -> Optional[ParsedReference]:
	key = f"biblio:{raw_reference}"
	if key in cache:
		return cache[key]

	params: Dict[str, str] = {"query.bibliographic": raw_reference, "rows": "5"}
	if mailto:
		params["mailto"] = mailto

	payload = request_json(
		session,
		"https://api.crossref.org/works",
		params=params,
		timeout_seconds=timeout_seconds,
		retries=retries,
	)
	if not payload:
		cache[key] = None
		return None

	message = payload.get("message") or {}
	items = message.get("items") or []
	if not isinstance(items, list) or not items:
		cache[key] = None
		return None

	chosen = choose_crossref_candidate(raw_reference, expected_year, items)
	if not chosen:
		cache[key] = None
		return None

	item, item_score = chosen
	# Title similarity ratio is in [0, 1]. Convert to a conservative confidence.
	confidence = min(0.9, max(0.5, item_score))
	parsed = parse_from_crossref_item(item, source="crossref_biblio", confidence=confidence)
	cache[key] = parsed
	return parsed


def heuristic_authors(author_block: str) -> List[str]:
	block = normalize_space(author_block)
	if not block:
		return []
	block = re.sub(r"\bet al\.?\b", "", block, flags=re.IGNORECASE)
	block = re.sub(r"\s*&\s*", " and ", block)
	parts = re.split(r"\s+and\s+|;\s*", block)
	candidates: List[str] = []
	for part in parts:
		part = normalize_space(part.strip("., "))
		if not part:
			continue
		# If there are many comma-separated chunks, keep likely "Lastname, Initials" pairs.
		if "," in part and len(part.split(",")) > 2:
			pieces = [normalize_space(p.strip()) for p in part.split(",") if normalize_space(p.strip())]
			for i in range(0, len(pieces), 2):
				candidates.append(normalize_space(", ".join(pieces[i : i + 2])))
		else:
			candidates.append(part)

	seen = set()
	result: List[str] = []
	for c in candidates:
		key = c.lower()
		if key in seen:
			continue
		seen.add(key)
		result.append(c)
	return result[:20]


def heuristic_parse(raw_reference: str) -> ParsedReference:
	text = normalize_space(raw_reference)
	doi = extract_doi(text)
	year = extract_year(text)

	authors: List[str] = []
	title: Optional[str] = None

	author_block = text
	if year is not None:
		m = re.search(rf"(.*?)(?:\(|\b){year}(?:\)|\b)", text)
		if m:
			author_block = normalize_space(m.group(1).strip(" .,:;"))
			after = normalize_space(text[m.end() :].lstrip(" .,:;-"))
			if after:
				title = normalize_space(after.split(".")[0])
	else:
		first_dot = text.find(".")
		if first_dot > 0:
			author_block = normalize_space(text[:first_dot])
			rest = normalize_space(text[first_dot + 1 :])
			title = normalize_space(rest.split(".")[0]) if rest else None

	authors = heuristic_authors(author_block)

	if (not title or len(title) < 4) and text:
		# Fallback: grab a middle segment that usually carries the title.
		segments = [normalize_space(s) for s in text.split(".") if normalize_space(s)]
		if len(segments) >= 2:
			title = segments[1]

	confidence = 0.25
	if year is not None:
		confidence += 0.1
	if doi is not None:
		confidence += 0.2
	if title:
		confidence += 0.1
	if authors:
		confidence += 0.1

	return ParsedReference(
		title=title,
		authors=authors,
		year=year,
		doi=doi,
		parse_source="heuristic",
		confidence=min(0.75, confidence),
	)


def enrich_reference_item(
	item: Any,
	session: requests.Session,
	mailto: Optional[str],
	enable_anystyle: bool,
	anystyle_path: Optional[str],
	enable_crossref: bool,
	timeout_seconds: int,
	retries: int,
	sleep_seconds: float,
	cache: Dict[str, Optional[ParsedReference]],
) -> Dict[str, Any]:
	if isinstance(item, dict):
		number = item.get("number")
		raw_value = item.get("value")
	else:
		number = None
		raw_value = item

	raw_value = normalize_space(str(raw_value or ""))
	parsed: Optional[ParsedReference] = None
	year_hint = extract_year(raw_value)
	doi = extract_doi(raw_value)

	if enable_anystyle and raw_value:
		parsed = parse_with_anystyle(
			raw_reference=raw_value,
			timeout_seconds=timeout_seconds,
			anystyle_path=anystyle_path,
			cache=cache,
		)

	if enable_crossref and parsed is not None:
		cross_checked = choose_crossref_from_anystyle(
			parsed_any=parsed,
			raw_reference=raw_value,
			session=session,
			mailto=mailto,
			timeout_seconds=timeout_seconds,
			retries=retries,
			cache=cache,
		)
		if cross_checked is not None:
			parsed = cross_checked
		time.sleep(sleep_seconds)

	if parsed is None:
		if enable_crossref and doi:
			parsed = crossref_by_doi(
				session=session,
				doi=doi,
				timeout_seconds=timeout_seconds,
				retries=retries,
				cache=cache,
			)
			time.sleep(sleep_seconds)

		if enable_crossref and parsed is None and raw_value:
			parsed = crossref_by_bibliographic(
				session=session,
				raw_reference=raw_value,
				expected_year=year_hint,
				mailto=mailto,
				timeout_seconds=timeout_seconds,
				retries=retries,
				cache=cache,
			)
			time.sleep(sleep_seconds)

	if parsed is None:
		parsed = heuristic_parse(raw_value)

	if parsed.doi is None and doi is not None:
		parsed.doi = doi
		parsed.confidence = min(1.0, parsed.confidence + 0.05)

	return {
		"number": number,
		"value": raw_value,
		"title": parsed.title,
		"authors": parsed.authors,
		"year": parsed.year,
		"doi": parsed.doi,
		"parse_source": parsed.parse_source,
		"confidence": round(parsed.confidence, 4),
	}


def enrich_row(
	row: Dict[str, Any],
	session: requests.Session,
	mailto: Optional[str],
	enable_anystyle: bool,
	anystyle_path: Optional[str],
	enable_crossref: bool,
	timeout_seconds: int,
	retries: int,
	sleep_seconds: float,
	cache: Dict[str, Optional[ParsedReference]],
) -> Dict[str, Any]:
	references = row.get("references")
	if not isinstance(references, list):
		references = []

	enriched_refs = [
		enrich_reference_item(
			item=ref,
			session=session,
			mailto=mailto,
			enable_anystyle=enable_anystyle,
			anystyle_path=anystyle_path,
			enable_crossref=enable_crossref,
			timeout_seconds=timeout_seconds,
			retries=retries,
			sleep_seconds=sleep_seconds,
			cache=cache,
		)
		for ref in references
	]

	return {
		"id": row.get("id"),
		"references": enriched_refs,
	}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Enrich parsed references into structured metadata.")
	parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Input JSONL path.")
	parser.add_argument(
		"--output",
		type=Path,
		default=None,
		help="Output JSONL path. Default: <input>_enriched.jsonl",
	)
	parser.add_argument("--mailto", type=str, default=None, help="Email for Crossref polite pool.")
	parser.add_argument("--disable-anystyle", action="store_true", help="Skip AnyStyle parser stage.")
	parser.add_argument("--disable-crossref", action="store_true", help="Skip Crossref API lookups.")
	parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
	parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
	parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP_SECONDS, help="Sleep between API requests.")
	parser.add_argument("--max-rows", type=int, default=None, help="Process only first N rows.")
	parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_PATH, help="Log file path.")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	input_path: Path = args.input
	output_path = compute_output_path(input_path, args.output)
	enable_anystyle = not args.disable_anystyle
	enable_crossref = not args.disable_crossref
	anystyle_path = shutil.which("anystyle")
	logger = setup_logger(args.log_file)

	output_path.parent.mkdir(parents=True, exist_ok=True)

	headers = {
		"User-Agent": (
			"rag4repo-ref-enricher/1.0"
			+ (f" (mailto:{args.mailto})" if args.mailto else "")
		)
	}
	session = requests.Session()
	session.headers.update(headers)

	cache: Dict[str, Optional[ParsedReference]] = {}
	rows_in = 0
	refs_in = 0
	refs_with_doi = 0
	total_rows = count_jsonl_objects(input_path)
	if args.max_rows is not None:
		total_rows = min(total_rows, args.max_rows)

	logger.info(
		"start input=%s output=%s max_rows=%s anystyle_enabled=%s anystyle_available=%s crossref_enabled=%s",
		input_path,
		output_path,
		args.max_rows,
		enable_anystyle,
		bool(anystyle_path),
		enable_crossref,
	)

	if enable_anystyle and not anystyle_path:
		print("[warn] AnyStyle is enabled but `anystyle` command was not found. Falling back to Crossref/heuristic.")
		logger.warning("AnyStyle enabled but command not found. Falling back to Crossref/heuristic.")

	row_iterator: Iterable[Dict[str, Any]] = iter_jsonl(input_path)
	if tqdm is not None:
		row_iterator = tqdm(row_iterator, total=total_rows, desc="Enriching rows", unit="row")
	else:
		print("[warn] `tqdm` is not installed. Progress bar disabled.")
		logger.warning("tqdm not installed. Progress bar disabled.")

	with output_path.open("w", encoding="utf-8") as out_handle:
		for row in row_iterator:
			if args.max_rows is not None and rows_in >= args.max_rows:
				break
			rows_in += 1

			refs = row.get("references")
			if isinstance(refs, list):
				refs_in += len(refs)

			enriched = enrich_row(
				row=row,
				session=session,
				mailto=args.mailto,
				enable_anystyle=enable_anystyle,
				anystyle_path=anystyle_path,
				enable_crossref=enable_crossref,
				timeout_seconds=args.timeout,
				retries=args.retries,
				sleep_seconds=args.sleep,
				cache=cache,
			)

			for r in enriched["references"]:
				if isinstance(r.get("doi"), str) and r["doi"]:
					refs_with_doi += 1

			out_handle.write(json.dumps(enriched, ensure_ascii=True) + "\n")

			if rows_in % 20 == 0:
				print(f"[progress] rows={rows_in} refs={refs_in}")
				logger.info("progress rows=%d refs=%d refs_with_doi=%d", rows_in, refs_in, refs_with_doi)

	print(f"Wrote {rows_in} rows to {output_path}")
	print(f"References processed: {refs_in}")
	print(f"References with DOI: {refs_with_doi}")
	print(f"AnyStyle enabled: {enable_anystyle}")
	print(f"AnyStyle available: {bool(anystyle_path)}")
	print(f"Crossref enabled: {enable_crossref}")
	print(f"Log file: {args.log_file}")
	logger.info(
		"done rows=%d refs=%d refs_with_doi=%d output=%s",
		rows_in,
		refs_in,
		refs_with_doi,
		output_path,
	)


if __name__ == "__main__":
	main()
