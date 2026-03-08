"""
Align enriched SR references with OpenAlex works.

Input files:
- data/refs/sr_references_enriched.jsonl
- data/refs/all_reference_ids_enriched.parquet

Output:
- data/refs/sr_references_aligned_openalex.jsonl (default)

Matching strategy:
1) Local exact DOI match against enriched OpenAlex parquet.
2) Local title/year/author scoring against enriched OpenAlex parquet.
3) OpenAlex API search fallback for unmatched references.

Each reference in output gets alignment fields:
- openalex_id
- match_method
- match_score
- match_status
- matched_title
- matched_year
- matched_doi
- api_queried
- needs_review
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pyarrow.parquet as pq
import requests
from tqdm import tqdm


DEFAULT_INPUT_JSONL = Path("data/refs/sr_references_enriched.jsonl")
DEFAULT_OPENALEX_PARQUET = Path("data/refs/all_reference_ids_enriched.parquet")
DEFAULT_OUTPUT_JSONL = Path("data/refs/sr_references_aligned_openalex.jsonl")
DEFAULT_ENV_PATH = Path("/home/fhg/pie65738/pie65738/projects/rag4repo/.env")
DEFAULT_LOG_PATH = Path("logs/align_with_oax.log")

OPENALEX_BASE_URL = "https://api.openalex.org"
REQUEST_TIMEOUT_SECONDS = 30
REQUEST_RETRIES = 3
REQUEST_SLEEP_SECONDS = 0.12
API_TOP_K = 5

DOI_RE = re.compile(r"(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)", re.IGNORECASE)

THRESHOLDS: Dict[str, float] = {
	"conservative": 0.86,
	"balanced": 0.78,
	"aggressive": 0.70,
}


@dataclass
class MatchResult:
	openalex_id: Optional[str]
	match_method: str
	match_score: float
	match_status: str
	matched_title: Optional[str]
	matched_year: Optional[int]
	matched_doi: Optional[str]
	api_queried: bool
	needs_review: bool


def setup_logging(log_path: Path) -> logging.Logger:
	log_path.parent.mkdir(parents=True, exist_ok=True)
	logger = logging.getLogger("align_with_oax")
	logger.setLevel(logging.INFO)
	logger.handlers.clear()
	handler = logging.FileHandler(log_path, encoding="utf-8")
	handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
	logger.addHandler(handler)
	logger.propagate = False
	return logger


def load_env_file(env_path: Path) -> Dict[str, str]:
	values: Dict[str, str] = {}
	if not env_path.exists():
		return values

	with env_path.open("r", encoding="utf-8") as handle:
		for line in handle:
			line = line.strip()
			if not line or line.startswith("#") or "=" not in line:
				continue
			key, value = line.split("=", 1)
			key = key.strip()
			value = value.strip().strip('"').strip("'")
			if key:
				values[key] = value

	return values


def get_openalex_auth(env_path: Path) -> Tuple[Optional[str], Optional[str]]:
	file_vars = load_env_file(env_path)
	api_key = (
		os.getenv("OPENALEX_API_KEY")
		or file_vars.get("OPENALEX_API_KEY")
		or os.getenv("OPENALEX_KEY")
		or file_vars.get("OPENALEX_KEY")
	)
	email = os.getenv("OPENALEX_EMAIL") or file_vars.get("OPENALEX_EMAIL")
	return api_key, email


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
	with path.open("r", encoding="utf-8") as handle:
		for line in handle:
			line = line.strip()
			if not line:
				continue
			try:
				obj = json.loads(line)
			except json.JSONDecodeError:
				continue
			if isinstance(obj, dict):
				yield obj


def normalize_space(text: str) -> str:
	return re.sub(r"\s+", " ", (text or "")).strip()


def normalize_text_for_match(text: str) -> str:
	text = (text or "").lower()
	text = re.sub(r"https?://\S+", " ", text)
	text = re.sub(r"[^a-z0-9\s]", " ", text)
	return normalize_space(text)


def similarity(a: str, b: str) -> float:
	if not a or not b:
		return 0.0
	return SequenceMatcher(None, normalize_text_for_match(a), normalize_text_for_match(b)).ratio()


def normalize_openalex_id(value: Optional[str]) -> Optional[str]:
	if not isinstance(value, str) or not value.strip():
		return None
	v = value.strip()
	if "/" in v:
		v = v.rsplit("/", 1)[-1]
	if not v.upper().startswith("W"):
		return None
	return v.upper()


def normalize_doi(value: Optional[str]) -> Optional[str]:
	if not isinstance(value, str) or not value.strip():
		return None
	v = value.strip().lower()
	v = re.sub(r"^https?://(dx\.)?doi\.org/", "", v)
	v = v.strip(" .,:;()[]{}")
	match = DOI_RE.search(v)
	if match:
		return match.group(1).lower()
	return v if v.startswith("10.") else None


def safe_int(value: Any) -> Optional[int]:
	if isinstance(value, int):
		return value
	if isinstance(value, str) and value.isdigit():
		return int(value)
	return None


def normalize_author_tokens(authors: Any) -> List[str]:
	if not isinstance(authors, list):
		return []
	tokens: List[str] = []
	for author in authors:
		if not isinstance(author, str):
			continue
		author_clean = normalize_text_for_match(author)
		parts = author_clean.split()
		if parts:
			tokens.append(parts[-1])
	return [t for t in tokens if t]


def author_overlap_score(ref_authors: Any, oa_authors: Any) -> float:
	ref_set = set(normalize_author_tokens(ref_authors))
	oa_set = set(normalize_author_tokens(oa_authors))
	if not ref_set or not oa_set:
		return 0.0
	inter = len(ref_set & oa_set)
	union = len(ref_set | oa_set)
	if union == 0:
		return 0.0
	return inter / union


def year_score(ref_year: Optional[int], oa_year: Optional[int]) -> float:
	if ref_year is None or oa_year is None:
		return 0.0
	delta = abs(ref_year - oa_year)
	if delta == 0:
		return 1.0
	if delta == 1:
		return 0.7
	if delta == 2:
		return 0.4
	return 0.0


def combine_score(title_sim: float, y_score: float, auth_score: float) -> float:
	return (0.70 * title_sim) + (0.20 * y_score) + (0.10 * auth_score)


def load_openalex_rows(path: Path) -> List[Dict[str, Any]]:
	table = pq.read_table(path)
	return table.to_pylist()


def build_indexes(
	openalex_rows: List[Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[int, List[Dict[str, Any]]], List[Dict[str, Any]]]:
	doi_index: Dict[str, Dict[str, Any]] = {}
	year_index: Dict[int, List[Dict[str, Any]]] = {}
	valid_rows: List[Dict[str, Any]] = []

	for row in openalex_rows:
		if row.get("fetch_status") not in (None, "ok"):
			continue
		work_id = normalize_openalex_id(row.get("work_id") or row.get("reference_id"))
		title = row.get("title")
		if not work_id or not isinstance(title, str) or not title.strip():
			continue

		row = dict(row)
		row["work_id"] = work_id
		row["title"] = normalize_space(title)
		row["publication_year"] = safe_int(row.get("publication_year"))
		row["doi"] = normalize_doi(row.get("doi"))
		valid_rows.append(row)

		doi = row.get("doi")
		if doi and doi not in doi_index:
			doi_index[doi] = row

		year = row.get("publication_year")
		if isinstance(year, int):
			year_index.setdefault(year, []).append(row)

	return doi_index, year_index, valid_rows


def get_local_candidates(
	ref_year: Optional[int],
	year_index: Dict[int, List[Dict[str, Any]]],
	all_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
	if isinstance(ref_year, int):
		candidates: List[Dict[str, Any]] = []
		for y in (ref_year - 1, ref_year, ref_year + 1):
			candidates.extend(year_index.get(y, []))
		if candidates:
			return candidates
	return all_rows


def evaluate_best_candidate(
	ref_title: str,
	ref_year: Optional[int],
	ref_authors: Any,
	candidates: List[Dict[str, Any]],
	threshold: float,
	method: str,
	api_queried: bool,
) -> MatchResult:
	best_row: Optional[Dict[str, Any]] = None
	best_score = 0.0

	for row in candidates:
		title_sim = similarity(ref_title, row.get("title") or "")
		if title_sim < 0.45:
			continue
		y_score = year_score(ref_year, safe_int(row.get("publication_year")))
		a_score = author_overlap_score(ref_authors, row.get("authors"))
		score = combine_score(title_sim, y_score, a_score)
		if score > best_score:
			best_score = score
			best_row = row

	if best_row is None:
		return MatchResult(
			openalex_id=None,
			match_method="unmatched",
			match_score=0.0,
			match_status="unmatched",
			matched_title=None,
			matched_year=None,
			matched_doi=None,
			api_queried=api_queried,
			needs_review=True,
		)

	if best_score >= threshold:
		return MatchResult(
			openalex_id=best_row.get("work_id"),
			match_method=method,
			match_score=round(best_score, 4),
			match_status="matched",
			matched_title=best_row.get("title"),
			matched_year=safe_int(best_row.get("publication_year")),
			matched_doi=normalize_doi(best_row.get("doi")),
			api_queried=api_queried,
			needs_review=False,
		)

	return MatchResult(
		openalex_id=None,
		match_method="unmatched",
		match_score=round(best_score, 4),
		match_status="unmatched",
		matched_title=best_row.get("title"),
		matched_year=safe_int(best_row.get("publication_year")),
		matched_doi=normalize_doi(best_row.get("doi")),
		api_queried=api_queried,
		needs_review=True,
	)


def get_with_retry(
	session: requests.Session,
	url: str,
	params: Optional[Dict[str, str]],
	retries: int,
	timeout_seconds: int,
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


def search_openalex_candidates(
	session: requests.Session,
	ref_title: str,
	ref_year: Optional[int],
	api_key: Optional[str],
	mailto: Optional[str],
	max_items: int,
) -> List[Dict[str, Any]]:
	params: Dict[str, str] = {
		"search": ref_title,
		"per-page": str(max_items),
	}
	if api_key:
		params["api_key"] = api_key
	if mailto:
		params["mailto"] = mailto

	payload = get_with_retry(
		session=session,
		url=f"{OPENALEX_BASE_URL}/works",
		params=params,
		retries=REQUEST_RETRIES,
		timeout_seconds=REQUEST_TIMEOUT_SECONDS,
	)
	if not payload:
		return []

	results = payload.get("results")
	if not isinstance(results, list):
		return []

	candidates: List[Dict[str, Any]] = []
	for item in results:
		if not isinstance(item, dict):
			continue
		work_id = normalize_openalex_id(item.get("id"))
		title = item.get("display_name")
		if not work_id or not isinstance(title, str) or not title.strip():
			continue

		oax_year = safe_int(item.get("publication_year"))
		if isinstance(ref_year, int) and isinstance(oax_year, int) and abs(ref_year - oax_year) > 4:
			continue

		doi = normalize_doi(item.get("doi"))
		authorships = item.get("authorships") if isinstance(item.get("authorships"), list) else []
		authors = []
		for auth in authorships:
			if not isinstance(auth, dict):
				continue
			author = auth.get("author") if isinstance(auth.get("author"), dict) else {}
			name = author.get("display_name")
			if isinstance(name, str) and name.strip():
				authors.append(name.strip())

		candidates.append(
			{
				"work_id": work_id,
				"title": normalize_space(title),
				"publication_year": oax_year,
				"doi": doi,
				"authors": authors,
			}
		)

	return candidates


def align_reference(
	ref: Dict[str, Any],
	doi_index: Dict[str, Dict[str, Any]],
	year_index: Dict[int, List[Dict[str, Any]]],
	all_rows: List[Dict[str, Any]],
	session: requests.Session,
	api_key: Optional[str],
	mailto: Optional[str],
	threshold: float,
) -> MatchResult:
	ref_title = normalize_space(str(ref.get("title") or ""))
	ref_authors = ref.get("authors")
	ref_year = safe_int(ref.get("year"))
	ref_doi = normalize_doi(ref.get("doi"))

	if ref_doi and ref_doi in doi_index:
		row = doi_index[ref_doi]
		return MatchResult(
			openalex_id=row.get("work_id"),
			match_method="doi_exact",
			match_score=1.0,
			match_status="matched",
			matched_title=row.get("title"),
			matched_year=safe_int(row.get("publication_year")),
			matched_doi=normalize_doi(row.get("doi")),
			api_queried=False,
			needs_review=False,
		)

	if ref_title:
		candidates = get_local_candidates(ref_year, year_index, all_rows)
		local_result = evaluate_best_candidate(
			ref_title=ref_title,
			ref_year=ref_year,
			ref_authors=ref_authors,
			candidates=candidates,
			threshold=threshold,
			method="title_year_local",
			api_queried=False,
		)
		if local_result.match_status == "matched":
			return local_result

	if not ref_title:
		return MatchResult(
			openalex_id=None,
			match_method="unmatched",
			match_score=0.0,
			match_status="unmatched",
			matched_title=None,
			matched_year=None,
			matched_doi=None,
			api_queried=False,
			needs_review=True,
		)

	api_candidates = search_openalex_candidates(
		session=session,
		ref_title=ref_title,
		ref_year=ref_year,
		api_key=api_key,
		mailto=mailto,
		max_items=API_TOP_K,
	)
	time.sleep(REQUEST_SLEEP_SECONDS)

	if not api_candidates:
		return MatchResult(
			openalex_id=None,
			match_method="unmatched",
			match_score=0.0,
			match_status="unmatched",
			matched_title=None,
			matched_year=None,
			matched_doi=None,
			api_queried=True,
			needs_review=True,
		)

	return evaluate_best_candidate(
		ref_title=ref_title,
		ref_year=ref_year,
		ref_authors=ref_authors,
		candidates=api_candidates,
		threshold=threshold,
		method="openalex_api_search",
		api_queried=True,
	)


def count_total_references(path: Path, max_rows: Optional[int]) -> int:
	total = 0
	for idx, row in enumerate(iter_jsonl(path), start=1):
		if isinstance(max_rows, int) and max_rows > 0 and idx > max_rows:
			break
		references = row.get("references")
		if isinstance(references, list):
			total += len(references)
	return total


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Align SR references with OpenAlex IDs.")
	parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT_JSONL)
	parser.add_argument("--openalex-parquet", type=Path, default=DEFAULT_OPENALEX_PARQUET)
	parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT_JSONL)
	parser.add_argument("--env-path", type=Path, default=DEFAULT_ENV_PATH)
	parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
	parser.add_argument("--profile", choices=sorted(THRESHOLDS.keys()), default="balanced")
	parser.add_argument("--max-rows", type=int, default=None)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	logger = setup_logging(args.log_path)
	threshold = THRESHOLDS[args.profile]

	logger.info("Starting alignment")
	logger.info("Input JSONL: %s", args.input_jsonl)
	logger.info("OpenAlex parquet: %s", args.openalex_parquet)
	logger.info("Output JSONL: %s", args.output_jsonl)
	logger.info("Env path: %s", args.env_path)
	logger.info("Profile: %s | threshold=%.2f", args.profile, threshold)

	api_key, mailto = get_openalex_auth(args.env_path)
	logger.info("OpenAlex API key %s", "loaded" if api_key else "not found")
	logger.info("OpenAlex mailto %s", "loaded" if mailto else "not found")

	openalex_rows = load_openalex_rows(args.openalex_parquet)
	doi_index, year_index, all_rows = build_indexes(openalex_rows)
	logger.info(
		"OpenAlex rows loaded: %s | valid rows: %s | doi index: %s | year buckets: %s",
		len(openalex_rows),
		len(all_rows),
		len(doi_index),
		len(year_index),
	)

	total_refs = count_total_references(args.input_jsonl, args.max_rows)
	logger.info("Total references to process: %s", total_refs)

	args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

	stats: Dict[str, int] = {
		"docs": 0,
		"refs": 0,
		"matched": 0,
		"unmatched": 0,
		"doi_exact": 0,
		"title_year_local": 0,
		"openalex_api_search": 0,
		"api_queried": 0,
	}

	session = requests.Session()
	progress = tqdm(total=total_refs, desc="Align references", unit="ref")

	with args.output_jsonl.open("w", encoding="utf-8") as out_handle:
		for doc_idx, row in enumerate(iter_jsonl(args.input_jsonl), start=1):
			if isinstance(args.max_rows, int) and args.max_rows > 0 and doc_idx > args.max_rows:
				break

			stats["docs"] += 1
			references = row.get("references")
			if not isinstance(references, list):
				references = []

			aligned_refs: List[Dict[str, Any]] = []
			for ref in references:
				stats["refs"] += 1
				if isinstance(ref, dict):
					match = align_reference(
						ref=ref,
						doi_index=doi_index,
						year_index=year_index,
						all_rows=all_rows,
						session=session,
						api_key=api_key,
						mailto=mailto,
						threshold=threshold,
					)

					if match.api_queried:
						stats["api_queried"] += 1
					if match.match_status == "matched":
						stats["matched"] += 1
						stats[match.match_method] = stats.get(match.match_method, 0) + 1
					else:
						stats["unmatched"] += 1

					ref_out = dict(ref)
					ref_out.update(
						{
							"openalex_id": match.openalex_id,
							"match_method": match.match_method,
							"match_score": match.match_score,
							"match_status": match.match_status,
							"matched_title": match.matched_title,
							"matched_year": match.matched_year,
							"matched_doi": match.matched_doi,
							"api_queried": match.api_queried,
							"needs_review": match.needs_review,
						}
					)
					aligned_refs.append(ref_out)
				else:
					stats["unmatched"] += 1
					aligned_refs.append(
						{
							"openalex_id": None,
							"match_method": "unmatched",
							"match_score": 0.0,
							"match_status": "unmatched",
							"matched_title": None,
							"matched_year": None,
							"matched_doi": None,
							"api_queried": False,
							"needs_review": True,
						}
					)

				progress.update(1)

			out_row = dict(row)
			out_row["references"] = aligned_refs
			out_handle.write(json.dumps(out_row, ensure_ascii=False) + "\n")

			if stats["docs"] % 100 == 0:
				logger.info(
					"Processed docs=%s refs=%s matched=%s unmatched=%s api_queried=%s",
					stats["docs"],
					stats["refs"],
					stats["matched"],
					stats["unmatched"],
					stats["api_queried"],
				)

	progress.close()

	logger.info("Finished alignment")
	logger.info("Stats: %s", json.dumps(stats, ensure_ascii=False))

	print(f"Wrote aligned references to: {args.output_jsonl}")
	print(
		"Summary: "
		f"docs={stats['docs']} refs={stats['refs']} matched={stats['matched']} "
		f"unmatched={stats['unmatched']} api_queried={stats['api_queried']}"
	)
	print(
		"Methods: "
		f"doi_exact={stats.get('doi_exact', 0)} "
		f"title_year_local={stats.get('title_year_local', 0)} "
		f"openalex_api_search={stats.get('openalex_api_search', 0)}"
	)


if __name__ == "__main__":
	main()
