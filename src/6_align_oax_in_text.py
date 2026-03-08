"""Align numeric citations in structured JSON and tables with OpenAlex IDs."""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_STRUCTURED_DIR = Path("data/parsed/structured")
DEFAULT_TABLES_DIR = Path("data/parsed/structured/tables")
DEFAULT_OUTPUT_STRUCTURED_DIR = Path("data/parsed/structured_mapped")
DEFAULT_OUTPUT_TABLES_DIR = Path("data/parsed/structured_mapped/tables")
DEFAULT_ALIGNED_REFS = Path("data/refs/sr_references_aligned_openalex.jsonl")
DEFAULT_LOG_PATH = Path("logs/align_oax_in_text.log")

BRACKET_CITATION_RE = re.compile(r"\[(?P<content>[^\]]+)\]")
PAREN_CITATION_RE = re.compile(r"\((?P<content>[^)]+)\)")
NUM_LIST_RE = re.compile(r"^[0-9,;\s\-\u2013]+$")
RANGE_RE = re.compile(r"^(\d+)\s*[\-\u2013]\s*(\d+)$")


def setup_logger(log_path: Path) -> logging.Logger:
	log_path.parent.mkdir(parents=True, exist_ok=True)
	logger = logging.getLogger("align_oax_in_text")
	logger.setLevel(logging.INFO)
	logger.handlers.clear()
	file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
	file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
	logger.addHandler(file_handler)
	logger.propagate = False
	return logger


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


def normalize_openalex_id(value: Optional[str]) -> Optional[str]:
	if not isinstance(value, str) or not value.strip():
		return None
	clean = value.strip()
	if "/" in clean:
		clean = clean.rsplit("/", 1)[-1]
	if not clean.upper().startswith("W"):
		return None
	return clean.upper()


def build_openalex_mapping(aligned_path: Path) -> Dict[str, Dict[str, str]]:
	mapping: Dict[str, Dict[str, str]] = {}
	for row in iter_jsonl(aligned_path):
		sr_id = row.get("id")
		refs = row.get("references") or []
		if not isinstance(sr_id, str) or not isinstance(refs, list):
			continue
		per_ref: Dict[str, str] = {}
		for ref in refs:
			if not isinstance(ref, dict):
				continue
			num = ref.get("number")
			openalex_id = normalize_openalex_id(ref.get("openalex_id"))
			if isinstance(num, str) and openalex_id:
				per_ref[num.strip()] = openalex_id
		mapping[sr_id] = per_ref
	return mapping


def parse_citation_numbers(content: str) -> Optional[List[str]]:
	if not NUM_LIST_RE.match(content.strip()):
		return None
	results: List[str] = []
	for token in re.split(r"[\s,;]+", content.strip()):
		if not token:
			continue
		if token.isdigit():
			results.append(token)
			continue
		range_match = RANGE_RE.match(token)
		if not range_match:
			return None
		start = int(range_match.group(1))
		end = int(range_match.group(2))
		low, high = (start, end) if start <= end else (end, start)
		results.extend([str(value) for value in range(low, high + 1)])
	return results


def replace_citations_in_text(
	text: str,
	mapping: Dict[str, str],
	stats: Dict[str, int],
	replace_paren_single: bool,
	cleanup_whitespace: bool,
) -> str:
	def _replace(match: re.Match[str], open_bracket: str, close_bracket: str, allow_single: bool) -> str:
		content = match.group("content")
		numbers = parse_citation_numbers(content)
		if numbers is None:
			return match.group(0)
		if not allow_single and len(numbers) == 1 and numbers[0] == content.strip():
			return match.group(0)
		mapped = [mapping.get(num) for num in numbers if mapping.get(num)]
		unmapped = [num for num in numbers if not mapping.get(num)]
		if mapped:
			stats["text_replaced"] += len(mapped)
		if unmapped:
			stats["text_unmapped"] += len(unmapped)
		pieces = mapped + [f"UNMAPPED:{num}" for num in unmapped]
		return f"{open_bracket}{', '.join(pieces)}{close_bracket}"

	text = BRACKET_CITATION_RE.sub(lambda m: _replace(m, "[", "]", True), text)
	text = PAREN_CITATION_RE.sub(lambda m: _replace(m, "(", ")", replace_paren_single), text)
	if cleanup_whitespace:
		text = re.sub(r"[ \t]{2,}", " ", text)
		text = re.sub(r"[ \t]+([,.;:])", r"\1", text)
	return text


def update_citations_list(
	citations: List[str],
	mapping: Dict[str, str],
	stats: Dict[str, int],
) -> Tuple[List[str], List[str]]:
	mapped_list: List[str] = []
	unmapped_list: List[str] = []
	for num in citations:
		openalex_id = mapping.get(str(num))
		if openalex_id:
			mapped_list.append(openalex_id)
			stats["array_replaced"] += 1
		else:
			unmapped_list.append(str(num))
			stats["array_unmapped"] += 1
	return mapped_list, unmapped_list


def update_block(block: Dict[str, Any], mapping: Dict[str, str], stats: Dict[str, int]) -> None:
	text = block.get("text")
	if isinstance(text, str):
		block["text"] = replace_citations_in_text(
			text,
			mapping,
			stats,
			replace_paren_single=False,
			cleanup_whitespace=True,
		)

	citations = block.get("citations")
	if isinstance(citations, list):
		mapped, unmapped = update_citations_list([str(c) for c in citations], mapping, stats)
		block["citations"] = mapped
		if unmapped:
			block["citations_unmapped"] = unmapped
		else:
			block.pop("citations_unmapped", None)

	for sub_key in ("subsections", "subsubsections"):
		subs = block.get(sub_key)
		if isinstance(subs, list):
			for sub in subs:
				if isinstance(sub, dict):
					update_block(sub, mapping, stats)


def update_structured_json(
	input_path: Path,
	output_path: Path,
	mapping: Dict[str, str],
	stats: Dict[str, int],
) -> bool:
	with input_path.open("r", encoding="utf-8") as handle:
		data = json.load(handle)
	if not isinstance(data, dict):
		return False
	sections = data.get("sections")
	if isinstance(sections, list):
		for section in sections:
			if isinstance(section, dict):
				update_block(section, mapping, stats)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", encoding="utf-8") as handle:
		json.dump(data, handle, ensure_ascii=True, indent=2)
		handle.write("\n")
	return True


def update_tables_markdown(
	input_path: Path,
	output_path: Path,
	mapping: Dict[str, str],
	stats: Dict[str, int],
) -> bool:
	text = input_path.read_text(encoding="utf-8")
	updated = replace_citations_in_text(
		text,
		mapping,
		stats,
		replace_paren_single=False,
		cleanup_whitespace=False,
	)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text(updated, encoding="utf-8")
	return updated != text


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Replace numeric citations with OpenAlex IDs in text.")
	parser.add_argument("--structured-dir", type=Path, default=DEFAULT_STRUCTURED_DIR)
	parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES_DIR)
	parser.add_argument("--output-structured-dir", type=Path, default=DEFAULT_OUTPUT_STRUCTURED_DIR)
	parser.add_argument("--output-tables-dir", type=Path, default=DEFAULT_OUTPUT_TABLES_DIR)
	parser.add_argument("--aligned-refs", type=Path, default=DEFAULT_ALIGNED_REFS)
	parser.add_argument("--log", type=Path, default=DEFAULT_LOG_PATH)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	logger = setup_logger(args.log)
	openalex_map = build_openalex_mapping(args.aligned_refs)

	stats = {
		"json_files": 0,
		"table_files": 0,
		"array_replaced": 0,
		"array_unmapped": 0,
		"text_replaced": 0,
		"text_unmapped": 0,
	}

	for json_path in sorted(args.structured_dir.glob("*.json")):
		with json_path.open("r", encoding="utf-8") as handle:
			data = json.load(handle)
		sr_id = data.get("id") if isinstance(data, dict) else None
		if not isinstance(sr_id, str) or sr_id not in openalex_map:
			logger.warning("No OpenAlex mapping for structured JSON: %s", json_path.name)
			continue
		output_json_path = args.output_structured_dir / json_path.name
		if update_structured_json(json_path, output_json_path, openalex_map[sr_id], stats):
			stats["json_files"] += 1

	for table_path in sorted(args.tables_dir.glob("*_tables.md")):
		sr_id = table_path.stem.replace("_tables", "")
		if sr_id not in openalex_map:
			logger.warning("No OpenAlex mapping for tables: %s", table_path.name)
			continue
		output_table_path = args.output_tables_dir / table_path.name
		if update_tables_markdown(table_path, output_table_path, openalex_map[sr_id], stats):
			stats["table_files"] += 1

	logger.info("Structured JSON files updated: %d", stats["json_files"])
	logger.info("Table markdown files updated: %d", stats["table_files"])
	logger.info("Citation array replaced: %d", stats["array_replaced"])
	logger.info("Citation array unmapped: %d", stats["array_unmapped"])
	logger.info("Text citations replaced: %d", stats["text_replaced"])
	logger.info("Text citations unmapped: %d", stats["text_unmapped"])
	print("Done. See log for details.")


if __name__ == "__main__":
	main()
