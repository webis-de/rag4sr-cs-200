"""
Parse numbered references from markdown files and write a JSONL output.

Each output row contains:
- id: markdown filename without extension
- references: list of {"number": "N", "value": "reference text"}
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional


DEFAULT_INPUT_DIR = Path("data/raw/md")
DEFAULT_OUTPUT_PATH = Path("data/refs/sr_references.jsonl")
DEFAULT_LOG_PATH = Path("logs/parse_sr_refs.log")


HEADER_RE = re.compile(
	r"^\s*#{1,6}\s*(?:\d+(?:[\.)])?\s*)?(?:references?|reference\s+list)\b",
	re.IGNORECASE,
)
ANY_HEADER_RE = re.compile(r"^\s*#{1,6}\s+\S")
ITEM_RE = re.compile(r"^\s*(\d+)[\.)]\s+(\S.*)$")


def iter_markdown_files(input_dir: Path) -> Iterable[Path]:
	for path in sorted(input_dir.glob("*.md")):
		if path.is_file():
			yield path


def parse_references_from_lines(lines: List[str]) -> List[Dict[str, str]]:
	refs: List[Dict[str, str]] = []
	inside_refs = False
	current_number: Optional[str] = None
	current_value_parts: List[str] = []

	def flush_current() -> None:
		nonlocal current_number, current_value_parts
		if current_number and current_value_parts:
			value = " ".join(part.strip() for part in current_value_parts if part.strip())
			if value:
				refs.append({"number": current_number, "value": value})
		current_number = None
		current_value_parts = []

	for raw_line in lines:
		line = raw_line.rstrip("\n")

		if not inside_refs:
			if HEADER_RE.match(line):
				inside_refs = True
			continue

		if ANY_HEADER_RE.match(line):
			break

		item_match = ITEM_RE.match(line)
		if item_match:
			flush_current()
			current_number = item_match.group(1)
			current_value_parts = [item_match.group(2).strip()]
			continue

		if current_number is not None:
			if line.strip():
				current_value_parts.append(line.strip())
			continue

	flush_current()
	return refs


def parse_references_from_file(path: Path) -> List[Dict[str, str]]:
	with path.open("r", encoding="utf-8") as handle:
		lines = handle.readlines()
	return parse_references_from_lines(lines)


def write_jsonl(rows: Iterable[Dict[str, object]], output_path: Path) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", encoding="utf-8") as handle:
		for row in rows:
			handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def setup_logger(log_path: Path) -> logging.Logger:
	log_path.parent.mkdir(parents=True, exist_ok=True)
	logger = logging.getLogger("parse_sr_refs")
	logger.setLevel(logging.INFO)
	logger.handlers.clear()
	file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
	file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
	logger.addHandler(file_handler)
	logger.propagate = False
	return logger


def build_rows(input_dir: Path, logger: logging.Logger) -> List[Dict[str, object]]:
	rows: List[Dict[str, object]] = []
	for md_path in iter_markdown_files(input_dir):
		refs = parse_references_from_file(md_path)
		if not refs:
			logger.warning("No references found for markdown: %s", md_path.name)
		rows.append({"id": md_path.stem, "references": refs})
	return rows


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Parse numbered references from markdown files.")
	parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
	parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
	parser.add_argument("--log", type=Path, default=DEFAULT_LOG_PATH)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	logger = setup_logger(args.log)
	rows = build_rows(args.input_dir, logger)
	write_jsonl(rows, args.output)
	print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
	main()
