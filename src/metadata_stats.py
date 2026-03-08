from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List


DEFAULT_REFS_PATH = Path("data/refs/sr_references_aligned_openalex.jsonl")
DEFAULT_STRUCTURED_DIR = Path("data/parsed/structured_mapped")
DEFAULT_LOG_PATH = Path("logs/metadata_stats.log")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Compute metadata statistics for SR references JSONL and structured SR JSON files."
		)
	)
	parser.add_argument("--refs-path", type=Path, default=DEFAULT_REFS_PATH)
	parser.add_argument("--structured-dir", type=Path, default=DEFAULT_STRUCTURED_DIR)
	parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
	return parser.parse_args()


def safe_len(value: Any) -> int:
	return len(value) if isinstance(value, list) else 0


def summarize_numeric(values: List[int]) -> Dict[str, float | int | None]:
	if not values:
		return {
			"count": 0,
			"mean": None,
			"median": None,
			"min": None,
			"max": None,
		}

	count = len(values)
	return {
		"count": count,
		"mean": sum(values) / count,
		"median": median(values),
		"min": min(values),
		"max": max(values),
	}


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
	with path.open("r", encoding="utf-8") as handle:
		for line in handle:
			line = line.strip()
			if not line:
				continue
			yield json.loads(line)


def compute_reference_stats(refs_path: Path) -> Dict[str, Any]:
	refs_per_sr: List[int] = []
	matched_refs = 0
	unmatched_refs = 0
	needs_review_refs = 0

	rows = 0
	for row in iter_jsonl(refs_path):
		rows += 1
		references = row.get("references", [])
		refs_per_sr.append(safe_len(references))

		if isinstance(references, list):
			for ref in references:
				if not isinstance(ref, dict):
					continue
				status = ref.get("match_status")
				if status == "matched":
					matched_refs += 1
				elif status == "unmatched":
					unmatched_refs += 1

				if ref.get("needs_review") is True:
					needs_review_refs += 1

	return {
		"n_srs": rows,
		"references_per_sr": summarize_numeric(refs_per_sr),
		"extra": {
			"total_references": sum(refs_per_sr),
			"matched_references": matched_refs,
			"unmatched_references": unmatched_refs,
			"needs_review_references": needs_review_refs,
		},
	}


def compute_counts_from_sections(sections: Any) -> tuple[int, int]:
	if not isinstance(sections, list):
		return 0, 0

	section_count = len(sections)
	subsection_count = 0
	for section in sections:
		if not isinstance(section, dict):
			continue
		subsection_count += safe_len(section.get("subsections"))
	return section_count, subsection_count


def collect_citation_counts(sections: Any) -> tuple[int, int]:
	if not isinstance(sections, list):
		return 0, 0

	mapped = 0
	unmapped = 0
	for section in sections:
		if not isinstance(section, dict):
			continue
		mapped += safe_len(section.get("citations"))
		unmapped += safe_len(section.get("citations_unmapped"))
		for sub_key in ("subsections", "subsubsections"):
			subs = section.get(sub_key)
			sub_mapped, sub_unmapped = collect_citation_counts(subs)
			mapped += sub_mapped
			unmapped += sub_unmapped
	return mapped, unmapped


def compute_structured_stats(structured_dir: Path) -> Dict[str, Any]:
	n_sections_per_file: List[int] = []
	n_subsections_per_file: List[int] = []
	n_tables_per_file: List[int] = []
	mapped_citations_total = 0
	unmapped_citations_total = 0

	files = sorted(structured_dir.glob("*.json"))
	metadata_fallbacks = 0

	for json_path in files:
		with json_path.open("r", encoding="utf-8") as handle:
			doc = json.load(handle)

		metadata = doc.get("metadata") if isinstance(doc, dict) else None
		sections = doc.get("sections") if isinstance(doc, dict) else []
		tables = doc.get("tables") if isinstance(doc, dict) else []

		computed_sections, computed_subsections = compute_counts_from_sections(sections)
		computed_tables = safe_len(tables)
		mapped_citations, unmapped_citations = collect_citation_counts(sections)
		mapped_citations_total += mapped_citations
		unmapped_citations_total += unmapped_citations

		if isinstance(metadata, dict):
			n_sections = metadata.get("n_sections")
			n_subsections = metadata.get("n_subsections")
			n_tables = metadata.get("n_tables")
		else:
			n_sections = None
			n_subsections = None
			n_tables = None

		if not isinstance(n_sections, int):
			n_sections = computed_sections
			metadata_fallbacks += 1

		if not isinstance(n_subsections, int):
			n_subsections = computed_subsections
			metadata_fallbacks += 1

		if not isinstance(n_tables, int):
			n_tables = computed_tables
			metadata_fallbacks += 1

		n_sections_per_file.append(n_sections)
		n_subsections_per_file.append(n_subsections)
		n_tables_per_file.append(n_tables)

	return {
		"n_documents": len(files),
		"sections_per_document": summarize_numeric(n_sections_per_file),
		"subsections_per_document": summarize_numeric(n_subsections_per_file),
		"tables_per_document": summarize_numeric(n_tables_per_file),
		"extra": {
			"total_sections": sum(n_sections_per_file),
			"total_subsections": sum(n_subsections_per_file),
			"total_tables": sum(n_tables_per_file),
			"total_mapped_citations": mapped_citations_total,
			"total_unmapped_citations": unmapped_citations_total,
			"metadata_field_fallback_uses": metadata_fallbacks,
		},
	}


def format_number(value: Any) -> str:
	if value is None:
		return "n/a"
	if isinstance(value, float):
		return f"{value:.2f}"
	return str(value)


def render_report(stats: Dict[str, Any], refs_path: Path, structured_dir: Path) -> str:
	refs_stats = stats["references"]
	structured_stats = stats["structured"]

	refs_summary = refs_stats["references_per_sr"]
	section_summary = structured_stats["sections_per_document"]
	subsection_summary = structured_stats["subsections_per_document"]
	table_summary = structured_stats["tables_per_document"]

	lines = [
		"=== Metadata Stats Report ===",
		f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
		f"References file: {refs_path}",
		f"Structured dir: {structured_dir}",
		"",
		"[References per SR]",
		f"SR documents: {refs_stats['n_srs']}",
		f"Mean: {format_number(refs_summary['mean'])}",
		f"Median: {format_number(refs_summary['median'])}",
		f"Min: {format_number(refs_summary['min'])}",
		f"Max: {format_number(refs_summary['max'])}",
		f"Total references: {refs_stats['extra']['total_references']}",
		"",
		"[Sections per Structured JSON]",
		f"Structured documents: {structured_stats['n_documents']}",
		f"Mean: {format_number(section_summary['mean'])}",
		f"Median: {format_number(section_summary['median'])}",
		f"Min: {format_number(section_summary['min'])}",
		f"Max: {format_number(section_summary['max'])}",
		f"Total sections: {structured_stats['extra']['total_sections']}",
		"",
		"[Subsections per Structured JSON]",
		f"Mean: {format_number(subsection_summary['mean'])}",
		f"Median: {format_number(subsection_summary['median'])}",
		f"Min: {format_number(subsection_summary['min'])}",
		f"Max: {format_number(subsection_summary['max'])}",
		f"Total subsections: {structured_stats['extra']['total_subsections']}",
		"",
		"[Tables per Structured JSON]",
		f"Mean: {format_number(table_summary['mean'])}",
		f"Median: {format_number(table_summary['median'])}",
		f"Min: {format_number(table_summary['min'])}",
		f"Max: {format_number(table_summary['max'])}",
		f"Total tables: {structured_stats['extra']['total_tables']}",
		"",
		"[Citations in Structured JSON]",
		f"Mapped citations: {structured_stats['extra']['total_mapped_citations']}",
		f"Unmapped citations: {structured_stats['extra']['total_unmapped_citations']}",
		"",
		"[Additional Useful Counts]",
		f"Matched references: {refs_stats['extra']['matched_references']}",
		f"Unmatched references: {refs_stats['extra']['unmatched_references']}",
		f"Needs-review references: {refs_stats['extra']['needs_review_references']}",
		(
			"Metadata fallback uses (missing n_sections/n_subsections/n_tables): "
			f"{structured_stats['extra']['metadata_field_fallback_uses']}"
		),
	]
	return "\n".join(lines) + "\n"


def write_report(log_path: Path, report: str) -> None:
	log_path.parent.mkdir(parents=True, exist_ok=True)
	with log_path.open("w", encoding="utf-8") as handle:
		handle.write(report)


def main() -> None:
	args = parse_args()

	if not args.refs_path.exists():
		raise FileNotFoundError(f"References JSONL file not found: {args.refs_path}")
	if not args.structured_dir.exists():
		raise FileNotFoundError(f"Structured JSON directory not found: {args.structured_dir}")

	stats = {
		"references": compute_reference_stats(args.refs_path),
		"structured": compute_structured_stats(args.structured_dir),
	}

	report = render_report(stats, args.refs_path, args.structured_dir)
	write_report(args.log_path, report)
	print(report)
	print(f"Report written to: {args.log_path}")


if __name__ == "__main__":
	main()
