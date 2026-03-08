"""Parse markdown files into structured JSON with externalized tables."""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Iterable, List, Tuple


TABLE_PLACEHOLDER_FMT = "{{{{TABLE:{table_id}}}}}"


@dataclass
class TableBlock:
	table_id: str
	placeholder: str
	caption: str | None
	markdown: str


def _is_heading(line: str) -> Tuple[int, str] | None:
	match = re.match(r"^(#{1,6})\s+(.*)\s*$", line)
	if not match:
		return None
	level = len(match.group(1))
	text = match.group(2).strip()
	return level, text


def _is_table_separator(line: str) -> bool:
	if "|" not in line:
		return False
	return bool(re.match(r"^\s*\|?\s*[-:]+(?:\s*\|\s*[-:]+)+\s*\|?\s*$", line))


def _parse_caption_line(line: str) -> str | None:
	stripped = line.strip()
	if not stripped:
		return None

	patterns = [
		r"^\*\*\s*table\s*\d+[\s:.-]*(.*?)\*\*$",
		r"^\*\*\s*table\s*\d+\s*\*\*\s*[:.-]?\s*(.*)$",
		r"^table\s*\d+[\s:.-]+(.*)$",
		r"^\*table\s*\d+[\s:.-]+(.*)\*$",
	]
	for pattern in patterns:
		match = re.match(pattern, stripped, flags=re.IGNORECASE)
		if match:
			caption = match.group(1).strip()
			return caption or stripped
	return None


def _find_table_caption(lines: List[str], table_start: int, table_end: int) -> str | None:
	# Prefer a caption line just above the table.
	for idx in range(table_start - 1, max(-1, table_start - 4), -1):
		if idx < 0:
			break
		candidate = lines[idx].strip()
		if not candidate:
			continue
		caption = _parse_caption_line(lines[idx])
		if caption:
			return caption
		break

	# Fallback: caption line immediately after the table block.
	for idx in range(table_end, min(len(lines), table_end + 3)):
		candidate = lines[idx].strip()
		if not candidate:
			continue
		caption = _parse_caption_line(lines[idx])
		if caption:
			return caption
		break

	return None


def _extract_tables(lines: List[str]) -> Tuple[List[str], List[TableBlock]]:
	out_lines: List[str] = []
	tables: List[TableBlock] = []
	in_code_fence = False
	i = 0
	table_index = 1

	while i < len(lines):
		line = lines[i]
		stripped = line.strip()
		if stripped.startswith("```"):
			in_code_fence = not in_code_fence
			out_lines.append(line)
			i += 1
			continue

		if not in_code_fence and "|" in line and i + 1 < len(lines):
			next_line = lines[i + 1]
			if _is_table_separator(next_line):
				# Capture table block
				table_start = i
				table_lines = [line, next_line]
				i += 2
				while i < len(lines) and "|" in lines[i] and lines[i].strip() != "":
					table_lines.append(lines[i])
					i += 1
				table_end = i
				table_id = f"tbl_{table_index:02d}"
				table_index += 1
				placeholder = TABLE_PLACEHOLDER_FMT.format(table_id=table_id)
				caption = _find_table_caption(lines, table_start, table_end)
				tables.append(
					TableBlock(
						table_id=table_id,
						placeholder=placeholder,
						caption=caption,
						markdown="\n".join(table_lines).strip(),
					)
				)
				out_lines.append(placeholder)
				continue

		out_lines.append(line)
		i += 1

	return out_lines, tables


def _collect_citations(text: str) -> List[str]:
	ordered: List[str] = []
	seen = set()

	def _add(num: str) -> None:
		if num not in seen:
			seen.add(num)
			ordered.append(num)

	# Bracketed lists like [1] or [1, 2]
	for match in re.finditer(r"\[(\d+(?:\s*,\s*\d+)*)\]", text):
		for num in re.split(r"\s*,\s*", match.group(1).strip()):
			if num:
				_add(num)

	# Paren lists like (1) or (1, 2)
	for match in re.finditer(r"\((\d+(?:\s*,\s*\d+)*)\)", text):
		for num in re.split(r"\s*,\s*", match.group(1).strip()):
			if num:
				_add(num)

	return ordered


def _find_placeholders(text: str) -> List[str]:
	return re.findall(r"\{\{TABLE:(tbl_\d+)\}\}", text)


def _normalize_heading(text: str) -> str:
	cleaned = re.sub(r"^[\s\dIVXLCDMivxlcdm.:-]+", "", text).strip()
	return re.sub(r"\s+", " ", cleaned).lower()


def _parse_sections(lines: List[str]) -> Tuple[str, List[dict]]:
	title = ""
	sections: List[dict] = []
	section_index = 0
	subsection_index = 0
	subsubsection_index = 0
	skip_mode = False
	skip_prefixes = ("abstract", "references")
	has_level2_heading = any((_is_heading(line) or (0, ""))[0] == 2 for line in lines)
	section_heading_level = 2 if has_level2_heading else 3
	subsection_heading_level = 3 if has_level2_heading else None
	subsubsection_heading_level = 4

	current_section = None
	current_subsection = None
	current_subsubsection = None
	preface_lines: List[str] = []

	def _flush_subsubsection() -> None:
		nonlocal current_subsubsection
		if current_subsubsection is None:
			return
		text = "\n".join(current_subsubsection["_lines"]).strip()
		current_subsubsection["text"] = text
		current_subsubsection["citations"] = _collect_citations(text)
		current_subsubsection["tables_in_text"] = _find_placeholders(text)
		current_subsubsection.pop("_lines", None)
		if current_subsection is not None:
			current_subsection.setdefault("subsubsections", []).append(current_subsubsection)
		elif current_section is not None:
			current_section.setdefault("subsubsections", []).append(current_subsubsection)
		current_subsubsection = None

	def _flush_subsection() -> None:
		nonlocal current_subsection
		if current_subsection is None:
			return
		_flush_subsubsection()
		text = "\n".join(current_subsection["_lines"]).strip()
		current_subsection["text"] = text
		current_subsection["citations"] = _collect_citations(text)
		current_subsection["tables_in_text"] = _find_placeholders(text)
		current_subsection.pop("_lines", None)
		current_section["subsections"].append(current_subsection)
		current_subsection = None

	def _flush_section() -> None:
		nonlocal current_section
		if current_section is None:
			return
		_flush_subsection()
		text = "\n".join(current_section["_lines"]).strip()
		current_section["text"] = text
		current_section["citations"] = _collect_citations(text)
		current_section["tables_in_text"] = _find_placeholders(text)
		current_section.pop("_lines", None)
		sections.append(current_section)
		current_section = None

	for line in lines:
		heading = _is_heading(line)
		if heading:
			level, text = heading
			if level == 1 and not title:
				title = text
				continue
			if level == section_heading_level:
				_flush_section()
				normalized = _normalize_heading(text)
				skip_mode = normalized.startswith(skip_prefixes)
				if skip_mode:
					current_section = None
					current_subsection = None
					current_subsubsection = None
					preface_lines.clear()
					continue
				section_index += 1
				subsection_index = 0
				subsubsection_index = 0
				current_section = {
					"section_id": f"s_{section_index}",
					"section_index": section_index,
					"section_label": text,
					"_lines": [],
					"subsections": [],
					"subsubsections": [],
				}
				if preface_lines:
					current_section["_lines"].extend(preface_lines)
					preface_lines.clear()
				continue
			if (
				subsection_heading_level is not None
				and level == subsection_heading_level
				and current_section is not None
			):
				_flush_subsection()
				subsection_index += 1
				subsubsection_index = 0
				current_subsection = {
					"subsection_id": f"s_{section_index}_{subsection_index}",
					"subsection_index": subsection_index,
					"subsection_label": text,
					"_lines": [],
					"subsubsections": [],
				}
				continue
			if level == subsubsection_heading_level and current_section is not None:
				_flush_subsubsection()
				subsubsection_index += 1
				current_subsubsection = {
					"subsubsection_id": f"s_{section_index}_{subsection_index}_{subsubsection_index}",
					"subsubsection_index": subsubsection_index,
					"subsubsection_label": text,
					"parent_subsection_id": (
						current_subsection["subsection_id"]
						if current_subsection is not None
						else None
					),
					"_lines": [],
				}
				continue

		if skip_mode:
			continue
		if current_subsubsection is not None:
			current_subsubsection["_lines"].append(line)
		elif current_subsection is not None:
			current_subsection["_lines"].append(line)
		elif current_section is not None:
			current_section["_lines"].append(line)
		else:
			if line.strip():
				if has_level2_heading:
					preface_lines.append(line)
				else:
					# Fallback mode with no ## headings: keep content for first inferred section.
					preface_lines.append(line)

	_flush_section()
	return title, sections


def _build_tables_index(sections: List[dict], tables: List[TableBlock]) -> List[dict]:
	table_to_source = {}

	for section in sections:
		for table_id in section.get("tables_in_text", []):
			table_to_source.setdefault(
				table_id,
				{
					"source_section_id": section["section_id"],
					"source_subsection_id": None,
				},
			)
		for subsubsection in section.get("subsubsections", []):
			for table_id in subsubsection.get("tables_in_text", []):
				table_to_source.setdefault(
					table_id,
					{
						"source_section_id": section["section_id"],
						"source_subsection_id": None,
					},
				)
		for subsection in section.get("subsections", []):
			for table_id in subsection.get("tables_in_text", []):
				table_to_source.setdefault(
					table_id,
					{
						"source_section_id": section["section_id"],
						"source_subsection_id": subsection["subsection_id"],
					},
				)
			for subsubsection in subsection.get("subsubsections", []):
				for table_id in subsubsection.get("tables_in_text", []):
					table_to_source.setdefault(
						table_id,
						{
							"source_section_id": section["section_id"],
							"source_subsection_id": subsection["subsection_id"],
						},
					)

	table_index = []
	for table in tables:
		source_info = table_to_source.get(table.table_id, {})
		source_section_id = source_info.get("source_section_id")
		source_subsection_id = source_info.get("source_subsection_id")
		table_scope = "appendix" if source_section_id is None else "main_text"
		table_index.append(
			{
				"table_id": table.table_id,
				"placeholder": table.placeholder,
				"source_section_id": source_section_id,
				"source_subsection_id": source_subsection_id,
				"table_scope": table_scope,
			}
		)
	return table_index


def _write_tables_file(path: str, tables: Iterable[TableBlock]) -> None:
	with open(path, "w", encoding="utf-8") as handle:
		for table in tables:
			handle.write(f"<!-- {table.table_id} -->\n")
			if table.caption:
				handle.write(f"Caption: {table.caption}\n")
			handle.write(table.markdown)
			handle.write("\n\n")


def _process_file(input_path: str, output_path: str, tables_path: str) -> None:
	with open(input_path, "r", encoding="utf-8") as handle:
		content = handle.read()

	lines = content.splitlines()
	cleaned_lines, tables = _extract_tables(lines)
	title, sections = _parse_sections(cleaned_lines)

	table_index = _build_tables_index(sections, tables)
	doc_id = os.path.splitext(os.path.basename(input_path))[0]

	payload = {
		"id": doc_id,
		"metadata": {
			"title": title,
			"n_sections": len(sections),
			"n_subsections": sum(len(s.get("subsections", [])) for s in sections),
			"n_subsubsections": (
				sum(len(s.get("subsubsections", [])) for s in sections)
				+ sum(
					len(ss.get("subsubsections", []))
					for s in sections
					for ss in s.get("subsections", [])
				)
			),
			"n_tables": len(tables),
		},
		"sections": sections,
		"tables": table_index,
		"tables_file": os.path.basename(tables_path),
	}

	with open(output_path, "w", encoding="utf-8") as handle:
		json.dump(payload, handle, indent=2, ensure_ascii=True)
		handle.write("\n")

	_write_tables_file(tables_path, tables)


def main() -> int:
	parser = argparse.ArgumentParser(description="Parse markdown to structured JSON.")
	parser.add_argument(
		"--input-dir",
		default="data/raw/md",
		help="Directory containing markdown files.",
	)
	parser.add_argument(
		"--output-dir",
		default="data/parsed/structured",
		help="Directory for JSON outputs.",
	)
	parser.add_argument(
		"--tables-dir",
		default="data/parsed/structured/tables",
		help="Directory for markdown tables outputs.",
	)
	args = parser.parse_args()

	os.makedirs(args.output_dir, exist_ok=True)
	os.makedirs(args.tables_dir, exist_ok=True)

	md_files = sorted(
		f for f in os.listdir(args.input_dir) if f.lower().endswith(".md")
	)
	for filename in md_files:
		input_path = os.path.join(args.input_dir, filename)
		doc_id = os.path.splitext(filename)[0]
		output_path = os.path.join(args.output_dir, f"{doc_id}.json")
		tables_path = os.path.join(args.tables_dir, f"{doc_id}_tables.md")
		_process_file(input_path, output_path, tables_path)

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
