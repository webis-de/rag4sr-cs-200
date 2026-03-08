"""
This script extracts all unique reference ids from the input JSONL file and saves them in a Parquet file. 
The reference ids are collected from the "referenced_works" field in each record. 
The output Parquet file will contain a single column "reference_id" with all unique reference ids sorted alphabetically.
"""
import json
from typing import Iterable, Set

import pyarrow as pa
import pyarrow.parquet as pq

INPUT_PATH = "data/raw/sr4all_computer_science_subset.jsonl"
OUTPUT_PATH = "data/refs/all_reference_ids.parquet"


def iter_reference_ids(jsonl_path: str) -> Iterable[str]:
	with open(jsonl_path, "r", encoding="utf-8") as handle:
		for line in handle:
			line = line.strip()
			if not line:
				continue
			try:
				record = json.loads(line)
			except json.JSONDecodeError:
				continue
			references = record.get("referenced_works") or []
			for ref in references:
				if isinstance(ref, str) and ref:
					yield ref


def collect_unique_references(jsonl_path: str) -> Set[str]:
	return set(iter_reference_ids(jsonl_path))


def write_parquet(reference_ids: Set[str], output_path: str) -> None:
	table = pa.table({"reference_id": sorted(reference_ids)})
	pq.write_table(table, output_path)


def main() -> None:
	reference_ids = collect_unique_references(INPUT_PATH)
	write_parquet(reference_ids, OUTPUT_PATH)
	print(f"Unique reference ids: {len(reference_ids)}")


if __name__ == "__main__":
	main()

