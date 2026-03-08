"""
Enrich OpenAlex reference IDs with metadata from the OpenAlex API.

Input parquet must include a `reference_id` column with values like:
- https://openalex.org/W123456789
- W123456789

Output is a parquet with one row per input id and selected metadata fields.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq
import requests
from tqdm import tqdm


INPUT_PATH = "data/refs/all_reference_ids.parquet"
OUTPUT_PATH = "data/refs/all_reference_ids_enriched.parquet"
ENV_PATH = Path("/home/fhg/pie65738/pie65738/projects/rag4repo/.env")
LOG_PATH = Path("logs/get_oax_refs.log")

OPENALEX_BASE_URL = "https://api.openalex.org"
REQUEST_TIMEOUT_SECONDS = 30
REQUEST_RETRIES = 3
REQUEST_SLEEP_SECONDS = 0.12
PROGRESS_EVERY = 100
MAX_RECORDS = None
INCLUDE_SOURCE_PAYLOAD = False


def setup_logging(log_path: Path) -> logging.Logger:
	log_path.parent.mkdir(parents=True, exist_ok=True)
	logger = logging.getLogger("get_oax_refs")
	logger.setLevel(logging.INFO)
	logger.handlers.clear()
	handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
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


def get_openalex_api_key(env_path: Path) -> Optional[str]:
	file_vars = load_env_file(env_path)
	return (
		os.getenv("OPENALEX_API_KEY")
		or file_vars.get("OPENALEX_API_KEY")
		or os.getenv("OPENALEX_KEY")
		or file_vars.get("OPENALEX_KEY")
	)


def load_reference_ids(parquet_path: str) -> List[str]:
	table = pq.read_table(parquet_path, columns=["reference_id"])
	rows = table.column("reference_id").to_pylist()
	return [row for row in rows if isinstance(row, str) and row.strip()]


def extract_work_id(reference_id: str) -> str:
	reference_id = reference_id.strip()
	if "/" in reference_id:
		return reference_id.rsplit("/", 1)[-1]
	return reference_id


def parse_abstract(inverted_index: Optional[Dict[str, List[int]]]) -> Optional[str]:
	if not inverted_index:
		return None

	max_pos = -1
	for positions in inverted_index.values():
		if positions:
			max_pos = max(max_pos, max(positions))

	if max_pos < 0:
		return None

	words = [""] * (max_pos + 1)
	for token, positions in inverted_index.items():
		for pos in positions:
			if 0 <= pos < len(words) and not words[pos]:
				words[pos] = token

	text = " ".join(word for word in words if word)
	return text.strip() or None


def get_with_retry(
	session: requests.Session,
	url: str,
	params: Optional[Dict[str, str]] = None,
) -> Optional[Dict[str, Any]]:
	for attempt in range(1, REQUEST_RETRIES + 1):
		try:
			response = session.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
			if response.status_code == 200:
				return response.json()
			if response.status_code == 404:
				return None
		except requests.RequestException:
			pass

		if attempt < REQUEST_RETRIES:
			time.sleep(1.0 * attempt)

	return None


def extract_authors(authorships: Any) -> List[str]:
	if not isinstance(authorships, list):
		return []
	authors: List[str] = []
	for item in authorships:
		if not isinstance(item, dict):
			continue
		author = item.get("author") or {}
		name = author.get("display_name")
		if isinstance(name, str) and name:
			authors.append(name)
	return authors


def extract_concepts(concepts: Any, max_items: int = 10) -> List[str]:
	if not isinstance(concepts, list):
		return []
	labels: List[str] = []
	for item in concepts[:max_items]:
		if not isinstance(item, dict):
			continue
		name = item.get("display_name")
		if isinstance(name, str) and name:
			labels.append(name)
	return labels


def enrich_one(reference_id: str, payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
	now = datetime.now(timezone.utc).isoformat()
	work_id = extract_work_id(reference_id)

	if payload is None:
		return {
			"reference_id": reference_id,
			"work_id": work_id,
			"title": None,
			"publication_year": None,
			"publication_date": None,
			"work_type": None,
			"language": None,
			"doi": None,
			"is_open_access": None,
			"open_access_status": None,
			"best_oa_url": None,
			"journal": None,
			"journal_type": None,
			"publisher": None,
			"authors": [],
			"authors_count": 0,
			"abstract": None,
			"has_abstract": False,
			"has_fulltext": None,
			"best_pdf_url": None,
			"cited_by_count": None,
			"referenced_works_count": None,
			"concepts": [],
			"primary_topic": None,
			"primary_field": None,
			"source_payload": None,
			"fetch_status": "missing_or_failed",
			"fetched_at_utc": now,
		}

	open_access = payload.get("open_access") or {}
	primary_location = payload.get("primary_location") or {}
	source = primary_location.get("source") or {}
	authors = extract_authors(payload.get("authorships"))
	abstract = parse_abstract(payload.get("abstract_inverted_index"))
	best_oa_location = payload.get("best_oa_location") or {}
	primary_topic = payload.get("primary_topic") or {}
	primary_field = (primary_topic.get("field") or {}).get("display_name")

	return {
		"reference_id": reference_id,
		"work_id": work_id,
		"title": payload.get("display_name"),
		"publication_year": payload.get("publication_year"),
		"publication_date": payload.get("publication_date"),
		"work_type": payload.get("type"),
		"language": payload.get("language"),
		"doi": payload.get("doi"),
		"is_open_access": open_access.get("is_oa"),
		"open_access_status": open_access.get("oa_status"),
		"best_oa_url": open_access.get("oa_url"),
		"journal": source.get("display_name"),
		"journal_type": source.get("type"),
		"publisher": source.get("host_organization_name"),
		"authors": authors,
		"authors_count": len(authors),
		"abstract": abstract,
		"has_abstract": bool(abstract),
		"has_fulltext": payload.get("has_fulltext"),
		"best_pdf_url": best_oa_location.get("pdf_url"),
		"cited_by_count": payload.get("cited_by_count"),
		"referenced_works_count": payload.get("referenced_works_count"),
		"concepts": extract_concepts(payload.get("concepts")),
		"primary_topic": primary_topic.get("display_name"),
		"primary_field": primary_field,
		"source_payload": json.dumps(payload, ensure_ascii=False) if INCLUDE_SOURCE_PAYLOAD else None,
		"fetch_status": "ok",
		"fetched_at_utc": now,
	}


def enrich_references(
	reference_ids: List[str],
	api_key: Optional[str],
	logger: logging.Logger,
) -> List[Dict[str, Any]]:
	session = requests.Session()
	rows: List[Dict[str, Any]] = []
	total = len(reference_ids)
	params = {"api_key": api_key} if api_key else None

	for idx, reference_id in enumerate(tqdm(reference_ids, desc="OpenAlex fetch", unit="ref"), start=1):
		work_id = extract_work_id(reference_id)
		url = f"{OPENALEX_BASE_URL}/works/{work_id}"
		payload = get_with_retry(session, url, params=params)
		rows.append(enrich_one(reference_id, payload))

		if idx % PROGRESS_EVERY == 0 or idx == total:
			logger.info("Processed %s/%s references", idx, total)

		time.sleep(REQUEST_SLEEP_SECONDS)

	return rows


def write_enriched_parquet(rows: List[Dict[str, Any]], output_path: str) -> None:
	table = pa.Table.from_pylist(rows)
	pq.write_table(table, output_path)


def main() -> None:
	logger = setup_logging(LOG_PATH)
	logger.info("Starting OpenAlex enrichment")
	logger.info("Input path: %s", INPUT_PATH)
	logger.info("Output path: %s", OUTPUT_PATH)
	logger.info("Using env file: %s", ENV_PATH)

	api_key = get_openalex_api_key(ENV_PATH)
	logger.info("OpenAlex API key %s", "loaded" if api_key else "not found")

	reference_ids = load_reference_ids(INPUT_PATH)
	if isinstance(MAX_RECORDS, int) and MAX_RECORDS > 0:
		reference_ids = reference_ids[:MAX_RECORDS]
	print(f"Loaded reference ids: {len(reference_ids)}")
	logger.info("Loaded reference ids: %s", len(reference_ids))

	enriched_rows = enrich_references(reference_ids, api_key, logger)
	write_enriched_parquet(enriched_rows, OUTPUT_PATH)
	logger.info("Wrote enriched parquet: %s", OUTPUT_PATH)

	ok_count = sum(1 for row in enriched_rows if row.get("fetch_status") == "ok")
	print(f"Enriched rows written: {len(enriched_rows)}")
	print(f"Successful fetches: {ok_count}")
	print(f"Output: {OUTPUT_PATH}")
	logger.info("Enriched rows written: %s", len(enriched_rows))
	logger.info("Successful fetches: %s", ok_count)
	logger.info("Finished OpenAlex enrichment")


if __name__ == "__main__":
	main()
