# RAG4SR-CS-200

A data and code repository for the paper:

**A Pipeline to Bootstrap the Evaluation of Retrieval-Augmented Generation for the Automation of Systematic Reviews in Computer Science**

## What this repository contains

This repository provides:

- **Dataset artifacts** for 200 computer science systematic reviews (SRs), centered on structured review text and aligned references.
- **Pipeline scripts** (`src/`) used to build and align references and structured JSON data.

---

## Data folder guide

Current top-level layout:

```text
data/
  raw/
    sr4all_computer_science_subset.jsonl
    md/
      W1505282872.md
      ... (200 markdown files)
  refs/
    sr_references_aligned_openalex_slim.jsonl
  structured/
    W1505282872.json
    W1837512326.json
    ... (200 files total)
    tables/
```

### 1) `data/structured/` (main dataset)

This folder contains one JSON file per SR, named by OpenAlex work ID (e.g., `W1505282872.json`).

Each file is the structured full text of a review, with:

- hierarchical sections/subsections/subsubsections,
- in-text citations mapped to OpenAlex IDs where possible,
- table placeholders and links to table snippets,
- metadata counts.

Typical top-level schema:

```json
{
  "id": "W1505282872",
  "metadata": {
    "title": "...",
    "n_sections": 3,
    "n_subsections": 3,
    "n_subsubsections": 7,
    "n_tables": 19
  },
  "sections": [
    {
      "section_id": "s_1",
      "section_label": "1. Introduction",
      "text": "... [W2168894761, UNMAPPED:3] ...",
      "citations": ["W2168894761", "W220935706"],
      "citations_unmapped": ["3"],
      "tables_in_text": ["tbl_01"],
      "subsections": [...],
      "subsubsections": [...]
    }
  ],
  "tables": [
    {
      "table_id": "tbl_01",
      "placeholder": "{{TABLE:tbl_01}}",
      "source_section_id": "s_2",
      "source_subsection_id": "s_2_1",
      "table_scope": "main_text"
    }
  ],
  "tables_file": "W1505282872_tables.md"
}
```

#### How to read citation fields

- `citations`: list of mapped **OpenAlex Work IDs** used in that text block.
- `citations_unmapped`: original numeric references that could not be confidently mapped.
- In `text`, unresolved references appear as markers like `UNMAPPED:14`.

This makes evaluation explicit: you can distinguish grounded vs unresolved evidence.

### 2) `data/structured/tables/`

Companion markdown files like `W1505282872_tables.md` containing extracted table blocks.

In `data/structured/*.json`, tables are referenced via placeholders such as:

- `{{TABLE:tbl_01}}`

Use `tables` + `tables_file` to resolve placeholders back to their table content.

### 3) `data/refs/sr_references_aligned_openalex_slim.jsonl`

JSONL with one row per SR and a simplified reference list.

Each row contains:

- `id`: SR/OpenAlex ID,
- `references`: array with fields:
  - `number` (original in-review reference number),
  - `title`, `authors`, `year`, `doi`,
  - `openalex_id` (or `null` if not aligned).

Example row structure:

```json
{
  "id": "W1505282872",
  "references": [
    {
      "number": "1",
      "title": "Procedures for performing systematic reviews",
      "authors": ["B. Kitchenham"],
      "year": null,
      "doi": null,
      "openalex_id": "W2168894761"
    }
  ]
}
```

### 4) `data/raw/` (raw inputs used by the pipeline)

This folder contains the pre-structured inputs used to generate the released artifacts.

#### `data/raw/md/`

- One markdown file per SR (`W*.md`), e.g. `W1505282872.md`.
- These files include the original review narrative, numbered in-text citations (`[1]`, `[2]`, ...), and reference lists.
- They are the input for:
  - `src/0_md2json.py` (structure extraction),
  - `src/1_parse_sr_refs.py` (reference parsing).

#### `data/raw/sr4all_computer_science_subset.jsonl` (hint)

This is a JSONL subset of candidate SR works in Computer Science (one work per line), used by `src/2_get_all_ref.py` to collect global cited OpenAlex IDs.

Typical fields include:

- **Bibliographic/core metadata**: `id`, `title`, `doi`, `abstract`, `year`, `type`, `source`, `language`, `field`, `subfield`, `authors`.
- **Citation graph fields**: `referenced_works_count`, `referenced_works` (list of OpenAlex work URLs), `cited_by_count`.
- **Discovery/retrieval helpers**: `topics`, `keywords`, `pdf_url`.
- **SR-specific enrichment fields** (when available): `objective`, `research_questions`, `inclusion_criteria`, `exclusion_criteria`, `n_studies_initial`, `n_studies_final`, `year_range`, `year_range_normalized`, `exact_boolean_queries`, etc.

In short:

- `data/raw/md/` = raw review full text,
- `data/raw/sr4all_computer_science_subset.jsonl` = raw OpenAlex-centered SR metadata + outbound citations,
- `data/structured/` and `data/refs/` = processed/aligned outputs for downstream RAG4SR evaluation.

---

## Running the pipeline (from raw markdown)

> If you only need the released dataset in `data/structured` and `data/refs`, you can skip this section.

### Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install pyarrow requests tqdm
```

Optional tools:

- `anystyle` CLI (used by `4_enrich_sr_references.py` unless disabled)
- OpenAlex API key in environment (optional but recommended)

### Input assumptions used by scripts

Most scripts assume an intermediate folder layout such as:

- `data/raw/md/*.md`
- `data/raw/sr4all_computer_science_subset.jsonl`
- outputs under `data/parsed/` and `data/refs/`

Two scripts (`3_get_oax_refs.py`, `5_align_with_oax.py`) default to an absolute `.env` path. Override with CLI options where available or edit defaults before running.

### Recommended run order

1. Parse markdown into structured JSON and table files

```bash
python src/0_md2json.py \
  --input-dir data/raw/md \
  --output-dir data/parsed/structured \
  --tables-dir data/parsed/structured/tables
```

2. Parse numbered references from markdown

```bash
python src/1_parse_sr_refs.py \
  --input-dir data/raw/md \
  --output data/refs/sr_references.jsonl
```

3. Build global OpenAlex reference ID list from SR corpus

```bash
python src/2_get_all_ref.py
```

4. Enrich global OpenAlex IDs via OpenAlex API

```bash
python src/3_get_oax_refs.py
```

5. Enrich parsed SR references (AnyStyle + Crossref + heuristics)

```bash
python src/4_enrich_sr_references.py \
  --input data/refs/sr_references.jsonl \
  --output data/refs/sr_references_enriched.jsonl
```

(If AnyStyle is not installed)

```bash
python src/4_enrich_sr_references.py \
  --disable-anystyle \
  --input data/refs/sr_references.jsonl \
  --output data/refs/sr_references_enriched.jsonl
```

6. Align enriched SR references with OpenAlex

```bash
python src/5_align_with_oax.py \
  --input-jsonl data/refs/sr_references_enriched.jsonl \
  --openalex-parquet data/refs/all_reference_ids_enriched.parquet \
  --output-jsonl data/refs/sr_references_aligned_openalex.jsonl \
  --profile balanced
```

7. Replace numeric in-text citations with OpenAlex IDs in structured text

```bash
python src/6_align_oax_in_text.py \
  --structured-dir data/parsed/structured \
  --tables-dir data/parsed/structured/tables \
  --output-structured-dir data/parsed/structured_mapped \
  --output-tables-dir data/parsed/structured_mapped/tables \
  --aligned-refs data/refs/sr_references_aligned_openalex.jsonl
```

8. Produce slim aligned references

```bash
python src/7_slim_refs.py
```

9. (Optional) Compute metadata statistics

```bash
python src/metadata_stats.py \
  --refs-path data/refs/sr_references_aligned_openalex.jsonl \
  --structured-dir data/parsed/structured_mapped
```

---

## Script summary

- `src/0_md2json.py`: markdown → structured JSON + extracted tables.
- `src/1_parse_sr_refs.py`: parse numbered reference list from markdown.
- `src/2_get_all_ref.py`: aggregate unique OpenAlex reference IDs from raw SR corpus.
- `src/3_get_oax_refs.py`: fetch OpenAlex metadata for aggregated IDs.
- `src/4_enrich_sr_references.py`: parse/enrich SR reference strings.
- `src/5_align_with_oax.py`: align SR references to OpenAlex IDs.
- `src/6_align_oax_in_text.py`: rewrite in-text numeric citations to OpenAlex IDs.
- `src/7_slim_refs.py`: produce compact reference JSONL.
- `src/metadata_stats.py`: generate dataset statistics report.

---

## Citation

If you use this repository, please cite the paper:

**Achkar, P., Gollub, T., Simons, A., Scells, H., Fröbe, M., Potthast, M.**
*A Pipeline to Bootstrap the Evaluation of Retrieval-Augmented Generation for the Automation of Systematic Reviews in Computer Science.*
