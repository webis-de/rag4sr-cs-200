import json
from pathlib import Path


def main() -> None:
    src = Path("data/refs/sr_references_aligned_openalex.jsonl")
    out = Path("data/refs/sr_references_aligned_openalex.slim.jsonl")

    with src.open("r", encoding="utf-8") as fin, out.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            refs = []
            for ref in obj.get("references", []):
                refs.append(
                    {
                        "number": ref.get("number"),
                        "title": ref.get("title"),
                        "authors": ref.get("authors"),
                        "year": ref.get("year"),
                        "doi": ref.get("doi"),
                        "openalex_id": ref.get("openalex_id"),
                    }
                )
            slim = {
                "id": obj.get("id"),
                "references": refs,
            }
            fout.write(json.dumps(slim, ensure_ascii=False) + "\n")

    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
