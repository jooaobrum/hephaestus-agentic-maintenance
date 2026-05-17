"""Extract and enrich procedure chunks from OCR JSON files.

Pipeline:
  1. Read OCR JSON files from --json-dir (output of parse_procedures_pdf.py)
  2. Split each document into semantic sections (H1/H2/H3)
  3. Generate a 2-3 sentence LLM context per chunk (gpt-4.1-nano)
  4. Save enriched chunks to CSV

Usage:
    uv run python scripts/ingestion/extract_procedures.py \\
        [--json-dir data/procedures_extracted] \\
        [--output-csv data/procedure_chunks.csv]
"""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field

CONTEXT_MODEL = "gpt-4.1-nano"
SECTION_PATTERN = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)

CONTEXT_SYSTEM_PROMPT = """You are a technical document analyst for industrial maintenance procedures.

Given the FULL DOCUMENT (for context) and a SPECIFIC CHUNK extracted from it, write a short 2-3 sentence context summary that situates the chunk within its source document.

Your summary must include:
- The machine ID and machine type (e.g. "HX-200, a 200-ton hydraulic press")
- The document purpose (troubleshooting procedures)
- What this specific section covers (e.g. fault code E-001 diagnostic steps, PPE requirements, emergency procedures)
- Any severity level if applicable (CRITICAL, WARNING, INFO)

Write concisely. This context will be prepended to the chunk text before embedding for search retrieval."""


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    chunk_number: int
    file_name: str
    section_title: str
    text: str
    page_numbers: list[int]
    contains_table: bool
    contains_image: bool
    image_paths: list[str]
    prev_chunk: int | None = None
    next_chunk: int | None = None
    context: str = ""


def _build_table_lookup(pages: list[dict]) -> dict[str, str]:
    lookup = {}
    for page in pages:
        for tbl in page.get("tables", []):
            lookup[tbl["id"]] = tbl["content"]
    return lookup


def _inline_tables(text: str, table_lookup: dict[str, str]) -> tuple[str, bool]:
    has_table = False
    for tbl_id, tbl_content in table_lookup.items():
        pattern = re.compile(rf"\[{re.escape(tbl_id)}\]\({re.escape(tbl_id)}\)")
        if pattern.search(text):
            has_table = True
            text = pattern.sub(tbl_content, text)
    return text, has_table


def _collect_image_paths(pages: list[dict]) -> dict[int, list[str]]:
    images = {}
    for page in pages:
        idx = page.get("index", 0)
        images[idx] = [img.get("id", "") for img in page.get("images", [])]
    return images


def _strip_header_footer(markdown: str) -> str:
    return "\n".join(markdown.splitlines()).strip()


def chunk_document(ocr_data: dict, file_name: str) -> list[Chunk]:
    pages = ocr_data.get("pages", [])
    table_lookup = _build_table_lookup(pages)
    images_by_page = _collect_image_paths(pages)

    full_markdown = "\n\n".join(_strip_header_footer(p["markdown"]) for p in pages)
    matches = list(SECTION_PATTERN.finditer(full_markdown))

    chunks = []
    for i, m in enumerate(matches):
        title = m.group(2).strip()
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_markdown)
        section_text = full_markdown[start:end].strip()
        section_text, has_table = _inline_tables(section_text, table_lookup)

        page_numbers: list[int] = []
        offset = 0
        for page in pages:
            page_md = _strip_header_footer(page["markdown"])
            if full_markdown[start:end] in full_markdown[offset: offset + len(page_md) + 10]:
                page_numbers.append(page.get("index", 0))
            offset += len(page_md) + 2

        has_image = any(img for imgs in images_by_page.values() for img in imgs)

        chunks.append(
            Chunk(
                chunk_number=i,
                file_name=file_name,
                section_title=title,
                text=section_text,
                page_numbers=page_numbers or [0],
                contains_table=has_table,
                contains_image=has_image,
                image_paths=[],
            )
        )

    for i, chunk in enumerate(chunks):
        chunk.prev_chunk = chunks[i - 1].chunk_number if i > 0 else None
        chunk.next_chunk = chunks[i + 1].chunk_number if i < len(chunks) - 1 else None

    return chunks


# ---------------------------------------------------------------------------
# LLM Context Enrichment
# ---------------------------------------------------------------------------

class ChunkContext(BaseModel):
    context: str = Field(
        description=(
            "A 2-3 sentence summary that situates this chunk within the full document. "
            "Include: machine ID, machine type, document title, and what this specific "
            "section covers (fault code, procedure type, PPE, emergency, etc.). "
            "Write as if prepending this to the chunk for a search index."
        )
    )


def generate_context(chunk: Chunk, full_doc_text: str, oai: OpenAI) -> str:
    user_prompt = (
        f"DOCUMENT FILE: {chunk.file_name}\n\nFULL DOCUMENT:\n{full_doc_text[:3000]}"
        f"\n\n---\n\nCHUNK TO CONTEXTUALIZE:\n{chunk.text[:2000]}"
    )
    resp = oai.beta.chat.completions.parse(
        model=CONTEXT_MODEL,
        messages=[
            {"role": "system", "content": CONTEXT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format=ChunkContext,
        temperature=0.0,
    )
    return resp.choices[0].message.parsed.context


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def extract_procedures(json_dir: Path, output_csv: Path) -> None:
    if not json_dir.exists():
        raise FileNotFoundError(f"JSON directory not found: {json_dir}")

    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {json_dir}.")
        return

    oai = OpenAI()
    all_chunks: list[Chunk] = []

    for json_path in json_files:
        print(f"\nProcessing {json_path.name} ...")
        ocr_data = json.loads(json_path.read_text())
        chunks = chunk_document(ocr_data, json_path.stem)

        full_doc_text = "\n\n".join(p.get("markdown", "") for p in ocr_data.get("pages", []))

        for j, chunk in enumerate(chunks):
            chunk.context = generate_context(chunk, full_doc_text, oai)
            print(f"  [{j + 1}/{len(chunks)}] context generated for: {chunk.section_title[:60]}")

        all_chunks.extend(chunks)

    print(f"\nSaving {len(all_chunks)} chunks to CSV ...")
    df = pd.DataFrame(
        [
            {
                "chunk_number": c.chunk_number,
                "file_name": c.file_name,
                "section_title": c.section_title,
                "context": c.context,
                "text": c.text,
                "page_numbers": c.page_numbers,
                "contains_table": c.contains_table,
                "contains_image": c.contains_image,
                "image_paths": c.image_paths,
                "prev_chunk": c.prev_chunk,
                "next_chunk": c.next_chunk,
            }
            for c in all_chunks
        ]
    )
    df.to_csv(output_csv, index=False)
    print(f"Done — {len(all_chunks)} chunks saved to {output_csv}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and enrich chunks from procedure OCR JSONs.")
    parser.add_argument("--json-dir", type=Path, default=Path("data/procedures_extracted"))
    parser.add_argument("--output-csv", type=Path, default=Path("data/procedure_chunks.csv"))
    args = parser.parse_args()

    extract_procedures(args.json_dir, args.output_csv)


if __name__ == "__main__":
    main()

