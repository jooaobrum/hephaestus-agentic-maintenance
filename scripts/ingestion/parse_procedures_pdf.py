"""Parse procedure PDFs via Mistral OCR and save raw JSON output.

Reads all PDFs from an input directory, runs Mistral OCR on each, and saves
one JSON file per PDF. Skips files that already have output (re-run safe).

Usage:
    uv run python scripts/ingestion/parse_procedures_pdf.py \\
        [--input-dir data/procedures] \\
        [--output-dir data/procedures_extracted]
"""

import argparse
import base64
import json
from pathlib import Path

try:
    from mistralai import Mistral
except ImportError:
    from mistralai.client import Mistral


def encode_pdf(pdf_path: Path) -> str:
    return (
        "data:application/pdf;base64,"
        + base64.b64encode(pdf_path.read_bytes()).decode()
    )


def extract_pdf(client: Mistral, pdf_path: Path) -> dict:
    response = client.ocr.process(
        model="mistral-ocr-latest",
        document={"type": "document_url", "document_url": encode_pdf(pdf_path)},
        include_image_base64=True,
        table_format="markdown",
        extract_header=False,
        extract_footer=False,
    )
    return response.model_dump()


def parse_procedures(input_dir: Path, output_dir: Path) -> None:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted(input_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {input_dir}.")
        return

    client = Mistral()

    for pdf_path in pdf_files:
        out_path = output_dir / (pdf_path.stem + ".json")
        if out_path.exists():
            print(f"Skipping {pdf_path.name} (already extracted).")
            continue

        print(f"Extracting {pdf_path.name} ...")
        result = extract_pdf(client, pdf_path)
        out_path.write_text(json.dumps(result, indent=2))
        pages = len(result.get("pages", []))
        print(f"  Saved {pages} pages → {out_path}")

    print("\nDone.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse procedure PDFs with Mistral OCR."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("data/procedures"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/procedures_extracted")
    )
    args = parser.parse_args()

    parse_procedures(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
