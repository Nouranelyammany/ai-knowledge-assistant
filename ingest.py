from pathlib import Path
import pymupdf

# Project folders
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"


def extract_text_from_pdf(pdf_path: Path) -> list[dict]:
    """Extract text from each page of a PDF."""
    pages = []

    try:
        with pymupdf.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text("text").strip()

                if text:
                    pages.append({
                        "source": pdf_path.name,
                        "page": page_number,
                        "text": text,
                    })

    except Exception as error:
        print(f"Error reading {pdf_path.name}: {error}")

    return pages


def main():
    pdf_files = sorted(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in: {DATA_DIR}")
        return

    all_pages = []

    for pdf_path in pdf_files:
        print(f"\nReading: {pdf_path.name}")

        pages = extract_text_from_pdf(pdf_path)
        all_pages.extend(pages)

        print(f"Pages with extracted text: {len(pages)}")

        if pages:
            print("\nText preview:")
            print(pages[0]["text"][:500])
        else:
            print("No extractable text found.")

    print("\n--- Extraction Summary ---")
    print(f"PDF files found: {len(pdf_files)}")
    print(f"Pages with extracted text: {len(all_pages)}")
    print(f"Total characters extracted: {sum(len(p['text']) for p in all_pages)}")


if __name__ == "__main__":
    main()
