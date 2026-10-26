
from pathlib import Path

import chromadb
import pymupdf
from sentence_transformers import SentenceTransformer

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
DB_DIR = PROJECT_DIR / "chroma_db"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
BATCH_SIZE = 64

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def extract_pages():
    """Extract text from each PDF page and retain its source."""
    pages = []

    for pdf_path in sorted(DATA_DIR.glob("*.pdf")):
        print(f"Reading {pdf_path.name}...")

        with pymupdf.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text("text").strip()

                if text:
                    pages.append({
                        "source": pdf_path.name,
                        "page": page_number,
                        "text": text,
                    })

    return pages


def split_into_chunks(text):
    """Split text into overlapping character-based chunks."""
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - CHUNK_OVERLAP

    return chunks


def main():
    pages = extract_pages()

    if not pages:
        print("No extractable PDF text found.")
        return

    documents = []
    metadatas = []
    ids = []

    for page in pages:
        chunks = split_into_chunks(page["text"])

        for chunk_number, chunk in enumerate(chunks, start=1):
            documents.append(chunk)
            metadatas.append({
                "source": page["source"],
                "page": page["page"],
                "chunk": chunk_number,
            })
            ids.append(
                f"{page['source']}_page{page['page']}_chunk{chunk_number}"
            )

    print(f"\nPages processed: {len(pages)}")
    print(f"Chunks created: {len(documents)}")
    print(f"Loading embedding model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient(path=str(DB_DIR))

    collection = client.get_or_create_collection(
        name="knowledge_base",
        metadata={"hnsw:space": "cosine"},
    )

    for start in range(0, len(documents), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(documents))
        batch_docs = documents[start:end]

        embeddings = model.encode(
            batch_docs,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        collection.upsert(
            ids=ids[start:end],
            documents=batch_docs,
            metadatas=metadatas[start:end],
            embeddings=embeddings.tolist(),
        )

        print(f"Indexed {end}/{len(documents)} chunks")

    print("\n--- Indexing Summary ---")
    print(f"Chunks created: {len(documents)}")
    print(f"Chunks stored in ChromaDB: {collection.count()}")
    print(f"Database location: {DB_DIR}")


if __name__ == "__main__":
    main()
