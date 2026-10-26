
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

PROJECT_DIR = Path(__file__).resolve().parent
DB_DIR = PROJECT_DIR / "chroma_db"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    # Connect to the existing vector database
    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_collection(name="knowledge_base")

    # Load the same embedding model used during indexing
    model = SentenceTransformer(MODEL_NAME)

    question = input("\nAsk a question about your documents: ").strip()

    if not question:
        print("Please enter a question.")
        return

    # Convert the question into an embedding
    question_embedding = model.encode(
        question,
        normalize_embeddings=True,
    ).tolist()

    # Find the 3 most relevant chunks
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=3,
        include=["documents", "metadatas", "distances"],
    )

    print("\n--- Retrieved Passages ---")

    for i, (document, metadata, distance) in enumerate(
        zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ),
        start=1,
    ):
        print(f"\nResult {i}")
        print(f"Source: {metadata['source']}")
        print(f"Page: {metadata['page']}")
        print(f"Cosine distance: {distance:.4f}")
        print(f"Text: {document[:700]}")

    print("\nRetrieval test completed.")


if __name__ == "__main__":
    main()
