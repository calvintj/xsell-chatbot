from src.retrieval.vector_store import get_raw_pinecone_index


def main():
    index = get_raw_pinecone_index()

    # Fetch all vectors (with a limit of 100 for safety)
    results = index.query(
        vector=[0] * 1536, top_k=100, include_metadata=True  # dummy vector
    )

    # Print each result
    for match in results.matches:
        print("\n--- Document ---")
        print(f"ID: {match.id}")
        print(f"Source: {match.metadata.get('source', 'N/A')}")
        print(f"Text: {match.metadata.get('text', 'N/A')[:200]}...")  # First 200 chars
        print(f"Score: {match.score}")


if __name__ == "__main__":
    main()
