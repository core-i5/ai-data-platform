import os
from dotenv import load_dotenv
import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

load_dotenv()


def load_faiss_index(index_path):
    """
    Load FAISS index from disk
    """
    return faiss.read_index(index_path)


def load_metadata(metadata_path):
    """
    Load metadata (chunk text mapping)
    """
    return pd.read_parquet(metadata_path)


def normalize_vectors(vectors):
    """
    Normalize vectors for cosine similarity

    WHY:
    - Prevent division by zero
    - Ensure stable cosine similarity
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    return vectors / norms


def search(query, index, metadata, bi_encoder, cross_encoder, top_k=5):
    """
    Core retrieval with reranking

    Steps:
    1. Query → embedding (bi-encoder)
    2. FAISS retrieval (fast, approximate)
    3. Cross-encoder reranking (accurate)
    """

    # Step 1: Query embedding (bi-encoder)
    query_vector = bi_encoder.encode([query])
    query_vector = normalize_vectors(query_vector).astype("float32")

    # Step 2: FAISS search (over-fetch)
    distances, indices = index.search(query_vector, top_k * 5)

    candidates = []

    for i, idx in enumerate(indices[0]):
        if idx == -1:
            continue

        candidates.append({
            "faiss_score": float(distances[0][i]),
            "text": metadata.iloc[idx]["chunk_text"],
            "chunk_id": int(metadata.iloc[idx]["chunk_id"]),
            "Id": int(metadata.iloc[idx]["Id"])
        })

    if not candidates:
        return []

    # Step 3: Cross-encoder reranking
    # Prepare (query, text) pairs
    pairs = [(query, c["text"]) for c in candidates]

    # real semantic scoring
    ce_scores = cross_encoder.predict(pairs)

    for i, c in enumerate(candidates):
        c["cross_score"] = float(ce_scores[i])

    # Sort by cross-encoder score
    candidates.sort(key=lambda x: x["cross_score"], reverse=True)

    return candidates[:top_k]


def main():
    index_path = os.getenv("INDEX_PATH")
    metadata_path = os.getenv("META_DATA_PATH")

    # Load FAISS + metadata
    index = load_faiss_index(index_path)
    metadata = load_metadata(metadata_path)

    # 🔹 Bi-encoder (for FAISS retrieval)
    bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")

    # Cross-encoder (for reranking)
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    print("Models loaded: bi-encoder + cross-encoder")

    while True:
        query = input("\n Enter query (or 'exit'): ")

        if query.lower() == "exit":
            break

        results = search(query, index, metadata, bi_encoder, cross_encoder)

        print("\n Top Results:\n")
        for r in results:
            print(f"Cross Score: {r['cross_score']:.4f}")
            print(f"FAISS Score: {r['faiss_score']:.4f}")
            print(f"Text: {r['text'][:200]}...")
            print("-" * 50)


if __name__ == "__main__":
    main()