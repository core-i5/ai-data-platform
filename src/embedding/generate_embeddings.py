import sys
import pandas as pd
import faiss
import numpy as np
import time
from .chunking import create_chunks

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

from dotenv import load_dotenv
load_dotenv()


def load_data(file_path: str):
    return pd.read_parquet(file_path)


def generate_embeddings(texts, batch_size=100, max_retries=3):
    """
    Generate embeddings using SentenceTransformers.

    Improvements:
    - Batching (memory control)
    - Retry logic (robustness)
    """

    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        for attempt in range(max_retries):
            try:
                emb = model.encode(
                    batch,
                    show_progress_bar=False
                )
                all_embeddings.extend(emb)
                break

            except Exception as e:
                print(f"Retry {attempt+1} failed: {e}")
                time.sleep(1)

                if attempt == max_retries - 1:
                    raise e

    embeddings = np.array(all_embeddings).astype("float32")

    print("Embeddings shape:", embeddings.shape)

    return embeddings


def create_faiss_index(embeddings: np.ndarray, use_ivf=False):
    """
    Create FAISS index.

    Improvements:
    - Vector normalization (for cosine similarity)
    - Option to use IVF (scalable)
    """

    # normalize vectors (cosine similarity)
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    if use_ivf:
        # scalable index
        nlist = 100  # number of clusters
        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)

        index.train(embeddings)
        index.add(embeddings)

    else:
        # use INNER PRODUCT (cosine)
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)

    return index


def embeddings(input_path: str, index_path: str, metadata_path: str):

    df = load_data(input_path)

    # Step 1: Chunking
    chunked_df = create_chunks(df)
    print(f"Created {len(chunked_df)} chunks")

    # Step 2: Embeddings
    embeddings = generate_embeddings(chunked_df["chunk_text"].tolist())

    # Step 3: FAISS index
    index = create_faiss_index(embeddings, use_ivf=False)

    # Save index
    faiss.write_index(index, index_path)

    # Save metadata
    chunked_df.to_parquet(metadata_path, index=False)

    print("FAISS index and metadata saved")