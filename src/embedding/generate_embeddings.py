import sys
import pandas as pd
import faiss
import numpy as np
# from openai import OpenAI
from .chunking import create_chunks

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

from dotenv import load_dotenv
load_dotenv()

# client = OpenAI()

def load_data(file_path: str):
    return pd.read_parquet(file_path)

# def generate_embeddings(texts, batch_size=100):
#     """
#     Generate embeddings using OpenAI.

#     Model:
#     text-embedding-3-small (cheap + good)
#     """

#     embeddings = []

#     for i in range(0, len(texts), batch_size):
#         batch = texts[i:i+batch_size]

#         response = client.embeddings.create(
#             model="text-embedding-3-small",
#             input=batch
#         )

#         embeddings.extend([item.embedding for item in response.data])

#         print(f"Processed {i + len(batch)} / {len(texts)}")

#     return embeddings


def generate_embeddings(texts, batch_size=100):
    """
    Generate embeddings using SentenceTransformers (local model).

    Model:
    all-MiniLM-L6-v2 (fast + lightweight)
    """

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True
    )
    print(embeddings.shape)

    return embeddings.tolist()


def embeddings(input_path: str, index_path: str, metadata_path: str):

    df = load_data(input_path)

    # Step 1: Chunking
    chunked_df = create_chunks(df)

    print(f"Created {len(chunked_df)} chunks")

    # Step 2: Embeddings
    embeddings = generate_embeddings(chunked_df["chunk_text"].tolist())

    embeddings_np = np.array(embeddings).astype("float32")

    # Step 3: FAISS index
    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings_np)

    # Save index
    faiss.write_index(index, index_path)

    # Save metadata
    chunked_df.to_parquet(metadata_path, index=False)

    print("FAISS index and metadata saved")
