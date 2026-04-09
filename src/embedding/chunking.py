import pandas as pd


def chunk_text(text, chunk_size=500, overlap=100):
    """
    Split text into overlapping chunks.

    Why:
    - Prevent context loss
    - Maintain semantic continuity
    """

    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]
        chunks.append(" ".join(chunk))

        start += chunk_size - overlap

    return chunks


def create_chunks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply chunking to DataFrame.

    Returns:
    - One row per chunk
    """

    records = []

    for _, row in df.iterrows():
        text = row["Text"]

        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            records.append({
                "Id": row["Id"],
                "chunk_id": i,
                "chunk_text": chunk
            })

    return pd.DataFrame(records)