import pandas as pd
import re


def clean_text(text: str) -> str:
    """
    Basic text cleaning.

    WHY:
    - Remove HTML noise (<br />, etc.)
    - Normalize whitespace
    - Improve embedding quality
    """
    if not isinstance(text, str):
        return ""

    text = re.sub(r"<.*?>", " ", text)  # remove HTML tags
    text = re.sub(r"\s+", " ", text).strip()  # normalize spaces
    return text


def chunk_text(text, chunk_size=500, overlap=100):
    """
    Split text into overlapping chunks.

    WHY:
    - Prevent context loss
    - Maintain semantic continuity
    - Cleaner input improves embedding quality
    """

    text = clean_text(text)  

    if not text:
        return []  

    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]

        if len(chunk) < 20:
            break

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
        text = row.get("Text", "")  
        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            records.append({
                "Id": row["Id"],
                "chunk_id": i,
                "chunk_text": chunk
            })

    return pd.DataFrame(records)