import sys
import pandas as pd

def enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce strict schema.

    Why:
    - Never trust upstream data
    - Prevent silent failures (wrong types, missing columns)
    """

    required_schema = {
        "Id": "int64",
        "ProductId": "object",
        "UserId": "object",
        "Score": "int64",
        "Time": "int64",
        "Text": "object"
    }

    for col, dtype in required_schema.items():
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

        # Enforce type
        try:
            df[col] = df[col].astype(dtype)
        except Exception:
            raise ValueError(f"Column {col} has incorrect type")

    print("Schema enforcement passed")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleaning + Feature Engineering

    Includes:
    - Controlled null removal
    - Duplicate removal
    - Text normalization
    """

    print("\n Cleaning data...")

    # Drop critical nulls
    df = df.dropna(subset=["Text", "Score", "ProductId"])

    # Remove duplicates
    df = df.drop_duplicates()

    # Normalize text
    df["Text"] = df["Text"].str.lower().str.strip()

    print("Cleaning completed")
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features.

    Why:
    - Raw data is rarely useful directly
    - Features improve downstream ML / search

    Features:
    - review_length: length of text
    - sentiment_label: basic heuristic (not ML yet)
    """

    print("\n Feature engineering...")

    # Review length
    df["review_length"] = df["Text"].apply(len)

    # Simple sentiment heuristic
    df["sentiment"] = df["Score"].apply(
        lambda x: "positive" if x >= 4 else ("negative" if x <= 2 else "neutral")
    )

    print("Feature engineering completed")
    return df


def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Time column to datetime.

    Why:
    - Unix timestamps are useless for analytics
    - Datetime enables time-based queries
    """

    df["Time"] = pd.to_datetime(df["Time"], unit="s")
    print("Converted Time → datetime")
    return df


def save_parquet(df: pd.DataFrame, output_path: str):
    """
    Save as Parquet.

    Why:
    - Columnar storage → faster analytics
    - Compression → smaller size
    - Schema preserved
    """

    try:
        df.to_parquet(output_path, index=False, engine="pyarrow")
        print(f"\n Saved as Parquet: {output_path}")
    except Exception as e:
        print(f"\n Failed to save Parquet: {e}")
        sys.exit(1)


def cleaning(df: pd.DataFrame, output_path: str):
    df = enforce_schema(df)
    df = clean_data(df)
    df = cast_types(df)
    df = feature_engineering(df)
    save_parquet(df, output_path)