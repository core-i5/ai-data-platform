import sys
import pandas as pd


def load_csv(file_path: str) -> pd.DataFrame:
    """Load CSV into pandas DataFrame"""
    try:
        df = pd.read_csv(file_path)
        print(f"\n File loaded successfully: {file_path}")
        return df
    except Exception as e:
        print(f"\n Failed to load file: {e}")
        sys.exit(1)


def print_schema(df: pd.DataFrame):
    """Print schema (column names + data types)"""
    print("\n Schema:")
    print(df.dtypes)

    print("\n Columns:")
    print(list(df.columns))


def validate_data(df: pd.DataFrame):
    """Basic validation checks"""
    print("\n Running basic validation...")

    # Check if empty
    if df.empty:
        print("DataFrame is empty")
        sys.exit(1)

    # Check null values
    null_counts = df.isnull().sum()
    print("\n Null Values per column:")
    print(null_counts)

    # Check duplicates
    duplicates = df.duplicated().sum()
    print(f"\n Duplicate rows: {duplicates}")

    print("\n Basic validation completed")


def main():
    file_path = "data/raw/Reviews.csv"

    df = load_csv(file_path)
    print_schema(df)
    validate_data(df)


if __name__ == "__main__":
    main()