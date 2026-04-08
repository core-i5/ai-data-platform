import os
from dotenv import load_dotenv
from src.ingestion.load_data import load_csv, print_schema, validate_data
from src.processing.clean_data import enforce_schema, clean_data, cast_types, feature_engineering, save_parquet

load_dotenv()

input_path = os.getenv("INPUT_FILE_PATH")
output_path = os.getenv("OUTPUT_FILE_PATH")
def main():


    df = load_csv(input_path)
    print_schema(df)
    validate_data(df)
    df = enforce_schema(df)
    df = clean_data(df)
    df = cast_types(df)
    df = feature_engineering(df)
    save_parquet(df, output_path)


if __name__ == "__main__":
    main()