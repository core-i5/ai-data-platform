import os
from dotenv import load_dotenv
from src.ingestion.load_data import validate
from src.processing.clean_data import cleaning
from src.embedding.generate_embeddings import embeddings

load_dotenv()

raw_data_path = os.getenv("RAW_FILE_PATH")
parquet_file_path = os.getenv("PARQUET_FILE_PATH")
index_path = os.getenv("INDEX_PATH")
meta_data_path = os.getenv("META_DATA_PATH")

def main():

    df = validate(raw_data_path)
    cleaning(df, parquet_file_path)
    embeddings(parquet_file_path, index_path, meta_data_path)


if __name__ == "__main__":
    main()