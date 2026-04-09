describe AI-Data-Platform

python version: 3.10.20

Project Structure: 
ai-data-platform/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── src/
│   ├── ingestion/
│   ├── processing/
│   ├── embedding/
│   ├── retrieval/
│   ├── api/
│
├── utils/
├── vector_store/
├── main.py
├── .env
├── requirements.txt

Dataset:
Using amazon product reviews dataset, available on kaggle, [kaggle](https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews?resource=download).
Place it here:
data/raw/Reviews.csv

 Columns:
['Id', 'ProductId', 'UserId', 'ProfileName', 'HelpfulnessNumerator', 'HelpfulnessDenominator', 'Score', 'Time', 'Summary', 'Text']

check schema and perform basic validation (null, empty and duplicate values.)

    Enforce strict schema.

    Why:
    - Never trust upstream data
    - Prevent silent failures (wrong types, missing columns)

    Cleaning + Feature Engineering

    Includes:
    - Controlled null removal
    - Duplicate removal
    - Text normalization

    Create derived features.

    Why:
    - Raw data is rarely useful directly
    - Features improve downstream ML / search

    Features:
    - review_length: length of text
    - sentiment_label: basic heuristic (not ML yet)


Convert Time column to datetime.

    Why:
    - Unix timestamps are useless for analytics
    - Datetime enables time-based queries

Save as Parquet.

    Why:
    - Columnar storage → faster analytics
    - Compression → smaller size
    - Schema preserved

processed data saved at data/processed/Reviews.parquet