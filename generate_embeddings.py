# generate_embeddings.py

import pandas as pd
from sentence_transformers import SentenceTransformer
import json
import os
from tqdm import tqdm  # progress bar

# -------- CONFIG --------
CSV_FILES = ["data\lahore_house_listings_zameen.csv", "data\zameen-property-data.csv"]  # Replace with your CSV file names
MODEL_NAME = "all-MiniLM-L6-v2"  # local, fast, multilingual
OUTPUT_FILE = "embeddings_output.json"
BATCH_SIZE = 100  # progress update interval
# ------------------------

# Load embedding model
print("🚀 Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)
print(f"✅ Model {MODEL_NAME} loaded.")

def get_price_bucket(price):
    """Optional: Bucket price column if it exists."""
    try:
        price = float(price)
        if price < 50000: return "<50k"
        if price < 500000: return "50k-5L"
        if price < 5000000: return "5L-50L"
        return ">50L"
    except:
        return "unknown"

def process_csv(file_path):
    """Reads CSV, generates embeddings, clusters per column."""
    df = pd.read_csv(file_path)
    all_fields = df.columns.tolist()
    print(f"\n📊 Processing '{file_path}' with columns: {all_fields}")
    results = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating embeddings"):
        row_dict = row.to_dict()
        text = " | ".join([f"{col}: {str(row_dict[col])}" for col in all_fields])

        # Generate embedding
        embedding = model.encode(text).tolist()

        # Cluster per column
        cluster = {}
        for col in all_fields:
            val = row_dict[col]
            if col.lower() == "price":
                cluster[col] = get_price_bucket(val)
            else:
                cluster[col] = str(val) if pd.notna(val) else "unknown"

        # Save the record
        record = {
            "data": row_dict,
            "cluster": cluster,
            "embedding": embedding
        }
        results.append(record)

    return results

if __name__ == "__main__":
    all_results = []
    for csv_file in CSV_FILES:
        if os.path.exists(csv_file):
            records = process_csv(csv_file)
            all_results.extend(records)
        else:
            print(f"⚠️ File '{csv_file}' not found. Skipping.")

    # Save all embeddings to JSON
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 Embeddings generation complete! Saved {len(all_results)} records to '{OUTPUT_FILE}'")
