import json
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer, util

FAISS_INDEX_FILE = "faiss.index"
METADATA_FILE = "metadata.jsonl"
TOP_K = 5

def load_metadata():
    metadata = []
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            metadata.append(json.loads(line))
    return metadata

def main():
    print("🚀 Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("📥 Loading FAISS index...")
    index = faiss.read_index(FAISS_INDEX_FILE)

    print("🗂 Loading metadata...")
    metadata = load_metadata()

    print("✅ Ready for RAG search!\n")

    while True:
        query = input("💬 Ask something (or 'exit'): ").strip()
        if query.lower() == "exit":
            break

        print("\n🔎 Processing query...")
        query_emb = model.encode(query, convert_to_tensor=False).astype("float32")
        query_emb = np.expand_dims(query_emb, axis=0)

        # normalize for cosine similarity
        faiss.normalize_L2(query_emb)

        # search top matches
        distances, indices = index.search(query_emb, TOP_K)

        print("\n📌 Top Results:\n")
        for rank, idx in enumerate(indices[0]):
            score = float(distances[0][rank])
            doc = metadata[idx]["data"]

            print(f"#{rank + 1} — Score: {score:.3f}")
            for k, v in doc.items():
                print(f"   {k}: {v}")
            print("-" * 60)

        print()

if __name__ == "__main__":
    main()
