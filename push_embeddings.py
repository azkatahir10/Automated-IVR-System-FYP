import json
import numpy as np
import faiss
import time

INPUT_FILE = "embeddings_output.json"
FAISS_INDEX_FILE = "faiss.index"
METADATA_FILE = "metadata.jsonl"

CHUNK_SIZE = 10000  # Notify progress every 10k items

def main():
    print("📂 Loading embeddings_output.json ...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    print(f"✅ Total embeddings: {total}")

    vectors = []
    metadata = []

    print("🚀 Starting FAISS index + metadata build...\n")

    start = time.time()

    for i, item in enumerate(data, 1):
        vectors.append(item["embedding"])
        metadata.append({
            "data": item.get("data", {}),
            "cluster": item.get("cluster")
        })

        if i % CHUNK_SIZE == 0 or i == total:
            pct = (i / total) * 100
            print(f"📦 Processed {i}/{total} ({pct:.1f}%) ✅")

    print("\n📐 Converting embeddings → numpy array...")
    vectors_np = np.array(vectors, dtype=np.float32)

    print("✨ Normalizing vectors...")
    faiss.normalize_L2(vectors_np)

    print("🎯 Creating FAISS index...")
    dim = vectors_np.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors_np)

    print(f"💾 Saving FAISS index → {FAISS_INDEX_FILE} ...")
    faiss.write_index(index, FAISS_INDEX_FILE)

    print(f"✍️ Writing metadata → {METADATA_FILE} ...")
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        for entry in metadata:
            f.write(json.dumps(entry) + "\n")

    end = time.time()
    print("\n🎉 Completed!")
    print(f"⏱️ Total time: {end - start:.2f} seconds")
    print(f"📌 FAISS Index Saved: {FAISS_INDEX_FILE}")
    print(f"📌 Metadata Saved: {METADATA_FILE}")


if __name__ == "__main__":
    main()
