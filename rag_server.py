from flask import Flask, request, jsonify
import json, faiss, numpy as np, os, re
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from deep_translator import GoogleTranslator
from dotenv import load_dotenv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
load_dotenv()

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)

FAISS_INDEX_FILE = "faiss.index"
METADATA_FILE = "metadata.jsonl"
TOP_K = 3  # Changed to 3 for top 3 results

# ----------------------------------------------
# LOAD MODELS + DATA
# ----------------------------------------------
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading FAISS index + metadata...")
index = faiss.read_index(FAISS_INDEX_FILE)

metadata = []
with open(METADATA_FILE, "r", encoding="utf-8") as f:
    for line in f:
        metadata.append(json.loads(line))

print("RAG Urdu Property Server Ready ✔️\n")


# ----------------------------------------------
# HELPERS
# ----------------------------------------------
def translate_to_urdu(text):
    """Translate English → Urdu only if not already Urdu."""
    try:
        if re.search(r"[\u0600-\u06FF]", text):
            return text
        return GoogleTranslator(source="auto", target="ur").translate(text)
    except:
        return text


def preprocess_results(raw_docs):
    """Convert property dict into Urdu key/value readable lines."""
    translated_docs = []
    for doc in raw_docs:
        urdu_pairs = []
        for k, v in doc.items():
            k_ur = translate_to_urdu(str(k))
            v_ur = translate_to_urdu(str(v))
            urdu_pairs.append(f"{k_ur}: {v_ur}")
        translated_docs.append(", ".join(urdu_pairs))
    return translated_docs


def is_goodbye(text):
    """Detect end-of-call terms."""
    endings = ["اللہ حافظ", "خدا حافظ", "bye", "thanks", "شکریہ"]
    text = text.strip().replace("۔", "")
    return any(e in text.lower() for e in endings)


def is_specific_query(query):
    """Detect if user is asking for specific/detailed property listings."""
    specific_keywords = [
        "تفصیل", "details", "options", "آپشن", "دکھائیں", "show", 
        "بتائیں", "tell", "کون سی", "which", "کتنی", "how many",
        "list", "لسٹ", "سب", "all", "مزید", "more", "تین", "3", "three"
    ]
    return any(keyword in query.lower() for keyword in specific_keywords)


# ----------------------------------------------
# LOCATION FILTERING (prevents mixed city results)
# ----------------------------------------------
def filter_by_location(metadata, index, important_words):
    filtered_meta = []
    filtered_embeddings = []

    for i, item in enumerate(metadata):
        text = json.dumps(item["data"], ensure_ascii=False)
        if any(word.lower() in text.lower() for word in important_words):
            filtered_meta.append(item["data"])
            filtered_embeddings.append(index.reconstruct(i))

    # If no match → return everything
    if len(filtered_embeddings) == 0:
        return metadata, index

    filtered_index = faiss.IndexFlatL2(len(filtered_embeddings[0]))
    filtered_index.add(np.array(filtered_embeddings))

    return filtered_meta, filtered_index


# ----------------------------------------------
# AI SUMMARY GENERATION (URDU NATURAL AGENT)
# ----------------------------------------------
def generate_urdu_summary(raw_results, query, history, is_specific=False):
    try:
        joined_text = "\n".join(raw_results)

        if is_specific:
            # Detailed listing prompt for specific queries
            prompt = f"""
آپ ایک دوستانہ رئیل اسٹیٹ سیلز ایجنٹ ہیں جو تفصیلی پراپرٹیز کی لسٹ فراہم کر رہے ہیں۔

------------------------------------------------------------------
پچھلی گفتگو:
{history}

گاہک کا سوال:
{query}

RAG سے نکلے ہوئے Top 3 پراپرٹیز:
{joined_text}
------------------------------------------------------------------

آپ نے یہ کرنا ہے:

1️⃣ پہلے ایک friendly opening:
- "جی بالکل، میں آپ کو تین بہترین آپشن بتاتا ہوں..."
- "ٹھیک ہے، میرے پاس آپ کے لیے تین اچھی پراپرٹیز ہیں..."

2️⃣ پھر ہر پراپرٹی کو CLEARLY الگ کر کے بیان کریں:

**پہلی پراپرٹی:**
[تفصیلات - رقبہ، بیڈ روم، باتھ روم، قیمت، لوکیشن]

**دوسری پراپرٹی:**
[تفصیلات - رقبہ، بیڈ روم، باتھ روم، قیمت، لوکیشن]

**تیسری پراپرٹی:**
[تفصیلات - رقبہ، بیڈ روم، باتھ روم، قیمت، لوکیشن]

3️⃣ آخر میں ایک soft closing:
- "ان میں سے کوئی آپ کو پسند آیا؟"
- "مزید تفصیل کے لیے بتائیں۔"

❗ ہر پراپرٹی کو نمبر یا heading سے الگ کریں
❗ صاف اور پڑھنے میں آسان format استعمال کریں
"""
        else:
            # General summary prompt (original)
            prompt = f"""
آپ ایک انتہائی دوستانہ، تجربہ کار اور قدرتی لہجے میں بات کرنے والے رئیل اسٹیٹ سیلز ایجنٹ ہیں۔

آپ ہمیشہ:
- نرم، خوش اخلاق، conversational لہجے میں بات کرتے ہیں
- کوئی روبوٹک جملہ، کوئی سخت لائن نہیں بولتے
- صرف ایک جامع، فلو میں بہتا ہوا انسانی جواب دیتے ہیں

------------------------------------------------------------------
پچھلی گفتگو:
{history}

گاہک کا سوال:
{query}

RAG سے نکلا ہوا پراپرٹی ڈیٹا:
{joined_text}
------------------------------------------------------------------

آپ نے ایسا کرنا ہے:

1️⃣ سب سے پہلے ایک human-style waiting لائن کہیں:
- "ایک لمحہ دیں، میں چیک کرتا ہوں…"
- "ذرا رکیں، میں دیکھ کر بتاتا ہوں…"
- "کچھ دیر ٹھہریں، میں دیکھ لیتا ہوں…"
(ہر بار مختلف لائن ہو)

2️⃣ پھر RAG کی معلومات کو خوبصورت انسانی جملوں میں بیان کریں:
- رقبہ ہو → "یہ جگہ تقریباً ___ مرلے/کنال کے آس پاس ہے"
- بیڈ روم → "اس میں ___ بیڈ روم ہیں"
- باتھ روم → "اس میں ___ باتھ روم ہیں"
- قیمت → "قیمت تقریباً ___ کے آس پاس ہے"
- لوکیشن/مارکیٹ/پارکنگ → نارمل انداز میں بتائیں

❗ اگر کوئی فیلڈ نہیں ہے تو اسے بالکل مت ذکر کریں۔

3️⃣ اس جواب میں:
- کوئی وزٹ آفر نہیں
- کوئی بکنگ نہیں
- کوئی نام یا نمبر نہیں مانگنا
- کوئی کال ٹو ایکشن نہیں

4️⃣ آخر میں صرف ایک soft friendly closing لائن:
- "اگر آپ چاہیں تو میں مزید options بھی چیک کر سکتا ہوں۔"
یا
- "مزید detail چاہیے ہو تو بتا دیں۔"

❗ انتہائی اہم:
ایک ہی جواب دیں۔
کوئی ڈپلیکیٹ، کوئی دو جواب، کوئی repeat لائن نہیں۔

اب ان ہدایات کے مطابق ایک ہی خوبصورت، مکمل اور انسانی انداز میں جواب لکھیں۔
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print("⚠️ Urdu summary error:", e)
        return "معاف کیجئے، میں اس وقت مکمل معلومات حاصل نہیں کر سکا۔"


# ----------------------------------------------
# MAIN SEARCH ROUTE
# ----------------------------------------------
@app.route("/search", methods=["POST"])
def search():
    data = request.get_json()

    query = data.get("query", "").strip()
    history = data.get("history", "")

    if not query:
        return jsonify({"error": "No query provided"}), 400

    # Goodbye detection
    if is_goodbye(query):
        return jsonify({
            "results": [],
            "summary": "ٹھیک ہے، اللہ حافظ! اپنا خیال رکھیں۔"
        })

    # Check if user wants specific details
    wants_specific = is_specific_query(query)

    # Location keywords (modify based on your data)
    location_words = ["dha", "ڈی ایچ اے", "garden", "گارڈن", "phase", "فیز"]

    # Filter metadata by location context
    filtered_meta, filtered_index = filter_by_location(metadata, index, location_words)

    # Encode user query
    query_emb = model.encode(query, convert_to_tensor=False).astype("float32")
    query_emb = np.expand_dims(query_emb, axis=0)
    faiss.normalize_L2(query_emb)

    # Search for top 3
    distances, indices = filtered_index.search(query_emb, TOP_K)

    # Extract top documents
    raw_docs = [filtered_meta[i] for i in indices[0]]
    urdu_ready = preprocess_results(raw_docs)

    # Generate Urdu property summary (with ranking if specific)
    urdu_summary = generate_urdu_summary(urdu_ready, query, history, is_specific=wants_specific)

    return jsonify({
        "results": urdu_ready,
        "summary": urdu_summary,
        "is_specific": wants_specific,  # Flag to indicate response type
        "count": len(urdu_ready)  # Number of results returned
    })


# ----------------------------------------------
# RUN SERVER
# ----------------------------------------------
if __name__ == "__main__":
    app.run(port=5001)
