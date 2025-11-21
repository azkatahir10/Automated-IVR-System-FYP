from flask import Flask, request, jsonify
import json, faiss, numpy as np, os, re
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from deep_translator import GoogleTranslator

client = OpenAI()

app = Flask(__name__)

FAISS_INDEX_FILE = "faiss.index"
METADATA_FILE = "metadata.jsonl"
TOP_K = 3

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading FAISS index and metadata...")
index = faiss.read_index(FAISS_INDEX_FILE)
metadata = []
with open(METADATA_FILE, "r", encoding="utf-8") as f:
    for line in f:
        metadata.append(json.loads(line))
print("RAG Urdu property service ready!\n")


# TRANSLATION HELPERS

def translate_to_urdu(text):
    """Translate English words to Urdu unless Urdu already present."""
    try:
        if re.search(r"[\u0600-\u06FF]", text):
            return text
        return GoogleTranslator(source="auto", target="ur").translate(text)
    except:
        return text


def preprocess_results(raw_docs):
    """Translate all key/value pairs into Urdu."""
    translated_docs = []
    for doc in raw_docs:
        urdu_pairs = []
        for k, v in doc.items():
            k_ur = translate_to_urdu(str(k))
            v_ur = translate_to_urdu(str(v))
            urdu_pairs.append(f"{k_ur}: {v_ur}")
        translated_docs.append(", ".join(urdu_pairs))
    return translated_docs


# AI RESPONSE GENERATION
def generate_urdu_summary(raw_results, query):
    try:
        joined_text = "\n".join(raw_results)

        prompt = f"""
        آپ ایک نہایت دوستانہ، تجربہ کار اور پروفیشنل رئیل اسٹیٹ سیلز ایجنٹ ہیں۔
        آپ کا انداز ہمیشہ:
        - مؤدبانہ
        - انسان جیسا
        - مکمل وضاحت کے ساتھ
        - گاہک کو سمجھانے والا
        ہونا چاہیے۔

        گاہک نے جو بھی سوال پوچھا ہے، اس کا جواب *بالکل اسی کے مطابق* دیں۔
         اگر گاہک نے جگہ، قیمت یا سائز پوچھا ہو تو کہیں:
        "کچھ دیر ٹھہریں، میں چیک کر کے بتاتا ہوں…"

        📌 اگر گاہک نے **کسی ایک مخصوص گھر/پلاٹ کی تفصیل** پوچھی ہو، تو RAG سے ملی تمام معلومات کو خوبصورت، کہانی جیسے انداز میں مکمل طور پر بیان کریں۔

        نیچے دی گئی RAG معلومات سے یہ تفصیل نکال کر سمجھائیں:
        - رقبہ (Area)
        - قیمت (Price)
        - بیڈ رومز
        - باتھ رومز
        - منزلیں
        - کار پورچ / گیراج
        - نقشہ / Facing
        - لوکیشن
        - اور کوئی بھی Extra معلومات

        ⚠ اگر کوئی فیلڈ موجود نہیں ہے تو اس کا نام نہ لیں۔

        آخر میں لازمی کہیں:
        “اگر آپ چاہیں تو میں اس گھر کا وزٹ بھی شیڈول کر سکتا ہوں۔ کیا آپ کل شام 5 بجے دیکھنا پسند کریں گے؟”

        ----------------------------------
        گاہک کا سوال:
        {query}

        RAG سے ملی ہوئی معلومات:
        {joined_text}
        ----------------------------------

        اب ایک مکمل قدرتی، دوستانہ، انسانی انداز میں جواب لکھیں۔
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print("⚠️ Urdu summary error:", e)
        return "معاف کیجئے گا، میں فی الحال مکمل تفصیل حاصل نہیں کر سکا۔"


#  SEARCH ENDPOINT
@app.route("/search", methods=["POST"])
def search():
    data = request.get_json()
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "No query provided"}), 400

    query_emb = model.encode(query, convert_to_tensor=False).astype("float32")
    query_emb = np.expand_dims(query_emb, axis=0)
    faiss.normalize_L2(query_emb)

    distances, indices = index.search(query_emb, TOP_K)

    raw_docs = [metadata[idx]["data"] for idx in indices[0]]
    urdu_ready_results = preprocess_results(raw_docs)

    urdu_summary = generate_urdu_summary(urdu_ready_results, query)

    return jsonify({
        "results": urdu_ready_results,
        "summary": urdu_summary
    })


if __name__ == "__main__":
    app.run(port=5001)
