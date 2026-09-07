"""
rag_server_local.py  —  Urdu RAG Property Search Server (Transcription/STT module)
Port: 5001

Architecture:
  Layer 1 — SQLite WHERE clause  (location, area, bedrooms, price)
  Layer 2 — FAISS cosine re-rank (on small candidate set from Layer 1)
  Layer 3 — Local template-based Urdu summary  (100% offline, zero API cost)
  Visit  —  Regex-based extraction             (100% offline, zero API cost)

Supabase used ONLY for: rag_searches log + client_visits (fully optional).
Property search + response generation are 100% local — no OpenAI, no network calls.
Designed for: STT -> normalize -> /search -> TTS pipeline.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json, sqlite3, faiss, numpy as np, os, re, math, datetime, logging
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# ── Optional: Supabase (logging only) ────────────────────────────────────────
try:
    from supabase import create_client, Client
    _SUPABASE_AVAILABLE = True
except ImportError:
    _SUPABASE_AVAILABLE = False

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['HF_HUB_OFFLINE'] = '1'
load_dotenv()

# -----------------------------------------------
# LOGGING
# -----------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# -----------------------------------------------
# SUPABASE CLIENT  (optional — only for logging)
# -----------------------------------------------
supabase = None
if _SUPABASE_AVAILABLE and os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SERVICE_KEY"):
    supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))
    log.info("✅ Supabase ready (search/visit logging enabled)")
else:
    log.info("ℹ️  Supabase not configured — logging disabled (non-fatal)")

app = Flask(__name__)
CORS(app)

SQLITE_FILE       = "properties.db"
FAISS_FILE        = "faiss.index"
FAISS_IDS_FILE    = "faiss_ids.json"
SQLITE_CANDIDATES = 50

# -----------------------------------------------
# LOAD MODEL + INDEX
# -----------------------------------------------
log.info("⏳ Loading multilingual embedding model...")
embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
log.info("✅ Embedding model ready!")

log.info("⏳ Loading FAISS index...")
faiss_index = faiss.read_index(FAISS_FILE)
log.info(f"✅ FAISS index loaded ({faiss_index.ntotal} vectors)")

log.info("⏳ Loading FAISS ID map...")
with open(FAISS_IDS_FILE, "r") as f:
    faiss_ids: list[int] = json.load(f)
log.info(f"✅ FAISS ID map loaded ({len(faiss_ids)} entries)")

_c = sqlite3.connect(SQLITE_FILE)
_total = _c.execute("SELECT COUNT(*) FROM properties").fetchone()[0]
_dha6  = _c.execute("SELECT COUNT(*) FROM properties WHERE location LIKE '%dha phase 6%'").fetchone()[0]
_c.close()
log.info(f"✅ SQLite ready: {_total:,} rows | DHA Phase 6: {_dha6:,}")
log.info("🚀 RAG Server ready on port 5001\n")

# -----------------------------------------------
# DB HELPER
# -----------------------------------------------
def get_db():
    conn = sqlite3.connect(SQLITE_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

# -----------------------------------------------
# SHARED HELPERS
# -----------------------------------------------
def _safe_int(val):
    try:
        if val is None or val == "" or (isinstance(val, float) and math.isnan(val)):
            return None
        return int(round(float(str(val))))
    except:
        return None

def sanitize(obj):
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(i) for i in obj]
    return obj

# -----------------------------------------------
# FORMAT FOR URDU DISPLAY
# -----------------------------------------------
def to_urdu(doc: dict) -> str:
    FIELDS = [
        ("title",             "عنوان"),
        ("location",          "مقام"),
        ("type",              "قسم"),
        ("price_crore",       "قیمت"),
        ("area_marla",        "رقبہ"),
        ("bedrooms",          "بیڈ روم"),
        ("bathrooms",         "باتھ روم"),
        ("kitchens",          "کچن"),
        ("store_rooms",       "اسٹور روم"),
        ("servant_quarters",  "ملازم کوارٹر"),
        ("built_year",        "تعمیری سال"),
        ("purpose",           "مقصد"),
        ("date_posted",       "تاریخ اشاعت"),
    ]
    AMENITIES = [
        ("furnished",          "فرنشڈ"),
        ("gym",                "جم"),
        ("study_room",         "اسٹڈی روم"),
        ("drawing_room",       "ڈرائنگ روم"),
        ("dining_room",        "ڈائننگ روم"),
        ("lawn_garden",        "لان/باغیچہ"),
        ("swimming_pool",      "سوئمنگ پول"),
        ("electricity_backup", "بجلی بیک اپ"),
        ("lounge",             "لاؤنج"),
    ]
    lines = []
    for key, label in FIELDS:
        val = doc.get(key)
        if val is None or val == "" or val == "unknown":
            continue
        if key == "price_crore" and isinstance(val, (int, float)):
            val = f"{val:.2f} کروڑ" if val >= 1 else f"{round(val*100):.0f} لاکھ"
        elif key == "area_marla" and isinstance(val, (int, float)):
            if val >= 20 and val % 20 == 0:
                val = f"{int(val//20)} کنال"
            elif val >= 20:
                val = f"{val/20:.1f} کنال"
            else:
                val = f"{int(val)} مرلے"
        elif isinstance(val, float) and not math.isnan(val) and val == int(val):
            val = int(val)
        lines.append(f"{label}: {val}")
    am = [u for k, u in AMENITIES if doc.get(k)]
    if am:
        lines.append("سہولیات: " + "، ".join(am))
    return "\n".join(lines)

# -----------------------------------------------
# 4-PASS QUERY NORMALISATION
# -----------------------------------------------
WHISPER_MAP = {
    "ڈی اے چے": "dha", "ڈی اے جے": "dha", "ڈی ایچ اے": "dha", "ڈی ایچ": "dha",
    "دی ایچ اے": "dha", "ری ایچ": "dha", "ڈیفنس": "defence", "ڈیفینس": "defence",
    "دی اے چے": "dha",
    "دی ایچ اے فیز": "dha phase", "دی ایچ اے فیس": "dha phase",
    "ڈی ایچ اے فیز": "dha phase", "ڈی ایچ اے فیس": "dha phase",
    "دی ایچ فیز": "dha phase", "دی ایچ فیس": "dha phase",
    "ڈی ایچ فیز": "dha phase", "ڈی ایچ فیس": "dha phase",
    "دی اے فیز": "dha phase", "دی اے فیس": "dha phase",
    "دیئے تو پھر": "dha phase", "دی فیز": "dha phase", "دی فیس": "dha phase",
    "اے فیز": "dha phase", "اے فیس": "dha phase",
    "فیز ون": "phase 1", "فیس ون": "phase 1",
    "فیز ٹو": "phase 2", "فیس ٹو": "phase 2",
    "فیز تھری": "phase 3", "فیس تھری": "phase 3",
    "فیز فور": "phase 4", "فیس فور": "phase 4",
    "فیز فائیو": "phase 5", "فیز فائف": "phase 5",
    "فیس فائیو": "phase 5", "فیس فائف": "phase 5",
    "فیز سکس": "phase 6", "فیس سکس": "phase 6",
    "فیس اکس": "phase 6", "فیس سِکس": "phase 6",
    "فیز ڈکس": "phase 6", "فیز ڈیکس": "phase 6",
    "فیز دکس": "phase 6", "فیز سکز": "phase 6",
    "فیز زکس": "phase 6", "فیس ڈکس": "phase 6",
    "فیس ڈیکس": "phase 6", "فیس دکس": "phase 6",
    "فیز کس": "phase 6", "فیس کس": "phase 6",
    "فیز سیون": "phase 7", "فیس سیون": "phase 7",
    "فیز ایٹ": "phase 8", "فیس ایٹ": "phase 8",
    "فیز نائن": "phase 9", "فیس نائن": "phase 9",
    "فیز دس": "phase 10", "فیس دس": "phase 10",
    "phase ون": "phase 1", "phase ٹو": "phase 2",
    "phase تھری": "phase 3", "phase فور": "phase 4",
    "phase فائیو": "phase 5", "phase فائف": "phase 5",
    "phase سکس": "phase 6", "phase سِکس": "phase 6",
    "phase سیون": "phase 7", "phase ایٹ": "phase 8",
    "phase نائن": "phase 9", "phase دس": "phase 10",
    "جوہر ٹاؤن": "johar town", "جوہر": "johar",
    "جوہا": "johar", "جو ہاٹ": "johar",
    "جو ہار ٹاؤن": "johar town", "جوہا ٹاؤن": "johar town",
    "بحریہ": "bahria", "بہریہ": "bahria",
    "گلبرگ": "gulberg", "گلبرک": "gulberg",
    "گارڈن ٹاؤن": "garden town", "گارڈن": "garden town",
    "ماڈل ٹاؤن": "model town", "ماڈل": "model town",
    "والنشیا": "valencia", "ولنشیا": "valencia",
    "فیصل ٹاؤن": "faisal town", "ٹاؤن شپ": "township",
    "واپڈا ٹاؤن": "wapda town",
    "لاہور": "lahore", "اسلام آباد": "islamabad", "اسلام ابد": "islamabad",
    "کراچی": "karachi", "راولپنڈی": "rawalpindi",
    "پشاور": "peshawar", "ملتان": "multan",
    "نالے": "kanal", "نالا": "kanal", "کنالا": "kanal", "کنالو": "kanal",
}

def normalize_query(q):
    for k in sorted(WHISPER_MAP, key=len, reverse=True):
        q = q.replace(k, WHISPER_MAP[k])
    return q

def normalize_phase_numbers(q):
    NUM = [
        ("ون","1"), ("one","1"), ("ٹو","2"), ("two","2"),
        ("تھری","3"), ("three","3"), ("فور","4"), ("four","4"),
        ("فائیو","5"), ("فائف","5"), ("five","5"), ("پائیو","5"), ("پائف","5"),
        ("سکس","6"), ("سِکس","6"), ("six","6"), ("اکس","6"),
        ("ڈکس","6"), ("ڈیکس","6"), ("دکس","6"), ("کس","6"),
        ("سیون","7"), ("seven","7"), ("ایٹ","8"), ("eight","8"),
        ("نائن","9"), ("nine","9"), ("دس","10"), ("ten","10"),
    ]
    for w, d in NUM:
        q = re.sub(rf'(phase)\s+{re.escape(w)}', rf'\g<1> {d}', q, flags=re.IGNORECASE)
    return q

def normalize_spoken_numbers(q):
    NW = [
        ("اکیس","21"), ("بائیس","22"), ("تئیس","23"), ("چوبیس","24"), ("پچیس","25"),
        ("چھبیس","26"), ("ستائیس","27"), ("اٹھائیس","28"), ("انتیس","29"), ("تیس","30"),
        ("چالیس","40"), ("پچاس","50"), ("ساٹھ","60"), ("ستر","70"),
        ("اسی","80"), ("نوے","90"),
        ("اٹھارہ","18"), ("سترہ","17"), ("سولہ","16"), ("پندرہ","15"), ("چودہ","14"),
        ("تیرہ","13"), ("بارہ","12"), ("گیارہ","11"),
        ("ڈیڑھ","1.5"), ("اڑھائی","2.5"),
        ("ساڑھے تین","3.5"), ("ساڑھے چار","4.5"),
        ("ساڑھے پانچ","5.5"), ("ساڑھے سات","7.5"),
        ("اک","1"), ("اِک","1"), ("دس","10"), ("نو","9"),
        ("آٹھ","8"), ("سات","7"), ("چھ","6"), ("پانچ","5"),
        ("چار","4"), ("تین","3"), ("دو","2"), ("ایک","1"),
    ]
    for w, d in NW:
        q = re.sub(rf'(?<![\w\d]){re.escape(w)}(?![\w\d])', d, q)
    return q

ROMAN_MAP = {
    "ghr": "house", "makan": "house", "flat": "flat",
    "dukan": "shop", "plot": "plot", "villa": "villa",
    "portion": "portion", "office": "office",
    "kenal": "kanal", "knal": "kanal", "knaal": "kanal",
    "mrla": "marla", "marley": "marla",
    "chahiya": "for sale", "chahye": "for sale",
    "kiraya": "for rent", "bikao": "for sale",
    "kamra": "bedroom", "kamray": "bedroom",
    "ma": "in", "mein": "in",
    "ka": "", "ki": "", "ke": "", "wala": "", "wali": "",
}

def normalize_roman(q):
    words = q.lower().split()
    return " ".join(" ".join(ROMAN_MAP.get(w, w) for w in words).split())

def full_normalize(query: str) -> str:
    return normalize_roman(
        normalize_spoken_numbers(
            normalize_phase_numbers(
                normalize_query(query)
            )
        )
    )

# -----------------------------------------------
# LOCATION EXTRACTION
# -----------------------------------------------
LOCATION_TOKENS = [
    "dha phase 6","dha phase 7","dha phase 8","dha phase 5","dha phase 4",
    "dha phase 3","dha phase 2","dha phase 1","dha phase 9","dha phase 10",
    "dha 9 town","dha 11 rahbar","dha","defence",
    "bahria town","bahria","johar town","johar","gulberg",
    "garden town","model town","valencia town","valencia",
    "park view city","park view","lake city","paragon city","central park",
    "wapda town","faisal town","township","iqbal town","allama iqbal",
    "sabzazar","green town","lda avenue","cantt","cantonment",
    "gulshan","clifton","nazimabad","north nazimabad",
    "fb area","federal b area","gulistan e jauhar","malir","landhi","korangi",
    "f-6","f-7","f-8","f-10","f-11","g-6","g-7","g-8","g-9","g-10","g-11",
    "e-7","e-11","i-8","i-10","i-14","d-12",
    "f6","f7","f8","f10","f11","g6","g7","g8","g9","g10","g11","e7","e11","i8","i10",
    "blue area","margalla","bahria rawalpindi","saddar","chaklala","pwd","media town",
    "phase 1","phase 2","phase 3","phase 4","phase 5",
    "phase 6","phase 7","phase 8","phase 9","phase 10",
    "block a","block b","block c","block d","block e","block f","block g",
    "block h","block j","block k","block l","block m","block n",
    "lahore","karachi","islamabad","rawalpindi","peshawar",
    "multan","faisalabad","gujranwala","sialkot","quetta","بلاک",
]

def extract_locations(query: str) -> list[str]:
    q = query.lower()
    matched = [
        tok for tok in LOCATION_TOKENS
        if re.search(r'(?<![\w])' + re.escape(tok) + r'(?![\w])', q)
    ]
    if not matched:
        return []
    matched.sort(key=len, reverse=True)
    final = []
    for tok in matched:
        if not any(tok in longer for longer in final):
            final.append(tok)
    return final

# -----------------------------------------------
# NUMERIC FILTERS
# -----------------------------------------------
def parse_filters(query: str) -> dict:
    f = {}
    b = re.search(r'(\d+)\s*(?:bed|bedroom|بیڈ|کمر)', query.lower())
    if b:
        f["Bedrooms"] = int(b.group(1))
    c = re.search(r'(\d+(?:\.\d+)?)\s*(?:crore|کروڑ)', query.lower())
    l = re.search(r'(\d+(?:\.\d+)?)\s*(?:lakh|لاکھ)', query.lower())
    if c:
        f["max_price_crore"] = float(c.group(1))
    elif l:
        f["max_price_crore"] = float(l.group(1)) / 100
    m = re.search(
        r'(\d+(?:\.\d+)?)\s{0,2}(?:marla|mrla|مرلہ|مرلے|مرلا|مرلو|مرلوں|مرلوہ|ملے|ملہ|ملا|ملو|ملوں|مرل)',
        query.lower()
    )
    k = re.search(
        r'(\d+(?:\.\d+)?)\s{0,2}(?:kanal|canal|کنال|کنالوں|کنالہ|کنالے)',
        query.lower()
    )
    if m:
        f["area_marla"] = float(m.group(1))
    elif k:
        f["area_marla"] = float(k.group(1)) * 20
    return f

# -----------------------------------------------
# LAYER 1: SQLITE
# -----------------------------------------------
def sqlite_search(loc_words: list, filters: dict, include_area=True) -> list[dict]:
    conds, params = [], []
    if loc_words:
        conds.append("(" + " OR ".join(["location LIKE ?"] * len(loc_words)) + ")")
        params += [f"%{t}%" for t in loc_words]
    if "Bedrooms" in filters:
        conds.append("bedrooms = ?")
        params.append(filters["Bedrooms"])
    if "max_price_crore" in filters:
        conds.append("(price_crore IS NULL OR price_crore <= ?)")
        params.append(filters["max_price_crore"])
    if include_area and "area_marla" in filters:
        am = filters["area_marla"]
        conds.append("(area_marla IS NULL OR (area_marla BETWEEN ? AND ?))")
        params += [am * 0.75, am * 1.25]
    where = ("WHERE " + " AND ".join(conds)) if conds else ""
    sql = f"""SELECT * FROM properties {where}
              ORDER BY CASE WHEN link IS NOT NULL AND link != '' THEN 0 ELSE 1 END, id
              LIMIT {SQLITE_CANDIDATES}"""
    conn = get_db()
    try:
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()

# -----------------------------------------------
# LAYER 2: FAISS RE-RANK
# -----------------------------------------------
def faiss_rerank(candidates: list[dict], q_vec: np.ndarray, top_n=5) -> list[dict]:
    if not candidates:
        return []
    if len(candidates) <= top_n:
        return candidates
    id_to_pos = {rid: pos for pos, rid in enumerate(faiss_ids)}
    vecs, valid = [], []
    for doc in candidates:
        pos = id_to_pos.get(doc["id"])
        if pos is not None and pos < faiss_index.ntotal:
            vecs.append(faiss_index.reconstruct(pos))
            valid.append(doc)
    if not vecs:
        return candidates[:top_n]
    va = np.array(vecs, dtype="float32")
    faiss.normalize_L2(va)
    tmp = faiss.IndexFlatIP(va.shape[1])
    tmp.add(va)
    q = q_vec.copy().reshape(1, -1).astype("float32")
    faiss.normalize_L2(q)
    _, idxs = tmp.search(q, min(top_n, tmp.ntotal))
    return [valid[i] for i in idxs[0] if i < len(valid)]

# -----------------------------------------------
# LAYER 3: LOCAL TEMPLATE-BASED URDU SUMMARY
# 100% offline — no OpenAI, no network calls.
# -----------------------------------------------
ORDINALS_UR = ["پہلی", "دوسری", "تیسری", "چوتھی", "پانچویں"]

def trim_history(h, max_turns=4):
    if not h:
        return ""
    lines = h.strip().split("\n")
    return "\n".join(lines[-(max_turns * 2):])

def build_summary(
    docs: list[str],
    query: str,
    history: str,
    is_spec: bool,
    no_results=False,
    no_results_reason="",
    requested_size="",
) -> str:
    """
    Builds the Urdu reply with zero external calls.
    `docs` are pre-formatted to_urdu() blocks — RAG-grounded by construction,
    so only fields that actually exist in the DB are ever mentioned.
    """

    # ── No results ────────────────────────────────────────────────────────
    if no_results:
        if no_results_reason == "location":
            return (
                "معاف کیجئے، اس مقام پر فی الحال ہمارے پاس کوئی پراپرٹی دستیاب نہیں ہے۔ "
                "کیا آپ کسی اور علاقے کے بارے میں پوچھنا چاہیں گے؟"
            )
        return (
            "معاف کیجئے، آپ کی دی گئی تفصیلات سے ملتی پراپرٹی فی الحال دستیاب نہیں ہے۔ "
            "اگر آپ بجٹ یا سائز میں تھوڑی تبدیلی کریں تو میں بہتر آپشن تلاش کر سکتا ہوں۔"
        )

    if not docs:
        return "معاف کیجئے، اس وقت کوئی متعلقہ پراپرٹی نہیں ملی۔"

    # ── Intro ─────────────────────────────────────────────────────────────
    if requested_size:
        intro = (
            f"معاف کیجئے، بالکل {requested_size} کی پراپرٹی فی الحال دستیاب نہیں ہے، "
            "لیکن یہ قریب ترین آپشنز ہیں:\n\n"
        )
    elif is_spec:
        intro = "ٹھیک ہے، یہ تفصیلات ملاحظہ کریں:\n\n"
    else:
        intro = "جی، آپ کے سوال کے مطابق یہ چند پراپرٹیز دستیاب ہیں:\n\n"

    # ── Property blocks ───────────────────────────────────────────────────
    blocks = []
    for i, doc in enumerate(docs):
        label = ORDINALS_UR[i] if i < len(ORDINALS_UR) else f"{i+1}ویں"
        if is_spec or requested_size:
            blocks.append(f"**{label} پراپرٹی:**\n{doc}")
        else:
            blocks.append(doc)

    body = "\n\n".join(blocks)

    # ── Closing ───────────────────────────────────────────────────────────
    closing = (
        "\n\nکیا یہ آپشنز آپ کے لیے مناسب ہو سکتے ہیں؟"
        if requested_size
        else "\n\nان میں سے کوئی پراپرٹی پسند آئی، یا مزید تفصیل چاہیں گے؟"
    )

    return intro + body + closing

# -----------------------------------------------
# MISC HELPERS
# -----------------------------------------------
def is_goodbye(text: str) -> bool:
    return any(
        e in text.strip().lower().replace("۔", "")
        for e in ["اللہ حافظ", "خدا حافظ", "bye", "thanks", "شکریہ", "alvida"]
    )

def is_specific(query: str) -> bool:
    return any(
        k in query.lower()
        for k in [
            "تفصیل","details","options","آپشن","دکھائیں","show","بتائیں",
            "tell","کون سی","which","list","لسٹ","سب","all","مزید","more","کونسا",
        ]
    )

def save_search_log(conv_id, query, summary, props, specific):
    if not supabase:
        return
    try:
        supabase.table("rag_searches").insert({
            "conversation_id": conv_id if conv_id else None,
            "query":           query,
            "summary":         summary,
            "properties":      json.dumps(sanitize(props)),
            "is_specific":     specific,
            "result_count":    len(props),
            "created_at":      datetime.datetime.utcnow().isoformat(),
        }).execute()
    except Exception as e:
        log.warning(f"⚠️ rag_searches log error (non-fatal): {e}")

# -----------------------------------------------
# VISIT EXTRACTION — pure regex, zero API calls
# -----------------------------------------------
def extract_visit_regex(text: str) -> dict:
    """
    Extracts visit details from conversation text using regex only.
    Returns a dict with the same keys the old GPT extractor used.
    """
    vd: dict = {
        "client_name":       None,
        "phone_number":      None,
        "visit_date":        None,
        "visit_time":        None,
        "selected_property": None,
        "confirmed":         False,
    }

    # Phone number  (Pakistani format: 03XX-XXXXXXX or +923XX...)
    phone = re.search(r'(\+92|0092|0)[\s-]?(3\d{2})[\s-]?(\d{7})', text)
    if phone:
        vd["phone_number"] = f"0{phone.group(2)}{phone.group(3)}"

    # Date  (YYYY-MM-DD or DD/MM/YYYY or DD-MM-YYYY)
    date = re.search(r'(\d{4}-\d{2}-\d{2})', text)
    if not date:
        date = re.search(r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})', text)
        if date:
            vd["visit_date"] = f"{date.group(3)}-{date.group(2).zfill(2)}-{date.group(1).zfill(2)}"
    else:
        vd["visit_date"] = date.group(1)

    # Time  (HH:MM, optional AM/PM)
    time_ = re.search(r'(\d{1,2}):(\d{2})\s*(?:am|pm|AM|PM|بجے)?', text)
    if time_:
        vd["visit_time"] = f"{time_.group(1).zfill(2)}:{time_.group(2)}"

    # Urdu/English time words  (e.g. "3 بجے", "5 o'clock")
    if not vd["visit_time"]:
        t_word = re.search(r'(\d{1,2})\s*(?:بجے|o\'?clock)', text)
        if t_word:
            vd["visit_time"] = f"{t_word.group(1).zfill(2)}:00"

    # Confirmation keywords
    CONFIRM_KW = ["confirm", "ٹھیک ہے", "ہاں", "yes", "بالکل", "ضرور", "ok", "okay"]
    if any(k in text.lower() for k in CONFIRM_KW):
        vd["confirmed"] = True

    # Name — look for "میرا نام X ہے" or "name is X" patterns
    name_ur = re.search(r'(?:میرا\s*نام|نام)\s+([^\s،,۔.]{2,20})', text)
    name_en = re.search(r'(?:my\s+name\s+is|name\s*[:\-]?\s*)([A-Za-z]{2,30})', text, re.I)
    if name_ur:
        vd["client_name"] = name_ur.group(1).strip()
    elif name_en:
        vd["client_name"] = name_en.group(1).strip()

    return vd

def save_visit_log(conv_id, vd, valid_links):
    if not supabase:
        log.info(f"ℹ️  Visit extracted (Supabase disabled): {vd}")
        return
    try:
        supabase.table("client_visits").insert({
            "conversation_id":   conv_id if conv_id else None,
            "client_name":       vd.get("client_name"),
            "phone_number":      vd.get("phone_number"),
            "visit_date":        vd.get("visit_date"),
            "visit_time":        vd.get("visit_time"),
            "selected_property": vd.get("selected_property"),
            "property_link":     json.dumps(valid_links) if valid_links else None,
            "confirmed":         vd.get("confirmed", False),
            "created_at":        datetime.datetime.utcnow().isoformat(),
        }).execute()
        log.info(f"📅 Visit saved: {vd.get('client_name')} | {vd.get('visit_date')} | links: {valid_links}")
    except Exception as e:
        log.warning(f"⚠️ client_visits insert error (non-fatal): {e}")

# -----------------------------------------------
# SEARCH ROUTE
# -----------------------------------------------
@app.route("/search", methods=["POST"])
def search():
    body    = request.get_json(force=True, silent=True) or {}
    query   = body.get("query",   "").strip()
    history = body.get("history", "")
    conv_id = body.get("conversation_id", "")

    if not query:
        return jsonify({"error": "No query"}), 400

    if is_goodbye(query):
        return jsonify({
            "results": [], "summary": "ٹھیک ہے، اللہ حافظ! اپنا خیال رکھیں۔",
            "matched_properties": [],
        })

    q_norm  = full_normalize(query)
    spec    = is_specific(q_norm)
    filters = parse_filters(q_norm)
    locs    = extract_locations(q_norm)

    log.info(f"🔍 Query: {query!r}")
    log.info(f"📐 Normalised: {q_norm!r}")
    log.info(f"🏙️ Locations: {locs} | Filters: {filters}")

    q_vec = embed_model.encode(q_norm, convert_to_tensor=False).astype("float32")

    # Layer 1 — SQLite
    candidates = sqlite_search(locs, filters, include_area=True)

    # Location not found
    if locs and not candidates:
        msg = build_summary([], q_norm, history, spec,
                            no_results=True, no_results_reason="location")
        return jsonify({
            "results": [], "summary": msg, "matched_properties": [],
            "no_results": True, "no_results_reason": "location",
        })

    # Area mismatch — retry without area constraint
    area_mismatch, size_str = False, ""
    if not candidates and "area_marla" in filters:
        area_mismatch = True
        am = filters["area_marla"]
        size_str = (
            f"{int(am)} مرلے" if am < 20
            else (f"{int(am//20)} کنال" if am % 20 == 0 else f"{am/20:.1f} کنال")
        )
        candidates = sqlite_search(locs, filters, include_area=False)

    # Still nothing
    if not candidates:
        msg = build_summary([], q_norm, history, spec,
                            no_results=True, no_results_reason="filters")
        return jsonify({
            "results": [], "summary": msg, "matched_properties": [],
            "no_results": True, "no_results_reason": "filters",
        })

    # Layer 2 — FAISS re-rank
    top = faiss_rerank(candidates, q_vec, top_n=5)
    top = sorted(
        top[:5],
        key=lambda d: 0 if (d.get("link") and d["link"] not in ("", "unknown")) else 1
    )[:3]

    urdu_lines = [to_urdu(d) for d in top]

    # Layer 3 — local template summary
    summary = build_summary(
        urdu_lines, q_norm, history, spec,
        requested_size=size_str if area_mismatch else "",
    )

    log.info(f"✅ {len(urdu_lines)} results returned")

    if conv_id:
        save_search_log(conv_id, query, summary, top, spec)

    return jsonify({
        "results":            urdu_lines,
        "summary":            summary,
        "is_specific":        spec,
        "count":              len(urdu_lines),
        "matched_properties": sanitize(top),
    })

# -----------------------------------------------
# VISIT EXTRACTION ROUTE  (pure regex — no OpenAI)
# -----------------------------------------------
@app.route("/extract-visit", methods=["POST"])
def extract_visit():
    body    = request.get_json(force=True, silent=True) or {}
    text    = body.get("text",             "").strip()
    conv_id = body.get("conversation_id",  "")
    links   = body.get("property_links",   [])

    if not text:
        return jsonify({"saved": False, "error": "No text"}), 400

    VISIT_KW = [
        "وزٹ","وزیر","visit","ملنا","آنا","شیڈول","schedule","confirm",
        "نام","نمبر","موبائل","بجے","جمعرات","جمعہ","ہفتہ","اتوار","پیر",
        "منگل","بدھ","کل","پرسوں","وقت رکھیں","ٹائم رکھیں","بکنگ","appointment",
    ]
    if not any(k in text.lower() for k in VISIT_KW):
        return jsonify({"saved": False, "reason": "No visit keywords"})

    vd          = extract_visit_regex(text)
    valid_links = [l for l in links if l and l != "unknown"]

    if not any([vd.get("client_name"), vd.get("phone_number"),
                vd.get("visit_date"),  vd.get("confirmed")]):
        return jsonify({"saved": False, "reason": "No visit data extracted"})

    save_visit_log(conv_id, vd, valid_links)

    return jsonify({
        "saved":       True,
        "client_name": vd.get("client_name"),
        "visit_date":  vd.get("visit_date"),
        "visit_time":  vd.get("visit_time"),
    })

# -----------------------------------------------
# HEALTH
# -----------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    conn  = get_db()
    total = conn.execute("SELECT COUNT(*) FROM properties").fetchone()[0]
    dha6  = conn.execute(
        "SELECT COUNT(*) FROM properties WHERE location LIKE '%dha phase 6%'"
    ).fetchone()[0]
    conn.close()
    return jsonify({
        "status":      "ok",
        "sqlite_rows": total,
        "dha_phase_6": dha6,
        "faiss_vecs":  faiss_index.ntotal,
        "supabase":    supabase is not None,
        "openai":      False,   # intentionally disabled
    })

if __name__ == "__main__":
    log.info("🚀 RAG Server → http://127.0.0.1:5001")
    app.run(host="127.0.0.1", port=5001, debug=False, threaded=True)





#     from flask import Flask, request, jsonify
# import json, faiss, numpy as np, os, time, re
# from sentence_transformers import SentenceTransformer
# from dotenv import load_dotenv
# from flask_cors import CORS
# import logging
# import math
# from collections import Counter
# load_dotenv()

# FAISS_INDEX_FILE = "faiss.index"
# METADATA_FILE    = "metadata.jsonl"
# TOP_K            = 3   # increased: gives more candidates before purpose/beds filtering

# _TYPE_URDU = {
#     "house": "مکان", "home": "گھر", "flat": "فلیٹ", "apartment": "اپارٹمنٹ",
#     "plot": "پلاٹ", "commercial": "کمرشل", "upper portion": "اپر پورشن",
#     "lower portion": "لوئر پورشن", "portion": "پورشن", "room": "کمرہ",
#     "office": "آفس", "shop": "دکان", "farm house": "فارم ہاؤس",
#     "villa": "ولا", "studio": "اسٹوڈیو", "penthouse": "پینٹ ہاؤس",
#     "townhouse": "ٹاؤن ہاؤس", "building": "عمارت",
# }
# _PURPOSE_URDU = {
#     "for sale": "فروخت کے لیے", "sale": "فروخت کے لیے",
#     "for rent": "کرایہ پر", "rent": "کرایہ پر",
# }

# def _localize_type(s: str) -> str:
#     return _TYPE_URDU.get(s.lower().strip(), s) if s else ""

# def _localize_purpose(s: str) -> str:
#     return _PURPOSE_URDU.get(s.lower().strip(), s) if s else ""

# def _urdu_size(size_str: str) -> str:
#     """Convert English size units to Urdu — 5 Marla → 5 مرلہ, 1 Kanal → 1 کنال"""
#     if not size_str: return ""
#     s = re.sub(r'(\d+(?:\.\d+)?)\s*[Mm]arla', r'\1 مرلہ', size_str)
#     s = re.sub(r'(\d+(?:\.\d+)?)\s*[Kk]anal', r'\1 کنال', s)
#     s = re.sub(r'(\d+(?:\.\d+)?)\s*[Ss]q\.?\s*[Ff]t', r'\1 مربع فٹ', s)
#     return s

# app = Flask(__name__)
# CORS(app)
# logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
# log = logging.getLogger(__name__)

# log.info("🔹 Loading embedding model...")
# embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
# log.info("✅ Multilingual embedding model ready!")

# log.info("🔹 Loading FAISS + metadata...")
# if not os.path.exists(FAISS_INDEX_FILE): raise FileNotFoundError(FAISS_INDEX_FILE)
# if not os.path.exists(METADATA_FILE):    raise FileNotFoundError(METADATA_FILE)

# index    = faiss.read_index(FAISS_INDEX_FILE)
# metadata = []
# with open(METADATA_FILE, "r", encoding="utf-8") as f:
#     for line in f:
#         line = line.strip()
#         if line:
#             metadata.append(json.loads(line))
# log.info(f"✅ {len(metadata)} records loaded")

# if metadata:
#     first = metadata[0].get("data", metadata[0])
#     log.info(f"🔍 First record keys: {list(first.keys())}")
#     log.info(f"🔍 First record sample: {dict(list(first.items())[:10])}")

# # ================= FUZZY CITY DETECTION =================
# CITY_MATCHERS = [
#     {
#         "city": "lahore",
#         "patterns": [
#             re.compile(r"لاہور"), re.compile(r"لاحور"), re.compile(r"لاہاور"),
#             re.compile(r"لہور"), re.compile(r"لحور"),
#             re.compile(r"اللہ\s*ہور"), re.compile(r"اللہ\s*اور"),
#             re.compile(r"lahore", re.I), re.compile(r"lahor\b", re.I),
#             re.compile(r"lahor[ae]", re.I), re.compile(r"la\s*hor", re.I),
#             re.compile(r"lhr\b", re.I),
#         ],
#     },
#     {
#         "city": "karachi",
#         "patterns": [
#             re.compile(r"کراچی"), re.compile(r"کراچ"),
#             re.compile(r"karachi", re.I), re.compile(r"krachi", re.I),
#             re.compile(r"karachy", re.I), re.compile(r"khi\b", re.I),
#         ],
#     },
#     {
#         "city": "islamabad",
#         "patterns": [
#             re.compile(r"اسلام\s*آباد"), re.compile(r"اسلام\s*اباد"),
#             re.compile(r"اسلامہ\s*ماہ"), re.compile(r"اسلاماباد"),
#             re.compile(r"islamabad", re.I), re.compile(r"islamaabad", re.I),
#             re.compile(r"islambad", re.I), re.compile(r"\bisb\b", re.I),
#         ],
#     },
#     {
#         "city": "rawalpindi",
#         "patterns": [
#             re.compile(r"راولپنڈی"), re.compile(r"راول\s*پنڈی"),
#             re.compile(r"rawalpindi", re.I), re.compile(r"rawlpindi", re.I),
#             re.compile(r"\bpindi\b", re.I), re.compile(r"\brwp\b", re.I),
#         ],
#     },
#     {
#         "city": "faisalabad",
#         "patterns": [
#             re.compile(r"فیصل\s*آباد"), re.compile(r"فیصل\s*اباد"),
#             re.compile(r"faisalabad", re.I), re.compile(r"faisalabd", re.I),
#             re.compile(r"lyallpur", re.I),
#         ],
#     },
#     {"city": "multan",   "patterns": [re.compile(r"ملتان"), re.compile(r"multan", re.I)]},
#     {"city": "peshawar", "patterns": [re.compile(r"پشاور"), re.compile(r"peshawar", re.I), re.compile(r"peshwar", re.I)]},
#     {"city": "quetta",   "patterns": [re.compile(r"کوئٹہ"), re.compile(r"quetta", re.I), re.compile(r"kwetta", re.I)]},
# ]

# NEIGHBOURHOOD_CITY_MAP = [
#     (re.compile(r"گلبر[گک]|gulberg", re.I),                         "lahore"),
#     (re.compile(r"جوہر\s*ٹاؤن|johar\s*town", re.I),                "lahore"),
#     (re.compile(r"جوکر\s*ٹاؤن|joker\s*town", re.I),                "lahore"),
#     (re.compile(r"جو[ہک]\s*ر?\s*ٹ[اآ]ؤ?ن", re.I),                 "lahore"),
#     (re.compile(r"ماڈل\s*ٹاؤن|model\s*town", re.I),                "lahore"),
#     (re.compile(r"بحریہ\s*ٹاؤن\s*(لاہور)?|bahria\s*town\s*(lahore)?", re.I), "lahore"),
#     (re.compile(r"dha\s*(lahore|لاہور)?", re.I),                    "lahore"),
#     (re.compile(r"علامہ\s*اقبال\s*ٹاؤن|allama\s*iqbal\s*town", re.I), "lahore"),
#     (re.compile(r"فیصل\s*ٹاؤن|faisal\s*town", re.I),               "lahore"),
#     (re.compile(r"گارڈن\s*ٹاؤن|garden\s*town", re.I),              "lahore"),
#     (re.compile(r"واپڈا\s*ٹاؤن|wapda\s*town", re.I),               "lahore"),
#     (re.compile(r"سمن\s*آباد|samanabad|samnabad", re.I),            "lahore"),
#     (re.compile(r"اقبال\s*ٹاؤن|iqbal\s*town", re.I),               "lahore"),
#     (re.compile(r"ٹاؤن\s*شپ|township\s*(lahore)?", re.I),          "lahore"),
#     (re.compile(r"رائے\s*ونڈ|raiwind", re.I),                       "lahore"),
#     (re.compile(r"کلفٹن|clifton", re.I),                             "karachi"),
#     (re.compile(r"گلشن(\s*اقبال)?|gulshan(\s*iqbal)?", re.I),       "karachi"),
#     (re.compile(r"dha\s*(defence|karachi|کراچی)", re.I),             "karachi"),
#     (re.compile(r"ڈیفنس|defence\s*housing", re.I),                   "karachi"),
#     (re.compile(r"نارتھ\s*ناظم\s*آباد|north\s*nazimabad", re.I),    "karachi"),
#     (re.compile(r"ناظم\s*آباد|nazimabad", re.I),                     "karachi"),
#     (re.compile(r"کورنگی|korangi", re.I),                            "karachi"),
#     (re.compile(r"لانڈھی|landhi", re.I),                             "karachi"),
#     (re.compile(r"ملیر|malir", re.I),                                "karachi"),
#     (re.compile(r"بہادر\s*آباد|bahadurabad", re.I),                  "karachi"),
#     (re.compile(r"پی\s*ای\s*سی\s*ایچ\s*ایس|pechs", re.I),          "karachi"),
#     (re.compile(r"فیڈرل\s*بی\s*ایریا|federal\s*b\s*area", re.I),   "karachi"),
#     (re.compile(r"[fف]-?10\b", re.I),                                "islamabad"),
#     (re.compile(r"[gگ]-?9\b",  re.I),                                "islamabad"),
#     (re.compile(r"[gگ]-?11\b", re.I),                                "islamabad"),
#     (re.compile(r"[gگ]-?10\b", re.I),                                "islamabad"),
#     (re.compile(r"[eای]-?7\b",  re.I),                               "islamabad"),
#     (re.compile(r"[fف]-?7\b",   re.I),                               "islamabad"),
#     (re.compile(r"[fف]-?6\b",   re.I),                               "islamabad"),
#     (re.compile(r"بحریہ\s*ٹاؤن\s*(اسلام\s*آباد|اسلامہ\s*ماہ)?|bahria\s*town\s*(islamabad)?", re.I), "islamabad"),
#     (re.compile(r"پارک\s*انکلیو|park\s*enclave", re.I),             "islamabad"),
#     (re.compile(r"سٹیلائٹ\s*ٹاؤن|satellite\s*town", re.I),         "rawalpindi"),
#     (re.compile(r"گلزار\s*قائد|gulzar\s*e?\s*quaid", re.I),        "rawalpindi"),
#     (re.compile(r"چکلالہ|chaklala", re.I),                           "rawalpindi"),
#     (re.compile(r"بحریہ\s*ٹاؤن\s*(راولپنڈی)?|bahria\s*town\s*(rawalpindi)?", re.I), "rawalpindi"),
#     (re.compile(r"جڑانوالہ|jaranwala", re.I),                        "faisalabad"),
#     (re.compile(r"سمندری|samundri", re.I),                           "faisalabad"),
# ]

# STT_AREA_CORRECTIONS = [
#     (re.compile(r"جوکر\s*ٹاؤن", re.I), "johar town lahore"),
#     (re.compile(r"جوہر\s*ٹاؤن", re.I), "johar town lahore"),
#     (re.compile(r"گلبر[گک]",     re.I), "gulberg lahore"),
#     (re.compile(r"ماڈل\s*ٹاؤن", re.I), "model town lahore"),
#     (re.compile(r"واپڈا\s*ٹاؤن", re.I),"wapda town lahore"),
#     (re.compile(r"فیصل\s*ٹاؤن", re.I), "faisal town lahore"),
#     (re.compile(r"گارڈن\s*ٹاؤن", re.I),"garden town lahore"),
#     (re.compile(r"ٹاؤن\s*شپ",   re.I), "township lahore"),
#     (re.compile(r"کلفٹن",       re.I), "clifton karachi"),
#     (re.compile(r"گلشن\s*اقبال",re.I), "gulshan iqbal karachi"),
#     (re.compile(r"ناظم\s*آباد", re.I), "nazimabad karachi"),
#     (re.compile(r"نارتھ\s*ناظم\s*آباد", re.I), "north nazimabad karachi"),
# ]

# def correct_stt_areas(text: str) -> str:
#     corrected = text
#     for pattern, replacement in STT_AREA_CORRECTIONS:
#         corrected = pattern.sub(replacement, corrected)
#     if corrected != text:
#         log.info(f"🔧 STT area corrected: '{text}' → '{corrected}'")
#     return corrected

# def detect_query_city(text: str) -> str | None:
#     for matcher in CITY_MATCHERS:
#         if any(p.search(text) for p in matcher["patterns"]):
#             log.info(f"🏙️ City detected (direct): '{matcher['city']}'")
#             return matcher["city"]
#     for pattern, city in NEIGHBOURHOOD_CITY_MAP:
#         if pattern.search(text):
#             log.info(f"🏙️ City detected (neighbourhood): '{city}'")
#             return city
#     return None

# # ================= PURPOSE DETECTION FROM QUERY =================
# # Detects whether user wants to rent or buy, extracted from the query text.
# # This is used to pre-filter FAISS candidates so rental queries don't return
# # sale listings and vice versa.

# _RENT_PATTERNS = re.compile(
#     r"کرای[ہے]|کرانا|کرائے|کرائیں|rent|rental|for\s*rent|kiraya|kiraye|"
#     r"ماہانہ|monthly|لیز|lease|کرایہ\s*پر",
#     re.I
# )
# _BUY_PATTERNS = re.compile(
#     r"خریدنا|خریدنے|خریدیں|خرید|buy|purchase|for\s*sale|فروخت|sell|selling|"
#     r"بیچنا|بیچیں|sale",
#     re.I
# )

# def detect_purpose_from_query(text: str) -> str | None:
#     """Returns 'rent', 'sale', or None if ambiguous."""
#     has_rent = bool(_RENT_PATTERNS.search(text))
#     has_buy  = bool(_BUY_PATTERNS.search(text))
#     if has_rent and not has_buy:
#         log.info("🎯 Purpose detected: rent")
#         return "rent"
#     if has_buy and not has_rent:
#         log.info("🎯 Purpose detected: sale")
#         return "sale"
#     return None   # ambiguous — don't filter

# # ================= BEDS DETECTION FROM QUERY =================
# # Extracts bedroom count from the query (e.g. "3 bed", "تین کمرے").
# # Used to soft-filter candidates.

# _BEDS_PATTERNS = [
#     (re.compile(r"\b1\s*bed|\bایک\s*کمر|\bone\s*bed", re.I),   1),
#     (re.compile(r"\b2\s*bed|\bدو\s*کمر|\btwo\s*bed", re.I),    2),
#     (re.compile(r"\b3\s*bed|\bتین\s*کمر|\bthree\s*bed", re.I), 3),
#     (re.compile(r"\b4\s*bed|\bچار\s*کمر|\bfour\s*bed", re.I),  4),
#     (re.compile(r"\b5\s*bed|\bپانچ\s*کمر|\bfive\s*bed", re.I), 5),
# ]

# def detect_beds_from_query(text: str) -> int | None:
#     for pattern, count in _BEDS_PATTERNS:
#         if pattern.search(text):
#             log.info(f"🛏️ Beds detected: {count}")
#             return count
#     return None

# # ================= FIELD NORMALIZATION =================
# CITY_IN_STRING_PATTERNS = [
#     (re.compile(r"lahore|لاہور", re.I),         "lahore"),
#     (re.compile(r"karachi|کراچی", re.I),         "karachi"),
#     (re.compile(r"islamabad|اسلام آباد", re.I),  "islamabad"),
#     (re.compile(r"rawalpindi|راولپنڈی", re.I),   "rawalpindi"),
#     (re.compile(r"faisalabad|فیصل آباد", re.I),  "faisalabad"),
#     (re.compile(r"multan|ملتان", re.I),           "multan"),
#     (re.compile(r"peshawar|پشاور", re.I),         "peshawar"),
#     (re.compile(r"quetta|کوئٹہ", re.I),           "quetta"),
# ]

# def extract_city_from_string(text: str) -> str:
#     if not text:
#         return ""
#     for pattern, city in CITY_IN_STRING_PATTERNS:
#         if pattern.search(text):
#             return city
#     return ""

# # ================= SIZE DETECTION =================
# _MARLA_NUM_RE = re.compile(r'(\d+(?:[./]\d+)?)\s*(?:مرل[ہے]|مر\s+ل[ہے]|marla)', re.I)

# _KANAL_NUM_RE = re.compile(r'(\d+(?:[./]\d+)?)\s*(?:کنال|kanal)',    re.I)
# _MARLA_WORDS = [
#     (re.compile(r'ایک\s*(?:مرل|مر\s*ل)',    re.I),  1),
#     (re.compile(r'دو\s*(?:مرل|مر\s*ل)',     re.I),  2),
#     (re.compile(r'تین\s*(?:مرل|مر\s*ل)',    re.I),  3),
#     (re.compile(r'چار\s*(?:مرل|مر\s*ل)',    re.I),  4),
#     (re.compile(r'پانچ\s*(?:مرل|مر\s*ل)',   re.I),  5),
#     (re.compile(r'چھ[ے]?\s*(?:مرل|مر\s*ل)', re.I),  6),
#     (re.compile(r'سات\s*(?:مرل|مر\s*ل)',    re.I),  7),
#     (re.compile(r'ساتھ\s*(?:مرل|مر\s*ل)',   re.I),  7),   # ← STT: سات → ساتھ
#     (re.compile(r'آٹھ\s*(?:مرل|مر\s*ل)',    re.I),  8),
#     (re.compile(r'نو\s*(?:مرل|مر\s*ل)',     re.I),  9),
#     (re.compile(r'دس\s*(?:مرل|مر\s*ل)',     re.I), 10),
#     (re.compile(r'گیارہ\s*(?:مرل|مر\s*ل)', re.I), 11),
#     (re.compile(r'بارہ\s*(?:مرل|مر\s*ل)',   re.I), 12),
#     (re.compile(r'پندرہ\s*(?:مرل|مر\s*ل)',  re.I), 15),
#     (re.compile(r'بیس\s*(?:مرل|مر\s*ل)',    re.I), 20),
#     # STT error variants (Whisper mishears مرلے as ملے / مندے / منڈے)
#     (re.compile(r'پانچ\s*(ملے|ملی|مندے|مند|منڈے|منلے)', re.I), 5),
#     (re.compile(r'دس\s*(ملے|ملی|مندے|مند|منڈے|منلے)',   re.I), 10),
#     (re.compile(r'سات\s*(ملے|ملی|مندے|مند|منڈے|منلے)',  re.I), 7),
#     (re.compile(r'ساتھ\s*(ملے|ملی|مندے|مند|منڈے|منلے)', re.I), 7),
#     (re.compile(r'تین\s*(ملے|ملی|مندے|مند|منڈے|منلے)',  re.I), 3),
#     (re.compile(r'چار\s*(ملے|ملی|مندے|مند|منڈے|منلے)',  re.I), 4),
#     (re.compile(r'آٹھ\s*(ملے|ملی|مندے|مند|منڈے|منلے)',  re.I), 8),
#     (re.compile(r'ایک\s*(ملے|ملی|مندے|مند|منڈے|منلے)',  re.I), 1),
#     (re.compile(r'دو\s*(ملے|ملی|مندے|مند|منڈے|منلے)',   re.I), 2),
# ]

# def detect_size_marla(text: str) -> float | None:
#     """Returns requested size in marla (1 kanal = 20 marla), or None."""
#     m = _MARLA_NUM_RE.search(text)
#     if m:
#         try: return float(m.group(1).replace('/', '.'))
#         except: pass
#     m = _KANAL_NUM_RE.search(text)
#     if m:
#         try: return float(m.group(1).replace('/', '.')) * 20
#         except: pass
#     for pat, val in _MARLA_WORDS:
#         if pat.search(text):
#             return float(val)
#     return None

# def parse_size_to_marla(size_str: str) -> float | None:
#     """Parse a property's size string into marla for comparison."""
#     if not size_str: return None
#     s = size_str.lower()
#     m = _KANAL_NUM_RE.search(s)
#     if m:
#         try: return float(m.group(1).replace('/', '.')) * 20
#         except: pass
#     m = _MARLA_NUM_RE.search(s)
#     if m:
#         try: return float(m.group(1).replace('/', '.'))
#         except: pass
#     # sq ft fallback (1 marla ≈ 272 sq ft)
#     m = re.search(r'(\d+(?:\.\d+)?)\s*(?:sq\.?\s*ft|sqft)', s)
#     if m:
#         try: return round(float(m.group(1)) / 272, 1)
#         except: pass
#     return None

# def _size_matches(size_str: str, query_marla: float | None, tol: float = 0.35) -> bool:
#     """True if property size is within tol% of the requested marla size."""
#     if query_marla is None: return True
#     prop_marla = parse_size_to_marla(size_str)
#     if prop_marla is None: return True          # unknown → don't filter out
#     return abs(prop_marla - query_marla) / query_marla <= tol

# def normalize_doc(raw: dict) -> dict:
#     def pick(*keys):
#         for k in keys:
#             v = raw.get(k)
#             if v is None: continue
#             # ← ADD: reject float NaN directly, not just the string "nan"
#             if isinstance(v, float) and math.isnan(v): continue
#             s = str(v).strip()
#             if s not in ("", "nan", "None", "unknown", "N/A", "n/a", "NaN", "inf", "-inf"):
#                 return s
#         return ""

#     city_direct   = pick("city", "City", "CITY", "city_name", "City_Name", "CityName")
#     location_area = pick(
#         "location", "Location", "LOCATION",
#         "area", "Area", "AREA",
#         "neighbourhood", "Neighbourhood", "neighborhood", "Neighborhood",
#         "locality", "Locality",
#         "address", "Address", "ADDRESS",
#         "sub_locality", "SubLocality",
#     )
#     city_canonical = (
#         extract_city_from_string(city_direct)
#         or extract_city_from_string(location_area)
#         or extract_city_from_string(pick("address", "Address", "full_address"))
#     )
#     if not city_canonical:
#         log.debug(f"⚠️  No city resolved. Raw keys: {list(raw.keys())[:12]}")

#     purpose_raw = pick(
#         "purpose", "Purpose", "PURPOSE",
#         "listing_purpose", "for_sale_or_rent", "type", "Type",
#         "listing_type", "Listing_Type",
#     ).lower()

#     # Normalise purpose to canonical "rent" or "sale"
#     if any(kw in purpose_raw for kw in ("rent", "کرای", "lease", "leas")):
#         purpose_canonical = "rent"
#     elif any(kw in purpose_raw for kw in ("sale", "sell", "buy", "خرید", "فروخت")):
#         purpose_canonical = "sale"
#     else:
#         purpose_canonical = purpose_raw  # keep raw if unrecognised

#     beds_raw = pick(
#         "bedrooms", "Bedrooms", "BEDROOMS", "beds", "Beds",
#         "bed", "Bed", "bedroom", "Bedroom",
#         "no_of_bedrooms", "num_bedrooms",
#     )
#     # Try to parse beds as integer for filtering
#     try:
#         beds_int = int(re.sub(r"[^\d]", "", beds_raw)) if beds_raw else None
#     except ValueError:
#         beds_int = None

#     return {
#         "city":             city_canonical,
#         "location_str":     location_area,
#         "size_str":         pick(
#             "area", "Area", "AREA", "plot_size", "Plot_Size",
#             "covered_area", "Covered_Area", "size", "Size",
#             "marla", "kanal", "sq_ft", "sqft",
#         ),
#         "size_marla": parse_size_to_marla(pick(   # ← ADD
#             "area", "Area", "AREA", "plot_size", "Plot_Size",
#             "covered_area", "size", "Size", "marla", "kanal", "sq_ft",
#         )),
#         "beds_str":         beds_raw,
#         "beds_int":         beds_int,
#         "price_str":        pick(
#             "price", "Price", "PRICE",
#             "asking_price", "Asking_Price",
#             "rent", "Rent", "RENT",
#             "monthly_rent", "Monthly_Rent",
#             "sale_price", "Sale_Price",
#             "total_price", "Total_Price",
#         ),
#         "type_str":         pick(
#             "property_type", "Property_Type", "PropertyType",
#             "Type", "type", "TYPE",
#             "category", "Category",
#         ),
#         "purpose_str":      purpose_raw,
#         "purpose":          purpose_canonical,   # ← canonical: "rent" | "sale" | ""
#         # ── URL FIELD ──────────────────────────────────────────────────────────
#         # Add your actual field name(s) here if different from these defaults.
#         # Check the "First record keys" log line at startup to find the exact name.
#         "url_str":          pick(
#             "url", "URL", "Url",
#             "link", "Link", "LINK",
#             "property_url", "Property_URL", "PropertyUrl",
#             "listing_url", "Listing_URL", "ListingUrl",
#             "detail_url", "Detail_URL",
#             "page_url", "Page_URL",
#             "zameen_url", "olx_url", "graana_url",
#         ),
#         "_raw": raw,
#     }

# def format_pkr_urdu(price_str: str) -> str:
#     """
#     Convert any price representation to Urdu lakh/crore.
#     Handles: raw digits (15000000), worded English (3.25 Crore, 89 Lakh),
#     and comma-formatted (1,50,00,000).
#     """
#     if not price_str: return price_str
#     s = price_str.strip()

#     # Already-worded English amounts (from dataset): "3.25 Crore", "89 Lakh"
#     crore_m = re.search(r'(\d+(?:\.\d+)?)\s*[Cc]ro(?:re)?', s)
#     lakh_m  = re.search(r'(\d+(?:\.\d+)?)\s*[Ll]a(?:kh|c)', s)
#     if crore_m:
#         val = float(crore_m.group(1))
#         return f"{int(val) if val == int(val) else val} کروڑ"
#     if lakh_m:
#         val = float(lakh_m.group(1))
#         return f"{int(val) if val == int(val) else val} لاکھ"

#     # Raw numeric amount
#     clean = re.sub(r"[,\s]", "", s)
#     m = re.search(r"(\d+(?:\.\d+)?)", clean)
#     if not m: return price_str
#     num = float(m.group(1))

#     if num >= 1_00_00_000:
#         val = round(num / 1_00_00_000, 2)
#         return f"{int(val) if val == int(val) else val} کروڑ"
#     if num >= 1_00_000:
#         val = round(num / 1_00_000, 2)
#         return f"{int(val) if val == int(val) else val} لاکھ"
#     if num >= 1_000:
#         val = round(num / 1_000, 1)
#         return f"{int(val) if val == int(val) else val} ہزار"
#     return price_str


# def format_property(norm: dict) -> str:
#     """Builds a fully Urdu property summary string for the LLM context."""
#     city    = norm["city"].capitalize() if norm["city"] else ""
#     loc     = norm["location_str"]
#     size    = _urdu_size(norm["size_str"])              # ← localized
#     beds    = norm["beds_str"]
#     price   = format_pkr_urdu(norm["price_str"])
#     ptype   = _localize_type(norm["type_str"])          # ← localized
#     purpose = _localize_purpose(norm["purpose_str"])    # ← localized

#     parts = []
#     if city and loc:
#         parts.append(f"{city}، {loc}" if city.lower() not in loc.lower() else loc)
#     elif city: parts.append(city)
#     elif loc:  parts.append(loc)

#     if ptype and purpose: parts.append(f"{ptype} ({purpose})")
#     elif ptype:           parts.append(ptype)
#     if size:                             parts.append(size)
#     if beds and beds not in ("0", ""):   parts.append(f"{beds} کمرے")
#     if price:                            parts.append(f"قیمت {price}")
#     return " — ".join(parts) if parts else "تفصیل دستیاب نہیں"

# def build_plain_summary(formatted_list: list[str]) -> str:
#     if not formatted_list:
#         return "معاف کیجیے، کوئی پراپرٹی نہیں ملی۔"
#     lines = [f"{i}️⃣ {p}" for i, p in enumerate(formatted_list, 1)]
#     lines.append("\nکیا آپ کسی پراپرٹی کا وزٹ شیڈول کرنا چاہیں گے؟")
#     return "\n".join(lines)


# # ── City inventory at startup ──────────────────────────────────────────────────
# log.info("🔹 Scanning city inventory...")
# _city_inventory: Counter = Counter()
# for _r in metadata:
#     _norm = normalize_doc(_r.get("data", _r))
#     if _norm["city"]:
#         _city_inventory[_norm["city"]] += 1

# if not _city_inventory:
#     log.error(
#         "❌ CITY INVENTORY IS EMPTY! normalize_doc() cannot extract city. "
#         "Check the field names logged above and add them to normalize_doc()."
#     )
# else:
#     log.info(f"✅ City inventory: {dict(_city_inventory)}")

# # ── Purpose inventory (logged for debugging) ─────────────────────────────────
# _purpose_inventory: Counter = Counter()
# _url_count = 0
# for _r in metadata:
#     _norm = normalize_doc(_r.get("data", _r))
#     if _norm["purpose"]: _purpose_inventory[_norm["purpose"]] += 1
#     if _norm["url_str"]: _url_count += 1
# log.info(f"✅ Purpose inventory: {dict(_purpose_inventory)}")
# log.info(f"✅ Properties with URL: {_url_count} / {len(metadata)}")


# def city_in_inventory(city: str) -> bool:
#     return _city_inventory.get(city, 0) > 0


# # ================= PURPOSE / BEDS FILTER HELPERS =================
# def _purpose_matches(norm_purpose: str, query_purpose: str) -> bool:
#     """True if the property purpose is compatible with what the user asked for."""
#     if not norm_purpose or not query_purpose:
#         return True   # unknown → don't filter out
#     return norm_purpose == query_purpose


# def _beds_matches(beds_int: int | None, query_beds: int | None, tolerance: int = 1) -> bool:
#     """
#     True if property beds are within tolerance of requested beds.
#     tolerance=1 means we accept ±1 bedroom to avoid being too strict.
#     """
#     if beds_int is None or query_beds is None:
#         return True   # unknown → don't filter out
#     return abs(beds_int - query_beds) <= tolerance

# def sanitize_for_json(obj):
#     """
#     Recursively replace float NaN / Infinity with None so Flask jsonify
#     produces valid JSON. Python's json module emits bare NaN which is
#     not valid JSON and causes JSON.parse to throw in Node.js.
#     """
#     if isinstance(obj, dict):
#         return {k: sanitize_for_json(v) for k, v in obj.items()}
#     if isinstance(obj, list):
#         return [sanitize_for_json(v) for v in obj]
#     if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
#         return None
#     return obj

# # ================= SEARCH ENDPOINT =================
# @app.route("/search", methods=["POST"])
# def search():
#     t0   = time.time()
#     body = request.get_json(force=True, silent=True) or {}
#     query      = (body.get("query")              or "").strip()
#     city_hint  = (body.get("city_hint")          or "").strip().lower() or None
#     neigh_hint = (body.get("neighbourhood_hint") or "").strip() or None

#     if not query:
#         return jsonify({"results": [], "urls": [], "raw_docs": [], "count": 0,
#             "summary": "معاف کیجیے گا، کوئی سوال موصول نہیں ہوا۔",
#             "city_detected": None, "city_used": None,
#             "city_match": False, "city_available": False, "elapsed": 0})

#     log.info(f"🔍 Query: {query!r}  |  city_hint: {city_hint!r}  |  neigh_hint: {neigh_hint!r}")

#     try:
#         city_in_query  = detect_query_city(query)
#         query_purpose  = detect_purpose_from_query(query)
#         query_beds     = detect_beds_from_query(query)
#         query_marla    = detect_size_marla(query)          # ← NEW
#         city           = city_in_query or city_hint
#         city_available = city_in_inventory(city) if city else True

#         log.info(f"🎯 Intent — city: {city} | purpose: {query_purpose} | beds: {query_beds} | marla: {query_marla}")

#         corrected_query = correct_stt_areas(query)
#         embed_parts = [corrected_query]
#         if neigh_hint:
#             embed_parts.append(neigh_hint)
#         if city and city not in corrected_query:
#             embed_parts.append(city)
#         # Only append purpose/beds as tight phrases, not loose keyword bags
#         if query_purpose == "rent":
#             embed_parts.append("rent kiraya")
#         elif query_purpose == "sale":
#             embed_parts.append("sale for sale")
#         if query_beds:
#             embed_parts.append(f"{query_beds} bed")
#         # Size appended as a normalized phrase
#         if query_marla:
#             marla_int = int(query_marla) if query_marla == int(query_marla) else query_marla
#             embed_parts.append(f"{marla_int} marla")
#         embed_query = " ".join(embed_parts)
#         log.info(f"📐 Embed query: '{embed_query}'")

#         qvec = embed_model.encode(embed_query).astype("float32")
#         qvec = np.expand_dims(qvec, 0)
#         faiss.normalize_L2(qvec)
#         distances, indices = index.search(qvec, TOP_K)

#         candidates = []
#         for idx, dist in zip(indices[0], distances[0]):
#             if idx < 0 or idx >= len(metadata) or dist < 0.05: continue
#             raw  = metadata[idx].get("data", metadata[idx])
#             norm = normalize_doc(raw)
#             candidates.append({"norm": norm, "raw": raw, "dist": float(dist)})

#         log.info(f"📦 {len(candidates)} candidates")

#         seen_keys = {}
#         deduped   = []
#         for c in candidates:
#             loc  = (c["norm"]["location_str"] or "").lower().strip()
#             purp = c["norm"]["purpose"] or ""
#             size = c["norm"]["size_str"] or ""
#             key  = f"{loc}|{purp}|{size}"
#             if key not in seen_keys:
#                 seen_keys[key] = True
#                 deduped.append(c)
#         if len(deduped) < len(candidates):
#             log.info(f"🧹 Deduplicated {len(candidates)} → {len(deduped)} candidates")
#         candidates = deduped

#         # ── Soft-filter: purpose + beds + size ───────────────────────────────
#         if query_purpose or query_beds or query_marla:
#             filtered = [
#                 c for c in candidates
#                 if _purpose_matches(c["norm"]["purpose"], query_purpose)
#                 and _beds_matches(c["norm"]["beds_int"], query_beds)
#                 and _size_matches(c["norm"]["size_str"], query_marla)   # ← NEW
#             ]
#             if len(filtered) >= 2:
#                 log.info(f"✅ After purpose/beds/size filter: {len(filtered)} (was {len(candidates)})")
#                 candidates = filtered
#             elif filtered:
#                 log.warning(f"⚠️ Only {len(filtered)} after filter — relaxing size tolerance")
#                 # Retry with looser size tolerance (50%)
#                 filtered2 = [
#                     c for c in candidates
#                     if _purpose_matches(c["norm"]["purpose"], query_purpose)
#                     and _beds_matches(c["norm"]["beds_int"], query_beds)
#                     and _size_matches(c["norm"]["size_str"], query_marla, tol=0.50)
#                 ]
#                 candidates = filtered2 if len(filtered2) >= 2 else candidates
#             else:
#                 log.warning("⚠️ 0 after filter — skipping (no matching data)")

#         # ── City-aware + neighbourhood preference scoring ─────────────────────
#         def neigh_score(norm):
#             """Bonus if property location contains the requested neighbourhood."""
#             if not neigh_hint: return 0
#             loc = (norm.get("location_str") or "").lower()
#             return 1 if neigh_hint.lower() in loc else 0

#         city_match = False
#         hits       = []
#         if city and city_available:
#             city_hits = [c for c in candidates if c["norm"]["city"] == city]
#             # ── Sort by neighbourhood match first, then embedding score ───────
#             city_hits.sort(key=lambda c: (-neigh_score(c["norm"]), -c["dist"]))  # ← NEW
#             log.info(f"🏙️ {len(city_hits)} '{city}' matches (neigh_hint={neigh_hint!r})")

#             if len(city_hits) >= 2:
#                 hits = city_hits[:3]; city_match = True
#             elif len(city_hits) == 1:
#                 others = [c for c in candidates if c["norm"]["city"] != city]
#                 hits   = city_hits + others[:2]; city_match = False
#                 log.warning("⚠️ Only 1 city hit — padded with others")
#             else:
#                 log.warning(f"⚠️ 0 '{city}' — widening search")
#                 wider_query = f"{neigh_hint} {city} property" if neigh_hint else f"{city} real estate property"
#                 if query_purpose: wider_query += f" for {query_purpose}"
#                 if query_marla:   wider_query += f" {query_marla} marla"
#                 q2 = embed_model.encode(wider_query).astype("float32")
#                 q2 = np.expand_dims(q2, 0)
#                 faiss.normalize_L2(q2)
#                 d2, i2 = index.search(q2, TOP_K * 3)
#                 wider = []
#                 for idx2, dist2 in zip(i2[0], d2[0]):
#                     if idx2 < 0 or idx2 >= len(metadata) or dist2 < 0.03: continue
#                     raw2  = metadata[idx2].get("data", metadata[idx2])
#                     norm2 = normalize_doc(raw2)
#                     if norm2["city"] == city:
#                         wider.append({"norm": norm2, "raw": raw2, "dist": float(dist2)})
#                 if wider:
#                     wider.sort(key=lambda c: (-neigh_score(c["norm"]), -c["dist"]))  # ← NEW
#                     hits = wider[:3]; city_match = True
#                 else:
#                     hits = candidates[:3]; city_match = False
#         elif city and not city_available:
#             hits = candidates[:3]; city_match = False
#         else:
#             hits = candidates[:3]; city_match = True

#         if not hits:
#             return jsonify({"results": [], "urls": [], "raw_docs": [], "count": 0,
#                 "summary": "معاف کیجیے، آپ کی تلاش کے مطابق کوئی پراپرٹی نہیں ملی۔",
#                 "city_detected": city_in_query, "city_used": city,
#                 "city_match": False, "city_available": city_available,
#                 "query_purpose": query_purpose, "query_beds": query_beds,
#                 "query_marla": query_marla, "elapsed": round(time.time() - t0, 2)})

#         formatted = [format_property(h["norm"]) for h in hits]
#         raw_docs  = [sanitize_for_json(h["raw"]) for h in hits]   # ← sanitize here
#         urls      = [h["norm"]["url_str"] for h in hits]
#         summary   = build_plain_summary(formatted)
#         elapsed   = round(time.time() - t0, 2)

#         log.info(f"✅ {len(formatted)} results | city='{city}' | purpose='{query_purpose}' | beds={query_beds} | marla={query_marla} | {elapsed}s")
#         for i, (f, u) in enumerate(zip(formatted, urls), 1):
#             log.info(f"   {i}. {f}")
#             if u: log.info(f"      🔗 {u}")

#         return jsonify({
#             "results": formatted, "urls": urls, "raw_docs": raw_docs,
#             "summary": summary, "count": len(formatted),
#             "city_detected": city_in_query, "city_used": city,
#             "city_match": city_match, "city_available": city_available,
#             "query_purpose": query_purpose, "query_beds": query_beds,
#             "query_marla": query_marla,    # ← NEW
#             "elapsed": elapsed,
#         })
#     except Exception as e:
#         log.error(f"❌ Search error: {e}", exc_info=True)
#         return jsonify({"results": [], "urls": [], "raw_docs": [], "count": 0,
#             "summary": "معاف کیجیے گا، پراپرٹی تلاش کرنے میں مسئلہ آ گیا۔",
#             "city_detected": None, "city_used": None,
#             "city_match": False, "city_available": False,
#             "elapsed": round(time.time() - t0, 2)}), 500

# @app.route("/health")
# def health():
#     return jsonify({
#         "status": "ok", "properties_count": len(metadata),
#         "faiss_total": index.ntotal, "faiss_dimension": index.d,
#         "city_inventory": dict(_city_inventory),
#         "purpose_inventory": dict(_purpose_inventory),
#         "url_count": _url_count,
#     })

# @app.route("/stats")
# def stats():
#     return jsonify({
#         "total_properties": len(metadata), "index_dimension": index.d,
#         "index_size": index.ntotal, "city_inventory": dict(_city_inventory),
#         "purpose_inventory": dict(_purpose_inventory),
#     })

# if __name__ == "__main__":
#     log.info("🚀 RAG Server → http://127.0.0.1:5001")
#     log.info(f"📊 {len(metadata)} properties | FAISS dim={index.d}")
#     log.info(f"🏙️ Cities: {dict(_city_inventory)}")
#     app.run(host="127.0.0.1", port=5001, debug=False, threaded=True)