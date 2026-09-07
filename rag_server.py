"""
rag_server.py  —  Urdu RAG Property Search Server (Port 5001)
Architecture:
  Layer 1 — SQLite WHERE clause  (location, area, bedrooms, price, purpose)
  Layer 2 — FAISS cosine re-rank (on small candidate set)
  Layer 3 — Local template Urdu summary  (zero API cost)
  Visit  —  Regex + relative-date extraction (offline)
Supabase: rag_searches + client_visits tables (optional, non-fatal if absent)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json, sqlite3, faiss, numpy as np, os, re, math, datetime, logging
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

try:
    from supabase import create_client
    _SUPABASE_AVAILABLE = True
except ImportError:
    _SUPABASE_AVAILABLE = False

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['HF_HUB_OFFLINE'] = '1'
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Supabase ──────────────────────────────────────────────────────────────────
supabase = None
if _SUPABASE_AVAILABLE and os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SERVICE_KEY"):
    supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))
    log.info("✅ Supabase ready")
else:
    log.info("ℹ️  Supabase not configured — logging disabled (non-fatal)")

app = Flask(__name__)
CORS(app)

# ── Config ────────────────────────────────────────────────────────────────────
SQLITE_FILE       = os.getenv("SQLITE_FILE",    "properties.db")
FAISS_FILE        = os.getenv("FAISS_FILE",     "faiss.index")
FAISS_IDS_FILE    = os.getenv("FAISS_IDS_FILE", "faiss_ids.json")
SQLITE_CANDIDATES = 50

log.info("⏳ Loading multilingual embedding model...")
embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
log.info("✅ Embedding model ready!")

log.info("⏳ Loading FAISS index...")
faiss_index = faiss.read_index(FAISS_FILE)
log.info(f"✅ FAISS: {faiss_index.ntotal} vectors")

with open(FAISS_IDS_FILE, "r") as f:
    faiss_ids: list[int] = json.load(f)
log.info(f"✅ FAISS ID map: {len(faiss_ids)} entries")

# Startup count — close connection immediately
_c = sqlite3.connect(SQLITE_FILE)
_total = _c.execute("SELECT COUNT(*) FROM properties").fetchone()[0]
_c.close()   # FIX: was not closing startup connection
log.info(f"✅ SQLite ready: {_total:,} rows")
log.info("🚀 RAG Server ready on port 5001\n")


def get_db():
    conn = sqlite3.connect(SQLITE_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


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


def to_urdu(doc: dict) -> str:
    FIELDS = [
        ("title",            "عنوان"),
        ("location",         "مقام"),
        ("type",             "قسم"),
        ("price_crore",      "قیمت"),
        ("area_marla",       "رقبہ"),
        ("bedrooms",         "بیڈ روم"),
        ("bathrooms",        "باتھ روم"),
        ("kitchens",         "کچن"),
        ("store_rooms",      "اسٹور روم"),
        ("servant_quarters", "ملازم کوارٹر"),
        ("built_year",       "تعمیری سال"),
        ("purpose",          "مقصد"),
        ("date_posted",      "تاریخ اشاعت"),
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


# ── Query normalisation (4 passes) ────────────────────────────────────────────

WHISPER_MAP = {
    "ڈی اے چے": "dha", "ڈی اے جے": "dha", "ڈی ایچ اے": "dha", "ڈی ایچ": "dha",
    "دی ایچ اے": "dha", "ری ایچ": "dha", "ڈیفنس": "defence", "ڈیفینس": "defence",
    "دی اے چے": "dha", "دی ایچ اے فیز": "dha phase", "ڈی ایچ اے فیز": "dha phase",
    "فیز ون": "phase 1", "فیس ون": "phase 1",
    "فیز ٹو": "phase 2", "فیس ٹو": "phase 2",
    "فیز تھری": "phase 3", "فیس تھری": "phase 3",
    "فیز فور": "phase 4",  "فیس فور": "phase 4",
    "فیز فائیو": "phase 5","فیز فائف": "phase 5","فیس فائیو": "phase 5",
    "فیز سکس": "phase 6", "فیس سکس": "phase 6", "فیس اکس": "phase 6",
    "فیز ڈکس": "phase 6", "فیز دکس": "phase 6", "فیز کس": "phase 6",
    "فیز سیون": "phase 7","فیس سیون": "phase 7",
    "فیز ایٹ": "phase 8", "فیس ایٹ": "phase 8",
    "فیز نائن": "phase 9","فیس نائن": "phase 9",
    "فیز دس": "phase 10", "فیس دس": "phase 10",
    "جوہر ٹاؤن": "johar town", "جوہر": "johar",
    "جوہا": "johar", "جو ہاٹ": "johar", "جوہا ٹاؤن": "johar town",
    "بحریہ": "bahria", "بہریہ": "bahria",
    "گلبرگ": "gulberg", "گلبرک": "gulberg",
    "گارڈن ٹاؤن": "garden town", "ماڈل ٹاؤن": "model town",
    "والنشیا": "valencia", "فیصل ٹاؤن": "faisal town",
    "ٹاؤن شپ": "township", "واپڈا ٹاؤن": "wapda town",
    "لاہور": "lahore", "اسلام آباد": "islamabad", "اسلام ابد": "islamabad",
    "کراچی": "karachi", "راولپنڈی": "rawalpindi",
    "پشاور": "peshawar", "ملتان": "multan",
    "نالے": "kanal", "نالا": "kanal", "کنالا": "kanal",
}

def normalize_query(q):
    for k in sorted(WHISPER_MAP, key=len, reverse=True):
        q = q.replace(k, WHISPER_MAP[k])
    return q

def normalize_phase_numbers(q):
    NUM = [
        ("ون","1"),("one","1"),("ٹو","2"),("two","2"),
        ("تھری","3"),("three","3"),("فور","4"),("four","4"),
        ("فائیو","5"),("فائف","5"),("five","5"),
        ("سکس","6"),("سِکس","6"),("six","6"),("اکس","6"),("ڈکس","6"),("کس","6"),
        ("سیون","7"),("seven","7"),("ایٹ","8"),("eight","8"),
        ("نائن","9"),("nine","9"),("دس","10"),("ten","10"),
    ]
    for w, d in NUM:
        q = re.sub(rf'(phase)\s+{re.escape(w)}', rf'\g<1> {d}', q, flags=re.IGNORECASE)
    return q

def normalize_spoken_numbers(q):
    NW = [
        ("اکیس","21"),("بائیس","22"),("تئیس","23"),("چوبیس","24"),("پچیس","25"),
        ("چھبیس","26"),("ستائیس","27"),("اٹھائیس","28"),("انتیس","29"),("تیس","30"),
        ("چالیس","40"),("پچاس","50"),("ساٹھ","60"),("ستر","70"),
        ("اسی","80"),("نوے","90"),
        ("اٹھارہ","18"),("سترہ","17"),("سولہ","16"),("پندرہ","15"),("چودہ","14"),
        ("تیرہ","13"),("بارہ","12"),("گیارہ","11"),
        ("ڈیڑھ","1.5"),("اڑھائی","2.5"),
        ("ساڑھے تین","3.5"),("ساڑھے چار","4.5"),("ساڑھے پانچ","5.5"),
        ("اک","1"),("اِک","1"),("نو","9"),("آٹھ","8"),("سات","7"),
        ("چھ","6"),("پانچ","5"),("چار","4"),("تین","3"),("دو","2"),("ایک","1"),
    ]
    for w, d in NW:
        q = re.sub(rf'(?<![\w\d]){re.escape(w)}(?![\w\d])', d, q)
    return q

ROMAN_MAP = {
    "ghr":"house","makan":"house","flat":"flat","dukan":"shop","plot":"plot",
    "villa":"villa","portion":"portion","office":"office",
    "kenal":"kanal","knal":"kanal","knaal":"kanal",
    "mrla":"marla","marley":"marla",
    "chahiya":"for sale","chahye":"for sale",
    "kiraya":"for rent","bikao":"for sale",
    "kamra":"bedroom","kamray":"bedroom",
    "ma":"in","mein":"in","ka":"","ki":"","ke":"","wala":"","wali":"",
}
def normalize_roman(q):
    words = q.lower().split()
    return " ".join(" ".join(ROMAN_MAP.get(w, w) for w in words).split())

def full_normalize(query: str) -> str:
    return normalize_roman(normalize_spoken_numbers(normalize_phase_numbers(normalize_query(query))))


# ── Location extraction ────────────────────────────────────────────────────────

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
    matched = [tok for tok in LOCATION_TOKENS
               if re.search(r'(?<![\w])' + re.escape(tok) + r'(?![\w])', q)]
    if not matched: return []
    matched.sort(key=len, reverse=True)
    final = []
    for tok in matched:
        if not any(tok in longer for longer in final):
            final.append(tok)
    return final


# ── Filters ────────────────────────────────────────────────────────────────────

def parse_filters(query: str) -> dict:
    f = {}
    q = query.lower()
    b = re.search(r'(\d+)\s*(?:bed|bedroom|بیڈ|کمر)', q)
    if b: f["Bedrooms"] = int(b.group(1))
    c = re.search(r'(\d+(?:\.\d+)?)\s*(?:crore|کروڑ)', q)
    l = re.search(r'(\d+(?:\.\d+)?)\s*(?:lakh|لاکھ)', q)
    if c:   f["max_price_crore"] = float(c.group(1))
    elif l: f["max_price_crore"] = float(l.group(1)) / 100
    m = re.search(r'(\d+(?:\.\d+)?)\s{0,2}(?:marla|mrla|مرلہ|مرلے|مرلا|مرل|ملے|ملہ)', q)
    k = re.search(r'(\d+(?:\.\d+)?)\s{0,2}(?:kanal|canal|کنال|کنالوں)', q)
    if m:   f["area_marla"] = float(m.group(1))
    elif k: f["area_marla"] = float(k.group(1)) * 20
    RENT_KW = ["kiraya","کرایہ","kiraiye","rent","for rent","lease"]
    SALE_KW = ["khareedna","خریدنا","for sale","bikao","sale","buy"]
    if any(k in q for k in RENT_KW):   f["purpose"] = "for rent"
    elif any(k in q for k in SALE_KW): f["purpose"] = "for sale"
    return f


# ── Layer 1: SQLite ────────────────────────────────────────────────────────────

def sqlite_search(loc_words: list, filters: dict, include_area=True) -> list[dict]:
    conds, params = [], []
    if loc_words:
        conds.append("(" + " OR ".join(["location LIKE ?"] * len(loc_words)) + ")")
        params += [f"%{t}%" for t in loc_words]
    if "Bedrooms" in filters:
        conds.append("bedrooms = ?"); params.append(filters["Bedrooms"])
    if "max_price_crore" in filters:
        conds.append("(price_crore IS NULL OR price_crore <= ?)"); params.append(filters["max_price_crore"])
    if include_area and "area_marla" in filters:
        am = filters["area_marla"]
        conds.append("(area_marla IS NULL OR (area_marla BETWEEN ? AND ?))"); params += [am * 0.75, am * 1.25]
    if "purpose" in filters:
        conds.append("purpose = ?"); params.append(filters["purpose"])
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


# ── Layer 2: FAISS re-rank ─────────────────────────────────────────────────────

def faiss_rerank(candidates: list[dict], q_vec: np.ndarray, top_n=5) -> list[dict]:
    if not candidates: return []
    if len(candidates) <= top_n: return candidates
    id_to_pos = {rid: pos for pos, rid in enumerate(faiss_ids)}
    vecs, valid = [], []
    for doc in candidates:
        pos = id_to_pos.get(doc["id"])
        if pos is not None and pos < faiss_index.ntotal:
            vecs.append(faiss_index.reconstruct(pos)); valid.append(doc)
    if not vecs: return candidates[:top_n]
    va = np.array(vecs, dtype="float32")
    faiss.normalize_L2(va)
    tmp = faiss.IndexFlatIP(va.shape[1])
    tmp.add(va)
    q = q_vec.copy().reshape(1, -1).astype("float32")
    faiss.normalize_L2(q)
    _, idxs = tmp.search(q, min(top_n, tmp.ntotal))
    return [valid[i] for i in idxs[0] if i < len(valid)]


# ── Layer 3: Urdu summary ──────────────────────────────────────────────────────

ORDINALS_UR = ["پہلی", "دوسری", "تیسری", "چوتھی", "پانچویں"]

def build_summary(docs, query, history, is_spec, no_results=False,
                  no_results_reason="", requested_size="") -> str:
    if no_results:
        if no_results_reason == "location":
            return ("معاف کیجئے، اس مقام پر فی الحال ہمارے پاس کوئی پراپرٹی دستیاب نہیں ہے۔ "
                    "کیا آپ کسی اور علاقے کے بارے میں پوچھنا چاہیں گے؟")
        return ("معاف کیجئے، آپ کی دی گئی تفصیلات سے ملتی پراپرٹی فی الحال دستیاب نہیں ہے۔ "
                "اگر آپ بجٹ یا سائز میں تھوڑی تبدیلی کریں تو میں بہتر آپشن تلاش کر سکتا ہوں۔")
    if not docs:
        return "معاف کیجئے، اس وقت کوئی متعلقہ پراپرٹی نہیں ملی۔"

    if requested_size:
        intro = (f"معاف کیجئے، بالکل {requested_size} کی پراپرٹی فی الحال دستیاب نہیں ہے، "
                 "لیکن یہ قریب ترین آپشنز ہیں:\n\n")
    elif is_spec:
        intro = "ٹھیک ہے، یہ تفصیلات ملاحظہ کریں:\n\n"
    else:
        intro = "جی، آپ کے سوال کے مطابق یہ چند پراپرٹیز دستیاب ہیں:\n\n"

    blocks = []
    for i, doc in enumerate(docs):
        label = ORDINALS_UR[i] if i < len(ORDINALS_UR) else f"{i+1}ویں"
        # Strip markdown from summary (TTS reads ** aloud)
        clean_doc = re.sub(r'\*+', '', doc)
        if is_spec or requested_size:
            blocks.append(f"{label} پراپرٹی:\n{clean_doc}")
        else:
            blocks.append(clean_doc)

    closing = ("\n\nکیا یہ آپشنز آپ کے لیے مناسب ہو سکتے ہیں؟"
               if requested_size
               else "\n\nان میں سے کوئی پراپرٹی پسند آئی، یا مزید تفصیل چاہیں گے؟")
    return intro + "\n\n".join(blocks) + closing


# ── Relative date resolution ───────────────────────────────────────────────────

_URDU_DAYS = {
    "پیر":0,"monday":0,"منگل":1,"tuesday":1,"بدھ":2,"wednesday":2,
    "جمعرات":3,"thursday":3,"جمعہ":4,"friday":4,"ہفتہ":5,"saturday":5,"اتوار":6,"sunday":6,
}

def _resolve_relative_date(text: str) -> str | None:
    today = datetime.date.today()
    lower = text.lower()
    if "آج" in text or "today" in lower:    return today.isoformat()
    if "کل" in text or "tomorrow" in lower: return (today + datetime.timedelta(days=1)).isoformat()
    if "پرسوں" in text or "day after tomorrow" in lower:
        return (today + datetime.timedelta(days=2)).isoformat()
    for ur_day, weekday in _URDU_DAYS.items():
        if ur_day in text:
            days_ahead = (weekday - today.weekday()) % 7 or 7
            return (today + datetime.timedelta(days=days_ahead)).isoformat()
    return None


# ── Visit extraction ───────────────────────────────────────────────────────────

def extract_visit_regex(text: str) -> dict:
    vd = {"client_name":None,"phone_number":None,"visit_date":None,
          "visit_time":None,"selected_property":None,"confirmed":False}
    phone = re.search(r'(\+92|0092|0)[\s-]?(3\d{2})[\s-]?(\d{7})', text)
    if phone: vd["phone_number"] = f"0{phone.group(2)}{phone.group(3)}"
    date = re.search(r'(\d{4}-\d{2}-\d{2})', text)
    if date:
        vd["visit_date"] = date.group(1)
    else:
        date2 = re.search(r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})', text)
        if date2:
            vd["visit_date"] = f"{date2.group(3)}-{date2.group(2).zfill(2)}-{date2.group(1).zfill(2)}"
        else:
            vd["visit_date"] = _resolve_relative_date(text)
    time_ = re.search(r'(\d{1,2}):(\d{2})\s*(?:am|pm|AM|PM|بجے)?', text)
    if time_: vd["visit_time"] = f"{time_.group(1).zfill(2)}:{time_.group(2)}"
    if not vd["visit_time"]:
        t_word = re.search(r'(\d{1,2})\s*(?:بجے|o\'?clock)', text)
        if t_word: vd["visit_time"] = f"{t_word.group(1).zfill(2)}:00"
    CONFIRM_KW = ["confirm","ٹھیک ہے","ہاں","yes","بالکل","ضرور","ok","okay"]
    if any(k in text.lower() for k in CONFIRM_KW): vd["confirmed"] = True
    name_ur = re.search(r'(?:میرا\s*نام|نام)\s+([^\s،,۔.]{2,20})', text)
    name_en = re.search(r'(?:my\s+name\s+is|name\s*[:\-]?\s*)([A-Za-z]{2,30})', text, re.I)
    if name_ur:   vd["client_name"] = name_ur.group(1).strip()
    elif name_en: vd["client_name"] = name_en.group(1).strip()
    return vd


def save_visit_to_supabase(conv_id, vd, valid_links):
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
        log.info(f"📅 Visit saved: {vd.get('client_name')} | {vd.get('visit_date')}")
    except Exception as e:
        log.warning(f"⚠️ client_visits insert error (non-fatal): {e}")


def save_rag_search(conv_id, query, summary, props, specific):
    if not supabase: return
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


# ── Misc helpers ───────────────────────────────────────────────────────────────

def is_goodbye(text: str) -> bool:
    return any(e in text.strip().lower().replace("۔", "")
               for e in ["اللہ حافظ","خدا حافظ","bye","thanks","شکریہ","alvida"])

def is_specific(query: str) -> bool:
    return any(k in query.lower() for k in [
        "تفصیل","details","options","آپشن","دکھائیں","show","بتائیں",
        "tell","کون سی","which","list","لسٹ","سب","all","مزید","more","کونسا",
    ])


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/search", methods=["POST"])
def search():
    body    = request.get_json(force=True, silent=True) or {}
    query   = body.get("query",   "").strip()
    history = body.get("history", "")
    conv_id = body.get("conversation_id", "")

    if not query:
        return jsonify({"error": "No query"}), 400

    if is_goodbye(query):
        return jsonify({"results":[],"summary":"ٹھیک ہے، اللہ حافظ! اپنا خیال رکھیں۔","matched_properties":[]})

    q_norm  = full_normalize(query)
    spec    = is_specific(q_norm)
    filters = parse_filters(q_norm)
    locs    = extract_locations(q_norm)

    log.info(f"🔍 Query: {query!r}")
    log.info(f"📐 Normalised: {q_norm!r}")
    log.info(f"🏙️ Locations: {locs} | Filters: {filters}")

    q_vec = embed_model.encode(q_norm, convert_to_tensor=False).astype("float32")

    candidates = sqlite_search(locs, filters, include_area=True)

    if locs and not candidates:
        msg = build_summary([], q_norm, history, spec, no_results=True, no_results_reason="location")
        return jsonify({"results":[],"summary":msg,"matched_properties":[],"no_results":True,"no_results_reason":"location"})

    area_mismatch, size_str = False, ""
    if not candidates and "area_marla" in filters:
        area_mismatch = True
        am = filters["area_marla"]
        size_str = (f"{int(am)} مرلے" if am < 20
                    else (f"{int(am//20)} کنال" if am % 20 == 0 else f"{am/20:.1f} کنال"))
        candidates = sqlite_search(locs, filters, include_area=False)

    if not candidates:
        msg = build_summary([], q_norm, history, spec, no_results=True, no_results_reason="filters")
        return jsonify({"results":[],"summary":msg,"matched_properties":[],"no_results":True,"no_results_reason":"filters"})

    top = faiss_rerank(candidates, q_vec, top_n=5)
    top = sorted(top[:5], key=lambda d: 0 if (d.get("link") and d["link"] not in ("","unknown")) else 1)[:3]
    urdu_lines = [to_urdu(d) for d in top]
    summary    = build_summary(urdu_lines, q_norm, history, spec,
                               requested_size=size_str if area_mismatch else "")

    log.info(f"✅ {len(urdu_lines)} results returned")
    if conv_id:
        save_rag_search(conv_id, query, summary, top, spec)

    return jsonify({
        "results":            urdu_lines,
        "summary":            summary,
        "is_specific":        spec,
        "count":              len(urdu_lines),
        "no_results":         False,
        "matched_properties": sanitize(top),
    })


@app.route("/extract-visit", methods=["POST"])
def extract_visit():
    body    = request.get_json(force=True, silent=True) or {}
    text    = body.get("text",            "").strip()
    conv_id = body.get("conversation_id", "")
    links   = body.get("property_links",  [])

    if not text:
        return jsonify({"saved": False, "error": "No text"}), 400

    VISIT_KW = [
        "وزٹ","visit","ملنا","آنا","شیڈول","schedule","confirm",
        "نام","نمبر","موبائل","بجے","جمعرات","جمعہ","ہفتہ","اتوار","پیر",
        "منگل","بدھ","کل","پرسوں","آج","بکنگ","appointment",
    ]
    if not any(k in text.lower() for k in VISIT_KW):
        return jsonify({"saved": False, "reason": "No visit keywords"})

    vd          = extract_visit_regex(text)
    valid_links = [l for l in links if l and l != "unknown"]

    if not any([vd.get("client_name"), vd.get("phone_number"),
                vd.get("visit_date"),  vd.get("confirmed")]):
        return jsonify({"saved": False, "reason": "No visit data extracted"})

    save_visit_to_supabase(conv_id, vd, valid_links)
    return jsonify({"saved":True,"client_name":vd.get("client_name"),
                    "visit_date":vd.get("visit_date"),"visit_time":vd.get("visit_time")})


@app.route("/health")
def health():
    conn  = get_db()
    total = conn.execute("SELECT COUNT(*) FROM properties").fetchone()[0]
    conn.close()
    return jsonify({
        "status":      "ok",
        "sqlite_rows": total,
        "faiss_vecs":  faiss_index.ntotal,
        "supabase":    supabase is not None,
        "openai":      False,
    })


if __name__ == "__main__":
    log.info("🚀 RAG Server → http://127.0.0.1:5001")
    app.run(host="127.0.0.1", port=5001, debug=False, threaded=True)