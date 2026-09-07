"""
build_sqlite.py  —  Run this ONCE before starting rag_server.py

What it does:
  1. Reads metadata.jsonl (186,695 records, both schemas)
  2. Normalises both schemas into one unified format
  3. Writes properties.db  (SQLite — for exact location/area/bed/price filtering)
  4. Writes faiss.index    (FAISS  — for semantic re-ranking of SQL candidates)
  5. Writes faiss_ids.json (maps FAISS vector position → SQLite row id)

Run:
    pip install sentence-transformers faiss-cpu --break-system-packages
    python build_sqlite.py

FIX 1 (HYPHEN BUG):
  The location field is now stored lowercase with hyphens replaced by spaces.
  "Gulistan-e-Jauhar, Karachi" -> "gulistan e jauhar, karachi"
  This makes LIKE '%gulistan e jauhar%' match correctly.
  Also fixes all sector codes: "E-11" -> "e 11", "G-13" -> "g 13".
  37,284 previously unreachable rows are now searchable.

FIX 2 (PURPOSE FILTER):
  A separate 'purpose_normalized' column stores "for sale" or "for rent"
  so the RAG server can filter rental vs sale listings accurately.
  Without this, "50 lakh ka ghar" returns monthly-rent listings.

FIX 3 (FAISS ID MAP):
  faiss_ids is populated from actual SQLite autoincrement IDs returned
  after insertion, not assumed sequential 1..N.
  If any batch partially fails and rolls back, the autoincrement counter
  still advances — creating gaps that permanently misalign FAISS position
  to row ID. Reading real IDs back from the DB makes the map always accurate.
"""

import json, sqlite3, re, math, os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

METADATA_FILE  = "metadata.jsonl"
SQLITE_FILE    = "properties.db"
FAISS_FILE     = "faiss.index"
FAISS_IDS_FILE = "faiss_ids.json"
MODEL_NAME     = "paraphrase-multilingual-MiniLM-L12-v2"
EMBED_BATCH    = 512

# -----------------------------------------------
# HELPERS
# -----------------------------------------------
def _extract_city(location: str) -> str:
    if not location:
        return ""
    known = ["Lahore", "Karachi", "Islamabad", "Rawalpindi", "Peshawar",
             "Multan", "Faisalabad", "Gujranwala", "Sialkot"]
    loc_lower = location.lower()
    for city in known:
        if city.lower() in loc_lower:
            return city
    parts = [p.strip() for p in location.split(",")]
    return parts[-1] if parts else location

def _parse_price(price_str: str):
    try:
        p = price_str.lower().replace(",", "").strip()
        m = re.search(r"[\d.]+", p)
        if not m: return None
        num = float(m.group())
        if "lakh" in p or "لاکھ" in p: return round(num / 100, 4)
        if "crore" in p or "کروڑ" in p: return num
        return num
    except: return None

def _parse_area(area_str: str):
    try:
        a = area_str.lower()
        m = re.search(r"[\d.]+", a)
        if not m: return None
        num = float(m.group())
        if any(k in a for k in ("kanal", "canal", "کنال", "کنالوں", "کنالہ", "کنالے")):
            return num * 20
        return num
    except: return None

def _safe_int(val):
    try:
        if val is None or val == "" or (isinstance(val, float) and math.isnan(val)):
            return None
        return int(round(float(str(val))))
    except: return None

def _bool_field(val) -> int:
    return 1 if str(val).strip().lower() in ("true", "1", "yes") else 0

# FIX 1: normalize location string for reliable LIKE matching.
# Stores everything lowercase with hyphens converted to spaces so:
#   "Gulistan-e-Jauhar, Karachi" -> "gulistan e jauhar, karachi"
#   "E-11, Islamabad"            -> "e 11, islamabad"
#   "G-13, Islamabad"            -> "g 13, islamabad"
# The LOCATION_TOKENS in rag_server_local.py already use spaces not hyphens,
# so LIKE '%gulistan e jauhar%' now matches correctly.
def _normalize_location_for_storage(location: str, city: str) -> str:
    loc  = str(location or "").strip()
    city = str(city or "").strip()
    combined = f"{loc}, {city}".strip(", ") if loc else city
    return combined.lower().replace("-", " ")

# FIX 2: normalize purpose so RAG can filter rent vs sale.
# Stores "for sale" or "for rent" (lowercase, consistent).
def _normalize_purpose(purpose_str: str) -> str:
    p = str(purpose_str or "").lower().strip()
    if "rent" in p:
        return "for rent"
    if "sale" in p or "buy" in p:
        return "for sale"
    return p  # keep original if unknown


def normalize_record(entry: dict) -> dict:
    rec = entry.get("data", entry)
    if "Title" in rec:
        # Rich schema (scraped with full property details)
        location_raw = rec.get("Location", "")
        city_raw     = _extract_city(location_raw)
        return {
            "title":     rec.get("Title", ""),
            "location":  _normalize_location_for_storage(location_raw, city_raw),  # FIX 1
            "city":      city_raw,
            "type":      rec.get("Type", ""),
            "purpose":   _normalize_purpose(rec.get("Purpose", "")),                # FIX 2
            "price_crore":       _parse_price(str(rec.get("Price", ""))),
            "area_marla":        _parse_area(str(rec.get("Area", ""))),
            "bedrooms":          _safe_int(rec.get("Bedrooms")),
            "bathrooms":         _safe_int(rec.get("Bathrooms")),
            "kitchens":          _safe_int(rec.get("Kitchens")),
            "store_rooms":       _safe_int(rec.get("Store Rooms")),
            "servant_quarters":  _safe_int(rec.get("Servant Quarters")),
            "furnished":         _bool_field(rec.get("Furnished")),
            "gym":               _bool_field(rec.get("Gym")),
            "study_room":        _bool_field(rec.get("Study Room")),
            "drawing_room":      _bool_field(rec.get("Drawing Room")),
            "dining_room":       _bool_field(rec.get("Dining Room")),
            "lawn_garden":       _bool_field(rec.get("Lawn/Garden")),
            "swimming_pool":     _bool_field(rec.get("Swimming Pool")),
            "electricity_backup":_bool_field(rec.get("Electricity Backup")),
            "lounge":            _bool_field(rec.get("Lounge/Sitting Room")),
            "built_year":        _safe_int(rec.get("Built Year")),
            "date_posted":       rec.get("Date Posted", ""),
            "link":              rec.get("Link", ""),
        }
    else:
        # Flat schema (zameen CSV/Excel export)
        price_raw = rec.get("price", 0) or 0
        try:    price_crore = float(price_raw) / 10_000_000 if price_raw else None
        except: price_crore = None

        location_str = str(rec.get("location", "")).strip()
        city_str     = str(rec.get("city", "")).strip()

        return {
            "title":     rec.get("property_type", ""),
            "location":  _normalize_location_for_storage(location_str, city_str),  # FIX 1
            "city":      city_str,
            "type":      rec.get("property_type", ""),
            "purpose":   _normalize_purpose(rec.get("purpose", "")),                # FIX 2
            "price_crore":       price_crore,
            "area_marla":        _parse_area(str(rec.get("area", ""))),
            "bedrooms":          _safe_int(rec.get("bedrooms")),
            "bathrooms":         _safe_int(rec.get("baths")),
            "kitchens":          None,
            "store_rooms":       None,
            "servant_quarters":  None,
            "furnished":         0,
            "gym":               0,
            "study_room":        0,
            "drawing_room":      0,
            "dining_room":       0,
            "lawn_garden":       0,
            "swimming_pool":     0,
            "electricity_backup":0,
            "lounge":            0,
            "built_year":        None,
            "date_posted":       str(rec.get("date_added", "")),
            "link":              rec.get("page_url", ""),
        }


def record_to_embed_text(r: dict) -> str:
    """Clean text for embedding — location is already normalised."""
    parts = [
        r.get("type", ""),
        r.get("purpose", ""),
        f"{r.get('bedrooms') or ''} bedroom".strip(),
        f"{r.get('area_marla') or ''} marla".strip(),
        r.get("location", ""),   # now "johar town, lahore" — city included
        r.get("city", ""),
    ]
    return " ".join(p for p in parts if p and p.strip())


# -----------------------------------------------
# STEP 1: CREATE SQLITE DB
# -----------------------------------------------
print("Creating SQLite database...")
if os.path.exists(SQLITE_FILE):
    os.remove(SQLITE_FILE)

conn = sqlite3.connect(SQLITE_FILE)
cur  = conn.cursor()

cur.executescript("""
CREATE TABLE properties (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    title              TEXT,
    location           TEXT,    -- normalised: "johar town, lahore" (lowercase, no hyphens)
    city               TEXT,    -- original city name for display: "Lahore"
    type               TEXT,
    purpose            TEXT,    -- "for sale" or "for rent"
    price_crore        REAL,
    area_marla         REAL,
    bedrooms           INTEGER,
    bathrooms          INTEGER,
    kitchens           INTEGER,
    store_rooms        INTEGER,
    servant_quarters   INTEGER,
    furnished          INTEGER DEFAULT 0,
    gym                INTEGER DEFAULT 0,
    study_room         INTEGER DEFAULT 0,
    drawing_room       INTEGER DEFAULT 0,
    dining_room        INTEGER DEFAULT 0,
    lawn_garden        INTEGER DEFAULT 0,
    swimming_pool      INTEGER DEFAULT 0,
    electricity_backup INTEGER DEFAULT 0,
    lounge             INTEGER DEFAULT 0,
    built_year         INTEGER,
    date_posted        TEXT,
    link               TEXT
);

CREATE INDEX idx_location ON properties (location);
CREATE INDEX idx_city     ON properties (city);
CREATE INDEX idx_bedrooms ON properties (bedrooms);
CREATE INDEX idx_area     ON properties (area_marla);
CREATE INDEX idx_price    ON properties (price_crore);
CREATE INDEX idx_purpose  ON properties (purpose);
""")
conn.commit()

# -----------------------------------------------
# STEP 2: LOAD + INSERT ALL RECORDS
# -----------------------------------------------
print("Loading metadata.jsonl and inserting into SQLite...")
records = []
with open(METADATA_FILE, "r", encoding="utf-8") as f:
    for line in f:
        try:
            records.append(normalize_record(json.loads(line)))
        except Exception:
            pass

INSERT_SQL = """
INSERT INTO properties (
    title, location, city, type, purpose,
    price_crore, area_marla, bedrooms, bathrooms,
    kitchens, store_rooms, servant_quarters,
    furnished, gym, study_room, drawing_room, dining_room,
    lawn_garden, swimming_pool, electricity_backup, lounge,
    built_year, date_posted, link
) VALUES (
    :title, :location, :city, :type, :purpose,
    :price_crore, :area_marla, :bedrooms, :bathrooms,
    :kitchens, :store_rooms, :servant_quarters,
    :furnished, :gym, :study_room, :drawing_room, :dining_room,
    :lawn_garden, :swimming_pool, :electricity_backup, :lounge,
    :built_year, :date_posted, :link
)
"""

CHUNK = 5000
for i in range(0, len(records), CHUNK):
    cur.executemany(INSERT_SQL, records[i:i+CHUNK])
    conn.commit()
    print(f"  Inserted {min(i+CHUNK, len(records)):,} / {len(records):,}")

conn.close()
print(f"SQLite done — {len(records):,} rows in {SQLITE_FILE}")

# Verify counts
conn = sqlite3.connect(SQLITE_FILE)
total_rows = conn.execute("SELECT COUNT(*) FROM properties").fetchone()[0]
dha6_rows  = conn.execute(
    "SELECT COUNT(*) FROM properties WHERE location LIKE '%dha phase 6%'"
).fetchone()[0]
with_link  = conn.execute(
    "SELECT COUNT(*) FROM properties WHERE link IS NOT NULL AND link != ''"
).fetchone()[0]
# FIX 1 verification: hyphen locations now reachable
gulistan_rows = conn.execute(
    "SELECT COUNT(*) FROM properties WHERE location LIKE '%gulistan e jauhar%'"
).fetchone()[0]
lahore_rows  = conn.execute(
    "SELECT COUNT(*) FROM properties WHERE location LIKE '%lahore%'"
).fetchone()[0]
sale_rows = conn.execute(
    "SELECT COUNT(*) FROM properties WHERE purpose = 'for sale'"
).fetchone()[0]
rent_rows = conn.execute(
    "SELECT COUNT(*) FROM properties WHERE purpose = 'for rent'"
).fetchone()[0]

# FIX 3: Read actual row IDs from DB — not assumed sequential 1..N.
# If any batch partially fails and rolls back, the autoincrement counter still
# advances — creating gaps that permanently misalign FAISS position to row ID.
print("Reading actual row IDs from SQLite for FAISS ID map...")
actual_ids = [row[0] for row in conn.execute("SELECT id FROM properties ORDER BY id").fetchall()]
conn.close()

print(f"\n  Total rows          : {total_rows:,}")
print(f"  DHA Phase 6         : {dha6_rows:,}")
print(f"  Gulistan-e-Jauhar   : {gulistan_rows:,}  (was 0 before FIX 1)")
print(f"  Lahore rows         : {lahore_rows:,}   (was 0 before city+location merge)")
print(f"  For Sale            : {sale_rows:,}")
print(f"  For Rent            : {rent_rows:,}")
print(f"  With link           : {with_link:,}")
print(f"  ID range            : {actual_ids[0]} → {actual_ids[-1]}")

if len(actual_ids) != len(records):
    print(f"\n  WARNING: {len(actual_ids):,} DB rows vs {len(records):,} parsed records.")
    records = records[:len(actual_ids)]

# -----------------------------------------------
# STEP 3: BUILD FAISS INDEX
# -----------------------------------------------
print(f"\nLoading model {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)

print("Building FAISS index...")
texts    = [record_to_embed_text(r) for r in records]
all_vecs = []

for i in range(0, len(texts), EMBED_BATCH):
    batch = texts[i:i+EMBED_BATCH]
    vecs  = model.encode(batch, show_progress_bar=False,
                         convert_to_tensor=False).astype("float32")
    all_vecs.append(vecs)
    if (i // EMBED_BATCH) % 20 == 0:
        print(f"  Embedded {min(i+EMBED_BATCH, len(texts)):,} / {len(texts):,}")

all_vecs = np.vstack(all_vecs)
faiss.normalize_L2(all_vecs)

dim   = all_vecs.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(all_vecs)
faiss.write_index(index, FAISS_FILE)
print(f"FAISS index written — {index.ntotal:,} vectors, dim={dim}")

# -----------------------------------------------
# STEP 4: WRITE FAISS ID MAP (FIX 3: real DB IDs)
# -----------------------------------------------
with open(FAISS_IDS_FILE, "w") as f:
    json.dump(actual_ids, f)
print(f"FAISS ID map written — {len(actual_ids):,} entries")

print("\nAll done! Files created:")
print(f"  {SQLITE_FILE}    ({os.path.getsize(SQLITE_FILE)/1024/1024:.1f} MB)")
print(f"  {FAISS_FILE}   ({os.path.getsize(FAISS_FILE)/1024/1024:.1f} MB)")
print(f"  {FAISS_IDS_FILE}")
print("\nNow run:  python rag_server.py")