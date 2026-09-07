# =============================================================================
# speech.py  —  Python Worker (Port 5002)
# Provides: /stt  /tts_urdu_stream  /tts_urdu  /health
# =============================================================================

import asyncio, queue, os, io, tempfile, logging, threading
import numpy as np
import edge_tts
from faster_whisper import WhisperModel
from flask import Flask, request, send_file, jsonify, Response, stream_with_context
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL", "small")
COMPUTE_TYPE       = os.getenv("COMPUTE_TYPE",  "int8")
MIN_AUDIO_RMS      = float(os.getenv("MIN_AUDIO_RMS", "80"))
TTS_VOICE          = os.getenv("TTS_VOICE", "ur-PK-AsadNeural")

logging.info(f"⏳ Loading Whisper ({WHISPER_MODEL_SIZE}, {COMPUTE_TYPE})...")
whisper_model = WhisperModel(
    WHISPER_MODEL_SIZE, device="cpu", compute_type=COMPUTE_TYPE,
    cpu_threads=min(os.cpu_count() or 4, 8), num_workers=2,
)
logging.info("✅ Whisper ready!")

WHISPER_INITIAL_PROMPT = (
    "یہ ایک پاکستانی رئیل اسٹیٹ IVR سروس ہے۔ "
    "گھر، مکان، فلیٹ، دکان، پلاٹ، کمرہ، "
    "کرایہ، خریدنا، بیچنا، سرمایہ کاری، "
    "لاہور، کراچی، اسلام آباد، راولپنڈی، فیصل آباد، "
    "ڈیفنس، گلبرگ، بحریہ ٹاؤن، جوہر ٹاؤن، ماڈل ٹاؤن، "
    "مرلہ، کنال، لاکھ، کروڑ، بجٹ، قیمت۔"
)

# ── Shared async loop for TTS ──────────────────────────────────────────────────
_tts_loop   = asyncio.new_event_loop()
_tts_thread = threading.Thread(target=lambda: _tts_loop.run_forever(), daemon=True)
_tts_thread.start()
_STREAM_DONE = object()

# ── Urdu diacritics to strip (cause TTS mispronunciation) ─────────────────────
_URDU_DIACRITICS = (
    "\u064B\u064C\u064D\u064E\u064F\u0650"
    "\u0651\u0652\u0653\u0654\u0655\u0656"
    "\u0657\u0658\u0670"
)

def _clean_tts_text(text: str) -> str:
    import re
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"#+\s*", "", text)
    text = text.translate(str.maketrans("", "", _URDU_DIACRITICS))
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ── RMS energy gate ────────────────────────────────────────────────────────────
def compute_audio_rms(file_path: str) -> float:
    try:
        import subprocess
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", file_path, "-ar", "16000", "-ac", "1",
             "-f", "s16le", "-loglevel", "error", "pipe:1"],
            capture_output=True, timeout=10
        )
        if result.returncode != 0 or not result.stdout:
            return 0.0
        samples = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32)
        rms = float(np.sqrt(np.mean(samples ** 2))) if len(samples) else 0.0
        logging.info(f"🔊 Audio RMS: {rms:.1f} (gate={MIN_AUDIO_RMS})")
        return rms
    except Exception as e:
        logging.warning(f"⚠️  RMS check failed (fail-open): {e}")
        return float("inf")

# ── STT ────────────────────────────────────────────────────────────────────────
@app.route("/stt", methods=["POST"])
def stt():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file"}), 400
        audio_file = request.files["file"]
        suffix = os.path.splitext(audio_file.filename or ".webm")[1] or ".webm"
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp_path = tmp.name
                audio_file.save(tmp_path)

            rms = compute_audio_rms(tmp_path)
            if rms < MIN_AUDIO_RMS:
                return jsonify({"text": "", "language_prob": 0.0, "rejected": True, "reason": "noise_floor"})

            segments, info = whisper_model.transcribe(
                tmp_path, language="ur", task="transcribe",
                initial_prompt=WHISPER_INITIAL_PROMPT,
                beam_size=2, temperature=0.2,
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0, no_speech_threshold=0.65,
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters=dict(
                    threshold=0.50, min_speech_duration_ms=350,
                    max_speech_duration_s=float("inf"),
                    min_silence_duration_ms=400, speech_pad_ms=100,
                ),
                word_timestamps=False, without_timestamps=True,
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
            logging.info(f"✅ STT [{info.language} {info.language_probability:.0%}] '{text[:80]}'")

            if info.language_probability < 0.55 or len(text) < 3:
                return jsonify({"text": "", "language_prob": info.language_probability,
                                "rejected": True, "reason": "low_confidence"})
            return jsonify({"text": text, "language_prob": info.language_probability})
        finally:
            if tmp_path:
                try: os.unlink(tmp_path)
                except OSError: pass

    except Exception as e:
        logging.error(f"❌ STT Error: {e}", exc_info=True)
        return jsonify({"error": str(e), "text": ""}), 500

# ── TTS Streaming ──────────────────────────────────────────────────────────────
@app.route("/tts_urdu_stream", methods=["POST"])
def tts_urdu_stream():
    data = request.get_json(force=True, silent=True) or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text"}), 400
    text = _clean_tts_text(text)
    logging.info(f"🔊 TTS stream: '{text[:60]}'")

    chunk_q = queue.Queue()

    async def _fill_queue():
        try:
            communicate = edge_tts.Communicate(text, TTS_VOICE, rate="+20%", pitch="+0Hz")
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    chunk_q.put(chunk["data"])
        except Exception as e:
            logging.error(f"❌ TTS async error: {e}")
        finally:
            chunk_q.put(_STREAM_DONE)

    asyncio.run_coroutine_threadsafe(_fill_queue(), _tts_loop)

    def _sync_generator():
        while True:
            try:
                item = chunk_q.get(timeout=60)
            except queue.Empty:
                logging.warning("⚠️ TTS stream timeout")
                break
            if item is _STREAM_DONE:
                break
            yield item

    return Response(
        stream_with_context(_sync_generator()),
        mimetype="audio/mpeg",
        headers={"X-Accel-Buffering": "no"},
    )

# ── TTS Non-streaming (greeting) ──────────────────────────────────────────────
@app.route("/tts_urdu", methods=["POST"])
def tts_urdu():
    data = request.get_json(force=True, silent=True) or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text"}), 400
    text = _clean_tts_text(text)

    async def _gen():
        buf = io.BytesIO()
        communicate = edge_tts.Communicate(text, TTS_VOICE, rate="+20%", pitch="+0Hz")
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])
        buf.seek(0)
        return buf.read()

    try:
        future      = asyncio.run_coroutine_threadsafe(_gen(), _tts_loop)
        audio_bytes = future.result(timeout=90)
        if not audio_bytes:
            return jsonify({"error": "TTS produced no audio"}), 500
        return send_file(io.BytesIO(audio_bytes), mimetype="audio/mpeg", as_attachment=False)
    except asyncio.TimeoutError:
        return jsonify({"error": "TTS timeout"}), 504
    except Exception as e:
        logging.error(f"❌ TTS Error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health():
    return jsonify({
        "status": "ok", "model": WHISPER_MODEL_SIZE,
        "compute": COMPUTE_TYPE, "voice": TTS_VOICE,
        "min_audio_rms": MIN_AUDIO_RMS,
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, threaded=True, use_reloader=False)