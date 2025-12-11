// server.js
import 'dotenv/config';
import express from "express";
import fetch from "node-fetch";
import cors from "cors";
import path from "path";
import { fileURLToPath } from "url";
import fs from "fs";
import pg from "pg";

const { Pool } = pg;
const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
});

const app = express();
app.use(cors());
// keep existing text parsers
app.use(express.text());
app.use(express.text({ type: "application/sdp" }));
// allow JSON for some endpoints if needed
app.use(express.json());

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/* -------------------------------------------------------------
   CREATE /conversations AND /conversations/audio DIRECTORY IF NOT EXISTS
------------------------------------------------------------- */
const conversationsDir = path.join(__dirname, "conversations");
const audioDir = path.join(conversationsDir, "audio");
if (!fs.existsSync(conversationsDir)) {
  fs.mkdirSync(conversationsDir);
  console.log("📁 conversations folder created.");
}
if (!fs.existsSync(audioDir)) {
  fs.mkdirSync(audioDir);
  console.log("📁 conversations/audio folder created.");
}

/* -------------------------------------------------------------
   CONVERSATION IN-MEMORY STATE (unchanged but extended for audio)
------------------------------------------------------------- */
let conversationState = {
  currentFile: null,
  messages: [],
  detectedPhones: new Set(),
  detectedIntents: new Set(),
  detectedLocations: new Set(),
  detectedVisitDates: new Set(),
  lastRAGMessage: "",
  // added for audio
  audioFilePath: null,   // file path (on disk) of the last uploaded mixed recording
  audioBuffer: null      // Buffer of last uploaded mixed recording (kept until finalize)
};

/* -------------------------------------------------------------
   REGEX HELPERS
------------------------------------------------------------- */
const PHONE_REGEX = /\+92\d{9}/g;
const PROPERTY_INTENT_REGEX = /plot|property|پلاٹ|زمین|phase|DHA|Bahria|فیز|مرلہ|marla/i;
const LOCATION_REGEX = /(DHA|Bahria|Gulberg|Phase\s?\d+|DHA Phase\s?\d+|Park View City)/ig;
const DATE_ISO_REGEX = /(\d{4}[-\/]\d{2}[-\/]\d{2})/g;
const DATE_DM_REGEX = /(\b\d{1,2}[-\/]\d{1,2}[-\/]\d{2,4}\b)/g;
const END_PHRASES_REGEX = /\b(thanks|thank you|bye|goodbye|khuda hafiz|خدا حافظ|شکریہ|الوداع)\b/i;

/* -------------------------------------------------------------
   STATIC FILE SERVER
------------------------------------------------------------- */
app.use(express.static(path.join(__dirname, "public")));

/* -------------------------------------------------------------
  START CONVERSATION (creates TXT in /conversations)
------------------------------------------------------------- */
app.post("/start-conversation", (req, res) => {
  try {
    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    const filename = `conversation-${timestamp}.txt`;
    const fullpath = path.join(conversationsDir, filename);

    fs.writeFileSync(
      fullpath,
      `New Urdu conversation started: ${new Date().toLocaleString()}\n\n`
    );

    conversationState = {
      currentFile: fullpath,
      messages: [],
      detectedPhones: new Set(),
      detectedIntents: new Set(),
      detectedLocations: new Set(),
      detectedVisitDates: new Set(),
      lastRAGMessage: "",
      audioFilePath: null,
      audioBuffer: null
    };

    console.log("🟢 Conversation started:", filename);
    res.send({ file: filename });

  } catch (err) {
    console.error("❌ start-conversation error:", err);
    res.status(500).send("Could not start conversation");
  }
});

/* -------------------------------------------------------------
  SAVE CONVERSATION MESSAGE (no DB insert here)
------------------------------------------------------------- */
app.post("/save-conversation", async (req, res) => {
  try {
    const raw = req.body || "";
    if (!raw) return res.status(400).send("No message provided");

    const roleMatch = raw.match(/^([A-Za-z_]+):\s*(.*)/);
    const role = roleMatch ? roleMatch[1].toLowerCase() : "user";
    const text = roleMatch ? roleMatch[2] : raw;
    const ts = new Date().toLocaleString();

    // Append to TXT
    if (conversationState.currentFile) {
      fs.appendFileSync(
        conversationState.currentFile,
        `[${ts}] ${role.toUpperCase()}: ${text}\n`,
        "utf-8"
      );
    } else {
      console.warn("No active conversation file to append to.");
    }

    conversationState.messages.push({ role, text, ts });

    // Extract phone, intent, location, dates
    (text.match(PHONE_REGEX) || []).forEach(p => conversationState.detectedPhones.add(p));
    if (PROPERTY_INTENT_REGEX.test(text)) conversationState.detectedIntents.add("property_query");
    if (/intent:.*\b(buy|sell|visit|inquiry)\b/i.test(text)) {
      conversationState.detectedIntents.add("explicit_intent");
    }

    Array.from(text.matchAll(LOCATION_REGEX)).forEach(m =>
      conversationState.detectedLocations.add(m[0].trim())
    );

    [...(text.match(DATE_ISO_REGEX) || []), ...(text.match(DATE_DM_REGEX) || [])]
      .forEach(d => conversationState.detectedVisitDates.add(d));

    // Handle property queries → RAG
    if (PROPERTY_INTENT_REGEX.test(text) && role === "user") {
      try {
        const ragSummary = await runRAGQuery(text);
        const reply = makeUrduReply(ragSummary);

        if (conversationState.currentFile) {
          fs.appendFileSync(
            conversationState.currentFile,
            `[${new Date().toLocaleString()}] AI: ${reply}\n`,
            "utf-8"
          );
        }

        conversationState.messages.push({
          role: "assistant",
          text: reply,
          ts: new Date().toLocaleString()
        });

        conversationState.lastRAGMessage = reply;

        return res.send("RAG completed.");
      } catch (err) {
        console.error("RAG error:", err);
        return res.send("RAG failed.");
      }
    }

    // Auto end via ending phrase
    if (END_PHRASES_REGEX.test(text) && role === "user") {
      console.log("🔚 End phrase detected → Ending conversation...");
      await finalizeAndSaveConversation("end-phrase-detected");
      return res.send("Conversation ended.");
    }

    res.send("Message appended.");

  } catch (err) {
    console.error("❌ save-conversation error:", err);
    res.status(500).send("Error saving message");
  }
});

/* -------------------------------------------------------------
   UPLOAD AUDIO (raw bytes) — saves file & stores buffer in memory
   client should POST application/octet-stream to this endpoint
------------------------------------------------------------- */
app.post('/upload-audio', express.raw({ type: ['audio/*', 'application/octet-stream'], limit: '60mb' }), async (req, res) => {
  try {
    if (!conversationState.currentFile) return res.status(400).send("No active conversation to attach audio.");

    const audioBuffer = Buffer.from(req.body); // req.body is a Buffer because of express.raw
    const baseName = path.basename(conversationState.currentFile, '.txt');
    const ts = Date.now();
    const audioFileName = `${baseName}-mixed-${ts}.webm`;
    const audioFilePath = path.join(audioDir, audioFileName);

    // write file to disk
    fs.writeFileSync(audioFilePath, audioBuffer);
    // save into in-memory state to insert into DB at finalize
    conversationState.audioFilePath = audioFilePath;
    conversationState.audioBuffer = audioBuffer;

    console.log(`🟦 Audio uploaded and saved: ${audioFileName} (${audioBuffer.length} bytes)`);
    res.send({ ok: true, file: audioFileName });
  } catch (err) {
    console.error("upload-audio error:", err);
    res.status(500).send("Audio upload failed");
  }
});

/* -------------------------------------------------------------
   MANUAL END OF CONVERSATION
------------------------------------------------------------- */
app.post("/end-conversation", async (req, res) => {
  try {
    if (!conversationState.currentFile)
      return res.status(400).send("No active conversation.");

    const reason = req.body || "manual_end";
    const result = await finalizeAndSaveConversation(reason);
    res.send(result);

  } catch (err) {
    console.error("❌ end-conversation error:", err);
    res.status(500).send("Error ending conversation");
  }
});

/* -------------------------------------------------------------
   CORE FINALIZATION (translation → RAG → extraction → DB)
------------------------------------------------------------- */
async function finalizeAndSaveConversation(reason = "manual") {
  const filePath = conversationState.currentFile;

  let fullText = "";
  try {
    fullText = fs.readFileSync(filePath, "utf-8");
  } catch {
    fullText = conversationState.messages
      .map(m => `[${m.ts}] ${m.role.toUpperCase()}: ${m.text}`)
      .join("\n");
  }

  /* ---------------- TRANSLATE URDU → ENGLISH ---------------- */
  let translatedText = "";
  try {
    const gptRes = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "gpt-4o-mini",
        messages: [
          { role: "system", content: "Translate Urdu conversation to English. No summary." },
          { role: "user", content: fullText }
        ]
      })
    });

    const gptData = await gptRes.json();
    translatedText = gptData.choices?.[0]?.message?.content || fullText;

  } catch (err) {
    console.error("Translation failed:", err);
    translatedText = fullText;
  }

  /* ---------------- RAG SUMMARY ON ENGLISH ---------------- */
  let finalSummary = "";
  try {
    finalSummary = await runRAGQuery(translatedText);
  } catch {
    finalSummary = "Summary unavailable.";
  }

  /* ---------------- EXTRACT FINAL STRUCTURED INFO ---------------- */
  const phone = Array.from(new Set([
    ...conversationState.detectedPhones,
    ...(translatedText.match(/\+92\d{9}/g) || [])
  ]))[0] || null;

  const intent =
    Array.from(conversationState.detectedIntents)[0] ||
    (/plot|property/i.test(translatedText) ? "property_query" : "general");

  const locations =
    translatedText.match(/DHA Phase \d+|DHA|Bahria Town|Gulberg|Park View City/gi) || [];

  Array.from(locations).forEach(l => conversationState.detectedLocations.add(l));

  const plot_location = Array.from(conversationState.detectedLocations)[0] || null;

  let visit_date = null;
  const rawDate = Array.from(conversationState.detectedVisitDates)[0];
  if (rawDate) {
    const iso = rawDate.match(/\d{4}[-\/]\d{2}[-\/]\d{2}/);
    if (iso) visit_date = iso[0];
  }

  const conversation_summary = {
    summary: finalSummary,
    translated_english_text: translatedText,
    reason_ended: reason,
    extracted: { phone, intent, plot_location, visit_date },
    last_rag: conversationState.lastRAGMessage,
    messages_preview: conversationState.messages.slice(-30),
    audio_file: conversationState.audioFilePath ? path.basename(conversationState.audioFilePath) : null
  };

  /* ---------------- SINGLE DB INSERT (includes audio_data BYTEA) ---------------- */
  try {
    await pool.query(
      `INSERT INTO call_records
        (conversation_file, phone_number, intent, plot_location, visit_date, conversation_summary, audio_data, faiss_ref)
       VALUES ($1,$2,$3,$4,$5,$6,$7,$8)`,
      [
        path.basename(filePath),
        phone,
        intent,
        plot_location,
        visit_date,
        JSON.stringify(conversation_summary),
        conversationState.audioBuffer || null, // Buffer -> BYTEA
        null
      ]
    );

    console.log("💾 Saved ONE DB row for:", path.basename(filePath));
  } catch (err) {
    console.error("Final DB insert error:", err);
    throw new Error("DB insert failed: " + err.message);
  }

  /* ---------------- RESET MEMORY ---------------- */
  conversationState = {
    currentFile: null,
    messages: [],
    detectedPhones: new Set(),
    detectedIntents: new Set(),
    detectedLocations: new Set(),
    detectedVisitDates: new Set(),
    lastRAGMessage: "",
    audioFilePath: null,
    audioBuffer: null
  };

  return "Conversation saved as ONE DB record.";
}

/* -------------------------------------------------------------
   RAG AND GPT HELPERS (unchanged)
------------------------------------------------------------- */
async function runRAGQuery(queryText) {
  try {
    const response = await fetch("http://127.0.0.1:5001/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: queryText }),
    });

    const data = await response.json();
    if (data.summary) return data.summary;
    if (data.results?.length > 0) return data.results.join("\n");
    return "کوئی متعلقہ معلومات نہیں ملیں۔";

  } catch (error) {
    console.error("RAG failed:", error);
    return "ڈیٹا حاصل کرنے میں مسئلہ پیش آیا۔";
  }
}

function makeUrduReply(ragText) {
  if (!ragText || ragText.length < 10)
    return "معاف کیجئے گا، فی الحال اس علاقے کی تازہ تفصیل دستیاب نہیں ہے۔";

  return `آپ کی درخواست کے مطابق میں نے ڈیٹا چیک کیا ہے۔ 
${ragText}
کیا آپ مزید قریبی علاقوں کے پلاٹس دیکھنا چاہیں گے؟`;
}

/* -------------------------------------------------------------
   REALTIME SESSION ENDPOINT (unchanged)
------------------------------------------------------------- */
app.post("/session", async (req, res) => {
  try {
    const offer = req.body;

    const gptResponse = await fetch(
      "https://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17",
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
          "Content-Type": "application/sdp",
        },
        body: offer,
      }
    );

    const answer = await gptResponse.text();
    res.set("Content-Type", "application/sdp");
    res.send(answer);

  } catch (err) {
    console.error("Realtime session error:", err);
    res.status(500).send("Error creating GPT session");
  }
});

/* -------------------------------------------------------------
   START SERVER
------------------------------------------------------------- */
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`🚀 Server running at http://localhost:${PORT}`);
  console.log(`📄 Conversations stored in /conversations`);
});
