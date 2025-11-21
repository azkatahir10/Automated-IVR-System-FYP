import 'dotenv/config';
import express from "express";
import fetch from "node-fetch";
import cors from "cors";
import path from "path";
import { fileURLToPath } from "url";
import fs from "fs";

const app = express();
app.use(cors());
app.use(express.text());
app.use(express.text({ type: "application/sdp" }));

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let currentFile = null;
let lastRAGMessage = "";

// Serve static frontend files
app.use(express.static(path.join(__dirname, "public")));

// --- Conversation setup ---
app.post("/start-conversation", (req, res) => {
  const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
  currentFile = path.join(__dirname, `conversation-${timestamp}.txt`);
  fs.writeFileSync(currentFile, " New Urdu conversation started\n");
  res.send({ file: path.basename(currentFile) });
});

// --- Save conversation + detect property queries ---
app.post("/save-conversation", async (req, res) => {
  if (!currentFile) return res.status(400).send("No active conversation file.");
  const message = req.body || "";
  fs.appendFileSync(currentFile, `[${new Date().toLocaleString()}] ${message}\n`);

  const isPropertyQuery = /dha|phase|plot|پلاٹ|زمین|فیز|property|خرید/i.test(message);

  if (isPropertyQuery) {
    console.log("Property query detected:", message);
    res.send("⏳ کچھ دیر ٹھہریں، میں ڈیٹا چیک کر کے بتاتا ہوں...");

    try {
      const ragSummary = await runRAGQuery(message);
      const fullReply = makeUrduReply(ragSummary);

      fs.appendFileSync(currentFile, `[AI]: ${fullReply}\n`);
      lastRAGMessage = fullReply;
    } catch (err) {
      console.error("Error fetching RAG summary:", err);
      lastRAGMessage = "معاف کیجئے گا، اس وقت ڈیٹا حاصل کرنے میں کچھ دشواری ہو رہی ہے۔";
    }

    return;
  }

  res.send(" Saved");
});

// --- Fetch Urdu summary from Python ---
async function runRAGQuery(queryText) {
  try {
    const response = await fetch("http://127.0.0.1:5001/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: queryText }),
    });

    const data = await response.json();
    if (data.summary) return data.summary;
    if (data.results && data.results.length > 0) return data.results.join("\n");
    return "کوئی متعلقہ معلومات نہیں ملیں۔";
  } catch (error) {
    console.error("⚠️ RAG request failed:", error);
    return "ڈیٹا حاصل کرنے میں مسئلہ پیش آیا۔";
  }
}

// --- Natural Urdu tone ---
function makeUrduReply(ragText) {
  if (!ragText || ragText.length < 10)
    return "معاف کیجئے گا، فی الحال اس علاقے کی تازہ تفصیل دستیاب نہیں ہے۔";

  return `آپ کی درخواست کے مطابق میں نے ڈیٹا چیک کیا ہے۔ 
${ragText}
کیا آپ چاہیں گی کہ میں آپ کو مزید قریبی علاقوں کے پلاٹس دکھاؤں؟`;
}

// --- Provide latest RAG response to frontend ---
app.get("/rag-latest", (req, res) => {
  res.json({ text: lastRAGMessage });
  lastRAGMessage = "";
});

// --- Realtime GPT session handling ---
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
    console.error(" Error creating session:", err);
    res.status(500).send("Error creating GPT session");
  }
});

// --- Start server ---
const PORT = 3000;
app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
  console.log(`Open http://localhost:${PORT}/index.html`);
});




















