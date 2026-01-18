import 'dotenv/config';
import express from "express";
import fetch from "node-fetch";
import cors from "cors";
import path from "path";
import { fileURLToPath } from "url";
import fs from "fs";

const app = express();
app.use(cors());
app.use(express.json());           // Important for JSON body
app.use(express.text({ type: "application/sdp" }));

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let currentFile = null;            // Stores current TXT filename
let lastRAGMessage = "";

// Serve static frontend
app.use(express.static(path.join(__dirname, "public")));

// --- Start conversation ---
app.post("/start-conversation", (req, res) => {
  const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
  currentFile = path.join(__dirname, `conversation-${timestamp}.txt`);
  fs.writeFileSync(currentFile, "New Urdu conversation started\n", { encoding: "utf8" });
  console.log("Conversation file created:", currentFile);
  res.json({ file: path.basename(currentFile) });
});

// --- Save conversation ---
app.post("/save-conversation", (req, res) => {
  if (!currentFile) return res.status(400).send("No active conversation file.");
  const { role, text } = req.body;
  if (!role || !text) return res.status(400).send("Missing role or text");

  fs.appendFileSync(currentFile, `[${new Date().toLocaleString()}] ${role.toUpperCase()}: ${text}\n`, { encoding: "utf8" });
  console.log("Saved message:", role.toUpperCase(), text);
  res.send("Saved");
});

// --- Provide latest RAG response (optional) ---
app.get("/rag-latest", (req, res) => {
  res.json({ text: lastRAGMessage });
  lastRAGMessage = "";
});

// --- GPT Realtime Session ---
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
    console.error("Error creating session:", err);
    res.status(500).send("Error creating GPT session");
  }
});

// --- Start server ---
const PORT = 3000;
app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
  console.log(`Open http://localhost:${PORT}/index.html`);
});













