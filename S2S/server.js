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

// resolve __dirname in ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Track current conversation file
let currentFile = null;

app.use(express.static(path.join(__dirname, "public")));

// --- Start a new conversation (create new file) ---
app.post("/start-conversation", (req, res) => {
  const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
  currentFile = path.join(__dirname, `conversation-${timestamp}.txt`);

  console.log("📝 Creating new conversation file:", currentFile);

  fs.writeFile(currentFile, "🔔 New conversation started\n", (err) => {
    if (err) {
      console.error("❌ Error creating conversation file:", err);
      return res.status(500).send("Error creating conversation file");
    }
    res.send({ file: path.basename(currentFile) });
  });
});

// --- Save a line in the current conversation ---
app.post("/save-conversation", (req, res) => {
  if (!currentFile) {
    console.error("⚠️ Tried to save but no file is active!");
    return res.status(400).send("No active conversation file. Start first.");
  }

  const message = req.body;
  const logLine = `[${new Date().toLocaleString()}] ${message}\n`;

  fs.appendFile(currentFile, logLine, (err) => {
    if (err) {
      console.error("❌ Error saving conversation:", err);
      return res.status(500).send("Error saving conversation");
    }
    res.send("✅ Saved");
  });
});

// --- Handle session creation (SDP exchange) ---
app.post("/session", async (req, res) => {
  try {
    const offer = req.body;
    const response = await fetch(
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

    const answer = await response.text();
    res.set("Content-Type", "application/sdp");
    res.send(answer);
  } catch (err) {
    console.error("❌ Error creating session:", err);
    res.status(500).send("Error creating session");
  }
});

// Start server
const PORT = 3000;
app.listen(PORT, () => {
  console.log(`✅ Server running at http://localhost:${PORT}`);
  console.log(`➡️  Open http://localhost:${PORT} in your browser`);
});




