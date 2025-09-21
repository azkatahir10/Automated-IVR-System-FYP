import 'dotenv/config';
import express from "express";
import fetch from "node-fetch";
import cors from "cors";
import path from "path";
import { fileURLToPath } from "url";
import fs from "fs";

const app = express();
app.use(cors());
app.use(express.text({ type: "application/sdp" }));

// resolve __dirname in ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Serve frontend (index.html in "public" folder)
app.use(express.static(path.join(__dirname, "public")));

// Handle session creation (SDP exchange)
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

// Optional: Save conversation transcript as sales lead
app.use(express.json());
app.post("/save-lead", (req, res) => {
  const lead = req.body; // { user: "...", assistant: "...", ... }
  const filePath = path.join(__dirname, "leads.json");

  let leads = [];
  if (fs.existsSync(filePath)) {
    leads = JSON.parse(fs.readFileSync(filePath, "utf-8"));
  }
  leads.push({ ...lead, timestamp: new Date().toISOString() });

  fs.writeFileSync(filePath, JSON.stringify(leads, null, 2));
  console.log("✅ Lead saved:", lead);

  res.json({ status: "ok" });
});

// Start server
const PORT = 3000;
app.listen(PORT, () => {
  console.log(`✅ Server running at http://localhost:${PORT}`);
  console.log(`➡️  Open http://localhost:${PORT} in your browser`);
});
