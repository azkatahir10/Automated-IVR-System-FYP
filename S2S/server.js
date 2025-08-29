import 'dotenv/config';
import express from "express";
import fetch from "node-fetch";
import cors from "cors";

const app = express();
app.use(cors());
app.use(express.text({ type: "application/sdp" }));

//Handle session creation
app.post("/session", async (req, res) => {
  try {
    const offer = req.body;

    const response = await fetch("https://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
        "Content-Type": "application/sdp",
      },
      body: offer,
    });

    const answer = await response.text();
    res.set("Content-Type", "application/sdp");
    res.send(answer);

  } catch (err) {
    console.error("❌ Error creating session:", err);
    res.status(500).send("Error creating session");
  }
});

const PORT = 3000;
app.listen(PORT, () => {
  console.log(`✅ Server running at http://localhost:${PORT}`);
});
