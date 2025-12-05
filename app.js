let pc, dc, audioEl, ragInterval;
let aiAudioAttached = false;
let currentConversationFile = null;
window.sessionUpdated = false;
window.ragPollingStarted = false;

// ------------------- Time update -------------------
function updateTime() {
  const now = new Date();
  const timeString = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  document.getElementById('current-time').textContent = timeString;
}
setInterval(updateTime, 1000);
updateTime();

// ------------------- Connection status -------------------
function updateStatus(connected) {
  const indicator = document.getElementById('status-indicator');
  const statusText = document.getElementById('status-text');
  if (connected) {
    indicator.className = 'status-indicator status-active';
    statusText.textContent = 'Connected - Listening';
  } else {
    indicator.className = 'status-indicator status-inactive';
    statusText.textContent = 'Disconnected';
  }
}

// ------------------- Logging -------------------
function log(msg, type = 'info') {
  const logDiv = document.getElementById('log');
  const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  const logEntry = document.createElement('div');
  logEntry.className = 'log-entry';
  const typeClass = type === 'error' ? 'log-type-error' : type === 'success' ? 'log-type-success' : 'log-type-info';
  logEntry.innerHTML = `<span class="log-timestamp">${timestamp}</span>
                        <span class="log-type ${typeClass}">${type.toUpperCase()}</span>
                        <span>${typeof msg === "object" ? JSON.stringify(msg) : msg}</span>`;
  if (logDiv.firstChild) logDiv.insertBefore(logEntry, logDiv.firstChild);
  else logDiv.appendChild(logEntry);
  console.log(msg);
}

// ------------------- Conversation handling -------------------
function addConversation(role, text) {
  if (!text) return;

  // Do NOT remove live AI messages
  removeLiveUserMessage();

  const convDiv = document.getElementById('conversation');
  const containsUrdu = /[\u0600-\u06FF]/.test(text);
  const textClass = containsUrdu ? 'message-text urdu-text' : 'message-text';

  const messageDiv = document.createElement('div');
  messageDiv.className = `message ${role}`;
  messageDiv.innerHTML = `<div class="message-header">
                            <i class="${role === "user" ? "fas fa-user" : "fas fa-robot"}"></i> ${role === "user" ? "You" : "Assistant"}
                          </div>
                          <div class="${textClass}">${text}</div>`;
  convDiv.appendChild(messageDiv);
  convDiv.scrollTop = convDiv.scrollHeight;

  log(`Saved conversation: ${role.toUpperCase()}: ${text}`, 'info');

  // Send to backend for saving
  if (currentConversationFile) {
    fetch("/save-conversation", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ role, text })
    }).catch(err => log("Save error: " + err, 'error'));
  }
}

// ------------------- Live preview helpers -------------------
function addLiveUserMessage(text) {
  let live = document.getElementById("liveUserMsg");
  if (!live) {
    live = document.createElement("div");
    live.id = "liveUserMsg";
    live.className = "message user";
    live.innerHTML = `<div class="message-header"><i class="fas fa-user"></i> You</div><div class="message-text">${text}</div>`;
    document.getElementById("conversation").appendChild(live);
  } else live.querySelector(".message-text").textContent = text;
  document.getElementById("conversation").scrollTop = document.getElementById("conversation").scrollHeight;
}
function removeLiveUserMessage() {
  const live = document.getElementById("liveUserMsg");
  if (live) live.remove();
}
function addLiveAIMessage(text) {
  // Always append new AI messages (do not replace or remove)
  const convDiv = document.getElementById('conversation');
  const messageDiv = document.createElement("div");
  messageDiv.className = "message assistant";
  messageDiv.innerHTML = `<div class="message-header"><i class="fas fa-robot"></i> Assistant</div><div class="message-text">${text}</div>`;
  convDiv.appendChild(messageDiv);
  convDiv.scrollTop = convDiv.scrollHeight;

  // Also save immediately
  addConversation("assistant", text);
}
function removeLiveAIMessage() {
  // Do nothing to preserve AI messages
}

// ------------------- Start conversation -------------------
async function start() {
  document.getElementById('start').disabled = true;
  document.getElementById('stop').disabled = false;
  updateStatus(true);
  log("شروع کیا جا رہا ہے....", 'info');

  try {
    const resp = await fetch("/start-conversation", { method: "POST" });
    const data = await resp.json();
    currentConversationFile = data.file;
    log("Conversation file: " + currentConversationFile, 'success');
  } catch(err) {
    log("Error starting conversation: " + err, 'error');
    return;
  }

  try {
    pc = new RTCPeerConnection();
    dc = pc.createDataChannel("oai-events");

    dc.onopen = () => {
      log("DataChannel is open", 'success');
      if (!window.sessionUpdated) {
        const sessionUpdate = {
          type: 'session.update',
          session: {
            instructions: `آپ ایک دوستانہ اور تجربہ کار رئیل اسٹیٹ سیلز ایجنٹ ہیں۔

            آپ نے ہمیشہ اس انداز میں بات کرنی ہے:
            - خوش اخلاق
            - نرم مزاج
            - انسانوں جیسی گفتگو
            - فون پر بات کرنے جیسے ہلکے اور فلو میں جملے
            - بلکل آسان، سادہ اور سمجھ میں آنے والی زبان
            - کبھی بھی کتابی، روبوٹک یا dry انداز نہیں

            -----------------------------------------------------------
            گاہک کے سوال کا مقصد سمجھیں:
            -----------------------------------------------------------
            ذہن میں یہ چار چیزیں check کریں:

            1) کیا وہ قیمت پوچھ رہا ہے؟  
            2) کیا وہ سائز / مرلہ / کنال پوچھ رہی ہے؟  
            3) کیا وہ علاقے / لوکیشن کے بارے میں پوچھ رہی ہے؟  
            4) کیا وہ کسی خاص گھر یا پلاٹ کی مکمل تفصیل مانگ رہی ہے؟

            ہر جواب گاہک کے سوال کے مطابق بنائیں — نہ کم، نہ زیادہ۔

            -----------------------------------------------------------
            اگر سوال پراپرٹی، قیمت، سائز، مرلہ، کنال یا علاقے سے متعلق ہو:
            -----------------------------------------------------------
            پہلے ایک دوستانہ delay line کہیں:
            - "کچھ دیر ٹھہریں، میں چیک کر کے دیکھتا ہوں…"
            یا
            - "جی ایک لمحہ دیں، میں ابھی چیک کرتا ہوں…"

            پھر RAG کے ڈیٹا کو انسانی انداز میں، خوبصورت لیکن مختصر جملوں میں سمجھائیں۔

            مثال:
            "آپ کے سوال کے مطابق، DHA فیز 6 میں 10 مرلے کے پلاٹس تقریباً 2 کروڑ کے آس پاس مل جاتے ہیں۔ قیمت لوکیشن اور پوزیشن کے مطابق تھوڑی کم زیادہ ہوتی ہے۔"

            -----------------------------------------------------------
            اگر وہ کسی خاص گھر/فلیٹ کی تفصیل مانگے:
            -----------------------------------------------------------
            RAG سے ملی ہوئی معلومات کو کہانی جیسے انداز میں بتائیں:

            - رقبہ  
            - بیڈ روم  
            - باتھ روم  
            - پارکنگ  
            - لوکیشن  
            - اور کوئی بھی extra info

            لیکن اگر کوئی فیلڈ موجود نہ ہو تو اس کا ذکر نہ کریں۔

            -----------------------------------------------------------
            وزٹ آفر کرنا کب ہے؟
            -----------------------------------------------------------
            صرف تب وزٹ آفر کریں جب گاہک شخصیت سے detail لے رہا ہو یا بجٹ, area, plot/house detail پوچھ رہا ہو۔

            لیکن ہر جواب میں وزٹ آفر مت دیں۔

            وزٹ کی لائن ہمیشہ آخر میں اور صرف ایک بار ہو:
            "اگر آپ چاہیں تو میں آپ کے لیے وزٹ کا وقت بھی رکھ سکتا ہوں۔"

            وزٹ کب finalize ہو؟  
            جب گاہک خود کہے:
            - "ٹھیک ہے وزٹ کروا دیں"
            - "آنا چاہتی ہوں"
            - "ملنا چاہتی ہوں"

            تب لازمی پوچھیں:
            - آپ کا نام؟  
            - آپ کا موبائل نمبر؟  
            - کس دن اور کس وقت وزٹ کرنا چاہیں گی؟

            پھر softly confirm کریں:
            "ٹھیک ہے، آپ کا وزٹ میں نے کل شام 5 بجے کے لیے نوٹ کر لیا ہے۔"

            -----------------------------------------------------------
            General Rules
            -----------------------------------------------------------
            - اگر user کی آواز unclear ہو تو کہیں:
              "مجھے آواز پوری طرح سمجھ نہیں آئی، آپ ایک بار پھر بتا دیں؟"

            - اگر user بے ربط جملے بولے (جیسے 'بہتری بہتری' یا 'ویڈیو ویڈیو')  
              تو politely clarification لیں، RAG نہ چلائیں۔

            - اگر user صرف greetings کرے → friendly greeting دیں۔

            - کبھی بھی لمبے پیراگراف نہ دیں۔  
            - گفتگو ہمیشہ warm, short, helpful اور human ہونی چاہیے۔
              `, // keep your full Urdu instructions here
            voice: 'alloy',
            input_audio_transcription: { model: 'whisper-1', language: 'ur' },
            output_audio_format: 'pcm16',
            turn_detection: { type: "server_vad", threshold: 0.5, silence_duration_ms: 200, create_response: true, interrupt_response: true }
          }
        };
        log("Sending session update to server...", 'info');
        dc.send(JSON.stringify(sessionUpdate));
        window.sessionUpdated = true;
      }
    };

    dc.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        if (data.type === "conversation.item.input_audio_transcription.completed" && data.transcript) {
          addLiveUserMessage(data.transcript);
          if (data.is_final) addConversation("user", data.transcript);
        }
        if (data.type === "response.audio_transcript.done" && data.transcript) {
          addLiveAIMessage(data.transcript);
        }
        if (data.type === "response.output_text.done" && data.text) {
          addConversation("assistant", data.text);
        }
      } catch(err) {
        log("Error parsing DataChannel message: " + err + " | Raw: " + e.data, 'error');
      }
    };

    const localStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    localStream.getTracks().forEach(t => pc.addTrack(t, localStream));
    log("Microphone stream added", 'success');

    audioEl = document.createElement("audio");
    audioEl.autoplay = true;
    pc.ontrack = (event) => {
      if (!aiAudioAttached) {
        audioEl.srcObject = event.streams[0];
        aiAudioAttached = true;
        log("Audio track received from server", 'success');
      }
    };

    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);
    const sdpResp = await fetch("/session", { method: "POST", headers: { "Content-Type": "application/sdp" }, body: offer.sdp });
    const answer = { type: "answer", sdp: await sdpResp.text() };
    await pc.setRemoteDescription(answer);
    log("SDP exchange completed", 'success');
    log("کنکشن تیار ہے، آپ بات کر سکتی ہیں-", 'success');

    startRAGPolling();
  } catch(err) {
    log("Error during start(): " + err, 'error');
    updateStatus(false);
    document.getElementById('start').disabled = false;
    document.getElementById('stop').disabled = true;
  }
}

// ------------------- RAG polling -------------------
async function startRAGPolling() {
  if (window.ragPollingStarted) return;
  window.ragPollingStarted = true;

  ragInterval = setInterval(async () => {
    try {
      const resp = await fetch("/rag-latest");
      if (!resp.ok) return log("RAG fetch failed: " + resp.status, 'error');
      const data = await resp.json();
      if (data.text) addLiveAIMessage(data.text);
    } catch(err) { log("Error fetching RAG data: " + err, 'error'); }
  }, 5000);

  log("RAG polling started every 5s", 'info');
}

// ------------------- Stop conversation -------------------
async function stop() {
  document.getElementById('start').disabled = false;
  document.getElementById('stop').disabled = true;
  updateStatus(false);

  if (pc) { pc.close(); pc = null; }
  if (ragInterval) { clearInterval(ragInterval); ragInterval = null; }

  aiAudioAttached = false;
  window.sessionUpdated = false;
  window.ragPollingStarted = false;

  removeLiveUserMessage();
  // Do NOT remove AI messages

  log("گفتگو ختم ہو گئی۔", 'info');
}

// ------------------- Event listeners -------------------
document.getElementById('start').onclick = start;
document.getElementById('stop').onclick = stop;

// Initial log
log("System ready. Click 'Start Conversation' to begin.", 'success');
