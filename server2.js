// =============================================================================
// server.js  —  WebSocket Backend
// Stack: Express + ws + Groq Whisper STT + Local RAG + Edge-TTS (via python worker)
// Supabase: conversations, messages, rag_searches, client_visits tables
// Port: 3000
// =============================================================================

import 'dotenv/config';
import express       from 'express';
import { WebSocketServer } from 'ws';
import Groq          from 'groq-sdk';
import { createClient } from '@supabase/supabase-js';
import fs            from 'fs';
import fetch         from 'node-fetch';
import FormData      from 'form-data';
import path          from 'path';
import crypto        from 'crypto';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);

// ── Clients ──────────────────────────────────────────────────────────────────
const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });

const supabase = (process.env.SUPABASE_URL && process.env.SUPABASE_SERVICE_KEY)
  ? createClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_KEY)
  : null;

if (!supabase) console.warn('⚠️  Supabase not configured — DB logging disabled');
else           console.log('✅ Supabase connected');

// ── Constants ─────────────────────────────────────────────────────────────────
const GROQ_MODEL   = 'llama-3.3-70b-versatile';
const WORKER_BASE  = process.env.WORKER_BASE  || 'http://127.0.0.1:5002';
const RAG_BASE     = process.env.RAG_BASE     || 'http://127.0.0.1:5001';

const GREETING_TEXT =
  'السلام علیکم! میں آپ کا رئیل اسٹیٹ اسسٹنٹ ہوں۔ ' +
  'براہ کرم بتائیں کہ آپ کو گھر خریدنا ہے، کرایہ پر لینا ہے، یا کوئی اور مدد چاہیے؟';

const WHISPER_PROMPT =
  'یہ ایک پاکستانی رئیل اسٹیٹ IVR سروس ہے۔ ' +
  'گھر، مکان، فلیٹ، پلاٹ، کرایہ، خریدنا، بیچنا، ' +
  'لاہور، کراچی، اسلام آباد، گلبرگ، بحریہ ٹاؤن، جوہر ٹاؤن، ' +
  'مرلہ، کنال، لاکھ، کروڑ، قیمت۔';

// Visit schedule karne ka message — RAG results ke baad pucha jata hai
const VISIT_PROMPT_TEXT =
  'کیا آپ ان میں سے کسی پراپرٹی کا وزٹ شیڈول کرنا چاہتے ہیں؟';

// ── Express ───────────────────────────────────────────────────────────────────
const app = express();
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

const tempDir = path.join(__dirname, 'temp');
if (!fs.existsSync(tempDir)) fs.mkdirSync(tempDir, { recursive: true });

// ── Session store ─────────────────────────────────────────────────────────────
// { history, lastActivity, lastRawDocs, lastUrls, conversationId, startedAt,
//   visitState: null | 'asked' | 'collecting_all' | 'collecting_property' | 'done',
//   visitRetries: number of times we've re-asked for missing fields }
const sessions = new Map();

function getOrCreateSession(clientId) {
  if (!sessions.has(clientId)) {
    sessions.set(clientId, {
      history:        [],
      lastActivity:   Date.now(),
      lastRawDocs:    [],
      lastUrls:       [],
      conversationId: null,
      startedAt:      new Date().toISOString(),
      visitState:     null,   // visit flow state machine
      visitData:      {},     // accumulated visit info
      visitRetries:   0,      // kitni baar missing fields ke liye dobara pucha
    });
  }
  const s = sessions.get(clientId);
  s.lastActivity = Date.now();
  return s;
}

// ── Safe WebSocket send ────────────────────────────────────────────────────────
function wsSend(ws, payload) {
  if (ws.readyState === 1) {
    try { ws.send(typeof payload === 'string' ? payload : JSON.stringify(payload)); }
    catch (e) { console.warn('⚠️ wsSend failed:', e.message); }
  }
}

// ── Temp file cleanup ──────────────────────────────────────────────────────────
function unlinkSafe(filePath) {
  if (!filePath) return;
  try { if (fs.existsSync(filePath)) fs.unlinkSync(filePath); } catch {}
}

// ── Garbage transcript filter ──────────────────────────────────────────────────
const URDU_CHAR_RE = /[\u0600-\u06FF]/g;
function isGarbageTranscript(text) {
  if (!text || text.length < 3) return { bad: true, reason: 'too_short' };
  const chars    = text.replace(/\s/g, '');
  const urduLen  = (text.match(URDU_CHAR_RE) || []).length;
  if (chars.length > 8 && urduLen / chars.length < 0.25)
    return { bad: true, reason: 'non_urdu' };
  const words  = text.trim().split(/\s+/);
  const unique = new Set(words.map(w => w.toLowerCase()));
  if (words.length >= 2 && unique.size === 1)       return { bad: true, reason: 'repetition' };
  if (words.length >= 4 && unique.size / words.length < 0.35)
    return { bad: true, reason: 'repetition' };
  return { bad: false };
}

// ── Detail intent detection ────────────────────────────────────────────────────
const EXPLICIT_DETAIL = /تفصیل|تفصیلات|پتہ|ایڈریس|address|رابطہ|contact|فون\s*نمبر/i;
function isDetailIntent(text) { return EXPLICIT_DETAIL.test(text); }

// ── Property search intent detection ───────────────────────────────────────────
// RAG ko sirf tab call karo jab user ka sawal kisi property dhoondne / khareedne /
// kiraye / location / size / price se related ho. Agar sawal general (greeting,
// chit-chat, ya kisi aur topic) ho to RAG skip karo — LLM seedha apne general
// real-estate-assistant prompt se jawab dega.
const PROPERTY_INTENT_RE =
  /گھر|مکان|فلیٹ|پلاٹ|پراپرٹی|پراپرٹیز|کرایہ|خرید|بیچ|فروخت|مرلہ|کنال|لاکھ|کروڑ|قیمت|بجٹ|علاقہ|ایریا|بیڈ\s*روم|باتھ\s*روم|سائز|دکھائیں|دکھاؤ|تلاش|چاہیے|ہاؤس|اپارٹمنٹ|لاہور|کراچی|اسلام\s*آباد|گلبرگ|بحریہ|جوہر\s*ٹاؤن|ڈی\s*ایچ\s*اے|model\s*town|house|flat|apartment|plot|property|rent|buy|sell|purchase|marla|kanal|lakh|crore|price|budget|bedroom|bathroom|location|area|size|lahore|karachi|islamabad|gulberg|bahria|dha|johar/i;

function isPropertySearchIntent(text) {
  return PROPERTY_INTENT_RE.test(text);
}

// ── Visit intent detection ─────────────────────────────────────────────────────
const VISIT_YES_RE  = /ہاں|جی|yes|بالکل|ضرور|okay|ok|ٹھیک\s*ہے|کریں|چاہتا|چاہتی|شیڈول|وزٹ|visit/i;
const VISIT_NO_RE   = /نہیں|no|نہ|نا|ابھی\s*نہیں|بعد\s*میں/i;

// ── Phone number extraction ────────────────────────────────────────────────────
function extractPhone(text) {
  const m = text.match(/(\+92|0092|0)[\s\-]?(3\d{2})[\s\-]?(\d{7})/);
  return m ? `0${m[2]}${m[3]}` : null;
}

// ── Date extraction ────────────────────────────────────────────────────────────
const URDU_MONTHS = {
  جنوری: 1, فروری: 2, مارچ: 3, اپریل: 4, مئی: 5, جون: 6,
  جولائی: 7, اگست: 8, ستمبر: 9, اکتوبر: 10, نومبر: 11, دسمبر: 12,
};

// "25 جون" ya "جون 25" jaisi tareekh ko parse karta hai (saal khud add karta hai)
function extractUrduMonthDate(text) {
  for (const [name, num] of Object.entries(URDU_MONTHS)) {
    const re = new RegExp(`(\\d{1,2})\\s*${name}|${name}\\s*(\\d{1,2})`);
    const m = text.match(re);
    if (m) {
      const day = parseInt(m[1] || m[2], 10);
      if (day < 1 || day > 31) continue;
      const today = new Date();
      let candidate = new Date(today.getFullYear(), num - 1, day);
      // agar tareekh guzar gayi ho to agle saal ki samjho
      if (candidate < new Date(today.toDateString())) {
        candidate = new Date(today.getFullYear() + 1, num - 1, day);
      }
      return candidate.toISOString().slice(0, 10);
    }
  }
  return null;
}

function extractDate(text) {
  const iso = text.match(/(\d{4}-\d{2}-\d{2})/);
  if (iso) return iso[1];
  const dmy = text.match(/(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{4})/);
  if (dmy) return `${dmy[3]}-${dmy[2].padStart(2,'0')}-${dmy[1].padStart(2,'0')}`;

  const today = new Date();
  if (/آج|today/i.test(text))   return today.toISOString().slice(0,10);
  if (/کل|tomorrow/i.test(text)) {
    const t = new Date(today); t.setDate(t.getDate()+1);
    return t.toISOString().slice(0,10);
  }
  if (/پرسوں/i.test(text)) {
    const t = new Date(today); t.setDate(t.getDate()+2);
    return t.toISOString().slice(0,10);
  }
  const DAYS = {پیر:1,monday:1,منگل:2,tuesday:2,بدھ:3,wednesday:3,
                جمعرات:4,thursday:4,جمعہ:5,friday:5,ہفتہ:6,saturday:6,اتوار:0,sunday:0};
  for (const [word, day] of Object.entries(DAYS)) {
    if (text.includes(word)) {
      const diff = (day - today.getDay() + 7) % 7 || 7;
      const t = new Date(today); t.setDate(t.getDate()+diff);
      return t.toISOString().slice(0,10);
    }
  }
  return extractUrduMonthDate(text);
}

// ── Time extraction ────────────────────────────────────────────────────────────
function extractTime(text) {
  const hm = text.match(/(\d{1,2}):(\d{2})\s*(?:am|pm|AM|PM|بجے)?/i);
  if (hm) return `${hm[1].padStart(2,'0')}:${hm[2]}`;
  const hr = text.match(/(\d{1,2})\s*(?:بجے|o'?clock)/i);
  if (hr) return `${hr[1].padStart(2,'0')}:00`;
  return null;
}

// ── Name extraction ────────────────────────────────────────────────────────────
function extractName(text) {
  const ur = text.match(/(?:میرا\s*نام|نام)\s+([^\s،,۔.]{2,20})/);
  if (ur) return ur[1].trim();
  const en = text.match(/(?:my\s+name\s+is|name\s*[:\-]?\s*)([A-Za-z]{2,30})/i);
  if (en) return en[1].trim();
  // agar koi aur word ho to woh bhi naam samajh lo
  const words = text.trim().split(/\s+/);
  if (words.length === 1 && words[0].length > 1) return words[0];
  return null;
}

// ── Combined one-shot extraction (name + phone + date + time together) ────────
// Pehle fast regex try karta hai; agar koi field reh jaye to Groq se behtar
// extraction karta hai (yeh "25 جون", "اگلے جمعہ", casual phrasing sab handle
// kar leta hai jo plain regex se miss ho jata tha).
async function extractVisitDetailsAll(text) {
  const regexResult = {
    client_name:  extractName(text),
    phone_number: extractPhone(text),
    visit_date:   extractDate(text),
    visit_time:   extractTime(text),
  };

  const allFound = regexResult.client_name && regexResult.phone_number &&
                    regexResult.visit_date  && regexResult.visit_time;
  if (allFound) return regexResult;

  try {
    const today = new Date().toISOString().slice(0, 10);
    const resp = await groq.chat.completions.create({
      model:       GROQ_MODEL,
      temperature: 0,
      max_tokens:  150,
      messages: [
        {
          role: 'system',
          content:
            `آج کی تاریخ ${today} ہے۔ صارف کے جملے سے یہ معلومات نکالیں: ` +
            `نام (client_name)، پاکستانی موبائل نمبر (phone_number — صرف ہندسے، 0 سے شروع، 11 ڈیجٹ)، ` +
            `وزٹ کی تاریخ (visit_date — format YYYY-MM-DD؛ "کل"، "پرسوں"، دن کے نام یا مہینے کا نام ہو تو خود حساب لگائیں)، ` +
            `اور وزٹ کا وقت (visit_time — format HH:MM، 24 گھنٹے کا فارمیٹ)۔ ` +
            `جو معلومات جملے میں موجود نہ ہو اس کی value null رکھیں۔ ` +
            `صرف ایک خالص JSON object واپس کریں — کوئی اور لفظ، وضاحت یا markdown نہیں۔ ` +
            `مثال: {"client_name":"علی","phone_number":"03001234567","visit_date":"2026-06-17","visit_time":"15:00"}`,
        },
        { role: 'user', content: text },
      ],
    });
    const raw   = resp.choices[0]?.message?.content || '{}';
    const clean = raw.replace(/```json|```/gi, '').trim();
    const parsed = JSON.parse(clean);
    return {
      client_name:  parsed.client_name  || regexResult.client_name  || null,
      phone_number: parsed.phone_number || regexResult.phone_number || null,
      visit_date:   parsed.visit_date   || regexResult.visit_date   || null,
      visit_time:   parsed.visit_time   || regexResult.visit_time   || null,
    };
  } catch (e) {
    console.warn('⚠️ LLM visit-extraction failed, using regex-only result:', e.message);
    return regexResult;
  }
}

// =============================================================================
// VISIT STATE MACHINE
// States: null → asked → collecting_all → [collecting_property] → done
// "collecting_all" ek hi message me naam, number, tareekh aur waqt collect
// karta hai — agar kuch reh jaye to ek consolidated follow-up question hota
// hai (har field ke liye alag se nahi pucha jata).
// =============================================================================

async function handleVisitFlow(session, userText, ws) {
  const vs = session.visitState;
  const vd = session.visitData;

  // ── State: asked — user ne haan/na kaha ──────────────────────────────────
  if (vs === 'asked') {
    if (VISIT_NO_RE.test(userText)) {
      session.visitState = null;
      const reply = 'ٹھیک ہے! اگر کبھی وزٹ کرنا ہو تو ضرور بتائیں۔ کیا کوئی اور مدد چاہیے؟';
      await sendAIReply(reply, ws, session);
      return true;
    }
    if (VISIT_YES_RE.test(userText)) {
      session.visitState   = 'collecting_all';
      session.visitRetries = 0;
      const reply =
        'بہت اچھا! براہ کرم ایک ساتھ بتائیں — اپنا نام، موبائل نمبر، اور وزٹ کی تاریخ اور وقت۔ ' +
        'مثلاً: "میرا نام علی ہے، نمبر 03001234567، کل دن 11 بجے آنا ہے۔"';
      await sendAIReply(reply, ws, session);
      return true;
    }
    // Unclear — ask again
    const reply = 'معاف کیجیے، کیا آپ وزٹ شیڈول کرنا چاہتے ہیں؟ ہاں یا نہیں بتائیں۔';
    await sendAIReply(reply, ws, session);
    return true;
  }

  // ── State: collecting_all — naam, number, tareekh, waqt sab ek sath ──────
  if (vs === 'collecting_all') {
    const extracted = await extractVisitDetailsAll(userText);
    // Sirf khaali fields ko fill karo — jo pehle se mil chuka hai usay mat ukharo
    vd.client_name  = vd.client_name  || extracted.client_name;
    vd.phone_number = vd.phone_number || extracted.phone_number;
    vd.visit_date   = vd.visit_date   || extracted.visit_date;
    vd.visit_time   = vd.visit_time   || extracted.visit_time;

    const missing = [];
    if (!vd.client_name)  missing.push('نام');
    if (!vd.phone_number) missing.push('موبائل نمبر');
    if (!vd.visit_date)   missing.push('تاریخ');
    if (!vd.visit_time)   missing.push('وقت');

    if (missing.length === 0) {
      return await proceedToPropertyOrFinalize(session, ws);
    }

    session.visitRetries = (session.visitRetries || 0) + 1;

    // Do baar consolidated follow-up ke baad bhi pura na mile to jo mila
    // usi se save kar do — taake conversation hamesha atki na rahe aur
    // DB mein kam az kam partial visit record to ban jaye.
    if (session.visitRetries >= 2) {
      return await proceedToPropertyOrFinalize(session, ws);
    }

    const reply = `صرف یہ بتا دیں: ${missing.join('، ')}۔`;
    await sendAIReply(reply, ws, session);
    return true;
  }

  // ── State: collecting_property ───────────────────────────────────────────
  if (vs === 'collecting_property') {
    // Number se property select karo
    const numMatch = userText.match(/(\d+)/);
    if (numMatch) {
      const idx = parseInt(numMatch[1]) - 1;
      if (session.lastUrls?.[idx]) {
        vd.selected_property = session.lastUrls[idx];
      }
    } else if (session.lastUrls?.length) {
      vd.selected_property = session.lastUrls[0]; // default first
    }
    session.visitState = 'done';
    await finalizeVisit(session, ws);
    return true;
  }

  return false; // not in visit flow
}

// ── Visit ka zaroori data mil gaya — ab ya property pochho ya seedha save karo
async function proceedToPropertyOrFinalize(session, ws) {
  const vd = session.visitData;
  if (session.lastUrls?.length === 1) {
    vd.selected_property = session.lastUrls[0];
    session.visitState = 'done';
    await finalizeVisit(session, ws);
  } else if (session.lastUrls?.length > 1) {
    session.visitState = 'collecting_property';
    const propList = session.lastUrls.map((u, i) => `${i + 1} نمبر پراپرٹی`).join('، ');
    const reply = `بہت شکریہ! کون سی پراپرٹی دیکھنا چاہتے ہیں؟ (${propList})`;
    await sendAIReply(reply, ws, session);
  } else {
    session.visitState = 'done';
    await finalizeVisit(session, ws);
  }
  return true;
}

// ── Visit finalize & save ─────────────────────────────────────────────────────
async function finalizeVisit(session, ws) {
  const vd = session.visitData;
  console.log('📅 Saving visit:', vd);

  // Save to Supabase
  const saved = await dbSaveVisit(session.conversationId, vd, session.lastUrls || []);

  const dateStr = vd.visit_date || 'طے شدہ تاریخ';
  const timeStr = vd.visit_time ? ` ${vd.visit_time} بجے` : '';
  const reply = saved
    ? `بہت خوب! ${vd.client_name || 'آپ'} کا وزٹ ${dateStr}${timeStr} کے لیے شیڈول ہو گیا۔ ہمارا نمائندہ آپ سے رابطہ کرے گا۔ شکریہ!`
    : `وزٹ کی درخواست موصول ہو گئی۔ ہمارا نمائندہ جلد آپ سے رابطہ کرے گا۔ شکریہ!`;

  session.visitState = null;
  session.visitData  = {};
  await sendAIReply(reply, ws, session);
}

// ── Shared reply sender (text → ws + TTS) ────────────────────────────────────
async function sendAIReply(text, ws, session) {
  wsSend(ws, { type: 'ai_response', text });
  await dbSaveMessage(session.conversationId, 'assistant', text);
  session.history.push({ role: 'assistant', content: text });
  wsSend(ws, { type: 'status', step: 'tts', message: 'آواز بن رہی ہے...' });
  await streamTTS(text, ws);
}

// ═══════════════════════════════════════════════
// SUPABASE HELPERS
// ═══════════════════════════════════════════════

async function dbCreateConversation(clientId) {
  if (!supabase) return null;
  try {
    const { data, error } = await supabase
      .from('conversations')
      .insert({
        client_id:  clientId,
        started_at: new Date().toISOString(),
        status:     'active',
      })
      .select('id')
      .single();
    if (error) throw error;
    console.log(`📝 Conversation created: ${data.id}`);
    return data.id;
  } catch (e) {
    console.warn('⚠️ dbCreateConversation failed (non-fatal):', e.message);
    return null;
  }
}

async function dbSaveMessage(conversationId, role, text) {
  if (!supabase || !conversationId) return;
  try {
    await supabase.from('messages').insert({
      conversation_id: conversationId,
      role,
      text,
      created_at: new Date().toISOString(),
    });
  } catch (e) {
    console.warn('⚠️ dbSaveMessage failed (non-fatal):', e.message);
  }
}

async function dbEndConversation(conversationId, durationSec) {
  if (!supabase || !conversationId) return;
  try {
    await supabase.from('conversations').update({
      ended_at:     new Date().toISOString(),
      duration_sec: durationSec,
      status:       'completed',
    }).eq('id', conversationId);
  } catch (e) {
    console.warn('⚠️ dbEndConversation failed (non-fatal):', e.message);
  }
}

async function dbSaveVisit(conversationId, visitData, propertyLinks) {
  if (!supabase) return false;
  try {
    const { error } = await supabase.from('client_visits').insert({
      conversation_id:   conversationId || null,
      client_name:       visitData.client_name       || null,
      phone_number:      visitData.phone_number      || null,
      visit_date:        visitData.visit_date        || null,
      visit_time:        visitData.visit_time        || null,
      selected_property: visitData.selected_property || null,
      property_link:     propertyLinks?.length ? JSON.stringify(propertyLinks) : null,
      confirmed:         true,
      created_at:        new Date().toISOString(),
    });
    if (error) throw error;
    console.log(`✅ Visit saved: ${visitData.client_name} | ${visitData.visit_date}`);
    return true;
  } catch (e) {
    console.warn('⚠️ dbSaveVisit failed (non-fatal):', e.message);
    return false;
  }
}

async function dbLogRAGSearch(conversationId, query, results, count, isSpecific) {
  if (!supabase) return;
  try {
    await supabase.from('rag_searches').insert({
      conversation_id: conversationId || null,
      query,
      result_count:    count,
      is_specific:     isSpecific,
      results_preview: results?.slice(0, 3).join(' | ').substring(0, 500) || null,
      created_at:      new Date().toISOString(),
    });
  } catch (e) {
    console.warn('⚠️ dbLogRAGSearch failed (non-fatal):', e.message);
  }
}

// ═══════════════════════════════════════════════
// STT — Groq (primary) + Local Whisper (fallback)
// ═══════════════════════════════════════════════

async function transcribeWithGroq(filePath) {
  try {
    const transcription = await groq.audio.transcriptions.create({
      file:            fs.createReadStream(filePath),
      model:           'whisper-large-v3-turbo',
      language:        'ur',
      prompt:          WHISPER_PROMPT,
      response_format: 'verbose_json',
    });
    const text = (transcription.text || '').trim();
    if (!text || text.length < 2) return { text: '', langProb: 0, rejected: true, reason: 'empty' };
    return { text, langProb: 0.95, rejected: false };
  } catch (err) {
    console.warn('⚠️ Groq STT failed:', err.message, '— falling back to local');
    return null;
  }
}

async function transcribeWithLocal(filePath) {
  try {
    const form = new FormData();
    form.append('file', fs.createReadStream(filePath), {
      filename: path.basename(filePath), contentType: 'audio/webm'
    });
    const res = await fetch(`${WORKER_BASE}/stt`, {
      method: 'POST', headers: form.getHeaders(), body: form,
      signal: AbortSignal.timeout(60000),
    });
    if (!res.ok) throw new Error('STT worker error: ' + res.status);
    const data = await res.json();
    if (data.rejected) return { text: '', langProb: 0, rejected: true, reason: data.reason };
    return { text: (data.text || '').trim(), langProb: data.language_prob || 0, rejected: false };
  } catch (e) {
    console.error('❌ Local STT error:', e.message);
    return { text: '', langProb: 0, rejected: false };
  }
}

async function transcribeAudio(filePath) {
  const result = await transcribeWithGroq(filePath);
  return result !== null ? result : transcribeWithLocal(filePath);
}

// ═══════════════════════════════════════════════
// RAG
// ═══════════════════════════════════════════════

async function queryRAG(userText, historyStr, conversationId) {
  try {
    const res = await fetch(`${RAG_BASE}/search`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ query: userText, history: historyStr, conversation_id: conversationId }),
      signal:  AbortSignal.timeout(10000),
    });
    if (!res.ok) return null;
    const data = await res.json();
    console.log(`🔍 RAG: ${data.count ?? 0} results | specific=${data.is_specific} | no_results=${data.no_results}`);
    return data;
  } catch (e) {
    console.warn('⚠️ RAG unreachable:', e.message);
    return null;
  }
}

function buildRAGContext(ragData) {
  if (!ragData?.results?.length) return null;
  return `ڈیٹابیس سے دستیاب پراپرٹیز:\n${ragData.results.map((r, i) => `${i + 1}. ${r}`).join('\n')}`;
}

// ═══════════════════════════════════════════════
// LLM SYSTEM PROMPTS
// ═══════════════════════════════════════════════

const SYSTEM_BASE = `آپ پاکستان کے رئیل اسٹیٹ IVR اسسٹنٹ ہیں۔
Speech-to-text سے آیا متن ملتا ہے — آواز کی غلطیاں context سے سمجھ کر جواب دیں۔
سخت قوانین:
1. صرف اردو میں، مکمل جملوں میں جواب دیں
2. 2 سے 4 جملے — مختصر اور واضح
3. صرف رئیل اسٹیٹ موضوعات
4. کوئی markdown نہیں
5. اپنی طرف سے کوئی پراپرٹی، قیمت، علاقہ یا خصوصیت مت بتائیں — صرف ڈیٹابیس کی معلومات
6. اگر ڈیٹابیس میں مطلوبہ پراپرٹی نہ ہو: "ابھی ہمارے پاس اس علاقے میں مطلوبہ پراپرٹی دستیاب نہیں، لیکن میں آپ کی تلاش رجسٹر کر سکتا ہوں۔"`;

function buildSystemPrompt(ragContext) {
  if (!ragContext) return SYSTEM_BASE;
  return `آپ پاکستان کے رئیل اسٹیٹ IVR اسسٹنٹ ہیں۔
Speech-to-text سے آیا متن ملتا ہے — آواز کی غلطیاں context سے سمجھ کر جواب دیں۔
${ragContext}
سخت قوانین — خلاف ورزی قابل قبول نہیں:
1. صرف اردو میں، مکمل جملوں میں جواب دیں
2. 2 سے 4 جملے — مختصر اور واضح
3. صرف اوپر دی گئی پراپرٹیز کا ذکر کریں
4. ہر پراپرٹی کا علاقہ، سائز اور قیمت بتائیں جیسا ڈیٹا میں ہے
5. بیڈروم، باتھ روم یا دیگر تفصیل خود سے مت بتائیں جب تک ڈیٹا میں واضح نہ ہو
6. اگر مانگا گیا سائز نہ ہو تو صاف بتائیں
7. کوئی markdown نہیں
8. قیمت ہمیشہ لاکھ یا کروڑ میں — مثال: 50 لاکھ، 1.5 کروڑ`;
}

function buildDetailSystemPrompt(rawDoc) {
  const SKIP = new Set(['nan', 'none', 'n/a', 'unknown', '', 'null', 'undefined']);
  const lines = Object.entries(rawDoc)
    .filter(([, v]) => { const s = String(v ?? '').trim(); return s.length > 0 && !SKIP.has(s.toLowerCase()); })
    .map(([k, v]) => `${k}: ${v}`)
    .join('\n');
  return `آپ پاکستان کے رئیل اسٹیٹ IVR اسسٹنٹ ہیں۔
صارف نے پراپرٹی کے بارے میں مزید تفصیل مانگی ہے۔ نیچے اس پراپرٹی کا مکمل ڈیٹا ہے:

${lines}

سخت قوانین:
1. صرف اردو میں جواب دیں
2. 3 سے 5 جملے — مختصر اور واضح
3. صرف اوپر دیے گئے ڈیٹا سے جواب دیں — کوئی چیز خود سے مت بتائیں
4. جو معلومات ڈیٹا میں نہ ہو: "یہ معلومات دستیاب نہیں"
5. رابطہ نمبر ہو تو بتائیں، نہ ہو تو کہیں: "رابطہ نمبر دستیاب نہیں"
6. کوئی markdown نہیں`;
}

// ── LLM calls ──────────────────────────────────────────────────────────────────

function trimHistory(history) {
  return history.length > 10 ? history.slice(-10) : history;
}

function cleanLLMText(raw) {
  return raw.replace(/\*\*/g, '').replace(/\*/g, '').replace(/^#+\s*/gm, '').replace(/\s+/g, ' ').trim();
}

async function queryGroq(session, userText, ragData) {
  try {
    const ragContext = buildRAGContext(ragData);
    session.history.push({ role: 'user', content: userText });
    session.history = trimHistory(session.history);

    const resp = await groq.chat.completions.create({
      model:       GROQ_MODEL,
      messages:    [{ role: 'system', content: buildSystemPrompt(ragContext) }, ...session.history],
      temperature: 0.2, max_tokens: 350, top_p: 0.85,
    });
    const text = cleanLLMText(resp.choices[0]?.message?.content || '');
    session.history.push({ role: 'assistant', content: text });
    return text || 'معاف کیجیے، دوبارہ کوشش کریں۔';
  } catch (e) {
    console.error('❌ Groq LLM error:', e.message);
    if (e.status === 429) return 'معاف کیجیے، سروس مصروف ہے۔ دوبارہ کوشش کریں۔';
    return 'معاف کیجیے، مسئلہ آگیا۔ دوبارہ کوشش کریں۔';
  }
}

async function queryGroqDetail(session, userText, rawDoc) {
  try {
    session.history.push({ role: 'user', content: userText });
    session.history = trimHistory(session.history);

    const resp = await groq.chat.completions.create({
      model:       GROQ_MODEL,
      messages:    [{ role: 'system', content: buildDetailSystemPrompt(rawDoc) }, ...session.history],
      temperature: 0.1, max_tokens: 400, top_p: 0.9,
    });
    const text = cleanLLMText(resp.choices[0]?.message?.content || '');
    session.history.push({ role: 'assistant', content: text });
    return text || 'معاف کیجیے، تفصیل نہیں مل سکی۔';
  } catch (e) {
    console.error('❌ Groq Detail error:', e.message);
    return 'معاف کیجیے، تفصیل نہیں مل سکی۔ دوبارہ کوشش کریں۔';
  }
}

// ═══════════════════════════════════════════════
// TTS
// ═══════════════════════════════════════════════

async function streamTTS(text, ws) {
  try {
    const res = await fetch(`${WORKER_BASE}/tts_urdu_stream`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }), signal: AbortSignal.timeout(60000),
    });
    if (!res.ok) throw new Error('TTS stream error: ' + res.status);
    const CHUNK = 8192;
    let buf = Buffer.alloc(0), total = 0;
    for await (const raw of res.body) {
      buf = Buffer.concat([buf, Buffer.from(raw)]);
      while (buf.length >= CHUNK) {
        const slice = buf.slice(0, CHUNK); buf = buf.slice(CHUNK); total += slice.length;
        wsSend(ws, { type: 'audio_chunk', mime: 'audio/mpeg', data: slice.toString('base64') });
      }
    }
    if (buf.length > 0) {
      total += buf.length;
      wsSend(ws, { type: 'audio_chunk', mime: 'audio/mpeg', data: buf.toString('base64') });
    }
    console.log(`✅ TTS streamed: ${total} bytes`);
  } catch (e) {
    console.error('❌ TTS stream error:', e.message);
  } finally {
    wsSend(ws, { type: 'audio_done' });
  }
}

async function generateTTS(text) {
  try {
    const res = await fetch(`${WORKER_BASE}/tts_urdu`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }), signal: AbortSignal.timeout(30000),
    });
    if (!res.ok) throw new Error('TTS error: ' + res.status);
    return Buffer.from(await res.arrayBuffer());
  } catch (e) {
    console.error('❌ TTS error:', e.message);
    return Buffer.alloc(0);
  }
}

// ═══════════════════════════════════════════════
// HEALTH CHECKS
// ═══════════════════════════════════════════════

async function checkWorker() {
  for (let i = 1; i <= 5; i++) {
    try {
      const data = await (await fetch(`${WORKER_BASE}/health`, { signal: AbortSignal.timeout(5000) })).json();
      console.log('✅ Python Worker:', data); return;
    } catch {
      if (i < 5) { console.log(`⏳ Worker not ready (${i}/5)…`); await new Promise(r => setTimeout(r, 2000)); }
      else console.error('❌ Python worker not reachable!');
    }
  }
}

(async () => {
  await checkWorker();
  try {
    const data = await (await fetch(`${RAG_BASE}/health`, { signal: AbortSignal.timeout(5000) })).json();
    console.log(`✅ RAG: ${data.sqlite_rows} props | FAISS: ${data.faiss_vecs} vecs | supabase=${data.supabase}`);
  } catch { console.warn('⚠️ RAG server not reachable — LLM-only fallback active'); }
})();

// ═══════════════════════════════════════════════
// REST ENDPOINTS
// ═══════════════════════════════════════════════

app.get('/api/conversations', async (req, res) => {
  if (!supabase) return res.json({ error: 'Supabase not configured' });
  try {
    const { data, error } = await supabase
      .from('conversations')
      .select('id, client_id, started_at, ended_at, duration_sec, status')
      .order('started_at', { ascending: false })
      .limit(20);
    if (error) throw error;
    res.json(data);
  } catch (e) { res.status(500).json({ error: e.message }); }
});

app.get('/api/visits', async (req, res) => {
  if (!supabase) return res.json({ error: 'Supabase not configured' });
  try {
    const { data, error } = await supabase
      .from('client_visits')
      .select('*')
      .order('created_at', { ascending: false })
      .limit(20);
    if (error) throw error;
    res.json(data);
  } catch (e) { res.status(500).json({ error: e.message }); }
});

// ═══════════════════════════════════════════════
// WEBSOCKET SERVER
// ═══════════════════════════════════════════════

const server = app.listen(3000, () => console.log('🚀 Server: http://localhost:3000'));
const wss    = new WebSocketServer({ server });

wss.on('connection', (ws) => {
  const clientId = `client_${Date.now()}_${crypto.randomBytes(4).toString('hex')}`;
  const state    = { audioChunks: [], isProcessing: false, totalBytes: 0 };
  console.log(`🔗 Connected: ${clientId}`);

  wsSend(ws, { type: 'connected', clientId,
    message: 'السلام علیکم! رئیل اسٹیٹ سے متعلق کوئی بھی سوال پوچھیں۔' });

  const session = getOrCreateSession(clientId);
  dbCreateConversation(clientId).then(id => { session.conversationId = id; });

  ws.on('message', async (data, isBinary) => {
    // Binary = audio chunk
    if (isBinary) {
      state.audioChunks.push(data);
      state.totalBytes += data.length;
      return;
    }

    let msg;
    try { msg = JSON.parse(data.toString()); } catch { return; }

    // ── Greeting ────────────────────────────────
    if (msg.type === 'play_greeting') {
      try {
        const buf = await generateTTS(GREETING_TEXT);
        wsSend(ws, { type: 'greeting_audio', mime: 'audio/mpeg', data: buf.toString('base64') });
        await dbSaveMessage(session.conversationId, 'assistant', GREETING_TEXT);
      } catch {
        wsSend(ws, { type: 'greeting_audio', data: '' });
      }
      return;
    }

    // ── Main speech pipeline ─────────────────────
    if (msg.type === 'end_speech') {
      if (state.isProcessing) {
        wsSend(ws, { type: 'info', message: 'پہلے والا سوال پراسیس ہو رہا ہے...' });
        return;
      }
      if (state.audioChunks.length === 0 || state.totalBytes < 5000) {
        wsSend(ws, { type: 'error', message: 'آواز بہت مختصر ہے، دوبارہ بولیں۔' });
        state.audioChunks = []; state.totalBytes = 0;
        return;
      }

      state.isProcessing = true;
      const chunks = [...state.audioChunks];
      state.audioChunks = []; state.totalBytes = 0;

      wsSend(ws, { type: 'processing', message: 'آواز پراسیس ہو رہی ہے...' });
      const tempFile = path.join(tempDir, `${clientId}_${Date.now()}.webm`);

      try {
        fs.writeFileSync(tempFile, Buffer.concat(chunks.map(c => Buffer.isBuffer(c) ? c : Buffer.from(c))));

        // 1. STT
        wsSend(ws, { type: 'status', step: 'stt', message: 'آواز سمجھی جا رہی ہے...' });
        const { text: userText, rejected, reason } = await transcribeAudio(tempFile);

        if (rejected || !userText || userText.length < 2) {
          wsSend(ws, { type: 'error', message: reason === 'noise_floor'
            ? 'پس منظر کی آواز آئی۔ واضح طور پر دوبارہ بولیں۔'
            : 'آواز سمجھ نہیں آئی۔ دوبارہ بولیں۔' });
          return;
        }

        const garbage = isGarbageTranscript(userText);
        if (garbage.bad) {
          wsSend(ws, { type: 'error', message: 'آواز واضح نہیں تھی، دوبارہ بولیں۔' });
          return;
        }

        wsSend(ws, { type: 'transcript', text: userText });
        await dbSaveMessage(session.conversationId, 'user', userText);

        // 2. Visit flow — agar active hai to pehle handle karo
        if (session.visitState) {
          const handled = await handleVisitFlow(session, userText, ws);
          if (handled) return;
        }

        // 3. Detail intent
        if (isDetailIntent(userText) && session.lastRawDocs?.length) {
          wsSend(ws, { type: 'status', step: 'llm', message: 'تفصیل تیار ہو رہی ہے...' });
          const aiText = await queryGroqDetail(session, userText, session.lastRawDocs[0]);
          wsSend(ws, { type: 'ai_response', text: aiText });
          await dbSaveMessage(session.conversationId, 'assistant', aiText);
          wsSend(ws, { type: 'status', step: 'tts', message: 'آواز بن رہی ہے...' });
          await streamTTS(aiText, ws);
          return;
        }

        // 4. RAG → LLM — RAG sirf tab call hoga jab sawal property-related ho
        let ragData = null;

        if (isPropertySearchIntent(userText)) {
          wsSend(ws, { type: 'status', step: 'rag', message: 'پراپرٹیز تلاش ہو رہی ہیں...' });
          const historyStr = session.history
            .map(m => `${m.role === 'user' ? 'صارف' : 'اسسٹنٹ'}: ${m.content}`)
            .join('\n');
          ragData = await queryRAG(userText, historyStr, session.conversationId);

          if (ragData?.matched_properties?.length) {
            session.lastRawDocs = ragData.matched_properties;
            session.lastUrls    = ragData.matched_properties
              .map(p => p.link || '')
              .filter(l => l && l !== 'unknown');
          }

          if (ragData) {
            wsSend(ws, {
              type:        'rag_result',
              count:       ragData.count       || 0,
              is_specific: ragData.is_specific || false,
              no_results:  ragData.no_results  || false,
              results:     ragData.results     || [],
              urls:        session.lastUrls    || [],
            });
            dbLogRAGSearch(
              session.conversationId, userText,
              ragData.results, ragData.count || 0, ragData.is_specific || false
            );
          }
        }

        wsSend(ws, { type: 'status', step: 'llm', message: 'جواب تیار ہو رہا ہے...' });
        const aiText = await queryGroq(session, userText, ragData);

        wsSend(ws, { type: 'ai_response', text: aiText });
        await dbSaveMessage(session.conversationId, 'assistant', aiText);

        wsSend(ws, { type: 'status', step: 'tts', message: 'آواز بن رہی ہے...' });
        await streamTTS(aiText, ws);

        // 5. Agar RAG ne results diye — visit schedule karne ka poochho
        if (ragData?.count > 0 && !ragData?.no_results) {
          // Thodi der baad visit prompt
          await new Promise(r => setTimeout(r, 800));
          session.visitState = 'asked';
          wsSend(ws, { type: 'ai_response', text: VISIT_PROMPT_TEXT });
          await dbSaveMessage(session.conversationId, 'assistant', VISIT_PROMPT_TEXT);
          wsSend(ws, { type: 'status', step: 'tts', message: 'آواز بن رہی ہے...' });
          await streamTTS(VISIT_PROMPT_TEXT, ws);
        }

      } catch (err) {
        console.error('❌ Pipeline error:', err.message);
        wsSend(ws, { type: 'error', message: 'معاف کیجیے، مسئلہ آگیا۔ دوبارہ کوشش کریں۔' });
        wsSend(ws, { type: 'audio_done' });
      } finally {
        unlinkSafe(tempFile);
        state.isProcessing = false;
      }
      return;
    }

    // ── Reset ───────────────────────────────────
    if (msg.type === 'reset_conversation') {
      const oldId  = session.conversationId;
      const durSec = Math.round((Date.now() - new Date(session.startedAt).getTime()) / 1000);
      await dbEndConversation(oldId, durSec);
      sessions.delete(clientId);
      const newSession = getOrCreateSession(clientId);
      newSession.conversationId = await dbCreateConversation(clientId);
      wsSend(ws, { type: 'reset_confirmed', message: 'نئی گفتگو شروع ہو گئی۔' });
      return;
    }

    if (msg.type === 'ping') { wsSend(ws, { type: 'pong' }); return; }
  });

  ws.on('close', async () => {
    console.log(`🔌 Disconnected: ${clientId}`);
    const s = sessions.get(clientId);
    if (s) {
      const dur = Math.round((Date.now() - new Date(s.startedAt).getTime()) / 1000);
      await dbEndConversation(s.conversationId, dur);
    }
    setTimeout(() => sessions.delete(clientId), 300_000);
  });

  ws.on('error', (e) => console.error(`❌ WS error ${clientId}:`, e.message));
});

// ── Cleanup intervals ──────────────────────────────────────────────────────────
setInterval(() => {
  try {
    let n = 0;
    fs.readdirSync(tempDir).forEach(f => {
      const fp = path.join(tempDir, f);
      try {
        if (Date.now() - fs.statSync(fp).mtimeMs > 3_600_000) { fs.unlinkSync(fp); n++; }
      } catch {}
    });
    if (n) console.log(`🧹 Cleaned ${n} temp files`);
  } catch {}
}, 600_000);

setInterval(() => {
  const now = Date.now(); let n = 0;
  for (const [id, s] of sessions.entries()) {
    if (now - s.lastActivity > 30 * 60 * 1000) { sessions.delete(id); n++; }
  }
  if (n) console.log(`🧹 Expired ${n} idle sessions`);
}, 5 * 60 * 1000);

process.on('SIGINT', () => {
  wss.clients.forEach(c => c.close(1000, 'Server shutting down'));
  server.close(() => process.exit(0));
  setTimeout(() => process.exit(1), 5000);
});