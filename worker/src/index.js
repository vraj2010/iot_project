/**
 * I2Sonic cloud orchestrator — Phase 3.2 (text in, voice out, free voices)
 *
 *   GET  /health   no auth. Shows the model and the voice chain.
 *
 *   All routes below need the header  X-Device-Token: <your token>
 *   and a JSON body  {"text": "..."}
 *
 *   POST /chat           -> {"reply": "...", "ms": 640}          (text only)
 *   POST /tts            -> audio/mpeg or audio/wav              (speak `text` as-is; laptop testing)
 *   POST /tts?provider=aura   force one voice provider (melotts | aura | gemini)
 *   POST /say            -> framed binary for the S3             (ask Gemini, then speak the reply)
 *
 *   POST /ask            body = raw 16 kHz s16le mono PCM (or a WAV file), not JSON.
 *                        Whisper -> Gemini -> voice; returns an I2S3 frame for the S3.
 *   POST /ask?debug=1    same, but returns JSON {heard, reply, ...} without speech (laptop testing)
 *
 *   Speech-to-text providers, in order (STT_PROVIDERS): workers (Workers AI Whisper
 *   large-v3-turbo, no extra key) then groq (Groq Whisper, needs GROQ_API_KEY secret).
 *
 *   Voice providers are tried in order (TTS_PROVIDERS in wrangler.toml).
 *   If one fails (quota, error), the next one is used:
 *     melotts  Workers AI MeloTTS: ~500 min/day free, returns MP3
 *     aura     Workers AI Deepgram Aura-1: ~50 replies/day free, 16 kHz PCM
 *     gemini   Gemini TTS: ~10/day free, WAV -> 16 kHz PCM
 *
 *   /say frame, all integers little-endian uint32:
 *     "I2S2" | format (1 = PCM s16le mono, 2 = MP3) | sample_rate (PCM only, else 0)
 *            | provider (1 melotts, 2 aura, 3 gemini) | chat_ms | tts_ms | text_len
 *            | text (UTF-8) | audio bytes...
 *
 *   /ask frame: "I2S3" | format | sample_rate | provider | chat_ms | tts_ms | stt_ms
 *            | heard_len | text_len | heard (UTF-8) | text (UTF-8) | audio bytes...
 *   If nothing usable was heard, heard is empty and the text is a short "didn't catch that".
 *
 *   Errors are always JSON {"error": "...", ...} with a non-200 status. If Gemini
 *   answers but every voice fails, /say returns the text in "reply" so the S3
 *   can print it and speak it with its own offline voice.
 */

const CHAT_URL =
  "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions";
const GEMINI_TTS_URL = "https://generativelanguage.googleapis.com/v1beta/interactions";

  const SYSTEM_PROMPT =
  // Who she is
  "You are Nexa, the voice assistant inside a small smart speaker. Everything you write is spoken aloud by a text-to-speech voice. " +
  "Personality: calm, quick and quietly confident; warm but efficient, with a dry sense of humour used sparingly and never at the user's expense. " +
  "You may call the user boss now and then, never more than once in a reply. " +

  // Answer quality
  "Lead with the answer in the first sentence, then add at most one useful detail. Keep replies to one to three short sentences unless the user asks for more; if a topic needs more, give the key point and offer to go on. " +
  "Be accurate and specific. Prefer a direct answer over hedging, but if you are unsure, say how sure you are in a few words rather than guessing. Never invent facts, numbers, names, quotes or sources. " +
  "If a request is ambiguous or sounds misheard, ask one short clarifying question instead of guessing. " +
  "For step-by-step tasks like recipes or directions, give the first two or three steps and offer the rest. " +

  // Written for the ear
  "Write for the ear: plain spoken English and short sentences, with no markdown, lists, emojis, symbols or URLs. " +
  "Say numbers, units and abbreviations the way a person would, for example twenty five degrees Celsius rather than 25°C. " +
  "Write your own name as Nexa, never in capitals. " +

  // Honest about limits
  "Be honest about what you can't do. You have no internet access, live data, clock or memory of earlier conversations, and you can't control devices yet. " +
  "For news, weather, prices, scores or today's date, say you can't check live information yet and share any general knowledge, noting it may be out of date. Never pretend to have done something you can't do. " +

  // Ethics and safety
  "Be kind, respectful and fair to everyone. On contested political or religious questions, give balanced facts rather than taking a side. " +
  "If asked to help with something dangerous, illegal or harmful to others, decline briefly without lecturing, and offer a safer alternative when there is one. " +
  "For health, legal or money questions, give useful general information and suggest a qualified professional for anything serious. " +
  "If someone mentions an emergency, a medical crisis or wanting to hurt themselves, respond with care, tell them to call 112 right away or Tele MANAS on 14416 for mental health support, and encourage them to reach someone they trust. " +
  "Never ask for passwords, card numbers or other sensitive personal details. " +
  "Don't say you are an AI or a language model unless asked; if asked, be truthful.";

const MAX_INPUT_CHARS = 500;
const UPSTREAM_TIMEOUT_MS = 20000;
const OUT_RATE = 16000; // the S3's speaker runs at 16 kHz
const LITTLE_ENDIAN = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;

const STT_MODEL = "@cf/openai/whisper-large-v3-turbo";
const GROQ_STT_URL = "https://api.groq.com/openai/v1/audio/transcriptions";
const MAX_AUDIO_SECONDS = 30;
const MIN_AUDIO_SECONDS = 0.3;
const NOT_HEARD_REPLY = "Sorry, I didn't catch that. Could you say it again?";
// Whisper sometimes "hears" these in silence or noise (subtitle credits from its
// training data). Nobody says them to a speaker, so a transcript that is only one
// of them counts as nothing heard. Real replies like "thanks" or "okay" are kept.
const HALLUCINATIONS = new Set([
  "", "you", "um", "uh", "hmm", "mm", "the end",
  "thanks for watching", "thank you for watching", "thanks for watching bye",
  "please subscribe", "subscribe", "like and subscribe",
  "subtitles by the amara org community", "transcribed by", "subtitles by",
]);

const FORMAT = { pcm: 1, mp3: 2 };
const PROVIDER_ID = { melotts: 1, aura: 2, gemini: 3 };
const PROVIDERS = { melotts: ttsMelo, aura: ttsAura, gemini: ttsGemini };
const STT_PROVIDERS = { workers: sttWorkers, groq: sttGroq };

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (url.pathname === "/health" && request.method === "GET") {
      return json({
        ok: true,
        model: env.GEMINI_MODEL || null,
        voices: providerChain(env),
        stt: sttChain(env),
        workers_ai: Boolean(env.AI),
        groq_key: Boolean(env.GROQ_API_KEY),
      });
    }

    const routes = { "/chat": handleChat, "/tts": handleTts, "/say": handleSay };
    const handler = routes[url.pathname];
    if (!handler && url.pathname !== "/ask") return json({ error: "not found" }, 404);
    if (request.method !== "POST") return json({ error: "use POST" }, 405);

    const authError = checkAuth(request, env);
    if (authError) return authError;

    if (url.pathname === "/ask") return handleAsk(request, env, url);   // audio body, not JSON

    const input = await readText(request);
    if (input.error) return json({ error: input.error }, 400);

    return handler(input.text, env, url);
  },
};

// ---------------------------------------------------------------- routes

async function handleChat(text, env) {
  const started = Date.now();
  const result = await askGemini(text, env);
  const ms = Date.now() - started;
  if (result.error) return json({ error: result.error, ms }, result.status);
  return json({ reply: result.reply, ms });
}

async function handleTts(text, env, url) {
  const only = url.searchParams.get("provider");
  if (only && !PROVIDERS[only]) {
    return json({ error: `unknown provider "${only}" (use melotts, aura or gemini)` }, 400);
  }
  const speech = await synthesize(text, env, only ? [only] : providerChain(env));
  if (speech.error) return json({ error: speech.error, tried: speech.tried }, speech.status);

  const headers = { "X-TTS-Ms": String(speech.ms), "X-TTS-Provider": speech.provider };
  if (speech.format === "mp3") {
    return new Response(speech.bytes, { headers: { ...headers, "Content-Type": "audio/mpeg" } });
  }
  return new Response(wavFile(speech.pcm, OUT_RATE), { headers: { ...headers, "Content-Type": "audio/wav" } });
}

async function handleSay(text, env) {
  const t0 = Date.now();
  const chat = await askGemini(text, env);
  const chatMs = Date.now() - t0;
  if (chat.error) return json({ error: chat.error, ms: chatMs }, chat.status);

  const speech = await synthesize(chat.reply, env, providerChain(env));
  if (speech.error) {
    return json({ error: `voice: ${speech.error}`, tried: speech.tried, reply: chat.reply, ms: chatMs },
      speech.status);
  }
  return new Response(sayFrame(chat.reply, speech, chatMs), {
    headers: { "Content-Type": "application/octet-stream", "X-TTS-Provider": speech.provider },
  });
}

async function handleAsk(request, env, url) {
  const debug = url.searchParams.get("debug") === "1";
  const audio = await readAudio(request);
  if (audio.error) return json({ error: audio.error }, 400);

  // 1) speech -> text
  const t0 = Date.now();
  const stt = await transcribe(audio.pcm, env);
  const sttMs = Date.now() - t0;
  if (stt.error) return json({ error: `hearing: ${stt.error}`, tried: stt.tried, ms: sttMs }, stt.status);
  const heard = cleanTranscript(stt.text);

  // 2) text -> answer (or a short "didn't catch that" without calling Gemini)
  let reply, chatMs = 0;
  if (!heard) {
    reply = NOT_HEARD_REPLY;
  } else {
    const t1 = Date.now();
    const chat = await askGemini(heard, env);
    chatMs = Date.now() - t1;
    if (chat.error) return json({ error: chat.error, heard, stt_ms: sttMs, ms: chatMs }, chat.status);
    reply = chat.reply;
  }

  if (debug) {
    return json({ heard, raw_transcript: stt.text, reply, stt_provider: stt.provider, stt_ms: sttMs,
                  chat_ms: chatMs, seconds: +(audio.pcm.length / OUT_RATE).toFixed(2) });
  }

  // 3) answer -> voice
  const speech = await synthesize(reply, env, providerChain(env));
  if (speech.error) {
    return json({ error: `voice: ${speech.error}`, tried: speech.tried, heard, reply, ms: chatMs }, speech.status);
  }
  return new Response(askFrame(heard, reply, speech, chatMs, sttMs), {
    headers: { "Content-Type": "application/octet-stream", "X-TTS-Provider": speech.provider,
               "X-STT-Provider": stt.provider },
  });
}

// ---------------------------------------------------------------- auth + input

function checkAuth(request, env) {
  if (!env.DEVICE_TOKEN || !env.GEMINI_API_KEY) {
    return json({ error: "server not configured: set DEVICE_TOKEN and GEMINI_API_KEY secrets" }, 500);
  }
  const given = request.headers.get("X-Device-Token") || "";
  if (!safeEqual(given, env.DEVICE_TOKEN)) {
    return json({ error: "bad or missing X-Device-Token" }, 401);
  }
  return null;
}

// Compare without returning early on the first different character.
function safeEqual(a, b) {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}

async function readText(request) {
  let body;
  try {
    body = await request.json();
  } catch {
    return { error: 'body must be JSON like {"text": "..."}' };
  }
  const text = typeof body?.text === "string" ? body.text.trim() : "";
  if (!text) return { error: "text is empty" };
  if (text.length > MAX_INPUT_CHARS) return { error: `text longer than ${MAX_INPUT_CHARS} characters` };
  return { text };
}

// /ask body: raw 16 kHz s16le mono PCM from the S3, or any 16-bit WAV (laptop tests).
async function readAudio(request) {
  const bytes = new Uint8Array(await request.arrayBuffer());
  if (bytes.length > MAX_AUDIO_SECONDS * OUT_RATE * 2 + 1024 && !isWav(bytes)) {
    return { error: `audio longer than ${MAX_AUDIO_SECONDS} s` };
  }
  let pcm;
  if (isWav(bytes)) {
    const decoded = toPcm16Mono(bytes, OUT_RATE);
    if (decoded.error) return { error: decoded.error };
    pcm = decoded.rate === OUT_RATE ? decoded.pcm : resample(decoded.pcm, decoded.rate, OUT_RATE);
  } else {
    const n = bytes.length >> 1;
    pcm = new Int16Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + n * 2));
  }
  const seconds = pcm.length / OUT_RATE;
  if (seconds < MIN_AUDIO_SECONDS) return { error: "audio too short" };
  if (seconds > MAX_AUDIO_SECONDS) return { error: `audio longer than ${MAX_AUDIO_SECONDS} s` };
  return { pcm };
}

// ---------------------------------------------------------------- Gemini: chat

async function askGemini(text, env) {
  const res = await callUpstream(CHAT_URL, {
    headers: { Authorization: `Bearer ${env.GEMINI_API_KEY}` },
    body: {
      model: env.GEMINI_MODEL || "gemini-3.5-flash-lite",
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        { role: "user", content: text },
      ],
      max_tokens: 1024,
    },
  });
  if (res.error) return res;

  const raw = res.data?.choices?.[0]?.message?.content;
  const reply = forSpeech(typeof raw === "string" ? raw : "", env);
  if (!reply) return { error: "gemini returned an empty answer", status: 502 };
  return { reply };
}

// ---------------------------------------------------------------- hearing: speech to text

function sttChain(env) {
  const list = (env.STT_PROVIDERS || "workers,groq")
    .split(",").map((s) => s.trim().toLowerCase()).filter((s) => STT_PROVIDERS[s]);
  return list.length ? list : ["workers"];
}

// Try each speech-to-text provider in turn: { text, provider } or { error, status, tried }.
async function transcribe(pcm, env) {
  const wav = wavFile(pcm, OUT_RATE);
  const tried = [];
  let lastStatus = 502;
  for (const name of sttChain(env)) {
    let out;
    try {
      out = await STT_PROVIDERS[name](wav, env);
    } catch (err) {
      out = errorFromException(err);
    }
    if (!out.error) return { text: out.text, provider: name };
    tried.push(`${name}: ${out.error}`);
    lastStatus = out.status || 502;
  }
  return { error: tried[tried.length - 1] || "no speech-to-text provider configured", status: lastStatus, tried };
}

async function sttWorkers(wav, env) {
  if (!env.AI) return { error: "Workers AI binding missing", status: 500 };
  const input = {
    audio: bytesToBase64(wav),
    task: "transcribe",
    language: env.STT_LANG || "en",
    vad_filter: true,                    // skip silence and noise before transcribing
    condition_on_previous_text: false,   // fewer runaway repetitions
  };
  if (env.STT_HINT) input.initial_prompt = env.STT_HINT;
  const result = await env.AI.run(STT_MODEL, input);
  const text = typeof result?.text === "string" ? result.text : null;
  if (text === null) return { error: "Whisper returned no text field", status: 502 };
  return { text };
}

async function sttGroq(wav, env) {
  if (!env.GROQ_API_KEY) return { error: "GROQ_API_KEY secret not set", status: 500 };
  const form = new FormData();
  form.append("file", new Blob([wav], { type: "audio/wav" }), "question.wav");
  form.append("model", env.GROQ_STT_MODEL || "whisper-large-v3-turbo");
  form.append("language", env.STT_LANG || "en");
  form.append("response_format", "json");
  form.append("temperature", "0");
  if (env.STT_HINT) form.append("prompt", env.STT_HINT);

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), UPSTREAM_TIMEOUT_MS);
  let res;
  try {
    res = await fetch(GROQ_STT_URL, {
      method: "POST",
      signal: controller.signal,
      headers: { Authorization: `Bearer ${env.GROQ_API_KEY}` },
      body: form,
    });
  } catch (err) {
    return { error: err && err.name === "AbortError" ? "timed out" : `unreachable: ${err}`, status: 504 };
  } finally {
    clearTimeout(timer);
  }
  if (res.status === 429) return { error: "rate limit hit", status: 429 };
  if (!res.ok) return { error: `HTTP ${res.status}: ${(await res.text()).slice(0, 200)}`, status: 502 };
  const data = await res.json().catch(() => null);
  if (typeof data?.text !== "string") return { error: "no text in the reply", status: 502 };
  return { text: data.text };
}

// Trim, and treat empty or known-hallucination transcripts as "nothing heard".
function cleanTranscript(text) {
  const t = String(text || "").replace(/\s+/g, " ").trim();
  const key = t.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();   // "Amara.org!" -> "amara org"
  if (HALLUCINATIONS.has(key) || key.length < 2) return "";
  return t.slice(0, MAX_INPUT_CHARS);
}

function bytesToBase64(bytes) {
  if (typeof bytes.toBase64 === "function") return bytes.toBase64();
  let out = "";
  for (let i = 0; i < bytes.length; i += 0x8000) {
    out += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  }
  return btoa(out);
}

// ---------------------------------------------------------------- voice: provider chain

function providerChain(env) {
  const list = (env.TTS_PROVIDERS || "melotts,aura,gemini")
    .split(",").map((s) => s.trim().toLowerCase()).filter((s) => PROVIDERS[s]);
  return list.length ? list : ["melotts"];
}

// Try each provider in turn. Returns
//   { format: "mp3", bytes, provider, ms }  or  { format: "pcm", pcm (16 kHz Int16Array), provider, ms }
// or { error, status, tried } when all of them failed.
async function synthesize(text, env, chain) {
  const tried = [];
  let lastStatus = 502;
  for (const name of chain) {
    const started = Date.now();
    let out;
    try {
      out = await PROVIDERS[name](text, env);
    } catch (err) {
      out = errorFromException(err);
    }
    if (!out.error) return { ...out, provider: name, ms: Date.now() - started };
    tried.push(`${name}: ${out.error}`);
    lastStatus = out.status || 502;
  }
  return { error: tried[tried.length - 1] || "no voice provider configured", status: lastStatus, tried };
}

// Workers AI quota errors arrive as exceptions; report them as 429 so the S3 says "limit hit".
function errorFromException(err) {
  const msg = String(err && err.message ? err.message : err).slice(0, 200);
  const quota = /limit|quota|neuron|3036|429|capacity/i.test(msg);
  return { error: msg, status: quota ? 429 : 502 };
}

async function ttsMelo(text, env) {
  if (!env.AI) return { error: "Workers AI binding missing (add [ai] binding = \"AI\")", status: 500 };
  const result = await env.AI.run("@cf/myshell-ai/melotts", { prompt: text, lang: env.MELO_LANG || "en" });
  return audioFromResult(await toBytes(result), env);
}

async function ttsAura(text, env) {
  if (!env.AI) return { error: "Workers AI binding missing (add [ai] binding = \"AI\")", status: 500 };
  const result = await env.AI.run("@cf/deepgram/aura-1", {
    text,
    speaker: env.AURA_SPEAKER || "orion",
    encoding: "linear16",
    container: "none",
    sample_rate: OUT_RATE,
  });
  const bytes = await toBytes(result);
  // Asked for headerless 16 kHz PCM; still accept WAV/MP3 if the model sends one.
  return audioFromResult(bytes, env, { rawRate: OUT_RATE });
}

async function ttsGemini(text, env) {
  const content = { type: "text", text };
  if (env.TTS_STYLE) content.annotations = [{ type: "speech_metadata", style: env.TTS_STYLE }];

  const res = await callUpstream(GEMINI_TTS_URL, {
    headers: { "x-goog-api-key": env.GEMINI_API_KEY },
    body: {
      model: env.GEMINI_TTS_MODEL || "gemini-3.8-flash-lite-tts",
      input: [{ type: "user_input", content: [content] }],
      response_format: { type: "audio", mime_type: "audio/wav", sample_rate: OUT_RATE },
      generation_config: { speech_config: [{ voice: env.GEMINI_VOICE || "Charon" }] },
    },
  });
  if (res.error) return res;
  const b64 = findAudio(res.data);
  if (!b64) return { error: "no audio in the Gemini TTS response", status: 502 };
  return audioFromResult(base64ToBytes(b64), env, { rawRate: OUT_RATE });
}

// ---------------------------------------------------------------- audio helpers

// Normalise whatever a provider returned into bytes. Handles a ReadableStream,
// a Response, an ArrayBuffer/typed array, base64 text, or an object holding any
// of those under "audio" / "result" (MeloTTS's response shape isn't documented).
async function toBytes(value) {
  if (value == null) return null;
  if (value instanceof Uint8Array) return value;
  if (value instanceof ArrayBuffer) return new Uint8Array(value);
  if (ArrayBuffer.isView(value)) return new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
  if (typeof Response !== "undefined" && value instanceof Response) return new Uint8Array(await value.arrayBuffer());
  if (typeof value.getReader === "function") return new Uint8Array(await new Response(value).arrayBuffer());
  if (typeof value === "string") return base64ToBytes(value.replace(/^data:[^,]*,/, ""));
  if (Array.isArray(value)) return Uint8Array.from(value);
  if (typeof value === "object") {
    for (const key of ["audio", "result", "data"]) {
      if (key in value) return toBytes(value[key]);
    }
  }
  return null;
}

function hasId3(b) {
  return b.length >= 3 && b[0] === 0x49 && b[1] === 0x44 && b[2] === 0x33; // "ID3" tag
}

// MP3 = ID3 tag, or a valid MPEG audio frame header (sync + legal version,
// layer, bitrate and sample-rate fields), so random PCM is not mistaken for it.
function isMp3(b) {
  if (hasId3(b)) return true;
  if (b.length < 4 || b[0] !== 0xff || (b[1] & 0xe0) !== 0xe0) return false;
  const version = (b[1] >> 3) & 3, layer = (b[1] >> 1) & 3;
  const bitrate = b[2] >> 4, rate = (b[2] >> 2) & 3;
  return version !== 1 && layer !== 0 && bitrate !== 0 && bitrate !== 15 && rate !== 3;
}

function isWav(b) {
  return b.length >= 12 && b[0] === 0x52 && b[1] === 0x49 && b[2] === 0x46 && b[3] === 0x46 &&
    b[8] === 0x57 && b[9] === 0x41 && b[10] === 0x56 && b[11] === 0x45;
}

// Bytes -> { format: "mp3", bytes } or { format: "pcm", pcm (16 kHz) }.
// With rawRate set we asked for headerless PCM: then only an ID3 tag counts as
// MP3, because PCM starting with a quiet negative sample (FF FF) looks like a sync.
function audioFromResult(bytes, env, { rawRate } = {}) {
  if (!bytes || bytes.length < 64) return { error: "voice provider returned no audio", status: 502 };
  if (rawRate ? hasId3(bytes) : isMp3(bytes)) return { format: "mp3", bytes };
  if (!isWav(bytes) && !rawRate) return { error: "voice provider returned an unknown audio format", status: 502 };

  const decoded = toPcm16Mono(bytes, rawRate || OUT_RATE);
  if (decoded.error) return { error: decoded.error, status: 502 };
  const pcm = decoded.rate === OUT_RATE ? decoded.pcm : resample(decoded.pcm, decoded.rate, OUT_RATE);
  return { format: "pcm", pcm };
}

// The Gemini audio is base64 in a content block of type "audio". Prefer the
// documented path (steps[].content[]) but search the whole reply. Last block wins.
function findAudio(data) {
  if (typeof data?.output_audio?.data === "string") return data.output_audio.data;
  let found = null;
  const visit = (node) => {
    if (Array.isArray(node)) return node.forEach(visit);
    if (!node || typeof node !== "object") return;
    if (node.type === "audio" && typeof node.data === "string") found = node.data;
    for (const v of Object.values(node)) if (v && typeof v === "object") visit(v);
  };
  visit(data?.steps ?? data);
  return found;
}

// Accept a WAV file or headerless 16-bit PCM; return mono Int16 samples + rate.
function toPcm16Mono(bytes, assumedRate) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const tag = (o) => String.fromCharCode(bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]);

  if (!isWav(bytes)) {
    const n = bytes.length >> 1; // headerless: s16le mono at the rate we asked for
    return { pcm: new Int16Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + n * 2)), rate: assumedRate };
  }

  let rate = assumedRate, channels = 1, bits = 16, dataOff = -1, dataLen = 0;
  for (let o = 12; o + 8 <= bytes.length; ) {
    const id = tag(o);
    const size = view.getUint32(o + 4, true);
    if (id === "fmt ") {
      channels = view.getUint16(o + 10, true);
      rate = view.getUint32(o + 12, true);
      bits = view.getUint16(o + 22, true);
    } else if (id === "data") {
      dataOff = o + 8;
      dataLen = Math.min(size, bytes.length - dataOff); // streamed WAVs may say 0xFFFFFFFF
      break;
    }
    o += 8 + size + (size & 1);
  }
  if (dataOff < 0) return { error: "WAV without a data chunk" };
  if (bits !== 16) return { error: `WAV is ${bits}-bit, expected 16` };

  const frames = Math.floor(dataLen / (2 * channels));
  if (channels === 1 && LITTLE_ENDIAN) {
    // Fast path: one memory copy, no per-sample work.
    const start = bytes.byteOffset + dataOff;
    return { pcm: new Int16Array(bytes.buffer.slice(start, start + frames * 2)), rate };
  }
  const pcm = new Int16Array(frames);
  for (let i = 0; i < frames; i++) {
    let sum = 0;
    for (let c = 0; c < channels; c++) sum += view.getInt16(dataOff + (i * channels + c) * 2, true);
    pcm[i] = Math.round(sum / channels);
  }
  return { pcm, rate };
}

// Linear resample. Only used if a provider ignores the 16 kHz request.
function resample(pcm, from, to) {
  const outLen = Math.floor((pcm.length * to) / from);
  const out = new Int16Array(outLen);
  const step = from / to;
  for (let i = 0; i < outLen; i++) {
    const pos = i * step;
    const j = Math.floor(pos);
    const frac = pos - j;
    const a = pcm[j] ?? 0;
    const b = pcm[j + 1] ?? a;
    out[i] = Math.round(a + (b - a) * frac);
  }
  return out;
}

function base64ToBytes(b64) {
  if (typeof Uint8Array.fromBase64 === "function") return Uint8Array.fromBase64(b64);
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

// ---------------------------------------------------------------- output formats

function sayFrame(text, speech, chatMs) {
  const textBytes = new TextEncoder().encode(text);
  const audio = speech.format === "mp3"
    ? speech.bytes
    : new Uint8Array(speech.pcm.buffer, speech.pcm.byteOffset, speech.pcm.byteLength);
  const head = 28;
  const out = new Uint8Array(head + textBytes.length + audio.byteLength);
  const v = new DataView(out.buffer);
  out.set([0x49, 0x32, 0x53, 0x32], 0);                       // "I2S2"
  v.setUint32(4, FORMAT[speech.format], true);
  v.setUint32(8, speech.format === "pcm" ? OUT_RATE : 0, true);
  v.setUint32(12, PROVIDER_ID[speech.provider] || 0, true);
  v.setUint32(16, chatMs, true);
  v.setUint32(20, speech.ms, true);
  v.setUint32(24, textBytes.length, true);
  out.set(textBytes, head);
  if (speech.format === "pcm" && !LITTLE_ENDIAN) {
    writePcmLE(out, head + textBytes.length, speech.pcm);
  } else {
    out.set(audio, head + textBytes.length);
  }
  return out;
}

function askFrame(heard, text, speech, chatMs, sttMs) {
  const enc = new TextEncoder();
  const heardBytes = enc.encode(heard);
  const textBytes = enc.encode(text);
  const audio = speech.format === "mp3"
    ? speech.bytes
    : new Uint8Array(speech.pcm.buffer, speech.pcm.byteOffset, speech.pcm.byteLength);
  const head = 36;
  const out = new Uint8Array(head + heardBytes.length + textBytes.length + audio.byteLength);
  const v = new DataView(out.buffer);
  out.set([0x49, 0x32, 0x53, 0x33], 0);                       // "I2S3"
  v.setUint32(4, FORMAT[speech.format], true);
  v.setUint32(8, speech.format === "pcm" ? OUT_RATE : 0, true);
  v.setUint32(12, PROVIDER_ID[speech.provider] || 0, true);
  v.setUint32(16, chatMs, true);
  v.setUint32(20, speech.ms, true);
  v.setUint32(24, sttMs, true);
  v.setUint32(28, heardBytes.length, true);
  v.setUint32(32, textBytes.length, true);
  out.set(heardBytes, head);
  out.set(textBytes, head + heardBytes.length);
  const audioAt = head + heardBytes.length + textBytes.length;
  if (speech.format === "pcm" && !LITTLE_ENDIAN) writePcmLE(out, audioAt, speech.pcm);
  else out.set(audio, audioAt);
  return out;
}

function wavFile(pcm, rate) {
  const out = new Uint8Array(44 + pcm.byteLength);
  const v = new DataView(out.buffer);
  const put = (o, s) => { for (let i = 0; i < 4; i++) out[o + i] = s.charCodeAt(i); };
  put(0, "RIFF"); v.setUint32(4, 36 + pcm.byteLength, true); put(8, "WAVE");
  put(12, "fmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, rate, true); v.setUint32(28, rate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  put(36, "data"); v.setUint32(40, pcm.byteLength, true);
  writePcmLE(out, 44, pcm);
  return out;
}

function writePcmLE(out, offset, pcm) {
  if (LITTLE_ENDIAN) {
    out.set(new Uint8Array(pcm.buffer, pcm.byteOffset, pcm.byteLength), offset);
    return;
  }
  const v = new DataView(out.buffer, out.byteOffset + offset, pcm.byteLength);
  for (let i = 0; i < pcm.length; i++) v.setInt16(i * 2, pcm[i], true);
}

// ---------------------------------------------------------------- shared helpers

async function callUpstream(url, { headers, body }) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), UPSTREAM_TIMEOUT_MS);
  let res;
  try {
    res = await fetch(url, {
      method: "POST",
      signal: controller.signal,
      headers: { "Content-Type": "application/json", ...headers },
      body: JSON.stringify(body),
    });
  } catch (err) {
    const timedOut = err && err.name === "AbortError";
    return { error: timedOut ? "gemini timed out" : `could not reach gemini: ${err}`, status: timedOut ? 504 : 502 };
  } finally {
    clearTimeout(timer);
  }

  if (res.status === 429) return { error: "gemini rate limit hit, try again shortly", status: 429 };
  if (!res.ok) {
    // Pass Gemini's own message through: it names a wrong model, voice or key.
    const detail = (await res.text()).slice(0, 300);
    return { error: `gemini ${res.status}: ${detail}`, status: 502 };
  }
  try {
    return { data: await res.json() };
  } catch {
    return { error: "gemini sent a non-JSON reply", status: 502 };
  }
}

// Strip anything that sounds wrong when spoken, and cap the length.
function forSpeech(s, env) {
  const limit = parseInt(env.MAX_REPLY_CHARS || "400", 10);
  let out = s
    .replace(/```[\s\S]*?```/g, " ")         // code blocks
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1") // [text](url) -> text
    .replace(/https?:\/\/\S+/g, " ")         // bare URLs
    .replace(/[*_#`>|~]/g, "")               // markdown symbols
    .replace(/^\s*[-•]\s+/gm, "")            // bullet markers
    .replace(/\s+/g, " ")
    .trim();

  if (out.length > limit) {
    const cut = out.slice(0, limit);
    const end = Math.max(cut.lastIndexOf(". "), cut.lastIndexOf("? "), cut.lastIndexOf("! "));
    out = end > limit / 2 ? cut.slice(0, end + 1) : cut.trimEnd() + "...";
  }
  return out;
}

function json(obj, status = 200) {
  return new Response(JSON.stringify(obj), {
    status,
    headers: { "Content-Type": "application/json; charset=utf-8" },
  });
}
