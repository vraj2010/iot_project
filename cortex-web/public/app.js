/* Cortex upload page. Loaded from index.html; no inline script (CSP: script-src 'self'). */
"use strict";
const $ = (id) => document.getElementById(id);
const state = { docs: [], maxMb: 50, running: new Map(), uploading: new Set(), timers: {} };

/* ---------- helpers ---------- */
function el(tag, attrs, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "class") n.className = v;
    else if (k === "text") n.textContent = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) if (kid != null && kid !== false) n.append(kid.nodeType ? kid : String(kid));
  return n;
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
function toast(msg, ms = 3500) {
  const t = $("toast"); t.textContent = msg; t.classList.remove("hidden");
  clearTimeout(state.timers.toast); state.timers.toast = setTimeout(() => t.classList.add("hidden"), ms);
}
function fmtBytes(b) {
  if (b == null) return "";
  if (b < 1024) return b + " B";
  if (b < 1048576) return (b / 1024).toFixed(0) + " KB";
  return (b / 1048576).toFixed(1) + " MB";
}
function ago(iso) {
  if (!iso) return "";
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  if (s < 60) return "just now";
  if (s < 3600) return Math.floor(s / 60) + " min ago";
  if (s < 86400) return Math.floor(s / 3600) + " h ago";
  return new Date(iso).toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });
}
const fileUrl = (id, page) => `/api/file/${encodeURIComponent(id)}` + (page ? `?page=${encodeURIComponent(page)}` : "");

class ApiError extends Error { constructor(msg, status) { super(msg); this.status = status; } }
async function api(path, { method = "GET", body } = {}) {
  let res;
  try {
    res = await fetch(path, {
      method, credentials: "same-origin",
      headers: body !== undefined ? { "Content-Type": "application/json" } : {},
      body: body !== undefined ? JSON.stringify(body) : undefined,
    });
  } catch (e) { throw new ApiError("Network error: " + e.message, 0); }
  let data = null;
  const text = await res.text();
  try { data = text ? JSON.parse(text) : null; } catch { data = null; }
  if (res.status === 401 && path !== "/api/login") { showLogin(); throw new ApiError("Signed out", 401); }
  if (!res.ok) {
    let msg = data && data.detail;
    if (Array.isArray(msg)) msg = msg.map((d) => d.msg).join("; ");
    throw new ApiError(msg || `HTTP ${res.status}`, res.status);
  }
  return data;
}

/* ---------- session ---------- */
function showLogin() {
  $("app").classList.add("hidden"); $("login").classList.remove("hidden");
  stopPolling(); setTimeout(() => $("password").focus(), 0);
}
function showApp() {
  $("login").classList.add("hidden"); $("app").classList.remove("hidden");
  refreshDocs(); refreshAnswers(); startPolling();
}
$("loginForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  $("loginErr").textContent = ""; $("loginBtn").disabled = true;
  try {
    await api("/api/login", { method: "POST", body: { password: $("password").value } });
    $("password").value = ""; showApp();
  } catch (err) {
    $("loginErr").textContent = err.status === 401 ? "Wrong password." : err.message;
  } finally { $("loginBtn").disabled = false; }
});
$("logoutBtn").addEventListener("click", async () => {
  try { await api("/api/logout", { method: "POST" }); } catch {}
  showLogin();
});

/* ---------- upload ---------- */
const drop = $("drop"), input = $("fileInput");
drop.addEventListener("click", () => input.click());
drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => addFiles(e.dataTransfer.files));
input.addEventListener("change", () => { addFiles(input.files); input.value = ""; });
// dropping a file elsewhere on the page shouldn't navigate away
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", (e) => e.preventDefault());

const OK_EXT = [".pdf", ".docx", ".txt"];
const uploadQueue = [];
let activeUploads = 0;

function addFiles(list) {
  const subject = $("subject").value.trim();
  for (const file of Array.from(list || [])) {
    const ext = (file.name.match(/\.[^.]+$/) || [""])[0].toLowerCase();
    const item = makeItem(file.name, fmtBytes(file.size));
    if (!OK_EXT.includes(ext)) { item.fail("Only PDF, DOCX or TXT files."); continue; }
    if (file.size === 0) { item.fail("The file is empty."); continue; }
    if (file.size > state.maxMb * 1048576) { item.fail(`Larger than ${state.maxMb} MB.`); continue; }
    item.set("Waiting…", 0);
    uploadQueue.push({ file, subject, item });
  }
  pump();
}

function pump() {
  while (activeUploads < 2 && uploadQueue.length) {
    const job = uploadQueue.shift();
    activeUploads++;
    runJob(job).finally(() => { activeUploads--; pump(); });
  }
}

function makeItem(name, size) {
  const stateEl = el("span", { class: "qstate" });
  const fill = el("i");
  const bar = el("div", { class: "bar" }, fill);
  const note = el("div", { class: "qnote muted" });
  const li = el("li", {}, el("div", { class: "qhead" }, el("span", { class: "qname", title: name, text: name }),
    el("span", { class: "qstate", text: size })), bar, el("div", { class: "qhead" }, stateEl), note);
  $("queue").prepend(li);
  return {
    set(label, frac, indet) {
      stateEl.textContent = label;
      bar.classList.toggle("indet", !!indet);
      if (frac != null) fill.style.width = Math.max(0, Math.min(100, frac * 100)).toFixed(1) + "%";
    },
    note(text, cls) { note.textContent = text || ""; note.className = "qnote " + (cls || "muted"); },
    done(label) { bar.classList.remove("indet"); bar.classList.add("ok"); fill.style.width = "100%"; stateEl.textContent = label; },
    pause(msg) { bar.classList.remove("indet"); stateEl.textContent = "Paused"; note.textContent = msg; note.className = "qnote warn"; },
    fail(msg) { bar.classList.remove("indet"); bar.classList.add("bad"); fill.style.width = "100%"; stateEl.textContent = "Failed"; note.textContent = msg; note.className = "qnote err"; },
  };
}

function putFile(url, file, contentType, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("PUT", url);
    xhr.setRequestHeader("Content-Type", contentType);
    xhr.setRequestHeader("Cache-Control", "max-age=3600");
    xhr.upload.onprogress = (e) => { if (e.lengthComputable) onProgress(e.loaded / e.total); };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) return resolve();
      let msg = `Storage HTTP ${xhr.status}`;
      try { const d = JSON.parse(xhr.responseText); msg += ": " + (d.message || d.error || ""); } catch {}
      reject(new Error(msg));
    };
    xhr.onerror = () => reject(new Error("Upload failed (network)"));
    xhr.send(file);
  });
}

async function runJob({ file, subject, item }) {
  try {
    item.set("Preparing…", 0);
    const signed = await api("/api/upload-url", { method: "POST", body: { filename: file.name, size: file.size, subject: subject || null } });
    state.uploading.add(signed.document_id);
    refreshDocs();
    try {
      await putFile(signed.upload_url, file, signed.content_type, (f) => item.set(`Uploading ${Math.round(f * 100)}%`, f));
    } catch (err) {
      // the file never reached storage: drop its empty row so it doesn't linger as "uploaded"
      api(`/api/documents/${signed.document_id}`, { method: "DELETE" }).catch(() => {}).finally(refreshDocs);
      throw err;
    } finally { state.uploading.delete(signed.document_id); }
    item.set("Uploaded · reading the file…", 0, true);
    const r = await ingestLoop(signed.document_id, progressTo(item));
    item.done(`Ready · ${r.total} passages`);
  } catch (err) {
    showStop(item, err);
  } finally { refreshDocs(); }
}

/* What the ingest loop reports, shown on a queue item. */
function progressTo(item) {
  return (p) => {
    if (p.total) item.set(`Embedding ${p.done} / ${p.total} passages`, p.done / p.total);
    let note = "";
    if (p.busy) note = "Another tab or call is processing this file; waiting…";
    else if (p.retry_after) note = `${p.note || "Gemini is busy"}; continuing in ${Math.ceil(p.retry_after)} s…`;
    else if (p.note) note = p.note;
    if (!note && p.duplicate_of && p.duplicate_of.length) note = "Same file as: " + p.duplicate_of.map((d) => d.filename).join(", ");
    item.note(note);
  };
}

function showStop(item, err) {
  if (err.status === 401) return;
  if (err.paused) item.pause(err.message);
  else item.fail(err.message);
}

class StopError extends Error { constructor(msg, extra) { super(msg); Object.assign(this, extra || {}); } }

const MAX_CALLS = 200;      // ingest calls per loop (each works up to ~4 min)
const MAX_STALLED = 20;     // calls in a row that made no progress (busy, rate limits, errors)

/* Calls /api/ingest until the document is ready, with limits so it can never loop forever. */
async function ingestLoop(id, onProgress) {
  if (state.running.has(id)) return state.running.get(id);
  const p = (async () => {
    let stalled = 0, lastDone = -1;
    for (let calls = 1; ; calls++) {
      if (calls > MAX_CALLS || stalled >= MAX_STALLED) {
        throw new StopError("Stopped retrying automatically; press Resume to continue later.", { paused: true });
      }
      let r;
      try {
        r = await api("/api/ingest", { method: "POST", body: { document_id: id } });
      } catch (err) {
        if (err.status === 401 || err.status === 404 || err.status === 400 || err.status === 422) throw err;
        stalled++;                                               // network error, function timeout…
        onProgress({ note: err.message, retry_after: Math.min(30, 2 * stalled) });
        await sleep(Math.min(30000, 2000 * stalled));
        continue;
      }
      onProgress(r);
      if (r.status === "ready") return r;
      if (r.status === "failed") throw new StopError(r.error || "Ingest failed");
      if (r.paused) throw new StopError(r.note || "Paused: Gemini quota used up, try later", { paused: true });
      stalled = r.done > lastDone ? 0 : stalled + 1;
      lastDone = Math.max(lastDone, r.done || 0);
      await sleep(r.retry_after ? r.retry_after * 1000 : 300);
      if (Date.now() - (state.lastDocs || 0) > 4000) refreshDocs();
    }
  })();
  state.running.set(id, p);
  try { return await p; } finally { state.running.delete(id); }
}

/* ---------- documents ---------- */
async function refreshDocs() {
  state.lastDocs = Date.now();
  try {
    const data = await api("/api/documents");
    state.docs = data.documents || [];
    renderDocs();
    $("docsUpdated").textContent = "updated " + new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  } catch (err) { if (err.status !== 401) $("docsUpdated").textContent = err.message; }
}

function renderDocs() {
  const rows = $("docRows");
  rows.replaceChildren();
  $("docCount").textContent = state.docs.length ? `· ${state.docs.length}` : "";
  if (!state.docs.length) {
    rows.append(el("tr", {}, el("td", { colspan: 6, class: "empty", text: "No documents yet. Upload a PDF, DOCX or TXT above." })));
  }
  for (const d of state.docs) {
    const uploading = state.uploading.has(d.id);
    const busy = uploading || state.running.has(d.id);
    const label = uploading ? "uploading" : (busy && d.status !== "ready" ? "processing" : d.status);
    const statusCell = el("td", {}, el("span", { class: "pill " + label, text: label }));
    if (label === "processing" || (label === "uploaded" && d.chunk_total)) {
      statusCell.append(el("div", { class: "bar minibar" }, el("i", { style: `width:${Math.round((d.progress || 0) * 100)}%` })));
    }
    if (d.error && d.status !== "ready") statusCell.append(el("div", { class: d.status === "failed" ? "errtext" : "errtext warn", text: d.error }));
    const actions = el("td", { class: "actions" });
    if (d.status !== "ready" && !busy) {
      actions.append(el("button", { class: "btn small", type: "button", title: "Continue processing",
        text: d.status === "failed" ? "Retry" : "Resume", onclick: () => resume(d) }), " ");
    }
    actions.append(el("button", { class: "btn small danger", type: "button", text: "Delete", onclick: () => removeDoc(d) }));
    const name = d.status === "ready"
      ? el("a", { href: fileUrl(d.id), target: "_blank", rel: "noopener", title: d.filename, text: d.filename })
      : el("span", { class: "n", title: d.filename, text: d.filename });
    rows.append(el("tr", {},
      el("td", { class: "name" }, name, el("small", { text: `${fmtBytes(d.bytes)} · ${ago(d.uploaded_at)}` })),
      el("td", { class: "hide-sm", text: d.subject || "—" }),
      el("td", { class: "num hide-sm", text: d.pages ?? "—" }),
      el("td", { class: "num", text: d.chunk_total ? (d.status === "ready" ? d.chunk_count : `${d.chunk_count}/${d.chunk_total}`) : (d.chunk_count || "—") }),
      statusCell, actions));
  }
  // document filter for the search box
  const sel = $("docFilter"), keep = sel.value;
  sel.replaceChildren(el("option", { value: "", text: "All documents" }),
    ...state.docs.filter((d) => d.status === "ready").map((d) => el("option", { value: String(d.id), text: d.filename })));
  sel.value = [...sel.options].some((o) => o.value === keep) ? keep : "";
}

async function resume(d) {
  const item = makeItem(d.filename, fmtBytes(d.bytes));
  item.set("Resuming…", d.progress || 0, !d.chunk_total);
  renderDocs();
  try {
    const r = await ingestLoop(d.id, progressTo(item));
    item.done(`Ready · ${r.total} passages`);
  } catch (err) { showStop(item, err); }
  refreshDocs();
}

async function removeDoc(d) {
  if (!confirm(`Delete "${d.filename}"? Friday will no longer find it.`)) return;
  try {
    const r = await api(`/api/documents/${d.id}`, { method: "DELETE" });
    toast(r.storage_deleted ? "Deleted." : "Deleted (the stored file could not be removed).");
  } catch (err) { if (err.status !== 401) toast(err.message); }
  refreshDocs();
}

/* ---------- search ---------- */
function passageView(p) {
  const text = el("div", { class: "ptext", text: p.content });
  const more = el("button", { class: "more", type: "button", text: "Show all", onclick: () => {
    text.classList.toggle("open"); more.textContent = text.classList.contains("open") ? "Show less" : "Show all"; } });
  return el("li", { class: "passage" },
    el("div", { class: "phead" },
      el("span", { class: "n", text: `[${p.n}]` }),
      el("a", { class: "src", href: fileUrl(p.document_id, p.page), target: "_blank", rel: "noopener",
        text: p.filename + (p.page ? ` · p. ${p.page}` : "") }),
      p.subject ? el("span", { class: "tag", text: p.subject }) : null,
      p.keyword_hit ? el("span", { class: "tag", text: "keyword match", title: "Also matched by words" }) : null,
      el("span", { class: "sim", title: "Cosine similarity to the question", text: `${(p.similarity * 100).toFixed(1)}% similar` })),
    text, p.content.length > 420 ? more : null);
}

$("searchForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const query = $("query").value.trim();
  if (!query) return;
  const out = $("searchOut");
  $("searchBtn").disabled = true;
  out.replaceChildren(el("p", { class: "muted", style: "margin:12px 0 0", text: "Searching…" }));
  const t0 = performance.now();
  try {
    const r = await api("/api/search", { method: "POST", body: { query, document_id: $("docFilter").value ? Number($("docFilter").value) : null } });
    const rtt = performance.now() - t0;
    const list = el("ul", { class: "passages" }, r.passages.map(passageView));
    out.replaceChildren(
      r.passages.length ? list : el("p", { class: "muted", style: "margin:12px 0 0", text: "Nothing relevant found in the ready documents." }),
      el("div", { class: "timing", text:
        `Embedding ${r.timings.embed_ms.toFixed(0)} ms · database search ${r.timings.search_ms.toFixed(0)} ms · round trip from this browser ${rtt.toFixed(0)} ms` }));
  } catch (err) {
    if (err.status !== 401) out.replaceChildren(el("p", { class: "err", style: "margin:12px 0 0", text: err.message }));
  } finally { $("searchBtn").disabled = false; }
});

/* ---------- answers ---------- */
function answerText(text, cites) {
  // "[n]" markers become superscript links to the cited page
  const byN = new Map((cites || []).map((c) => [String(c.n), c]));
  const node = el("div", { class: "atext" });
  const parts = String(text || "").split(/(\[\d+\])/g);
  for (const part of parts) {
    const m = part.match(/^\[(\d+)\]$/);
    if (m && byN.has(m[1])) {
      const c = byN.get(m[1]);
      node.append(el("sup", {}, el("a", { href: fileUrl(c.document_id, c.page), target: "_blank", rel: "noopener",
        title: `${c.filename}${c.page ? ", page " + c.page : ""}`, text: `[${m[1]}]` })));
    } else if (part) node.append(part);
  }
  return node;
}

function answerView(a) {
  const cites = Array.isArray(a.citations) ? a.citations : [];
  const lat = a.latency && typeof a.latency === "object" ? a.latency : null;
  const meta = [ago(a.created_at), a.mode ? `via ${a.mode}` : null,
    lat && lat.first_audio_ms != null ? `first audio ${Math.round(lat.first_audio_ms)} ms` : null,
    lat && lat.search_ms != null ? `search ${Math.round(lat.search_ms)} ms` : null].filter(Boolean).join(" · ");
  return el("li", { class: "answer" },
    el("div", { class: "question", text: a.question }),
    el("div", { class: "ameta", text: meta }),
    answerText(a.answer, cites),
    cites.length ? el("ol", { class: "cites" }, cites.map((c) => el("li", {},
      el("span", { class: "cn", text: `[${c.n}]` }),
      el("div", {},
        el("a", { href: fileUrl(c.document_id, c.page), target: "_blank", rel: "noopener",
          text: (c.filename || "document") + (c.page ? ` · page ${c.page}` : "") }),
        c.excerpt ? el("q", { text: c.excerpt }) : null)))) : null);
}

async function refreshAnswers() {
  try {
    const data = await api("/api/answers?limit=20");
    const list = $("answers");
    const rows = data.answers || [];
    list.replaceChildren(...(rows.length ? rows.map(answerView)
      : [el("li", { class: "empty", text: "No document answers yet. Ask Friday about your notes." })]));
    $("ansUpdated").textContent = "updated " + new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  } catch (err) { if (err.status !== 401) $("ansUpdated").textContent = err.message; }
}

/* ---------- polling while visible ---------- */
function startPolling() {
  stopPolling();
  state.timers.answers = setInterval(() => { if (document.visibilityState === "visible") refreshAnswers(); }, 15000);
  state.timers.docs = setInterval(() => {
    if (document.visibilityState !== "visible") return;
    // another tab or device may be ingesting: keep the table fresh while anything is unfinished
    if (state.docs.some((d) => d.status === "processing" || d.status === "uploaded")) refreshDocs();
  }, 10000);
}
function stopPolling() { clearInterval(state.timers.answers); clearInterval(state.timers.docs); }
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible" && !$("app").classList.contains("hidden")) refreshAnswers();
});

/* ---------- start ---------- */
(async function init() {
  try {
    const s = await api("/api/session");
    if (s.max_upload_mb) { state.maxMb = s.max_upload_mb; $("dropHint").textContent = `PDF, DOCX or TXT · up to ${s.max_upload_mb} MB each`; }
    s.authenticated ? showApp() : showLogin();
  } catch (err) {
    showLogin();
    $("loginErr").textContent = err.status === 500 ? err.message : "";
  }
})();
