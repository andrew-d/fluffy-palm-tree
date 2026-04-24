"use strict";

const ENTITY_TYPES = [
  "private_person",
  "private_email",
  "private_phone",
  "private_address",
  "private_date",
  "private_url",
];

const POLL_MS = 500;

function $(id) { return document.getElementById(id); }

function buildLegend() {
  const ul = $("legend");
  ul.innerHTML = "";
  for (const t of ENTITY_TYPES) {
    const li = document.createElement("li");
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = `var(--c-${t})`;
    const label = document.createElement("span");
    label.textContent = t;
    li.append(swatch, label);
    ul.appendChild(li);
  }
}

// Render the original text with <span class="ent"> wrappers for each entity.
// Uses character (rune) offsets; we operate on a code-point array so non-BMP
// characters are handled correctly.
function renderHighlighted(text, entities) {
  const out = $("output");
  out.innerHTML = "";
  const codePoints = Array.from(text);

  // Sort entities by start offset; drop overlaps defensively (keep the earlier).
  const ents = [...entities].sort((a, b) => a.start - b.start);
  const filtered = [];
  let prevEnd = -1;
  for (const e of ents) {
    if (e.start >= prevEnd) {
      filtered.push(e);
      prevEnd = e.end;
    }
  }

  let cursor = 0;
  for (const e of filtered) {
    if (e.start > cursor) {
      out.appendChild(document.createTextNode(codePoints.slice(cursor, e.start).join("")));
    }
    const span = document.createElement("span");
    span.className = "ent";
    span.dataset.type = e.entity_group;
    span.dataset.tooltip =
      `${e.entity_group}\nscore: ${e.score.toFixed(4)}\noffsets: [${e.start}, ${e.end})`;
    span.textContent = codePoints.slice(e.start, e.end).join("");
    out.appendChild(span);
    cursor = e.end;
  }
  if (cursor < codePoints.length) {
    out.appendChild(document.createTextNode(codePoints.slice(cursor).join("")));
  }
}

function renderThroughput(s) {
  const dl = $("throughput");
  dl.innerHTML = "";
  const rows = [
    ["duration",      `${s.duration_ms.toFixed(1)} ms`],
    ["chars",         `${s.chars}  (${fmt(s.chars_per_sec)} chars/sec)`],
    ["tokens",        `${s.tokens}  (${fmt(s.tokens_per_sec)} tokens/sec)`],
  ];
  for (const [k, v] of rows) {
    const dt = document.createElement("dt");
    dt.textContent = k;
    const dd = document.createElement("dd");
    dd.textContent = v;
    dl.append(dt, dd);
  }
}

function fmt(n) {
  if (!isFinite(n)) return "∞";
  if (n >= 100) return n.toFixed(0);
  if (n >= 10)  return n.toFixed(1);
  return n.toFixed(2);
}

function renderEntityList(entities) {
  const ul = $("entities");
  ul.innerHTML = "";
  if (entities.length === 0) {
    const li = document.createElement("li");
    li.innerHTML = `<span class="meta">(no entities detected)</span>`;
    ul.appendChild(li);
    return;
  }
  for (const e of entities) {
    const li = document.createElement("li");
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = `var(--c-${e.entity_group})`;
    const label = document.createElement("span");
    label.innerHTML =
      `<strong>${escapeHTML(e.word)}</strong> ` +
      `<span class="meta">${e.entity_group} · ` +
      `score=${e.score.toFixed(4)} · [${e.start},${e.end})</span>`;
    li.append(swatch, label);
    ul.appendChild(li);
  }
}

function escapeHTML(s) {
  return s.replace(/[&<>"']/g, c => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[c]);
}

function setStatus(msg, isError = false) {
  const el = $("status");
  el.textContent = msg;
  el.classList.toggle("error", isError);
}

async function submit(text) {
  const r = await fetch("/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!r.ok) throw new Error(`submit failed: ${r.status} ${await r.text()}`);
  return r.json();
}

async function fetchStatus(id) {
  const r = await fetch(`/jobs/${encodeURIComponent(id)}`);
  if (!r.ok) throw new Error(`status failed: ${r.status} ${await r.text()}`);
  return r.json();
}

function describeQueueState(s) {
  if (s.status === "active") return "classifying…";
  if (s.status === "queued") {
    const pos = s.position;
    return pos === 0
      ? "queued (next up)"
      : `queued (position ${pos + 1} of ${s.queue_length})`;
  }
  return s.status;
}

async function pollUntilDone(id) {
  for (;;) {
    const s = await fetchStatus(id);
    if (s.status === "done") return s;
    if (s.status === "error") throw new Error(s.error || "classification failed");
    setStatus(`job ${id}: ${describeQueueState(s)}`);
    await new Promise(r => setTimeout(r, POLL_MS));
  }
}

async function onRun() {
  const text = $("input").value;
  if (!text.trim()) {
    setStatus("enter some text first", true);
    return;
  }
  const btn = $("run");
  btn.disabled = true;
  $("result").hidden = true;
  setStatus("submitting…");
  try {
    const { id } = await submit(text);
    setStatus(`job ${id}: submitted`);
    const s = await pollUntilDone(id);
    renderHighlighted(s.text, s.entities);
    renderThroughput(s);
    renderEntityList(s.entities);
    $("result").hidden = false;
    setStatus(`job ${id}: done (${s.entities.length} entit${s.entities.length === 1 ? "y" : "ies"})`);
  } catch (err) {
    console.error(err);
    setStatus(String(err.message || err), true);
  } finally {
    btn.disabled = false;
  }
}

document.addEventListener("DOMContentLoaded", () => {
  buildLegend();
  $("run").addEventListener("click", onRun);
  $("input").addEventListener("keydown", e => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") onRun();
  });
});
