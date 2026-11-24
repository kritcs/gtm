// Intuition in a bottle: the reading, then the engine.
const $ = (s) => document.querySelector(s);
const api = async (path, opts = {}) => {
  const r = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts,
    body: opts.body ? JSON.stringify(opts.body) : undefined });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
};
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

const S = { sid: null, world: null, script: null, qIndex: 0, cards: [], cardIdx: 0, drafts: {}, events: [] };
const EYEBROW = { reflection: "Understood", hit: "What your customers share", discovery: "The one you didn't see",
  needed: "Needed", blindspot: "Where I can't see clearly", next_step: "The next step" };

function show(id) {
  document.querySelectorAll(".stage").forEach((s) => s.classList.toggle("active", s.id === id));
  window.scrollTo(0, 0);
}

// ---------------------------------------------------------------- enter
(async function boot() {
  for (let i = 0; i < 60; i++) {
    try { const h = await api("/api/health"); if (h.ok) break; } catch (e) {}
    await new Promise((r) => setTimeout(r, 500));
  }
  S.world = await api("/api/world");
  S.script = await api("/api/demo/script");
  $("#world-line").textContent =
    `${S.world.companies} companies · ${S.world.posts.toLocaleString()} posts · through ${S.world.through} · simulated market`;
  $("#sim-note").textContent = `Simulated market data (seed ${S.world.seed}) · Band: ${S.world.integrations.band.split(" ")[0]} · Flexprice: ${S.world.integrations.flexprice.split(" ")[0]}`;
})();

$("#begin").onclick = async () => {
  const r = await api("/api/sessions", { method: "POST" });
  S.sid = r.id;
  listen();
  ask(r.question);
  show("talk");
};

// ---------------------------------------------------------------- talk
function ask(q) {
  S.qIndex = q.index;
  $("#q-count").textContent = `Question ${q.index + 1} of ${q.total}`;
  $("#question").textContent = q.text;
  $("#answer").value = "";
  $("#answer").focus();
}

$("#demo-answer").onclick = () => { $("#answer").value = S.script.answers[S.qIndex] || ""; };

$("#send").onclick = async () => {
  const text = $("#answer").value.trim();
  if (!text) return;
  $("#send").disabled = true;
  try {
    const r = await api(`/api/sessions/${S.sid}/answer`, { method: "POST", body: { text } });
    settlePhrases(r.learned.phrases);
    renderBottle(r.bottle);
    if (r.next) ask(r.next); else runReading();
  } finally { $("#send").disabled = false; }
};
$("#answer").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) $("#send").click(); });

function settlePhrases(list) {
  list.forEach((p, i) => {
    const el = document.createElement("span");
    el.className = "phrase"; el.textContent = `“${p}”`;
    $("#phrases").appendChild(el);
    setTimeout(() => el.classList.add("in"), 300 + i * 700);
  });
}

function renderBottle(b) {
  const kinds = { pain: "Pain", trigger: "Trigger", buyer: "Buyer", segment: "Segment" };
  let html = "";
  if (b.customers.length) html += `<dt>Customers</dt><dd>${b.customers.map(esc).join(", ")}</dd>`;
  if (b.lost.length) html += `<dt>Lost deals</dt><dd>${b.lost.map(esc).join(", ")}</dd>`;
  for (const k of Object.keys(kinds)) {
    const rs = b.rules.filter((r) => r.kind === k);
    if (rs.length) html += `<dt>${kinds[k]}</dt>` + rs.map((r) => `<dd>${esc(r.text)}</dd>`).join("");
  }
  $("#bottle-list").innerHTML = html;
  $("#bottle-fill").style.height = `${Math.min(100, (b.answers.length / 4) * 100)}%`;
}

// speech: Web Speech API, with the textarea always available as a fallback
const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
let rec = null;
if (!SR) $("#mic").style.display = "none";
$("#mic").onclick = () => {
  if (rec) { rec.stop(); return; }
  rec = new SR(); rec.continuous = true; rec.interimResults = true; rec.lang = "en-US";
  const base = $("#answer").value;
  rec.onresult = (e) => {
    let t = ""; for (const r of e.results) t += r[0].transcript;
    $("#answer").value = (base + " " + t).trim();
  };
  rec.onend = () => { rec = null; $("#mic").classList.remove("listening"); $("#mic").textContent = "● Speak"; };
  rec.start(); $("#mic").classList.add("listening"); $("#mic").textContent = "■ Stop";
};

// ---------------------------------------------------------------- pause + feed
function listen() {
  const es = new EventSource(`/api/sessions/${S.sid}/events`);
  es.onmessage = (m) => {
    const ev = JSON.parse(m.data);
    S.events.push(ev);
    const k = document.createElement("div");
    k.innerHTML = `<b>${esc(ev.agent)}</b> · ${esc(ev.content)}`;
    $("#kitchen").appendChild(k);
    while ($("#kitchen").children.length > 7) $("#kitchen").firstChild.remove();
    renderFeed();
    if (ev.agent === "Webhook") onReply();
  };
}

async function runReading() {
  show("pause");
  const names = S.world.sample_names;
  let i = 0;
  const t = setInterval(() => { $("#shuffle").textContent = names[i++ % names.length]; }, 90);
  const started = Date.now();
  const r = await api(`/api/sessions/${S.sid}/reading`, { method: "POST" });
  const wait = Math.max(0, 2600 - (Date.now() - started)); // let the shuffle breathe
  setTimeout(() => { clearInterval(t); S.cards = r.cards; S.cardIdx = 0; renderCard(); show("reading"); }, wait);
}

// ---------------------------------------------------------------- reading
function cardFront(c) {
  const blur = (1 - c.clarity) * 3.2, op = 0.35 + 0.65 * c.clarity;
  let html = `<div class="front-text" style="filter: blur(${blur.toFixed(2)}px); opacity:${op.toFixed(2)}">${esc(c.front_text)}</div>`;
  if (c.companies?.length) html += `<div class="companies">${c.companies.slice(0, 10).map((x) => `<span class="chip">${esc(x.name)}</span>`).join("")}</div>`;
  return html;
}

function cardBack(c) {
  let html = `<p>${esc(c.back_text)}</p>`;
  for (const r of c.receipts) {
    html += `<div class="receipt"><q>${esc(r.quote)}</q><div class="meta">${r.company ? esc(r.company) + " · " : ""}${esc(r.date)} · ` +
      `<a href="${esc(r.url)}" target="_blank">post ${r.post_id}</a>${r.verified ? " · verified" : " · <span class=flag>unverified</span>"}</div></div>`;
  }
  const st = { ...c.stats }; delete st.rules;
  const fmt = (v) => typeof v === "number" ? (Math.abs(v) < 0.001 && v !== 0 ? v.toExponential(1) : +v.toPrecision(3))
    : Array.isArray(v) ? v.map(fmt).join(", ") : typeof v === "object" && v ? Object.entries(v).map(([a, b]) => `${a} ${fmt(b)}`).join(", ") : v;
  if (Object.keys(st).length) html += `<div class="stats">${Object.entries(st).map(([k, v]) => `${esc(k.replaceAll("_", " "))}: ${esc(fmt(v))}`).join("\n")}</div>`;
  return html;
}

function renderCard() {
  const c = S.cards[S.cardIdx];
  $("#card").classList.remove("flipped");
  $("#card-eyebrow").textContent = EYEBROW[c.kind] || c.kind;
  $("#card-front").innerHTML = cardFront(c);
  $("#card-back").innerHTML = cardBack(c);
  $("#card-diff").textContent = c.diff || "";
  $("#flip").style.display = c.receipts.length || Object.keys(c.stats).length ? "" : "none";
  for (const id of ["#yes", "#no", "#no-text"]) $(id).style.display = c.revisable ? "" : "none";
  $("#next-card").textContent = S.cardIdx === S.cards.length - 1 ? "Open the engine" : "Next";
  $("#dots").innerHTML = S.cards.map((_, i) => `<span class="${i === S.cardIdx ? "on" : ""}"></span>`).join("");
  $("#no-text").value = "";
}

$("#flip").onclick = () => $("#card").classList.toggle("flipped");
$("#next-card").onclick = () => {
  if (S.cardIdx < S.cards.length - 1) { S.cardIdx++; renderCard(); } else openDash();
};
async function sendFeedback(text) {
  const c = S.cards[S.cardIdx];
  const nc = await api(`/api/sessions/${S.sid}/cards/${c.id}/feedback`, { method: "POST", body: { text } });
  S.cards[S.cardIdx] = nc; renderCard();
}
$("#yes").onclick = () => sendFeedback("Yes, exactly");
$("#no").onclick = () => sendFeedback($("#no-text").value || "No");
$("#no-text").addEventListener("keydown", (e) => { if (e.key === "Enter") $("#no").click(); });
$("#no-text").addEventListener("focus", () => { if (!$("#no-text").value && S.cards[S.cardIdx].id === "discovery") $("#no-text").placeholder = S.script.correction; });

// ---------------------------------------------------------------- dashboard
$("#reread").onclick = () => show("reading");
document.querySelectorAll("#tabs button").forEach((b) => b.onclick = () => tab(b.dataset.tab));
function tab(name) {
  document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("on", b.dataset.tab === name));
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("on", t.id === name));
  ({ accounts: loadAccounts, pipeline: loadPipeline, "bottle-tab": loadBottle, proof: loadProof, feed: renderFeed })[name]?.();
}
function openDash() { show("dash"); tab("accounts"); }

async function loadAccounts() {
  const r = await api(`/api/sessions/${S.sid}/accounts`);
  meter(r.credits);
  $("#account-list").innerHTML = r.accounts.map((a) => `
    <div class="account">
      <div class="score"><b>${a.score.toFixed(2)}</b><br>market ${a.market.toFixed(2)} · you ${a.founder.toFixed(2)} · λ ${a.lambda.toFixed(2)}</div>
      <h3>${esc(a.name)}</h3>
      <div class="sub">${esc(a.industry)} · ${esc(a.size)} people · ${esc(a.stage.replace("_", " "))}${a.enriched ? ` · ${esc(a.contact.first)}, ${esc(a.contact.role)} &lt;${esc(a.contact.email)}&gt;` : ""}</div>
      <div class="chips">${a.chips.map((ch) => `<a class="chip ${ch.kind}" href="${esc(ch.receipt.url)}" target="_blank" title="${esc(ch.receipt.date)}">${esc(ch.label)}</a>`).join("")}</div>
      <button class="ghost" onclick="draftFor(${a.id})">Draft email</button>
    </div>`).join("");
}

function meter(credits, start = 30) {
  $("#meter-fill").style.width = `${Math.max(0, (credits / start) * 100)}%`;
  $("#meter-text").textContent = `${credits.toFixed(0)} of ${start} credits left · enrichment runs in order of score ÷ cost`;
}

$("#enrich").onclick = async () => {
  $("#enrich").disabled = true;
  const r = await api(`/api/sessions/${S.sid}/enrich`, { method: "POST" });
  meter(r.balance, r.start);
  await loadAccounts();
  if (r.skipped.length) $("#meter-text").textContent += ` · out of budget for ${r.skipped.join(", ")}`;
};

window.draftFor = async (id) => {
  const d = await api(`/api/sessions/${S.sid}/drafts`, { method: "POST", body: { account_id: id } });
  S.drafts[d.id] = d;
  tab("outreach");
  renderDraft(d);
};

function renderDraft(d) {
  $("#draft-area").innerHTML = `
    <p class="weight">${esc(d.why)}</p>
    <p class="fine">To ${esc(d.to_name)} &lt;${esc(d.to)}&gt; · ${esc(d.subject)}</p>
    <textarea id="draft-body">${esc(d.body)}</textarea>
    <div id="flags">${d.flags.length ? d.flags.map((f) => `<div class="flag">⚑ ${esc(f)}</div>`).join("") : `<p class="fine">No AI-sounding phrases found.</p>`}</div>
    <div class="row"><span class="grow"></span><button class="primary" id="approve">Approve and send</button></div>
    <p class="fine" id="sent-line"></p>`;
  $("#approve").onclick = async () => {
    const r = await api(`/api/sessions/${S.sid}/drafts/${d.id}/approve`, { method: "POST", body: { body: $("#draft-body").value } });
    S.drafts[d.id] = r;
    $("#approve").disabled = true;
    const link = `${location.origin}/reply/${d.id}`;
    $("#sent-line").innerHTML = `Sent via ${esc(r.delivery.channel)}. Open <a href="${link}" target="_blank">${link}</a> on a phone to reply as ${esc(d.to_name)}.`;
  };
}

async function loadPipeline(highlight) {
  const r = await api(`/api/sessions/${S.sid}/pipeline`);
  $("#kanban").innerHTML = Object.entries(r.columns).map(([col, items]) => `
    <div class="col"><h4>${col}</h4>${items.map((it) => `<div class="kcard ${highlight === it.account ? "just" : ""}">${esc(it.account)}</div>`).join("")}</div>`).join("");
}

async function onReply() {
  const last = S.events.filter((e) => e.agent === "Webhook").pop();
  const name = last?.content.split(" replied")[0];
  tab("pipeline");
  await loadPipeline(name);
}

async function loadBottle() {
  const b = await api(`/api/sessions/${S.sid}/bottle`);
  $("#bottle-inspect").innerHTML = `
    <h3>Rules and their trust, q ~ Beta(α, β)</h3>
    <table><tr><th>Kind</th><th>Rule</th><th>Trust</th><th>α</th><th>β</th></tr>
    ${b.rules.map((r) => `<tr><td>${esc(r.kind)}</td><td>${esc(r.text)}</td><td><span class="trust" style="width:${r.trust * 80}px"></span>${r.trust.toFixed(2)}</td><td>${r.alpha}</td><td>${r.beta}</td></tr>`).join("")}</table>
    <p class="fine">Experience set E: ${[...b.customers, ...b.lost].map(esc).join(", ")} · constraints: ${esc(b.constraints)}</p>
    <h3>Every feature the hit finder tested</h3>
    <table><tr><th>Feature</th><th>k/n</th><th>base rate</th><th>p</th><th>BH q</th><th>shrunk lift</th><th>said it?</th><th>shown</th></tr>
    ${b.hit_features.map((f) => `<tr><td>${esc(f.label)}</td><td>${f.k}/${f.n}</td><td>${(f.base_rate * 100).toFixed(1)}%</td><td>${f.p_chance.toExponential(1)}</td><td>${f.q_value.toFixed(3)}</td><td>${f.shrunk_lift.toFixed(1)}</td><td>${f.mentioned ? "yes" : ""}</td><td>${f.passes ? "✓" : ""}</td></tr>`).join("")}</table>`;
}

function renderFeed() {
  if (!S.world) return;
  $("#band-status").textContent = `Band: ${S.world.integrations.band}. Every agent step below is posted as a Band event.`;
  $("#feed-list").innerHTML = S.events.slice().reverse().map((e) => `
    <li><span class="agent">${esc(e.agent)}</span><span class="type ${e.type === "error" ? "error" : ""}">${esc(e.type)}</span><span>${esc(e.content)}</span><span class="band">${esc(e.band)}</span></li>`).join("");
}

async function loadProof() {
  $("#proof-area").innerHTML = `<p class="fine">Running the backtest…</p>`;
  const b = await api("/api/backtest");
  const verdict = b.combination_beats_both_ci ? "Market + founder beats both parts, outside the confidence intervals."
    : b.combination_beats_both_point ? "Market + founder beats both parts on the point estimate; the intervals still overlap."
    : "The combination does not beat both parts here. We don't claim it.";
  $("#proof-area").innerHTML = `
    <p class="weight">${esc(verdict)}</p>
    <p class="fine">Trained on data through ${esc(b.asof)}; predicts which of ${b.candidates} companies adopt Ledgerlight afterwards (${b.positives} did). Recall@${b.k}, 95% bootstrap CI. Simulated market.</p>
    <table><tr><th>Method</th><th>Recall@${b.k}</th><th>95% CI</th></tr>
    ${Object.entries(b.methods).map(([k, v]) => `<tr><td>${esc(k)}</td><td>${v.recall.toFixed(3)}</td><td>${v.ci95[0].toFixed(3)} to ${v.ci95[1].toFixed(3)}</td></tr>`).join("")}</table>
    <p class="fine">Leakage checks: ${b.leakage_checks.map(esc).join(" · ")}</p>`;
}
