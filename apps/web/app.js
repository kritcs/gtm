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
const EYEBROW = { reflection: "From the interview", hit: "Shared customer trait", discovery: "New segment",
  needed: "Public post", blindspot: "Low data", next_step: "First account" };

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
  $("#next-card").textContent = S.cardIdx === S.cards.length - 1 ? "Accounts" : "Next";
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
$("#yes").onclick = () => sendFeedback("Yes");
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

// ---------------------------------------------------------------- model & proof
const SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]; // categorical slots 1-3, fixed order
const BADGE_NAME = { synthetic: "Synthetic data", simulated: "Simulated", measured: "Measured" };
const badge = (b, caption) => `<span class="badge ${b}">${BADGE_NAME[b]}</span>` +
  (caption ? `<div class="badge-caption ${b}">${esc(caption)}</div>` : "");

// Line chart: <= 3 series, legend + direct end labels, crosshair tooltip. One y-axis only.
function lineChart({ title, series, xLabel, yLabel, yFmt = (v) => v.toFixed(2), ref }) {
  const W = 440, H = 200, L = 44, R = 92, T = 10, B = 30;
  const xs = series.flatMap((s) => s.points.map((p) => p[0]));
  const ys = series.flatMap((s) => s.points.map((p) => p[1])).concat(ref ? [ref.y] : []);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  let y0 = Math.min(0, ...ys), y1 = Math.max(...ys) * 1.08;
  const X = (v) => L + ((v - x0) / (x1 - x0 || 1)) * (W - L - R);
  const Y = (v) => T + (1 - (v - y0) / (y1 - y0 || 1)) * (H - T - B);
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => y0 + f * (y1 - y0));
  let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(title)}">`;
  for (const t of ticks) svg += `<line x1="${L}" x2="${W - R}" y1="${Y(t)}" y2="${Y(t)}" stroke="#ece8e1"/>` +
    `<text x="${L - 6}" y="${Y(t) + 4}" font-size="10" text-anchor="end" fill="#6f6a62">${yFmt(t)}</text>`;
  svg += `<text x="${(L + W - R) / 2}" y="${H - 4}" font-size="10" text-anchor="middle" fill="#6f6a62">${esc(xLabel)}</text>`;
  svg += `<text x="${L}" y="${H - 16}" font-size="10" fill="#6f6a62">${x0}</text><text x="${W - R}" y="${H - 16}" font-size="10" text-anchor="end" fill="#6f6a62">${x1}</text>`;
  if (ref) svg += `<line x1="${L}" x2="${W - R}" y1="${Y(ref.y)}" y2="${Y(ref.y)}" stroke="#6f6a62" stroke-dasharray="4 4"/>` +
    `<text x="${W - R + 6}" y="${Y(ref.y) + 4}" font-size="10" fill="#52514e">${esc(ref.label)}</text>`;
  series.forEach((s, i) => {
    const d = s.points.map((p, j) => `${j ? "L" : "M"}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join("");
    svg += `<path d="${d}" fill="none" stroke="${SERIES[i]}" stroke-width="2" stroke-linejoin="round"/>`;
    const last = s.points[s.points.length - 1];
    if (series.length > 1 || !ref) svg += `<text x="${X(last[0]) + 6}" y="${Y(last[1]) + 4 + i * 0}" font-size="10" fill="#0b0b0b">${esc(s.short || s.name)}</text>`;
  });
  svg += `<line class="xh" x1="0" x2="0" y1="${T}" y2="${H - B}" stroke="#b9b3a9" style="display:none"/>`;
  svg += `<rect x="${L}" y="${T}" width="${W - L - R}" height="${H - T - B}" fill="transparent"/></svg>`;
  const legend = series.length > 1 ? `<div class="legend">${series.map((s, i) => `<span><i style="background:${SERIES[i]}"></i>${esc(s.name)}</span>`).join("")}</div>` : "";
  const id = "c" + Math.random().toString(36).slice(2, 8);
  setTimeout(() => {
    const root = document.getElementById(id); if (!root) return;
    const sv = root.querySelector("svg"), tip = root.querySelector(".tip"), xh = sv.querySelector(".xh");
    sv.addEventListener("mousemove", (e) => {
      const r = sv.getBoundingClientRect(), vx = ((e.clientX - r.left) / r.width) * W;
      const xv = x0 + ((vx - L) / (W - L - R)) * (x1 - x0);
      const rows = series.map((s, i) => {
        const p = s.points.reduce((a, b) => Math.abs(b[0] - xv) < Math.abs(a[0] - xv) ? b : a);
        return { s, i, p };
      });
      const px = X(rows[0].p[0]);
      xh.setAttribute("x1", px); xh.setAttribute("x2", px); xh.style.display = "";
      tip.innerHTML = `<b>${esc(xLabel)} ${rows[0].p[0]}</b><br>` + rows.map((r) => `<i style="display:inline-block;width:8px;height:8px;background:${SERIES[r.i]};margin-right:5px"></i>${esc(r.s.name)}: ${yFmt(r.p[1])}`).join("<br>");
      tip.style.display = "block";
      tip.style.left = Math.min(r.width - 170, (px / W) * r.width + 10) + "px"; tip.style.top = "30px";
    });
    sv.addEventListener("mouseleave", () => { tip.style.display = "none"; xh.style.display = "none"; });
  });
  return `<div class="viz" id="${id}"><h4>${esc(title)}</h4>${legend}${svg}<div class="tip"></div><p class="fine" style="margin:4px 0 0">${esc(yLabel)}</p></div>`;
}

const recallTable = (methods, k) => `<table><tr><th>Method</th><th>Recall@${k}</th><th>95% CI</th></tr>` +
  Object.entries(methods).map(([m, v]) => `<tr><td>${esc(m)}</td><td>${v.recall.toFixed(3)}</td><td>${v.ci95[0].toFixed(3)} to ${v.ci95[1].toFixed(3)}</td></tr>`).join("") + `</table>`;

async function loadProof() {
  $("#proof-area").innerHTML = `<p class="fine">Loading…</p>`;
  let d;
  try { d = await api("/api/model-proof"); } catch (e) { $("#proof-area").innerHTML = `<p class="flag">Run <code>make sim</code> first.</p>`; return; }
  const mm = d.market_model, pr = d.prior_residual, ts = d.thompson, bt = d.backtest, ab = d.ablation, vo = d.voice_dpo;
  const pol = ts.policies;
  const rounds = (arr) => arr.map((v, i) => [i + 1, v]);
  $("#proof-area").innerHTML = `<div class="mp">
  <div class="say">This panel is the roadmap. Training ran on synthetic data; the voice fine-tune is simulated; there are no measured numbers yet.</div>
  <p class="lede">${badge("synthetic")} real training code ran on the seeded synthetic market (<code>make sim</code>, seed ${d.seed}, ${d.built_seconds}s on a laptop CPU). &nbsp; ${badge("simulated")} a mock-up of a step that was not run. &nbsp; ${badge("measured")} real public data. Badges are never mixed in one chart or table.</p>

  <section>${badge("synthetic", d.planted.caption)}
    <h3>Recovered the planted patterns</h3>
    <p class="lede">The generator plants known structure. These checks ask whether the pipeline finds it without being told.</p>
    <table><tr><th>Planted in the generator</th><th>Recovered</th><th>Evidence</th></tr>
    ${d.planted.items.map((i) => `<tr><td>${esc(i.planted)}</td><td class="ok">${i.recovered ? "✓ yes" : "✗ no"}</td><td>${esc(i.evidence)}</td></tr>`).join("")}</table>
  </section>

  <section>${badge("synthetic", mm.caption)}
    <h3>Market model training</h3>
    <p class="lede">${esc(mm.task)}. ${mm.train_edges.toLocaleString()} adoption edges, ${mm.companies} companies, ${mm.tools} tools, ${mm.eval_companies} companies in validation.</p>
    <div class="grid2">
      ${lineChart({ title: `LightGCN BPR training loss (${mm.lightgcn.seconds}s)`, xLabel: "epoch", yLabel: "BPR loss, mean of the last 5 epochs",
        series: [{ name: "LightGCN", points: mm.lightgcn.curve.map((r) => [r.epoch, r.loss]) }] })}
      ${lineChart({ title: "LightGCN validation Recall@10", xLabel: "epoch", yLabel: "dashed: popularity baseline",
        series: [{ name: "LightGCN", points: mm.lightgcn.curve.map((r) => [r.epoch, r.val_recall]) }],
        ref: { y: mm.popularity_recall, label: "popularity" } })}
    </div>
    <p class="fine">ALS (implicit, dim ${mm.als.dim}): regularization chosen on this validation set (${Object.entries(mm.als.reg_sweep).map(([k, v]) => `reg ${k}: ${v.toFixed(3)}`).join(" · ")}); final Recall@10 ${mm.als.curve.at(-1).val_recall.toFixed(3)} in ${mm.als.seconds}s. LightGCN ${mm.lightgcn.curve.at(-1).val_recall.toFixed(3)}, popularity ${mm.popularity_recall.toFixed(3)}: on this synthetic market tool adoption is mostly popularity, and the graph model only matches it.</p>
  </section>

  <section>${badge("synthetic", pr.caption)}
    <h3>Founder prior + learned correction</h3>
    <p class="lede"><code>${esc(pr.formula)}</code>. Train: features as of ${esc(pr.split.train_features_asof)} (${pr.train.n.toLocaleString()} rows, ${pr.train.positives} positives), labels ${esc(pr.split.train_labels)}. Test: features as of ${esc(pr.split.test_features_asof)}, ${esc(pr.split.test_labels)} (${pr.test.positives} of ${pr.test.n}). ${esc(pr.correction_model)}.</p>
    <div class="grid2"><div>${recallTable(pr.methods, pr.test.k)}</div>
      <div><table><tr><th>Largest correction weights</th><th>w</th></tr>${Object.entries(pr.weights).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 6).map(([f, v]) => `<tr><td>${esc(f)}</td><td>${v >= 0 ? "+" : ""}${v.toFixed(2)}</td></tr>`).join("")}</table></div></div>
    <p class="fine">The correction learns the planted RevOps trigger and moves weight toward logistics and away from saas-billing, against the founder's stated beliefs. Because those beliefs are miscalibrated, the correction alone still beats prior + correction here.</p>
  </section>

  <section>${badge("synthetic", ts.caption)}
    <h3>Beta priors + Thompson sampling</h3>
    <p class="lede">${esc(ts.setup)} Best planted segment: ${esc(ts.best_segment)}.</p>
    <div class="grid2">
      ${lineChart({ title: "Cumulative conversions", xLabel: "round", yLabel: "mean over runs", yFmt: (v) => v.toFixed(0),
        series: [{ name: pol.ts_calibrated.label, short: "rescaled priors", points: rounds(pol.ts_calibrated.cum_conversions) },
                 { name: pol.ts_flat.label, short: "flat priors", points: rounds(pol.ts_flat.cum_conversions) },
                 { name: pol.ts_founder.label, short: "as stated", points: rounds(pol.ts_founder.cum_conversions) }] })}
      <div><table><tr><th>Segment</th><th>Planted rate</th><th>Founder said</th><th>Rescaled</th></tr>
        ${Object.keys(ts.truth).map((s) => `<tr><td>${esc(s)}</td><td>${ts.truth[s].toFixed(3)}</td><td>${ts.founder_prior_mean[s].toFixed(2)}</td><td>${ts.calibrated_prior_mean[s].toFixed(3)}</td></tr>`).join("")}</table></div>
    </div>
    <table><tr><th>Policy, 40 rounds</th><th>Conversions</th><th>Regret vs always-best</th></tr>
      ${Object.values(pol).map((p) => `<tr><td>${esc(p.label)}</td><td>${p.cum_conversions.at(-1).toFixed(1)}</td><td>${p.cum_regret.at(-1).toFixed(1)}</td></tr>`).join("")}</table>
    <p class="fine">Stated founder beliefs are too high and miss logistics, so used as-is they cost conversions; rescaled to the base rate (same ranking and confidence) they finish best. Never updating the priors is worst after random.</p>
  </section>

  <section>${badge("synthetic", bt.caption)}
    <h3>Backtest, ${esc(bt.asof)} → after, and the ablation</h3>
    <p class="lede">${bt.candidates} candidates, ${bt.positives} new adopters. Recall@${bt.k}, 95% bootstrap CI. "Description match" stands in for an LLM ranking; it is not an LLM.</p>
    <div class="grid2"><div>${recallTable(bt.methods, bt.k)}</div><div><h4 style="margin:12px 0 0">Ablation</h4>${recallTable(ab.methods, ab.k)}</div></div>
    <p class="fine">${bt.combination_beats_both_ci ? "Market + founder beats both parts outside the intervals." : bt.combination_beats_both_point ? "Market + founder beats both parts on the point estimate; the intervals overlap." : "The combination does not beat both parts."} Leakage checks: ${bt.leakage_checks.map(esc).join(" · ")}.</p>
  </section>

  <section>${badge("synthetic", d.maturity.caption)}
    <h3>Maturity stages</h3>
    <div class="stage-cols">${d.maturity.stages.map((s) => `<div class="stage"><span class="badge ${s.badge}">${BADGE_NAME[s.badge]}</span><b>${esc(s.stage)}</b>${esc(s.method)}<p class="fine">${esc(s.evidence)}</p></div>`).join("")}</div>
  </section>

  <section>${badge("simulated", vo.caption)}
    <h3>Voice model: LoRA + DPO on edit pairs</h3>
    <p class="lede">${Object.entries(vo.config).map(([k, v]) => `<b>${esc(k)}</b>: ${esc(v)}`).join(" · ")}</p>
    <div class="grid2">
      ${lineChart({ title: "DPO loss (simulated)", xLabel: "step", yLabel: "simulated values from a seeded generator",
        series: [{ name: "DPO loss", points: vo.dpo_curve.map((r) => [r.step, r.loss]) }] })}
      ${lineChart({ title: "Reward margin, chosen − rejected (simulated)", xLabel: "step", yLabel: "simulated values from a seeded generator",
        series: [{ name: "reward margin", points: vo.dpo_curve.map((r) => [r.step, r.reward_margin]) }] })}
    </div>
    <p class="fine">Simulated eval: win rate vs base ${vo.eval.win_rate_vs_base.toFixed(2)} (95% CI ${vo.eval.ci95[0].toFixed(2)} to ${vo.eval.ci95[1].toFixed(2)}, n = ${vo.eval.n_judged}); AI-sounding flags per email ${vo.eval.ai_flags_per_email.base} → ${vo.eval.ai_flags_per_email.tuned}. Not run.</p>
  </section>

  <section>${badge("measured", d.badges.measured)}
    <h3>Measured on real public data</h3>
    <div class="empty">${esc(d.measured.note)}</div>
  </section>
  </div>`;
}
