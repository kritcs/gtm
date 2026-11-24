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
const EYEBROW = { reflection: "01 — Heard", hit: "02 — Pattern", discovery: "03 — New segment",
  needed: "04 — Public post", blindspot: "05 — Low data", next_step: "06 — Start here" };

// ---------------------------------------------------------------- motion helpers
const RM = matchMedia("(prefers-reduced-motion: reduce)").matches;
const EASE = "cubic-bezier(.2,.8,.2,1)";
const sleep = (ms) => new Promise((r) => setTimeout(r, RM ? 0 : ms));
const anim = (el, kf, o = {}) => (RM || !el ? null : el.animate(kf, { duration: 600, easing: EASE, fill: "both", ...o }));
const MAP_MODE = { enter: "ambient", talk: "ambient", pause: "scan", reading: "reading", dash: "dash" };

// Reveal text word by word; bare numbers and percentages count up.
function reveal(el, text, { delay = 0, stagger = 45 } = {}) {
  const words = String(text).split(/(\s+)/);
  el.innerHTML = words.map((w) => /^\s+$/.test(w) ? w
    : /^\d+(\.\d+)?%?[.,:]?$/.test(w) ? `<span class="w"><span class="num" data-to="${esc(w)}">${esc(w)}</span></span>`
    : `<span class="w">${esc(w)}</span>`).join("");
  if (RM) return;
  el.querySelectorAll(".w").forEach((w, i) => anim(w, [
    { opacity: 0, transform: "translateY(.45em)", filter: "blur(6px)" },
    { opacity: 1, transform: "none", filter: "blur(0)" }], { duration: 560, delay: delay + i * stagger }));
  el.querySelectorAll(".num").forEach((n) => {
    const raw = n.dataset.to, m = raw.match(/^(\d+(?:\.\d+)?)(.*)$/), to = parseFloat(m[1]), dec = (m[1].split(".")[1] || "").length;
    const t0 = performance.now() + delay + 150;
    const step = (now) => {
      const f = Math.min(1, Math.max(0, (now - t0) / 900)), e = 1 - Math.pow(1 - f, 3);
      n.textContent = (to * e).toFixed(dec) + m[2];
      if (f < 1) requestAnimationFrame(step);
    };
    n.textContent = (0).toFixed(dec) + m[2];
    requestAnimationFrame(step);
  });
}

function stagger(nodes, kf, o = {}) {
  [...nodes].forEach((n, i) => anim(n, kf, { delay: (o.delay || 0) + i * (o.step || 70), duration: o.duration || 560 }));
}

let current = "enter";
async function show(id) {
  const from = document.getElementById(current), to = document.getElementById(id);
  window.MarketMap?.mode(MAP_MODE[id]);
  if (from && from !== to && !RM) {
    await anim(from, [{ opacity: 1, transform: "none" }, { opacity: 0, transform: "translateY(-14px)" }], { duration: 320 })?.finished;
  }
  document.querySelectorAll(".stage").forEach((s) => s.classList.toggle("active", s.id === id));
  current = id;
  window.scrollTo(0, 0);
  anim(to, [{ opacity: 0, transform: "translateY(18px)" }, { opacity: 1, transform: "none" }], { duration: 520 });
}

// ---------------------------------------------------------------- enter
(async function boot() {
  for (let i = 0; i < 60; i++) {
    try { const h = await api("/api/health"); if (h.ok) break; } catch (e) {}
    await new Promise((r) => setTimeout(r, 500));
  }
  stagger(document.querySelectorAll("#enter .center > *"), [{ opacity: 0, transform: "translateY(16px)" }, { opacity: 1, transform: "none" }], { step: 120, duration: 800 });
  S.world = await api("/api/world");
  S.script = await api("/api/demo/script");
  $("#world-line").textContent = `${S.world.companies} companies · ${S.world.posts.toLocaleString()} posts · synthetic`;
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
  $("#q-count").textContent = `${q.index + 1}/${q.total}`;
  reveal($("#question"), q.text, { stagger: 28 });
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
    for (const c of r.learned.companies) if (c.role === "customer") window.MarketMap?.light(c.id);
    renderBottle(r.bottle);
    if (r.next) await sleep(500);
    if (r.next) ask(r.next); else runReading();
  } finally { $("#send").disabled = false; }
};
$("#answer").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) $("#send").click(); });

// phrases lift out of the answer box and settle above it
function settlePhrases(list) {
  const src = $("#answer").getBoundingClientRect();
  list.forEach((p, i) => {
    const el = document.createElement("span");
    el.className = "phrase in"; el.textContent = `“${p}”`;
    $("#phrases").appendChild(el);
    const dst = el.getBoundingClientRect();
    const dx = src.left + 20 - dst.left, dy = src.top + 20 - dst.top;
    anim(el, [{ opacity: 0, transform: `translate(${dx}px, ${dy}px) scale(.9)`, filter: "blur(4px)" },
              { opacity: 1, transform: "none", filter: "blur(0)" }], { duration: 1100, delay: 120 + i * 260 });
  });
}

function renderBottle(b) {
  const kinds = { pain: "pain", trigger: "trigger", buyer: "buyer", segment: "segment" };
  let html = "";
  if (b.customers.length) html += `<dt>customers</dt><dd>${b.customers.map(esc).join(", ")}</dd>`;
  if (b.lost.length) html += `<dt>lost</dt><dd>${b.lost.map(esc).join(", ")}</dd>`;
  for (const k of Object.keys(kinds)) {
    const rs = b.rules.filter((r) => r.kind === k);
    if (rs.length) html += `<dt>${kinds[k]}</dt>` + rs.map((r) => `<dd>${esc(r.text)}</dd>`).join("");
  }
  const before = new Set([...document.querySelectorAll("#bottle-list dd")].map((d) => d.textContent));
  $("#bottle-list").innerHTML = html;
  stagger([...document.querySelectorAll("#bottle-list dd")].filter((d) => !before.has(d.textContent)),
          [{ opacity: 0, transform: "translateX(-10px)", background: "#e5efe9" }, { opacity: 1, transform: "none", background: "transparent" }],
          { step: 120, duration: 900 });
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
  rec.onend = () => { rec = null; $("#mic").classList.remove("listening"); };
  rec.start(); $("#mic").classList.add("listening");
};

// ---------------------------------------------------------------- pause + feed
function listen() {
  const es = new EventSource(`/api/sessions/${S.sid}/events`);
  es.onmessage = (m) => {
    const ev = JSON.parse(m.data);
    S.events.push(ev);
    const k = document.createElement("div");
    k.innerHTML = `<span class="tick">✓</span><b>${esc(ev.agent)}</b> · ${esc(ev.content)}`;
    $("#kitchen").appendChild(k);
    anim(k, [{ opacity: 0, transform: "translateY(8px)" }, { opacity: 1, transform: "none" }], { duration: 380 });
    while ($("#kitchen").children.length > 7) $("#kitchen").firstChild.remove();
    if (ev.meta?.candidates) countTo($("#scored"), ev.meta.candidates);
    window.AV?.ingest(ev);
    renderFeed();
    if (ev.agent === "Webhook") onReply();
  };
}

async function runReading() {
  await show("pause");
  const names = S.world.sample_names;
  const deck = $("#shuffle");
  deck.innerHTML = Array.from({ length: 5 }, (_, i) => `<div class="deck-card" style="--i:${i}">${esc(names[i])}</div>`).join("");
  let i = 5;
  const t = setInterval(() => {
    const top = deck.lastElementChild;
    anim(top, [{ transform: "translateY(0) rotate(0)", opacity: 1 }, { transform: "translateY(-46px) rotate(-6deg)", opacity: 0 }], { duration: 220, fill: "none" })
      ?.finished.then(() => { top.textContent = names[i++ % names.length]; deck.prepend(top); });
    if (RM) { top.textContent = names[i++ % names.length]; }
  }, 160);
  const started = Date.now();
  const r = await api(`/api/sessions/${S.sid}/reading`, { method: "POST" });
  await sleep(Math.max(0, 3200 - (Date.now() - started))); // let the scan read as work
  clearInterval(t);
  S.cards = r.cards; S.cardIdx = 0;
  await show("reading");
  renderCard(0);
}

function countTo(el, to) {
  if (!el) return;
  const from = parseInt(el.textContent) || 0, t0 = performance.now();
  const step = (now) => { const f = Math.min(1, (now - t0) / 1200); el.textContent = Math.round(from + (to - from) * (1 - Math.pow(1 - f, 3))); if (f < 1) requestAnimationFrame(step); };
  requestAnimationFrame(step);
}

// ---------------------------------------------------------------- reading
function cardFront(c) {
  let html = `<div class="front-text"></div>`;
  if (c.companies?.length) html += `<div class="companies">${c.companies.slice(0, 10).map((x) => `<span class="chip" data-id="${x.id}">${esc(x.name)}</span>`).join("")}</div>`;
  return html;
}

// confidence shows as clarity: words resolve, then settle at the card's own focus
function playFront(c, delay = 0) {
  const ft = $("#card-front .front-text");
  const blur = (1 - c.clarity) * 3.2, op = 0.35 + 0.65 * c.clarity;
  reveal(ft, c.front_text, { delay });
  ft.style.filter = `blur(${blur.toFixed(2)}px)`; ft.style.opacity = op.toFixed(2);
  anim(ft, [{ filter: "blur(10px)", opacity: 1 }, { filter: `blur(${blur.toFixed(2)}px)`, opacity: op }], { duration: 1600, delay });
  stagger(document.querySelectorAll("#card-front .chip"), [{ opacity: 0, transform: "translateY(10px) scale(.92)" }, { opacity: 1, transform: "none" }],
          { delay: delay + 500, step: 60 });
}

function cardBack(c) {
  let html = c.back_text ? `<p class="back-line">${esc(c.back_text)}</p>` : "";
  for (const r of c.receipts.slice(0, 5)) {
    const q = r.quote.length > 90 ? r.quote.slice(0, 88).replace(/\s+\S*$/, "") + " …" : r.quote;
    html += `<a class="receipt" href="${esc(r.url)}" target="_blank"><q>${esc(q)}</q>` +
      `<span class="meta">${r.company ? esc(r.company) + " · " : ""}${esc(r.date)}${r.verified ? " ✓" : ""}</span></a>`;
  }
  return html;
}

async function renderCard(dir = 1, { swapOnly = false } = {}) {
  const c = S.cards[S.cardIdx];
  const scene = document.querySelector(".card-scene");
  if (!swapOnly && dir && $("#card-front").innerHTML && !RM) {
    await anim(scene, [{ transform: "none", opacity: 1 }, { transform: `translateX(${-70 * dir}px) rotate(${-3 * dir}deg)`, opacity: 0 }],
               { duration: 300, easing: "cubic-bezier(.4,0,1,1)" })?.finished;
  }
  $("#card").classList.remove("flipped");
  $("#card-eyebrow").textContent = EYEBROW[c.kind] || c.kind;
  $("#card-front").innerHTML = cardFront(c);
  $("#card-back").innerHTML = cardBack(c);
  $("#card-diff").textContent = c.diff || "";
  $("#flip").style.display = c.receipts.length || c.back_text ? "" : "none";
  for (const id of ["#yes", "#no", "#no-text"]) $(id).style.display = c.revisable ? "" : "none";
  $("#next-card").title = S.cardIdx === S.cards.length - 1 ? "Accounts" : "Next";
  $("#dots").innerHTML = S.cards.map((_, i) => `<span class="${i === S.cardIdx ? "on" : ""}"></span>`).join("");
  $("#no-text").value = "";
  window.MarketMap?.show(c.map || {});
  if (!swapOnly) anim(scene, [{ transform: `translateY(36px) rotate(${1.5 * dir}deg) scale(.97)`, opacity: 0 }, { transform: "none", opacity: 1 }], { duration: 700 });
  playFront(c, swapOnly ? 0 : 180);
  if (c.diff) anim($("#card-diff"), [{ opacity: 0, transform: "translateY(8px)" }, { opacity: 1, transform: "none" }], { duration: 500, delay: 300 });
}

$("#flip").onclick = () => $("#card").classList.toggle("flipped");
$("#next-card").onclick = () => {
  if (S.cardIdx < S.cards.length - 1) { S.cardIdx++; renderCard(1); } else openDash();
};
async function sendFeedback(text) {
  const c = S.cards[S.cardIdx];
  const nc = await api(`/api/sessions/${S.sid}/cards/${c.id}/feedback`, { method: "POST", body: { text } });
  // show what the correction removed before the card rewrites itself
  const gone = new Set(nc.map?.removed || []);
  const chips = [...document.querySelectorAll("#card-front .chip")].filter((ch) => gone.has(+ch.dataset.id));
  chips.forEach((ch) => ch.classList.add("struck"));
  if (chips.length) await sleep(900);
  await anim($("#card-front .front-text"), [{ opacity: 1 }, { opacity: 0, filter: "blur(6px)" }], { duration: 300 })?.finished;
  S.cards[S.cardIdx] = nc; renderCard(0, { swapOnly: true });
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
  anim(document.getElementById(name), [{ opacity: 0, transform: "translateY(10px)" }, { opacity: 1, transform: "none" }], { duration: 420 });
  ({ accounts: loadAccounts, pipeline: loadPipeline, system: loadSystem, "bottle-tab": loadBottle, proof: loadProof, feed: loadAgents })[name]?.();
}
async function openDash() { await show("dash"); tab("accounts"); }

async function loadAccounts({ animate = true } = {}) {
  const r = await api(`/api/sessions/${S.sid}/accounts`);
  meter(r.credits);
  const top = Math.max(...r.accounts.map((a) => a.score), 1);
  $("#account-list").innerHTML = r.accounts.map((a) => `
    <div class="account">
      <div class="score" title="market ${a.market.toFixed(2)} · founder ${a.founder.toFixed(2)} · λ ${a.lambda.toFixed(2)}">${a.score.toFixed(2)}</div>
      <h3>${esc(a.name)}</h3>
      <div class="scorebar"><i style="--w:${(a.score / top * 100).toFixed(1)}%"></i></div>
      <div class="sub" data-acct="${a.id}">${esc(a.industry)} · ${esc(a.size)} people · ${esc(a.stage.replace("_", " "))}${a.enriched ? ` · <span class="contact">${esc(a.contact.first)}, ${esc(a.contact.role)} &lt;${esc(a.contact.email)}&gt;</span>` : ""}</div>
      <div class="chips">${a.chips.map((ch) => `<a class="chip ${ch.kind}" href="${esc(ch.receipt.url)}" target="_blank" title="${esc(ch.receipt.date)}">${esc(ch.label)}</a>`).join("")}</div>
      <button class="btn" onclick="draftFor(${a.id})"><i class="i" style="--i:url(/static/vendor/icons/mail.svg)"></i>Draft</button>
    </div>`).join("");
  if (animate) {
    stagger(document.querySelectorAll(".account"), [{ opacity: 0, transform: "translateY(18px)" }, { opacity: 1, transform: "none" }], { step: 90 });
    document.querySelectorAll(".scorebar i").forEach((b, i) => anim(b, [{ width: "0%" }, { width: b.style.getPropertyValue("--w") }], { duration: 900, delay: 200 + i * 90 }));
    stagger(document.querySelectorAll(".account .chip"), [{ opacity: 0, transform: "scale(.9)" }, { opacity: 1, transform: "none" }], { delay: 350, step: 30 });
  }
}

function meter(credits, start = 30) {
  $("#meter-fill").style.width = `${Math.max(0, (credits / start) * 100)}%`;
  $("#meter-text").textContent = `${credits.toFixed(0)}/${start} credits`;
}

// credits drain one account at a time, in the order Flexprice spent them
$("#enrich").onclick = async () => {
  $("#enrich").disabled = true;
  const r = await api(`/api/sessions/${S.sid}/enrich`, { method: "POST" });
  let bal = r.start;
  for (const l of r.ledger) {
    bal = l.balance;
    meter(bal, r.start);
    $("#meter-text").textContent = `${bal.toFixed(0)}/${r.start} · ${l.account}`;
    const coin = document.createElement("span");
    coin.className = "coin"; coin.textContent = `−${l.credits}`;
    $(".meter").appendChild(coin);
    anim(coin, [{ opacity: 0, transform: "translateY(0)" }, { opacity: 1, offset: 0.2 }, { opacity: 0, transform: "translateY(42px)" }], { duration: 700 })?.finished.then(() => coin.remove());
    await sleep(380);
  }
  await loadAccounts({ animate: false });
  document.querySelectorAll(".contact").forEach((c) => anim(c, [{ filter: "blur(6px)", opacity: 0 }, { filter: "blur(0)", opacity: 1 }], { duration: 800 }));
  meter(r.balance, r.start);
  if (r.skipped.length) $("#meter-text").textContent += ` · ${r.skipped.length} skipped (budget)`;
};

window.draftFor = async (id) => {
  const d = await api(`/api/sessions/${S.sid}/drafts`, { method: "POST", body: { account_id: id } });
  S.drafts[d.id] = d;
  tab("outreach");
  renderDraft(d);
};

function renderDraft(d) {
  $("#draft-area").innerHTML = `
    <p class="weight">${esc(d.to_name)}, ${esc(d.to_role)} · ${esc(d.account)}</p>
    <div class="draft-wrap"><textarea id="draft-body"></textarea><div class="sweep"></div></div>
    <div id="flags"></div>
    <div class="row" style="display:flex;margin-top:10px"><span class="grow"></span><button class="btn primary" id="approve"><i class="i" style="--i:url(/static/vendor/icons/send.svg)"></i>Approve & send</button></div>
    <p class="fine" id="sent-line"></p>`;
  typeDraft(d);
  $("#approve").onclick = async () => {
    const r = await api(`/api/sessions/${S.sid}/drafts/${d.id}/approve`, { method: "POST", body: { body: $("#draft-body").value } });
    S.drafts[d.id] = r;
    $("#approve").disabled = true;
    const link = `${location.origin}/reply/${d.id}`;
    $("#sent-line").innerHTML = `<span class="small">Sent · reply at <a href="${link}" target="_blank">${esc(location.host)}/reply/${esc(d.id)}</a></span>`;
    anim($("#sent-line"), [{ opacity: 0, transform: "translateY(6px)" }, { opacity: 1, transform: "none" }]);
    await sleep(1400);
    tab("pipeline");
  };
}

async function typeDraft(d) {
  const ta = $("#draft-body"), btn = $("#approve");
  btn.disabled = true;
  if (RM) ta.value = d.body;
  else for (let i = 0; i <= d.body.length; i += 3) { ta.value = d.body.slice(0, i); await new Promise((r) => requestAnimationFrame(r)); }
  ta.value = d.body;
  const sw = document.querySelector(".sweep");
  await anim(sw, [{ transform: "translateY(0)", opacity: 0.9 }, { transform: `translateY(${ta.offsetHeight - 4}px)`, opacity: 0.9 }, { opacity: 0 }], { duration: 900, easing: "linear" })?.finished;
  $("#flags").innerHTML = d.flags.length ? d.flags.map((f) => `<div class="flag">⚑ ${esc(f)}</div>`).join("") : `<p class="fine">✓ no AI tells</p>`;
  anim($("#flags"), [{ opacity: 0 }, { opacity: 1 }], { duration: 400 });
  btn.disabled = false;
}

let PG = null;
async function loadPipeline(highlight) {
  const d = await api(`/api/sessions/${S.sid}/graph`);
  if (!PG) {
    PG = new PipelineGraph($("#pgraph"));
    document.querySelectorAll("#groupby button").forEach((b) => b.onclick = () => {
      document.querySelectorAll("#groupby button").forEach((x) => x.classList.toggle("on", x === b));
      PG.setGroup(b.dataset.g);
    });
  }
  PG.resize();
  const hit = highlight && d.accounts.find((a) => a.name === highlight);
  PG.update(d, { pulse: hit ? hit.id : null });
  const count = (st) => d.accounts.filter((a) => a.stage === st).length;
  const dot = (c) => `<i style="background:${c}"></i>`;
  $("#glegend").innerHTML =
    `<div>${dot(PIPELINE_STATUS.customer.color)}past customer ${d.customers.length}` +
    ["suggested", "drafted", "sent", "replied"].map((st) => `${dot(PIPELINE_STATUS[st].color)}${st} <b>${count(st)}</b>`).join("") +
    `${dot(PIPELINE_STATUS.segment.color)}new segment ${d.segment.length}</div>` +
    `<div>${["industry", "trigger", "tool", "pain"].map((k) => `<i class="ln" style="background:${PIPELINE_REL[k].color}"></i>${PIPELINE_REL[k].label}`).join("")}</div>`;
}

let SV = null;
async function loadSystem() {
  if (!SV) { SV = new SystemView($("#system-root")); }
  SV.setLive(await api(`/api/sessions/${S.sid}/system`));
  SV.show(SV.panel);
}

async function onReply() {
  const last = S.events.filter((e) => e.agent === "Webhook").pop();
  const name = last?.content.split(" replied")[0];
  if (!document.querySelector("#pipeline.on")) { tab("pipeline"); await sleep(450); }
  await loadPipeline(name);
}

async function loadBottle() {
  const b = await api(`/api/sessions/${S.sid}/bottle`);
  const KINDS = ["pain", "trigger", "buyer", "segment"];
  const gauge = (t, market) => {
    const a = Math.PI * (1 - t), x = 30 + 24 * Math.cos(a), y = 30 - 24 * Math.sin(a);
    return `<svg viewBox="0 0 60 36" class="gauge"><path d="M6,30 A24,24 0 0 1 54,30" class="g-bg"/>
      <path d="M6,30 A24,24 0 0 1 ${x.toFixed(1)},${y.toFixed(1)}" class="g-fg ${market ? "mk" : ""}" pathLength="1"/></svg>`;
  };
  const cols = KINDS.map((k) => {
    const rs = b.rules.filter((r) => r.kind === k).sort((x, y) => y.trust - x.trust);
    if (!rs.length) return "";
    return `<div class="rcol"><h4>${k}</h4>${rs.map((r) => {
      const market = (r.key || "").includes(":");
      return `<div class="rcard ${market ? "market" : ""}"><p>${esc(r.text.replace(" (found in the market data)", ""))}</p>
        <div class="rfoot"><span class="src">${market ? "from market data" : ""}</span>${gauge(r.trust, market)}<b>${r.trust.toFixed(2)}</b></div></div>`;
    }).join("")}</div>`;
  }).join("");

  const tr = b.hit_features.slice(0, 8);
  const RW = 900, L = 230, PW = 520, RH = 46, X = (v) => L + v * PW;
  const rows = tr.map((f, i) => {
    const y = 40 + i * RH, share = f.k / f.n, base = f.base_rate;
    return `<g class="trow ${f.passes ? "sig" : ""}" style="--d:${i * 90}ms">
      <text x="0" y="${y + 5}" class="tl">${esc(f.label)}</text>
      <line x1="${X(0)}" x2="${X(1)}" y1="${y}" y2="${y}" class="tgrid"/>
      <line x1="${X(base)}" x2="${X(share)}" y1="${y}" y2="${y}" class="tgap" pathLength="1"/>
      <circle cx="${X(base)}" cy="${y}" r="6" class="tbase"/>
      <circle cx="${X(share)}" cy="${y}" r="7" class="tshare" style="--from:${X(base) - X(share)}px"/>
      <text x="${X(1) + 24}" y="${y + 5}" class="tv">${f.k}/${f.n} <tspan class="muted">vs ${(base * 100).toFixed(0)}%</tspan></text>
      ${f.passes ? `<g transform="translate(${X(1) + 120},${y})"><rect x="0" y="-12" width="84" height="24" rx="12" class="tbadge"/><text x="42" y="4" class="tbadge-t">q ${f.q_value.toExponential(0).replace("e-", "e-")}</text></g>` : ""}
    </g>`;
  }).join("");
  const ticks = [0, .25, .5, .75, 1].map((t) => `<text x="${X(t)}" y="18" class="ttick">${t * 100}%</text>`).join("");
  $("#bottle-inspect").innerHTML = `
    <section class="bsec"><div class="bhead"><h3>What you believe</h3></div>
      <div class="rboard">${cols}</div></section>
    <section class="bsec"><div class="bhead"><h3>Your customers vs similar companies</h3>
      <span class="blegend"><i class="dot-c"></i>your customers <i class="dot-b"></i>similar companies</span></div>
      <svg viewBox="0 0 ${RW + 60} ${60 + tr.length * RH}" class="traits">${ticks}${rows}</svg></section>`;
  if (!RM) {
    document.querySelectorAll(".rcard").forEach((c, i) => anim(c, [{ opacity: 0, transform: "translateY(10px)" }, { opacity: 1, transform: "none" }], { delay: i * 50 }));
    document.querySelectorAll(".g-fg").forEach((g, i) => anim(g, [{ strokeDasharray: "0 1" }, { strokeDasharray: "1 0" }], { duration: 900, delay: 200 + i * 50 }));
    document.querySelectorAll(".trow").forEach((r, i) => {
      anim(r.querySelector(".tgap"), [{ strokeDasharray: "0 1" }, { strokeDasharray: "1 0" }], { duration: 900, delay: 300 + i * 90 });
      anim(r.querySelector(".tshare"), [{ transform: `translateX(${r.querySelector(".tshare").style.getPropertyValue("--from")})`, opacity: 0 }, { transform: "none", opacity: 1 }], { duration: 900, delay: 300 + i * 90 });
    });
  }
}

function loadAgents() {
  if (!window.AV) { window.AV = new AgentsView($("#agents-root")); AV.load(S.events); }
  renderFeed();
}

function renderFeed() {
  if (!S.world || !$("#feed-list")) return;
  const had = $("#feed-list").children.length;
  $("#feed-list").innerHTML = S.events.slice().reverse().map((e) => `
    <li><span class="agent">${esc(e.agent)}</span><span class="type ${e.type === "error" ? "error" : ""}">${esc(e.type)}</span><span>${esc(e.content)}</span><span class="band">${esc(e.band)}</span></li>`).join("");
  const fresh = [...$("#feed-list").children].slice(0, Math.max(0, $("#feed-list").children.length - had));
  if (document.querySelector("#feed.on")) stagger(fresh, [{ opacity: 0, transform: "translateX(-12px)" }, { opacity: 1, transform: "none" }], { step: 40, duration: 400 });
}

// ---------------------------------------------------------------- model & proof
const SERIES = ["#3987e5", "#d95926", "#199e70"]; // categorical slots 1-3, dark-surface steps, fixed order
const BADGE_NAME = { synthetic: "Synthetic", simulated: "Simulated", measured: "Measured" };
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
  for (const t of ticks) svg += `<line x1="${L}" x2="${W - R}" y1="${Y(t)}" y2="${Y(t)}" stroke="rgba(255,255,255,.07)"/>` +
    `<text x="${L - 6}" y="${Y(t) + 4}" font-size="10" text-anchor="end" fill="rgba(255,255,255,.45)">${yFmt(t)}</text>`;
  svg += `<text x="${(L + W - R) / 2}" y="${H - 4}" font-size="10" text-anchor="middle" fill="rgba(255,255,255,.45)">${esc(xLabel)}</text>`;
  svg += `<text x="${L}" y="${H - 16}" font-size="10" fill="rgba(255,255,255,.45)">${x0}</text><text x="${W - R}" y="${H - 16}" font-size="10" text-anchor="end" fill="rgba(255,255,255,.45)">${x1}</text>`;
  if (ref) svg += `<line x1="${L}" x2="${W - R}" y1="${Y(ref.y)}" y2="${Y(ref.y)}" stroke="rgba(255,255,255,.4)" stroke-dasharray="4 4"/>` +
    `<text x="${W - R + 6}" y="${Y(ref.y) + 4}" font-size="10" fill="rgba(255,255,255,.6)">${esc(ref.label)}</text>`;
  series.forEach((s, i) => {
    const d = s.points.map((p, j) => `${j ? "L" : "M"}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join("");
    svg += `<path class="draw" d="${d}" fill="none" stroke="${SERIES[i]}" stroke-width="2" stroke-linejoin="round" pathLength="1" stroke-dasharray="1" stroke-dashoffset="${RM ? 0 : 1}"/>`;
    const last = s.points[s.points.length - 1];
    if (series.length > 1 || !ref) svg += `<text x="${X(last[0]) + 6}" y="${Y(last[1]) + 4 + i * 0}" font-size="10" fill="rgba(255,255,255,.85)">${esc(s.short || s.name)}</text>`;
  });
  svg += `<line class="xh" x1="0" x2="0" y1="${T}" y2="${H - B}" stroke="rgba(255,255,255,.3)" style="display:none"/>`;
  svg += `<rect x="${L}" y="${T}" width="${W - L - R}" height="${H - T - B}" fill="transparent"/></svg>`;
  const legend = series.length > 1 ? `<div class="legend">${series.map((s, i) => `<span><i style="background:${SERIES[i]}"></i>${esc(s.name)}</span>`).join("")}</div>` : "";
  const id = "c" + Math.random().toString(36).slice(2, 8);
  setTimeout(() => {
    const root = document.getElementById(id); if (!root) return;
    new IntersectionObserver((ents, ob) => ents.forEach((en) => {
      if (!en.isIntersecting) return;
      root.querySelectorAll("path.draw").forEach((p, i) => anim(p, [{ strokeDashoffset: 1 }, { strokeDashoffset: 0 }], { duration: 1400, delay: i * 200 }));
      ob.disconnect();
    }), { threshold: 0.4 }).observe(root);
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
  return `<div class="viz" id="${id}"><h4>${esc(title)}</h4>${legend}${svg}<div class="tip"></div>${yLabel ? `<p class="fine" style="margin:4px 0 0">${esc(yLabel)}</p>` : ""}</div>`;
}

const recallTable = (methods, k) => `<table><tr><th></th><th>Recall@${k}</th><th>95% CI</th></tr>` +
  Object.entries(methods).map(([m, v]) => `<tr><td>${esc(m)}</td><td>${v.recall.toFixed(2)}</td><td class="muted">${v.ci95[0].toFixed(2)}–${v.ci95[1].toFixed(2)}</td></tr>`).join("") + `</table>`;

async function loadProof() {
  $("#proof-area").innerHTML = `<p class="fine">Loading…</p>`;
  let d;
  try { d = await api("/api/model-proof"); } catch (e) { $("#proof-area").innerHTML = `<p class="flag">Run <code>make sim</code> first.</p>`; return; }
  const mm = d.market_model, pr = d.prior_residual, ts = d.thompson, bt = d.backtest, vo = d.voice_dpo, pol = ts.policies;
  const rounds = (arr) => arr.map((v, i) => [i + 1, v]);
  const head = (title, b) => `<div class="mp-head"><h3>${esc(title)}</h3>${b === "synthetic" ? "" : `<span class="badge ${b}">${BADGE_NAME[b]}</span>`}</div>`;
  const planted = d.planted.items;
  const short = [
    ["RevOps trigger, tested", `q ${Number(planted[0].evidence.match(/q = ([^,]+)/)?.[1]).toExponential(0)}`],
    ["RevOps trigger, learned", `w ${pr.weights["RevOps hire, last 6 months"].toFixed(2)}`],
    ["Logistics segment", (planted[2].evidence.match(/^(\d+ of \d+)/)?.[1] || "").replace(" of ", "/")],
    ["Bad extractions caught", planted[3].evidence.match(/rejected (\d+)/)?.[1] || ""],
  ];
  const ws = Object.entries(pr.weights).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 5);
  const wmax = Math.max(...ws.map((w) => Math.abs(w[1])));
  const keep = ["popularity", "description match (LLM stand-in)", "market model alone", "founder alone", "market + founder"];
  $("#proof-area").innerHTML = `<div class="mp">
  <p class="mp-top">Trained on synthetic data, unless marked.</p>

  <section>${head("Planted patterns recovered", "synthetic")}
    <div class="kpis">${short.map(([l, v], i) => `<div class="kpi"><span class="kpi-v">${planted[i].recovered ? "✓" : "✗"} ${esc(v)}</span><span class="kpi-l">${esc(l)}</span></div>`).join("")}</div>
  </section>

  <section>${head("Market model", "synthetic")}
    <div class="grid2">
      ${lineChart({ title: "BPR loss", xLabel: "epoch", yLabel: "", series: [{ name: "LightGCN", points: mm.lightgcn.curve.map((r) => [r.epoch, r.loss]) }] })}
      ${lineChart({ title: "Recall@10", xLabel: "epoch", yLabel: "", series: [{ name: "LightGCN", points: mm.lightgcn.curve.map((r) => [r.epoch, r.val_recall]) }],
        ref: { y: mm.popularity_recall, label: "popularity" } })}
    </div>
  </section>

  <section>${head("Prior + correction", "synthetic")}
    <div class="grid2">
      <div>${recallTable(pr.methods, pr.test.k)}</div>
      <div>${ws.map(([f, v]) => `<div class="wbar"><span class="wl">${esc(f)}</span>
        <span class="wtrack"><i style="--w:${(Math.abs(v) / wmax * 50).toFixed(1)}%;${v < 0 ? "right:50%" : "left:50%"};background:${v < 0 ? "#d95926" : "#3987e5"}"></i></span>
        <span class="wv">${v > 0 ? "+" : ""}${v.toFixed(2)}</span></div>`).join("")}</div>
    </div>
  </section>

  <section>${head("Thompson sampling", "synthetic")}
    ${lineChart({ title: "Conversions", xLabel: "round", yLabel: "", yFmt: (v) => v.toFixed(0),
      series: [{ name: "rescaled priors", points: rounds(pol.ts_calibrated.cum_conversions) },
               { name: "flat priors", points: rounds(pol.ts_flat.cum_conversions) },
               { name: "priors as stated", points: rounds(pol.ts_founder.cum_conversions) }] })}
  </section>

  <section>${head(`Backtest · ${bt.asof} → after`, "synthetic")}
    ${recallTable(Object.fromEntries(keep.map((k) => [k.replace(" (LLM stand-in)", ""), bt.methods[k]])), bt.k)}
  </section>

  <section>${head("Voice model · LoRA + DPO", "simulated")}
    <div class="grid2">
      ${lineChart({ title: "DPO loss", xLabel: "step", yLabel: "", series: [{ name: "DPO loss", points: vo.dpo_curve.map((r) => [r.step, r.loss]) }] })}
      <div class="kpis one"><div class="kpi"><span class="kpi-v">${vo.eval.win_rate_vs_base.toFixed(2)}</span><span class="kpi-l">win rate vs base</span></div></div>
    </div>
  </section>

  <section>${head("Measured", "measured")}<p class="fine">None yet.</p></section>
  </div>`;
}
