// "How it works" pop-ups: each technique as a short animated scene made of color blocks.
(function () {
  const RM = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const NS = "http://www.w3.org/2000/svg";
  const C = { blue: "#3987e5", gray: "#2c2b3d", yellow: "#eda100", orange: "#d95926", green: "#22c55e", red: "#ef4444", ink: "#f5f5f4", dim: "rgba(255,255,255,.45)" };
  const W = 760, H = 344;
  let FAST = false; // true while skipping ahead to a chosen step
  const sleep = (ms) => new Promise((r) => setTimeout(r, RM || FAST ? 0 : ms));

  function el(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v);
    if (parent) parent.appendChild(e);
    return e;
  }
  const text = (svg, x, y, s, cls = "ex-t", anchor = "start") => { const t = el("text", { x, y, class: cls, "text-anchor": anchor }, svg); t.textContent = s; return t; };
  const fill = (e, color, ms = 500) => { e.style.transition = `fill ${RM ? 0 : ms}ms, opacity ${RM ? 0 : ms}ms, transform ${RM ? 0 : ms}ms`; e.setAttribute("fill", color); };
  const fade = (e, o, ms = 500) => { e.style.transition = `opacity ${RM ? 0 : ms}ms`; e.style.opacity = o; };
  // animate the individual `scale` property so it composes with SVG transform="translate(...)"
  const pop = (e, d = 0) => e.animate?.([{ opacity: 0, scale: ".6" }, { opacity: 1, scale: "1" }],
    { duration: RM ? 0 : 450, delay: RM ? 0 : d, easing: "cubic-bezier(.2,.8,.2,1.4)", fill: "backwards" });
  const grid = (svg, x0, y0, cols, rows, size = 30, gap = 8, color = C.gray) => {
    const out = [];
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
      const b = el("rect", { x: x0 + c * (size + gap), y: y0 + r * (size + gap), width: size, height: size, rx: 7, fill: color, class: "ex-b" }, svg);
      b.style.transformBox = "fill-box"; b.style.transformOrigin = "center";
      out.push(b);
    }
    return out;
  };
  const badge = (svg, x, y, label, color) => {
    const g = el("g", { transform: `translate(${x},${y})` }, svg);
    el("rect", { x: -78, y: -16, width: 156, height: 32, rx: 16, fill: "rgba(0,0,0,.55)", stroke: color, "stroke-width": 1.5 }, g);
    const t = text(g, 0, 5, label, "ex-badge", "middle"); t.setAttribute("fill", color);
    g.style.transformBox = "fill-box"; g.style.transformOrigin = "center"; pop(g);
    return g;
  };

  // ------------------------------------------------------------------ scenes
  const node = (svg, x, y, label, kind) => {
    const g = el("g", { transform: `translate(${x},${y})` }, svg);
    if (kind === "company") el("rect", { x: -26, y: -14, width: 52, height: 28, rx: 7, fill: C.gray, stroke: "rgba(255,255,255,.3)" }, g);
    else el("circle", { r: 15, fill: kind === "role" ? "#3b2a10" : "#14233d", stroke: kind === "role" ? C.yellow : C.blue }, g);
    text(g, 0, 4, label, "ex-node", "middle");
    pop(g);
    return g;
  };
  const edge = (svg, x1, y1, x2, y2, label) => {
    const p = el("line", { x1, y1, x2, y2, class: "ex-edge" }, svg);
    p.animate?.([{ opacity: 0 }, { opacity: 1 }], { duration: RM ? 0 : 500, fill: "backwards" });
    if (label) text(svg, (x1 + x2) / 2, (y1 + y2) / 2 - 6, label, "ex-elbl", "middle");
    return p;
  };
  const beta = (a, b) => { // Beta pdf on [0,1], 60 points, normalised to height 1
    const lg = (z) => { const c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
      if (z < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * z)) - lg(1 - z); z -= 1; let x = c[0]; for (let i = 1; i < 9; i++) x += c[i] / (z + i); const t = z + 7.5; return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x); };
    const ys = Array.from({ length: 59 }, (_, i) => { const x = (i + 1) / 60; return Math.exp((a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x) - (lg(a) + lg(b) - lg(a + b))); });
    const m = Math.max(...ys); return ys.map((y) => y / m);
  };
  const curve = (ys, x0, y0, w, h) => "M" + ys.map((v, i) => `${(x0 + ((i + 1) / 60) * w).toFixed(1)},${(y0 - v * h).toFixed(1)}`).join(" L");

  const SCENES = {
    bottle: {
      title: "Talk us through your instinct",
      adjusts: "What gets adjusted: α and β for each belief (Bayesian priors, not neural weights)",
      steps: [
        ["You talk. We extract beliefs: pain, trigger, buyer.", (s) => {
          s.chips = [["Pain", "“nobody trusted the numbers”"], ["Trigger", "“they had just raised”"], ["Buyer", "“the VP Finance signs”"]].map(([k, t], i) => {
            const g = el("g", { transform: `translate(40,${70 + i * 78})` }, s.svg);
            el("rect", { x: 0, y: 0, width: 250, height: 52, rx: 14, fill: "rgba(255,255,255,.05)", stroke: "rgba(255,255,255,.25)" }, g);
            text(g, 16, 20, k, "ex-small"); text(g, 16, 40, t, "ex-t");
            pop(g, i * 160); return g;
          });
        }],
        ["Each belief becomes a Beta(α, β) prior. Confidence sets the pseudo-counts.", (s) => {
          s.curves = [[14, 6, "quite sure"], [6, 4, "fairly sure"], [2, 2, "a hunch"]].map(([a, b, how], i) => {
            const y0 = 112 + i * 78;
            el("line", { x1: 330, x2: 560, y1: y0, y2: y0, stroke: "rgba(255,255,255,.15)" }, s.svg);
            const path = el("path", { d: curve(beta(a, b), 330, y0, 230, 44), class: "ex-beta" }, s.svg);
            path.animate?.([{ opacity: 0 }, { opacity: 1 }], { duration: RM ? 0 : 500, delay: RM ? 0 : i * 150, fill: "backwards" });
            const lab = text(s.svg, 580, y0 - 16, `Beta(${a}, ${b})`, "ex-mono");
            text(s.svg, 580, y0, `${how} = ${a + b} pseudo-counts`, "ex-small");
            return { path, lab, a, b, y0 };
          });
        }],
        ["Each reply is a success, α + 1. After three replies the posterior moves right.", async (s) => {
          const c = s.curves[0], d = el("circle", { cx: 450, cy: 30, r: 8, fill: C.green }, s.svg);
          text(s.svg, 470, 34, "× 3", "ex-small");
          d.animate?.([{ transform: "translateY(0)" }, { transform: "translateY(60px)" }], { duration: RM ? 0 : 500, fill: "forwards" });
          await sleep(550); fade(d, 0, 200);
          c.a += 3; c.path.setAttribute("d", curve(beta(c.a, c.b), 330, c.y0, 230, 44)); c.path.classList.add("up");
          c.lab.textContent = `Beta(${c.a}, ${c.b})`;
        }],
        ["Each silence or no is a failure, β + 1. Three misses pull a hunch down.", async (s) => {
          const c = s.curves[2], d = el("circle", { cx: 440, cy: 30, r: 8, fill: C.red }, s.svg);
          text(s.svg, 400, 34, "× 3", "ex-small");
          d.animate?.([{ transform: "translateY(0)" }, { transform: "translateY(216px)" }], { duration: RM ? 0 : 650, fill: "forwards" });
          await sleep(700); fade(d, 0, 200);
          c.b += 3; c.path.setAttribute("d", curve(beta(c.a, c.b), 330, c.y0, 230, 44)); c.path.classList.add("down");
          c.lab.textContent = `Beta(${c.a}, ${c.b})`;
        }],
      ],
    },

    engine: {
      title: "We test your instinct against the market",
      adjusts: "What gets adjusted: graph embeddings (LightGCN) and one vector for your product",
      steps: [
        ["We build a market knowledge graph from public posts. Every edge cites its post.", (s) => {
          const A = node(s.svg, 120, 120, "Acme", "company"), B = node(s.svg, 120, 240, "Borg", "company");
          void A; void B;
          node(s.svg, 330, 80, "Stripe", "tool"); node(s.svg, 330, 170, "SFDC", "tool"); node(s.svg, 330, 270, "RevOps", "role");
          edge(s.svg, 146, 120, 315, 82, "ADOPTED"); edge(s.svg, 146, 120, 315, 168, "ADOPTED");
          edge(s.svg, 146, 240, 315, 172, "ADOPTED"); edge(s.svg, 146, 240, 315, 268, "HIRING_FOR");
          const cite = el("g", { transform: "translate(560,170)" }, s.svg);
          el("rect", { x: -120, y: -42, width: 240, height: 84, rx: 12, fill: "rgba(255,255,255,.04)", stroke: "rgba(255,255,255,.25)", "stroke-dasharray": "4 4" }, cite);
          text(cite, 0, -14, "Source: HN post, 2024-03", "ex-small", "middle");
          text(cite, 0, 8, "“Stack: Stripe, Salesforce…”", "ex-t", "middle");
          text(cite, 0, 28, "Reviewer checks every edge", "ex-small", "middle");
          pop(cite, 300);
          s.kg = true;
        }],
        ["Backward: a time-split backtest. Train up to Dec 2024, test on what happened next.", (s) => {
          s.svg.innerHTML = "";
          const y = 170;
          el("rect", { x: 60, y: y - 8, width: 400, height: 16, rx: 8, fill: C.blue, opacity: .7 }, s.svg);
          el("rect", { x: 470, y: y - 8, width: 230, height: 16, rx: 8, fill: C.yellow, opacity: .7 }, s.svg);
          el("line", { x1: 465, x2: 465, y1: y - 40, y2: y + 40, stroke: "#fff", "stroke-dasharray": "3 4" }, s.svg);
          text(s.svg, 260, y - 22, "Train: graph + model fit on the past", "ex-t", "middle");
          text(s.svg, 585, y - 22, "Test: who adopted next", "ex-t", "middle");
          text(s.svg, 465, y + 58, "Dec 2024 cut-off. No future data leaks into training.", "ex-small", "middle");
        }],
        ["Each shared trait gets a hypothesis test against matched companies, with Benjamini–Hochberg correction.", async (s) => {
          s.svg.innerHTML = "";
          text(s.svg, 40, 40, "Your customers vs matched companies (same size, stage, months)", "ex-small");
          s.past = grid(s.svg, 40, 60, 12, 5, 26, 8);
          s.cust = [13, 30, 44];
          s.past.forEach((b, i) => pop(b, i * 5));
          s.cust.forEach((i) => fill(s.past[i], C.blue));
          await sleep(700);
          s.cust.forEach((i) => { s.past[i].setAttribute("stroke", C.yellow); s.past[i].setAttribute("stroke-width", 4); });
          fill(s.past[51], C.yellow);
          await sleep(500);
          badge(s.svg, 200, 300, "q ≈ 1 in 80,000 ✓", C.green);
          const many = [0, 4, 7, 9, 16, 20, 25, 28, 33, 37, 40, 47, 55, 58];
          await sleep(900);
          many.forEach((i, k) => setTimeout(() => fill(s.past[i], "rgba(237,161,0,.55)"), RM ? 0 : k * 40));
          await sleep(700);
          badge(s.svg, 540, 300, "q = 0.4: could be luck ✕", C.red);
        }],
        ["Forward: a graph neural network (LightGCN) embeds every company. Similar ones land close.", async (s) => {
          s.svg.innerHTML = "";
          text(s.svg, 40, 36, "Embedding space", "ex-h");
          const rnd = (seed) => { let x = Math.sin(seed) * 10000; return x - Math.floor(x); };
          const clusters = [[230, 170, C.gray], [470, 120, C.gray], [560, 250, C.orange], [300, 270, C.gray]];
          s.dots = [];
          for (let i = 0; i < 70; i++) {
            const [cx, cy, col] = clusters[i % 4];
            const d = el("circle", { cx: 380, cy: 180, r: 6, fill: col === C.orange ? C.gray : col }, s.svg);
            d.style.transition = `cx ${RM ? 0 : 1100}ms cubic-bezier(.2,.8,.2,1), cy ${RM ? 0 : 1100}ms cubic-bezier(.2,.8,.2,1), fill .6s`;
            const tx = cx + (rnd(i) - .5) * 120, ty = cy + (rnd(i + 99) - .5) * 80;
            setTimeout(() => { d.setAttribute("cx", tx); d.setAttribute("cy", ty); }, RM ? 0 : 20 + i * 8);
            s.dots.push({ d, cl: i % 4, tx, ty });
          }
          await sleep(1200);
          s.dots.filter((x) => x.cl === 0).slice(0, 3).forEach((x) => fill(x.d, C.blue));
        }],
        ["Cold start: your product's vector starts from its description, then is fitted to your customers.", async (s) => {
          const star = el("path", { d: "M0,-11 L3,-3 L11,-3 L5,2 L7,10 L0,5 L-7,10 L-5,2 L-11,-3 L-3,-3 Z", fill: "#fff" }, s.svg);
          star.setAttribute("transform", "translate(620,70)");
          star.style.transition = `transform ${RM ? 0 : 1400}ms cubic-bezier(.2,.8,.2,1)`;
          const lab = text(s.svg, 640, 74, "Your product", "ex-t");
          await sleep(300);
          star.setAttribute("transform", "translate(250,170)"); lab.setAttribute("x", 270); lab.setAttribute("y", 174);
          lab.style.transition = `x ${RM ? 0 : 1400}ms, y ${RM ? 0 : 1400}ms`;
        }],
        ["Score = market score + λ × your signal. λ is larger near companies you know.", async (s) => {
          const ring = el("circle", { cx: 250, cy: 170, r: 10, fill: "none", stroke: "#fff", "stroke-dasharray": "4 5" }, s.svg);
          ring.style.transition = `r ${RM ? 0 : 900}ms`;
          setTimeout(() => ring.setAttribute("r", 80), 50);
          await sleep(900);
          s.dots.filter((x) => x.cl === 2).forEach((x) => fill(x.d, C.orange));
          const t = text(s.svg, 560, 320, "Semantic match: same pain, different words", "ex-small", "middle"); pop(t);
          const near = s.dots.filter((x) => x.cl === 0).sort((a, b) => Math.hypot(a.tx - 250, a.ty - 170) - Math.hypot(b.tx - 250, b.ty - 170)).slice(3, 7);
          near.forEach((x, k) => { const r = text(s.svg, x.tx, x.ty - 10, String(k + 1), "ex-rank2", "middle"); pop(r, k * 120); x.d.setAttribute("stroke", "#fff"); });
        }],
      ],
    },

    voice: {
      title: "Unmistakably yours",
      adjusts: "What gets adjusted: LoRA adapter weights (the base model stays frozen)",
      steps: [
        ["Your sent emails become the training set.", (s) => {
          text(s.svg, 40, 40, "Sent emails", "ex-h");
          for (let i = 0; i < 4; i++) {
            const g = el("g", { transform: `translate(${40 + i * 9},${66 + i * 11})` }, s.svg);
            el("rect", { width: 150, height: 110, rx: 12, fill: "#17162a", stroke: "rgba(255,255,255,.2)" }, g);
            for (let l = 0; l < 5; l++) el("rect", { x: 14, y: 16 + l * 17, width: 122 - (l % 2) * 36, height: 6, rx: 3, fill: "rgba(255,255,255,.25)" }, g);
            pop(g, i * 140);
          }
        }],
        ["LoRA fine-tuning: the base model's weights stay frozen. A small adapter learns your style.", async (s) => {
          text(s.svg, 270, 40, "Base LLM weights (frozen)", "ex-small");
          s.base = grid(s.svg, 270, 54, 8, 8, 16, 4, "#24233a");
          text(s.svg, 450, 40, "LoRA adapter", "ex-small");
          s.lora = grid(s.svg, 450, 54, 2, 8, 16, 4, "#24233a");
          s.lora.forEach((b) => b.setAttribute("stroke", C.blue));
          el("path", { d: "M215,140 L262,140", class: "ex-arrow" }, s.svg);
          for (let k = 0; k < 6 && !RM; k++) {
            s.lora.forEach((b) => fill(b, `hsl(${215 + Math.random() * 40},70%,${30 + Math.random() * 35}%)`, 250));
            await sleep(260);
          }
        }],
        ["DPO on your edits: your version is preferred over the draft it replaced.", async (s) => {
          const g = el("g", { transform: "translate(560,70)" }, s.svg);
          el("rect", { width: 170, height: 40, rx: 10, fill: "rgba(34,197,94,.12)", stroke: C.green }, g); text(g, 12, 25, "Chosen: your edit", "ex-t");
          el("rect", { y: 52, width: 170, height: 40, rx: 10, fill: "rgba(239,68,68,.1)", stroke: C.red }, g); text(g, 12, 77, "Rejected: the draft", "ex-t");
          pop(g);
          for (let k = 0; k < 4 && !RM; k++) { s.lora.forEach((b) => fill(b, `hsl(${150 + Math.random() * 50},60%,${30 + Math.random() * 30}%)`, 250)); await sleep(260); }
        }],
        ["Every drafted line is scored by its likelihood under your voice model. Low scores get flagged.", async (s) => {
          s.svg.querySelectorAll("*").forEach((e) => { if (e !== s.svg) fade(e, .15, 400); });
          await sleep(300);
          text(s.svg, 300, 220, "Draft", "ex-h");
          const vals = [.92, .88, .28, .9];
          s.rows = vals.map((v, i) => {
            const y = 238 + i * 24;
            const line = el("rect", { x: 300, y, width: [230, 190, 240, 150][i], height: 9, rx: 4, fill: "rgba(255,255,255,.75)" }, s.svg);
            const sc = el("rect", { x: 545, y: y - 1, width: 0, height: 11, rx: 5, fill: v > .5 ? C.green : C.red }, s.svg);
            sc.style.transition = `width ${RM ? 0 : 500}ms`;
            setTimeout(() => sc.setAttribute("width", 100 * v), RM ? 0 : 200 + i * 150);
            text(s.svg, 655, y + 9, v > .5 ? "Sounds like you" : "Doesn't", "ex-small");
            return { line, sc };
          });
          await sleep(900);
          s.rows[2].line.setAttribute("fill", C.red);
        }],
        ["The weak line is rewritten. You approve before anything sends.", async (s) => {
          const r = s.rows[2];
          r.line.setAttribute("fill", "rgba(255,255,255,.75)"); r.sc.setAttribute("fill", C.green); r.sc.setAttribute("width", 100 * .9);
          const g = el("g", { transform: "translate(620,40)" }, s.svg);
          el("rect", { x: -70, y: -18, width: 140, height: 36, rx: 18, fill: C.ink }, g);
          text(g, 0, 6, "Approve & send", "ex-btn", "middle");
          pop(g);
        }],
      ],
    },
  };

  // ------------------------------------------------------------------ modal
  let token = 0;
  function close() {
    token++;
    const m = document.getElementById("howto");
    m?.classList.remove("open");
    setTimeout(() => m && (m.hidden = true), 250);
  }

  async function play(kind, startAt = 0) {
    const my = ++token, scene = SCENES[kind];
    const m = document.getElementById("howto");
    m.querySelector(".hx-title").textContent = scene.title;
    m.querySelector(".hx-adj").textContent = scene.adjusts;
    const dots = m.querySelector(".hx-dots");
    dots.innerHTML = scene.steps.map((_, i) => `<button data-i="${i}" aria-label="Step ${i + 1}"></button>`).join("");
    dots.querySelectorAll("button").forEach((b) => b.onclick = () => play(kind, +b.dataset.i));
    while (my === token) {
      const stage = m.querySelector(".hx-stage");
      stage.innerHTML = "";
      const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, class: "hx-svg" }, stage);
      const s = { svg };
      // fast-forward silently to the chosen step
      for (let i = 0; i < scene.steps.length && my === token; i++) {
        const [cap, run] = scene.steps[i];
        dots.querySelectorAll("button").forEach((b, j) => b.classList.toggle("on", j === i));
        const capEl = m.querySelector(".hx-cap");
        capEl.textContent = cap;
        capEl.animate?.([{ opacity: 0, transform: "translateY(6px)" }, { opacity: 1, transform: "none" }], { duration: 350 });
        FAST = i < startAt;
        await run(s);
        FAST = false;
        if (i < startAt) continue;
        await sleep(2300);
      }
      startAt = 0;
      await sleep(900);
    }
  }

  window.Explainers = {
    open(kind) {
      let m = document.getElementById("howto");
      if (!m) {
        m = document.createElement("div");
        m.id = "howto"; m.hidden = true;
        m.innerHTML = `<div class="hx-back"></div><div class="hx-card" role="dialog" aria-modal="true" aria-label="How it works">
          <div class="hx-head"><div><h3 class="hx-title"></h3><p class="hx-adj"></p></div><button class="btn icon hx-x" aria-label="Close">✕</button></div>
          <div class="hx-stage"></div><p class="hx-cap"></p><div class="hx-dots"></div></div>`;
        document.body.appendChild(m);
        m.querySelector(".hx-back").onclick = close;
        m.querySelector(".hx-x").onclick = close;
        addEventListener("keydown", (e) => { if (e.key === "Escape") close(); });
      }
      m.hidden = false;
      requestAnimationFrame(() => m.classList.add("open"));
      play(kind);
    },
    close,
  };
})();
