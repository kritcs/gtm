// "How it works" pop-ups: each technique as a short animated scene made of color blocks.
(function () {
  const RM = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const NS = "http://www.w3.org/2000/svg";
  const C = { blue: "#3987e5", gray: "#2c2b3d", yellow: "#eda100", orange: "#d95926", green: "#22c55e", red: "#ef4444", ink: "#f5f5f4", dim: "rgba(255,255,255,.45)" };
  const W = 760, H = 344;
  const sleep = (ms) => new Promise((r) => setTimeout(r, RM ? 0 : ms));

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
  const SCENES = {
    bottle: {
      title: "Talk us through your instinct",
      steps: [
        ["You talk about real customers.", (s) => {
          s.chips = ["Nobody trusted the numbers", "They had just raised", "The VP Finance signs"].map((t, i) => {
            const g = el("g", { class: "ex-chip" }, s.svg);
            g.setAttribute("transform", `translate(40,${80 + i * 62})`);
            el("rect", { x: 0, y: 0, width: 230, height: 40, rx: 20, fill: "rgba(255,255,255,.06)", stroke: "rgba(255,255,255,.3)" }, g);
            text(g, 16, 26, `“${t}”`, "ex-t");
            pop(g, i * 180);
            return g;
          });
          const shape = "M380,70 h80 v26 q0,14 22,30 v150 q0,22 -22,22 h-80 q-22,0 -22,-22 v-150 q22,-16 22,-30 z";
          const clip = el("clipPath", { id: "bclip" }, el("defs", {}, s.svg));
          el("path", { d: shape }, clip);
          s.liquid = el("rect", { x: 350, y: 298, width: 140, height: 0, fill: "rgba(57,135,229,.55)", "clip-path": "url(#bclip)" }, s.svg);
          s.liquid.style.transition = `y ${RM ? 0 : 1100}ms cubic-bezier(.2,.8,.2,1), height ${RM ? 0 : 1100}ms cubic-bezier(.2,.8,.2,1)`;
          s.bottle = el("path", { d: shape, class: "ex-bottle" }, s.svg);
        }],
        ["Each sentence becomes a belief with a confidence score.", async (s) => {
          s.chips.forEach((g, i) => g.animate?.([{ transform: `translate(40px,${80 + i * 62}px)`, opacity: 1 }, { transform: `translate(390px,${200 + i * 10}px) scale(.3)`, opacity: 0 }],
            { duration: RM ? 0 : 700, delay: RM ? 0 : i * 150, fill: "forwards", easing: "cubic-bezier(.5,0,.2,1)" }));
          setTimeout(() => { s.liquid.setAttribute("y", 170); s.liquid.setAttribute("height", 130); }, RM ? 0 : 300);
          await sleep(800);
          s.bars = [["Pain", .82, "sure"], ["Trigger", .5, "fairly sure"], ["Buyer", .3, "a hunch"]].map(([l, v, how], i) => {
            const y = 110 + i * 60;
            text(s.svg, 530, y - 8, `${l} · ${how}`, "ex-small");
            el("rect", { x: 530, y, width: 200, height: 18, rx: 9, fill: "rgba(255,255,255,.08)" }, s.svg);
            const bar = el("rect", { x: 530, y, width: 0, height: 18, rx: 9, fill: C.blue }, s.svg);
            bar.style.transition = `width ${RM ? 0 : 800}ms cubic-bezier(.2,.8,.2,1)`;
            requestAnimationFrame(() => bar.setAttribute("width", 200 * v));
            return { bar, v };
          });
        }],
        ["A reply that comes back raises the belief that found it.", async (s) => {
          const d = el("circle", { cx: 630, cy: 40, r: 9, fill: C.green }, s.svg);
          d.animate?.([{ transform: "translateY(0)" }, { transform: "translateY(79px)" }], { duration: RM ? 0 : 600, fill: "forwards", easing: "cubic-bezier(.5,0,.8,.6)" });
          await sleep(650); fade(d, 0, 300);
          s.bars[0].bar.setAttribute("width", 200 * .92); fill(s.bars[0].bar, C.green);
        }],
        ["Silence or a no lowers it. Your gut gets graded by outcomes.", async (s) => {
          const d = el("circle", { cx: 600, cy: 40, r: 9, fill: C.red }, s.svg);
          d.animate?.([{ transform: "translateY(0)" }, { transform: "translateY(199px)" }], { duration: RM ? 0 : 700, fill: "forwards", easing: "cubic-bezier(.5,0,.8,.6)" });
          await sleep(750); fade(d, 0, 300);
          s.bars[2].bar.setAttribute("width", 200 * .16); fill(s.bars[2].bar, C.red);
        }],
      ],
    },

    engine: {
      title: "Dual past-future engine",
      steps: [
        ["Backward: we replay what already happened.", (s) => {
          text(s.svg, 40, 40, "Past", "ex-h"); text(s.svg, 420, 40, "Today", "ex-h");
          el("line", { x1: 390, x2: 390, y1: 24, y2: 320, class: "ex-div" }, s.svg);
          s.past = grid(s.svg, 40, 70, 8, 6, 30, 9);
          s.cust = [9, 20, 34];
          s.past.forEach((b, i) => pop(b, i * 8));
          s.cust.forEach((i) => fill(s.past[i], C.blue));
          text(s.svg, 40, 334, "Blue: your customers. Gray: similar companies. Yellow: has the trait.", "ex-small");
        }],
        ["A trait your customers share, and others rarely do, is a real pattern.", async (s) => {
          const ring = (b) => { b.style.transition = "stroke-width .4s"; b.setAttribute("stroke", C.yellow); b.setAttribute("stroke-width", 4); };
          s.cust.forEach((i, k) => setTimeout(() => ring(s.past[i]), RM ? 0 : k * 160));
          setTimeout(() => fill(s.past[27], C.yellow), RM ? 0 : 500);
          await sleep(900);
          s.b1 = badge(s.svg, 205, 304, "Real pattern ✓", C.green);
        }],
        ["A trait that's common everywhere could be luck. We throw it out.", async (s) => {
          fade(s.b1, 0, 300);
          s.past.forEach((b, i) => { fill(b, s.cust.includes(i) ? C.blue : C.gray, 300); b.setAttribute("stroke-width", 0); });
          await sleep(500);
          const many = [0, 3, 5, 9, 12, 15, 18, 20, 22, 26, 29, 31, 34, 38, 41, 44, 46];
          many.forEach((i, k) => setTimeout(() => {
            if (s.cust.includes(i)) { s.past[i].setAttribute("stroke", C.yellow); s.past[i].setAttribute("stroke-width", 4); }
            else fill(s.past[i], C.yellow);
          }, RM ? 0 : k * 50));
          await sleep(1100);
          s.b2 = badge(s.svg, 205, 304, "Could be luck ✕", C.red);
          await sleep(700);
          many.forEach((i) => { fill(s.past[i], s.cust.includes(i) ? C.blue : C.gray, 400); s.past[i].setAttribute("stroke-width", 0); });
          fade(s.b2, 0, 400);
        }],
        ["Forward: we scan today's market for the real pattern.", async (s) => {
          s.now = grid(s.svg, 420, 70, 8, 6, 30, 9);
          s.now.forEach((b, i) => pop(b, i * 6));
          await sleep(500);
          const line = el("rect", { x: 412, y: 62, width: 3, height: 236, fill: C.ink, opacity: .8 }, s.svg);
          line.animate?.([{ transform: "translateX(0)" }, { transform: "translateX(310px)" }], { duration: RM ? 0 : 1400, fill: "forwards", easing: "linear" });
          s.hits = [2, 11, 17, 24, 30, 37, 45];
          s.hits.forEach((i) => setTimeout(() => fill(s.now[i], C.yellow), RM ? 0 : ((i % 8) / 8) * 1400));
          await sleep(1500); fade(line, 0, 300);
        }],
        ["It also finds companies that describe your problem in different words.", async (s) => {
          s.diff = [6, 15, 39, 47];
          s.diff.forEach((i, k) => setTimeout(() => fill(s.now[i], C.orange), RM ? 0 : k * 160));
          const chip = el("g", {}, s.svg);
          el("rect", { x: 560, y: 8, width: 170, height: 30, rx: 15, fill: "rgba(217,89,38,.15)", stroke: C.orange }, chip);
          text(chip, 645, 28, "“freight audit”", "ex-t", "middle");
          pop(chip);
        }],
        ["Both directions agree on who to contact next.", async (s) => {
          s.now.forEach((b, i) => { if (!s.hits.includes(i) && !s.diff.includes(i)) fade(b, .18, 500); });
          s.past.forEach((b) => fade(b, .25, 500));
          await sleep(500);
          [2, 11, 6, 24, 15].forEach((i, k) => {
            const b = s.now[i], x = +b.getAttribute("x") + 15, y = +b.getAttribute("y") + 21;
            const t = text(s.svg, x, y, String(k + 1), "ex-rank", "middle");
            pop(t, k * 140);
            b.setAttribute("stroke", "#fff"); b.setAttribute("stroke-width", 2);
          });
        }],
      ],
    },

    voice: {
      title: "Unmistakably yours",
      steps: [
        ["We learn your voice from the emails you've already sent.", (s) => {
          text(s.svg, 40, 40, "Your sent emails", "ex-h");
          for (let i = 0; i < 4; i++) {
            const g = el("g", { transform: `translate(${40 + i * 10},${70 + i * 12})` }, s.svg);
            el("rect", { width: 170, height: 120, rx: 12, fill: "#17162a", stroke: "rgba(255,255,255,.2)" }, g);
            for (let l = 0; l < 5; l++) el("rect", { x: 16, y: 18 + l * 18, width: 138 - (l % 2) * 40, height: 6, rx: 3, fill: "rgba(255,255,255,.25)" }, g);
            pop(g, i * 150);
          }
          s.model = el("rect", { x: 300, y: 120, width: 120, height: 90, rx: 18, fill: "#241f63", stroke: C.blue, "stroke-width": 2 }, s.svg);
          text(s.svg, 360, 160, "Your voice", "ex-t", "middle"); text(s.svg, 360, 180, "model", "ex-t", "middle");
          s.model.animate?.([{ opacity: .5 }, { opacity: 1 }], { duration: 1200, iterations: Infinity, direction: "alternate" });
          el("path", { d: "M235,165 L292,165", class: "ex-arrow" }, s.svg);
        }],
        ["It drafts each email the way you would write it.", async (s) => {
          text(s.svg, 470, 40, "Draft", "ex-h");
          el("path", { d: "M428,165 L462,165", class: "ex-arrow" }, s.svg);
          s.lines = [];
          for (let i = 0; i < 4; i++) {
            const y = 76 + i * 46;
            const line = el("rect", { x: 470, y, width: 0, height: 10, rx: 5, fill: "rgba(255,255,255,.7)" }, s.svg);
            line.style.transition = `width ${RM ? 0 : 500}ms`;
            setTimeout(() => line.setAttribute("width", [190, 160, 200, 120][i]), RM ? 0 : i * 350);
            s.lines.push({ line, y });
          }
          await sleep(1500);
        }],
        ["Every line gets a score for how likely you are to have written it.", async (s) => {
          s.scores = s.lines.map(({ y }, i) => {
            const v = [.92, .88, .31, .9][i];
            el("rect", { x: 680, y: y - 2, width: 50, height: 14, rx: 7, fill: "rgba(255,255,255,.08)" }, s.svg);
            const b = el("rect", { x: 680, y: y - 2, width: 0, height: 14, rx: 7, fill: v > .5 ? C.green : C.red }, s.svg);
            b.style.transition = `width ${RM ? 0 : 600}ms, fill ${RM ? 0 : 500}ms`;
            setTimeout(() => b.setAttribute("width", 50 * v), RM ? 0 : i * 200);
            return b;
          });
          await sleep(1000);
          s.flag = text(s.svg, 470, 196, "Doesn't sound like you", "ex-flag");
          pop(s.flag);
        }],
        ["Lines that don't sound like you get rewritten before you see them.", async (s) => {
          const l = s.lines[2].line;
          l.setAttribute("width", 0); fade(s.flag, 0, 300);
          await sleep(500);
          l.setAttribute("width", 175);
          s.scores[2].setAttribute("width", 50 * .9); s.scores[2].setAttribute("fill", C.green);
        }],
        ["You approve. Nothing sends without you.", async (s) => {
          const g = el("g", { transform: "translate(565,300)" }, s.svg);
          el("rect", { x: -70, y: -18, width: 140, height: 36, rx: 18, fill: C.ink }, g);
          const t = text(g, 0, 6, "Approve & send", "ex-btn", "middle");
          g.style.transformBox = "fill-box"; g.style.transformOrigin = "center"; pop(g);
          void t;
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
        await run(s);
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
          <div class="hx-head"><h3 class="hx-title"></h3><button class="btn icon hx-x" aria-label="Close">✕</button></div>
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
