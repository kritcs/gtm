// Peem's proposed architecture, made visible: the 8-step loop, the graph, and how learning works.
// Each step is mapped to what runs in this demo (live numbers) and marked honestly: in demo / partly / roadmap.
(function () {
  const RM = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const STATUS = { demo: ["#22c55e", "In demo"], partly: ["#eda100", "Partly in demo"], roadmap: ["#71717a", "Roadmap"] };
  const ICON = (name, size = 18) => `<foreignObject x="${-size / 2}" y="${-size / 2}" width="${size}" height="${size}">
    <div xmlns="http://www.w3.org/1999/xhtml" class="fo-i" style="width:${size}px;height:${size}px;--i:url(/static/vendor/icons/${name}.svg)"></div></foreignObject>`;
  const fmt = (v) => typeof v === "number" ? v.toLocaleString() : v;

  // From docs/peem-pipeline.txt. `kpis` pick live numbers from /system for this step.
  const PHASES = [
    { name: "understand", steps: [0, 1], color: "#3987e5" },
    { name: "decide", steps: [2, 3, 4], color: "#9085e9" },
    { name: "act", steps: [5, 6], color: "#eda100" },
    { name: "learn", steps: [7], color: "#22c55e" },
  ];
  const STEPS = [
    { n: 1, title: "Intake", icon: "mic", q: "What do you sell, and who do you think buys it?", status: "demo",
      math: "“fintech, 70%, quite sure” → Beta(14, 6)", kpis: [["bottle", "rules", "rules"], ["bottle", "mean trust", "mean trust"], ["interview", "answers", "answers"]] },
    { n: 2, title: "Market graph", icon: "network", q: "Map the market.", status: "partly",
      math: "Company · Person · Product · Competitor · Signal", kpis: [["posts", "companies", "companies"], ["graph", "adoption edges", "adoption edges"], ["review", "rejected", "claims rejected"]] },
    { n: 3, title: "Score", icon: "target", q: "Who is most likely to become a big, lasting customer?", status: "partly",
      math: "E[value] = prior × similarity × contract size", kpis: [["rank", "ranked", "companies ranked"], ["rank", "top", "top account"], ["model", "BPR loss", "training loss"]] },
    { n: 4, title: "Discover + enrich", icon: "users", q: "Find the real people.", status: "partly",
      math: "wallet check before each batch", kpis: [["credits", "left", "credits left"], ["credits", "spent on", "enriched"]] },
    { n: 5, title: "Signal packet", icon: "package", q: "Why reach out now?", status: "partly",
      math: "signals → packet → draft", kpis: [["hit", "feature", "top signal"], ["hit", "q", "odds of luck"], ["disc", "term", "new segment term"]] },
    { n: 6, title: "Draft + review", icon: "square-pen", q: "Write it, then check it.", status: "partly",
      math: "every sentence → signal → source", kpis: [["draft", "drafts", "drafts"], ["draft", "AI flags", "AI flags"]] },
    { n: 7, title: "Approve + send", icon: "send", q: "Human says yes.", status: "demo",
      math: "no send without approval", kpis: [["approve", "approved", "approved"]] },
    { n: 8, title: "Learn", icon: "rotate-ccw", q: "What actually worked?", status: "partly",
      math: "retained = 1 · trial ≈ 0.3 · churned = 0", kpis: [["reply", "replies", "replies"]] },
  ];

  const W = 760, H = 580, CX = 380, CY = 290, R = 228;
  const ang = (i) => -Math.PI / 2 + (i / STEPS.length) * Math.PI * 2;
  const pos = (i, r = R) => [CX + r * Math.cos(ang(i)), CY + r * Math.sin(ang(i))];
  const arc = (a0, a1, r) => {
    const [x0, y0] = [CX + r * Math.cos(a0), CY + r * Math.sin(a0)], [x1, y1] = [CX + r * Math.cos(a1), CY + r * Math.sin(a1)];
    return `M${x0},${y0} A${r},${r} 0 ${a1 - a0 > Math.PI ? 1 : 0} 1 ${x1},${y1}`;
  };

  class SystemView {
    constructor(root) {
      this.root = root; this.live = {}; this.running = false; this.sel = 0; this.panel = "loop"; this.story = 0;
      root.classList.add("wide");
      root.innerHTML = `
        <div class="sys-head">
          <div class="seg" id="sys-seg"><button data-p="loop" class="on">Loop</button><button data-p="graph">Graph</button><button data-p="learn">Learning</button></div>
          <span class="grow"></span>
          <span class="sys-legend">${Object.values(STATUS).map(([c, l]) => `<span><i style="background:${c}"></i>${l}</span>`).join("")}</span>
        </div>
        <div class="sys-body"><div class="sys-stage" id="sys-stage"></div><aside class="sys-drawer" id="sys-drawer"></aside></div>
        <p class="sys-credit">Architecture by Peem Puangmaha</p>`;
      root.querySelectorAll("#sys-seg button").forEach((b) => b.onclick = () => {
        root.querySelectorAll("#sys-seg button").forEach((x) => x.classList.toggle("on", x === b));
        this.show(b.dataset.p);
      });
    }

    setLive(live) { this.live = live || {}; }

    show(p) {
      this.panel = p; this.running = false;
      const st = this.root.querySelector("#sys-stage");
      if (p === "loop") this.loop(st);
      if (p === "graph") this.graph(st);
      if (p === "learn") this.learn(st);
      st.animate?.([{ opacity: 0 }, { opacity: 1 }], { duration: 400 });
    }

    // ------------------------------------------------------------------ 1. the loop
    loop(st) {
      const gap = (Math.PI * 2) / STEPS.length / 2;
      const phases = PHASES.map((ph) => {
        const a0 = ang(ph.steps[0]) - gap + 0.05, a1 = ang(ph.steps.at(-1)) + gap - 0.05, am = (a0 + a1) / 2;
        const [lx, ly] = [CX + (R - 46) * Math.cos(am), CY + (R - 46) * Math.sin(am)];
        return `<path d="${arc(a0, a1, R)}" class="phase" style="stroke:${ph.color}"/>
          <text x="${lx}" y="${ly}" class="phase-lbl" style="fill:${ph.color}">${cap(ph.name)}</text>`;
      }).join("");
      const nodes = STEPS.map((s, i) => {
        const [x, y] = pos(i), [col] = STATUS[s.status];
        const out = pos(i, R + 30);
        const anchor = Math.abs(out[0] - CX) < 20 ? "middle" : out[0] > CX ? "start" : "end";
        return `<g class="step" data-i="${i}" tabindex="0" transform="translate(${x},${y})">
          <circle class="halo" r="38"/><circle class="dot" r="26"/>${ICON(s.icon, 20)}
          <circle cx="19" cy="-19" r="5" fill="${col}" stroke="#0e0e0f" stroke-width="2.5"/></g>
          <text x="${out[0]}" y="${out[1] + 5}" text-anchor="${anchor}" class="lbl" data-i="${i}">${esc(s.title)}</text>`;
      }).join("");
      const [lx, ly] = pos(7, R - 28), [sx, sy] = pos(2, R - 28);
      st.innerHTML = `<div class="loop-wrap">
        <svg viewBox="0 0 ${W} ${H}" class="sys-svg loop-svg">
          <circle cx="${CX}" cy="${CY}" r="${R}" class="ring"/>
          <circle cx="${CX}" cy="${CY}" r="${R}" class="ring-flow"/>
          ${phases}
          <path id="back" d="M${lx},${ly} Q${CX},${CY - 120} ${sx},${sy}" class="back"/>
          <text x="${CX}" y="${CY - 74}" class="back-lbl">Outcomes flow back</text>
          <text x="${CX}" y="${CY + 22}" class="center-n" id="cn">1</text>
          <text x="${CX}" y="${CY + 58}" class="center-sub" id="cs">Forward pass</text>
          ${nodes}
          <circle r="8" class="token" id="tok" cx="${pos(0)[0]}" cy="${pos(0)[1]}" opacity="0"/>
          <circle r="8" class="token back-tok" id="btok" opacity="0"/>
        </svg>
        <div class="loop-detail" id="loop-detail"></div>
      </div>`;
      st.querySelectorAll(".step, .lbl").forEach((g) => {
        g.onclick = () => { this.running = false; this.select(+g.dataset.i); };
        g.onkeydown = (e) => { if (e.key === "Enter") g.onclick(); };
      });
      this.select(this.sel ?? 0);
    }

    select(i) {
      this.sel = i;
      this.root.querySelectorAll(".step").forEach((g) => g.classList.toggle("on", +g.dataset.i === i));
      this.root.querySelectorAll(".lbl").forEach((g) => g.classList.toggle("on", +g.dataset.i === i));
      const s = STEPS[i], [col, lab] = STATUS[s.status];
      const ph = PHASES.find((p) => p.steps.includes(i));
      const cn = this.root.querySelector("#cn"), cs = this.root.querySelector("#cs");
      if (cn) { cn.textContent = s.n; cs.textContent = cap(ph.name); cs.style.fill = ph.color; }
      const kpis = s.kpis.map(([sec, key, label]) => {
        let v = this.live[sec]?.[key];
        if (key === "q" && v) v = oddsText(v).replace(" odds of luck", "");
        if (key === "feature" && v) v = { "role:revops": "RevOps hire" }[v] || v;
        return [cap(label), v];
      }).filter(([, v]) => v !== undefined && v !== null && v !== "");
      const d = this.root.querySelector("#loop-detail");
      if (!d) return;
      d.innerHTML = `
        <p class="pill"><i style="background:${col}"></i>${lab}</p>
        <h2>${esc(s.title)}</h2>
        <p class="q">${esc(s.q)}</p>
        <p class="formula">${esc(s.math)}</p>
        <div class="dkpis">${kpis.map(([l, v]) => `<div><b>${esc(fmt(v))}</b><span>${esc(l)}</span></div>`).join("")}</div>
        <div class="d-nav"><button class="btn icon" id="d-prev" aria-label="Previous step"><i class="i flip-x" style="--i:url(/static/vendor/icons/arrow-right.svg)"></i></button>
          <button class="btn icon" id="d-next" aria-label="Next step"><i class="i" style="--i:url(/static/vendor/icons/arrow-right.svg)"></i></button>
          <span class="grow"></span>
          <button class="btn primary" id="sys-play"><i class="i" style="--i:url(/static/vendor/icons/play.svg)"></i>Run the loop</button></div>`;
      d.querySelector("#d-prev").onclick = () => { this.running = false; this.select((i + 7) % 8); };
      d.querySelector("#d-next").onclick = () => { this.running = false; this.select((i + 1) % 8); };
      d.querySelector("#sys-play").onclick = () => this.run();
      d.animate?.([{ opacity: 0, transform: "translateY(6px)" }, { opacity: 1, transform: "none" }], { duration: 300 });
    }

    async run() {
      if (this.running) return;
      this.running = true;
      const tok = this.root.querySelector("#tok"), btok = this.root.querySelector("#btok"), back = this.root.querySelector("#back");
      const wait = (ms) => new Promise((r) => setTimeout(r, RM ? 0 : ms));
      const move = (from, to, ms) => new Promise((res) => {
        const a0 = ang(from); let a1 = ang(to); if (a1 < a0) a1 += Math.PI * 2;
        if (RM) { const [x, y] = pos(to); tok.setAttribute("cx", x); tok.setAttribute("cy", y); return res(); }
        const t0 = performance.now();
        const f = (now) => {
          const t = Math.min(1, (now - t0) / ms), e = t < .5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2, a = a0 + (a1 - a0) * e;
          tok.setAttribute("cx", CX + R * Math.cos(a)); tok.setAttribute("cy", CY + R * Math.sin(a));
          t < 1 && this.running ? requestAnimationFrame(f) : res();
        };
        requestAnimationFrame(f);
      });
      tok.setAttribute("cx", pos(0)[0]); tok.setAttribute("cy", pos(0)[1]); tok.setAttribute("opacity", 1);
      for (let i = 0; i < STEPS.length && this.running; i++) {
        this.select(i);
        await wait(1600);
        if (i < STEPS.length - 1 && this.running) await move(i, i + 1, 600);
      }
      if (!this.running) { tok.setAttribute("opacity", 0); return; }
      back.classList.add("live"); tok.setAttribute("opacity", 0); btok.setAttribute("opacity", 1);
      const cs = this.root.querySelector("#cs"); cs.textContent = "Backward pass"; cs.style.fill = "#22c55e";
      const L = back.getTotalLength(), t0 = performance.now();
      await new Promise((res) => {
        const f = (now) => {
          const t = Math.min(1, (now - t0) / (RM ? 1 : 1800)), p = back.getPointAtLength(L * t);
          btok.setAttribute("cx", p.x); btok.setAttribute("cy", p.y);
          t < 1 ? requestAnimationFrame(f) : res();
        };
        requestAnimationFrame(f);
      });
      this.select(2);
      await wait(1400);
      back.classList.remove("live"); btok.setAttribute("opacity", 0);
      this.running = false;
    }

    // ------------------------------------------------------------------ 2. the graph
    graph(st) {
      const L = this.live;
      const N = { // id: [label, icon, x, y, zone, status, count]
        person: ["Person", "user", 120, 110, "public", "roadmap"],
        company: ["Company", "building-2", 330, 150, "public", "demo", L.posts?.companies],
        product: ["Product", "package", 600, 95, "public", "demo", L.graph?.tools],
        signal: ["Signal", "radar", 600, 235, "public", "demo", L.graph?.features],
        competitor: ["Competitor", "swords", 870, 150, "public", "roadmap"],
        outcome: ["Outcome", "reply", 330, 470, "private", "partly", L.reply?.replies],
        rule: ["Persona rule", "wand-sparkles", 600, 470, "private", "partly", L.bottle?.rules],
        email: ["Founder email", "mail", 870, 470, "private", "partly", L.draft?.drafts],
      };
      const E = { // id: [from, to, label]
        uses: ["company", "product", "Uses"], hiring: ["company", "signal", "Hiring for"], competes: ["product", "competitor", "Competes with"],
        works: ["person", "company", "Works at"], outcome: ["outcome", "company", "Outcome of"], credits: ["outcome", "signal", "Credits"],
        targets: ["rule", "company", "Targets"], trains: ["email", "rule", "Trains"],
      };
      const STORIES = [
        { edges: ["uses"], text: "Companies use products. Every edge cites the post it came from." },
        { edges: ["hiring"], text: "A hiring post becomes a signal, like a RevOps hire." },
        { edges: ["outcome", "credits"], text: "A reply becomes an outcome and credits the signal that found it." },
        { edges: ["targets", "trains"], text: "Your rules and emails stay private, and they still shape who you target." },
      ];
      const edgePath = (a, b) => {
        const A = N[a], B = N[b], mx = (A[2] + B[2]) / 2, my = (A[3] + B[3]) / 2;
        const bend = Math.abs(A[3] - B[3]) > 150 ? 0 : -24;
        return [`M${A[2]},${A[3]} Q${mx},${my + bend} ${B[2]},${B[3]}`, mx, my + bend / 2];
      };
      const edges = Object.entries(E).map(([id, [a, b, l]]) => {
        let [d, mx, my] = edgePath(a, b); const road = N[a][5] === "roadmap" || N[b][5] === "roadmap";
        if (Math.abs(my - 335) < 22) my = my < 335 ? 300 : 372;
        return `<path id="ge-${id}" d="${d}" class="gedge2 ${road ? "roadmap" : ""}"/>
          <g class="echip" id="ec-${id}" transform="translate(${mx},${my})"><rect x="${-l.length * 3.4 - 8}" y="-10" width="${l.length * 6.8 + 16}" height="20" rx="10"/><text y="4">${l}</text></g>`;
      }).join("");
      const nodes = Object.entries(N).map(([id, [l, ic, x, y, zone, s, c]]) => `<g class="gnode2 ${zone} ${s}" id="gn-${id}" transform="translate(${x},${y})">
          <rect x="-82" y="-30" width="164" height="60" rx="16"/>
          <g transform="translate(-54,0)">${ICON(ic, 20)}</g>
          <text x="-30" y="${c != null ? -3 : 5}" class="gt">${l}</text>
          ${c != null ? `<text x="-30" y="16" class="gc">${Number(c).toLocaleString()}</text>` : `<text x="-30" y="22" class="gc">Roadmap</text>`}</g>`).join("");
      st.innerHTML = `<div class="graph2">
        <svg viewBox="0 0 1000 570" class="sys-svg">
          <rect x="8" y="10" width="984" height="300" rx="22" class="zone public"/>
          <text x="32" y="42" class="zone-lbl">Public market · shared by everyone</text>
          <rect x="8" y="360" width="984" height="200" rx="22" class="zone private"/>
          <text x="32" y="392" class="zone-lbl private">Your private data · only you</text>
          <g transform="translate(32,335)">${ICON("lock", 16)}</g>
          <text x="50" y="340" class="lock-lbl">Private data never leaves your layer</text>
          <line x1="300" x2="980" y1="335" y2="335" class="boundary"/>
          ${edges}${nodes}
          <circle r="6" class="gtok" id="gtok" opacity="0"/>
        </svg>
        <div class="story"><p class="story-text" id="story-text"></p>
          <div class="story-dots">${STORIES.map((_, i) => `<button data-s="${i}" aria-label="Story ${i + 1}"></button>`).join("")}</div></div>
        <div class="gkpis">
          <div><b>${fmt(L.posts?.companies ?? "—")}</b><span>Companies</span></div>
          <div><b>${fmt(L.graph?.["adoption edges"] ?? "—")}</b><span>Adoption edges</span></div>
          <div><b>${fmt(L.review?.kept ?? "—")}</b><span>Facts kept</span></div>
          <div><b>${fmt(L.review?.rejected ?? "—")}</b><span>Rejected by Reviewer</span></div>
        </div></div>`;
      const play = (k) => {
        this.story = k;
        const s = STORIES[k];
        st.querySelectorAll(".gedge2, .echip").forEach((el) => el.classList.remove("hot", "dim"));
        st.querySelectorAll(".gnode2").forEach((el) => el.classList.add("dim"));
        Object.keys(E).forEach((id) => {
          const on = s.edges.includes(id);
          st.querySelector("#ge-" + id).classList.add(on ? "hot" : "dim");
          st.querySelector("#ec-" + id).classList.add(on ? "hot" : "dim");
          if (on) E[id].slice(0, 2).forEach((n) => st.querySelector("#gn-" + n).classList.remove("dim"));
        });
        const t = st.querySelector("#story-text");
        t.textContent = s.text;
        t.animate?.([{ opacity: 0, transform: "translateY(6px)" }, { opacity: 1, transform: "none" }], { duration: 400 });
        st.querySelectorAll(".story-dots button").forEach((b) => b.classList.toggle("on", +b.dataset.s === k));
        // token walks the story's path
        const tok = st.querySelector("#gtok"), paths = s.edges.map((id) => st.querySelector("#ge-" + id));
        let j = 0, t0 = performance.now();
        tok.setAttribute("opacity", 1);
        const f = (now) => {
          if (this.panel !== "graph" || this.story !== k) return;
          const p = paths[j], Lp = p.getTotalLength(), u = Math.min(1, (now - t0) / 1100), pt = p.getPointAtLength(Lp * u);
          tok.setAttribute("cx", pt.x); tok.setAttribute("cy", pt.y);
          if (u >= 1) { j = (j + 1) % paths.length; t0 = now; }
          if (!RM) requestAnimationFrame(f);
        };
        requestAnimationFrame(f);
      };
      st.querySelectorAll(".story-dots button").forEach((b) => b.onclick = () => { clearInterval(this.storyTimer); play(+b.dataset.s); });
      clearInterval(this.storyTimer);
      play(0);
      this.storyTimer = setInterval(() => { if (this.panel !== "graph") return clearInterval(this.storyTimer); play((this.story + 1) % STORIES.length); }, 4200);
    }

    learn(st) {
      const segs = [{ name: "fintech", belief: 0.7, conf: "quite sure", k: 20 }, { name: "healthcare", belief: 0.5, conf: "a hunch", k: 4 }];
      const STAGES = [["0", "outcomes", "priors + graph similarity"], ["1–99", "outcomes", "Beta updates"],
                      ["100+", "outcomes", "per-startup fine-tune"], ["many", "tenants", "shared model"]];
      st.innerHTML = `
        <div class="learn2">
          <div class="l-curves">
            ${segs.map((s, i) => `<div class="l-card" id="bc${i}">
              <p class="l-say">“${s.name}, ${Math.round(s.belief * 100)}%, ${s.conf}”</p>
              <svg viewBox="0 0 400 190" class="beta2">
                <path class="bprior"/><path class="bfill"/><path class="bline"/><line class="bmean" y1="10" y2="160"/>
                <line x1="10" x2="390" y1="160" y2="160" class="baxis"/>
                <text x="10" y="180" class="btick">0</text><text x="200" y="180" class="btick" text-anchor="middle">0.5</text><text x="390" y="180" class="btick" text-anchor="end">1</text>
              </svg>
              <div class="l-foot"><div><span class="l-mean">0.00</span><span class="l-ab"></span></div><div class="l-dots"></div></div>
            </div>`).join("")}
          </div>
          <div class="l-controls">
            <div class="seg2"><button data-o="1"><i class="d ok"></i>Retained</button><button data-o="0.3"><i class="d mid"></i>Trial</button><button data-o="0"><i class="d bad"></i>Churned</button></div>
            <button class="btn ghost" id="lreset">Reset</button>
          </div>
          <div class="l-bottom">
            <div class="l-panel">
              <p class="l-h">score = logit(prior) + correction</p>
              <div id="wbars"></div>
            </div>
            <div class="l-panel">
              <p class="l-h">Maturity</p>
              <div class="stepper">${STAGES.map(([n, u, m], i) => `<div class="stp" data-i="${i}"><span class="stp-dot"></span><b>${n} <small>${u}</small></b><span>${m}</span></div>`).join("")}</div>
            </div>
          </div>
        </div>`;
      const state = segs.map((s) => ({ a: s.belief * s.k, b: (1 - s.belief) * s.k, a0: s.belief * s.k, b0: (1 - s.belief) * s.k }));
      let outcomes = 0;
      const lgamma = (z) => {
        const g = 7, c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
          12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
        if (z < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * z)) - lgamma(1 - z);
        z -= 1; let x = c[0]; for (let i = 1; i < g + 2; i++) x += c[i] / (z + i);
        const t = z + g + 0.5; return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
      };
      const pdf = (x, a, b) => Math.exp((a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x) - (lgamma(a) + lgamma(b) - lgamma(a + b)));
      const curve = (a, b) => Array.from({ length: 119 }, (_, j) => pdf((j + 1) / 120, a, b));
      const toPath = (ys, mx) => "M" + ys.map((v, j) => `${(10 + (j + 1) / 120 * 380).toFixed(1)},${(160 - (v / mx) * 140).toFixed(1)}`).join(" L");
      const stepper = () => {
        const stage = outcomes >= 100 ? 2 : outcomes >= 1 ? 1 : 0;
        st.querySelectorAll(".stp").forEach((el) => el.classList.toggle("on", +el.dataset.i <= stage));
        st.querySelectorAll(".stp").forEach((el) => el.classList.toggle("cur", +el.dataset.i === stage));
      };
      const draw = (i, from) => {
        const card = st.querySelector("#bc" + i), s = state[i];
        const prior = curve(s.a0, s.b0), target = curve(s.a, s.b), prev = from || target;
        const mx = Math.max(...prior, ...target);
        card.querySelector(".bprior").setAttribute("d", toPath(prior, mx));
        card.querySelector(".bprior").style.opacity = s.a === s.a0 && s.b === s.b0 ? 0 : 1;
        const t0 = performance.now();
        const f = (now) => {
          const t = RM ? 1 : Math.min(1, (now - t0) / 700), e = 1 - Math.pow(1 - t, 3);
          const ys = target.map((v, j) => prev[j] + (v - prev[j]) * e), d = toPath(ys, mx);
          card.querySelector(".bline").setAttribute("d", d);
          card.querySelector(".bfill").setAttribute("d", d + " L390,160 L10,160 Z");
          const mean = s.a / (s.a + s.b), x = 10 + mean * 380;
          card.querySelector(".bmean").setAttribute("x1", x); card.querySelector(".bmean").setAttribute("x2", x);
          card.querySelector(".l-mean").textContent = mean.toFixed(2);
          card.querySelector(".l-ab").textContent = `Beta(${+s.a.toFixed(1)}, ${+s.b.toFixed(1)})`;
          if (t < 1) requestAnimationFrame(f); else card._ys = target;
        };
        requestAnimationFrame(f);
      };
      state.forEach((_, i) => draw(i));
      stepper();
      st.querySelectorAll("[data-o]").forEach((btn) => btn.onclick = () => {
        const o = +btn.dataset.o;
        outcomes++;
        state.forEach((s, i) => {
          const card = st.querySelector("#bc" + i), prev = card._ys;
          s.a += o; s.b += 1 - o;
          const dot = document.createElement("i");
          dot.className = "d " + (o === 1 ? "ok" : o === 0 ? "bad" : "mid");
          card.querySelector(".l-dots").appendChild(dot);
          dot.animate?.([{ transform: "scale(0)" }, { transform: "scale(1)" }], { duration: 300, easing: "cubic-bezier(.2,.8,.2,1.4)" });
          draw(i, prev);
        });
        stepper();
      });
      st.querySelector("#lreset").onclick = () => this.learn(st);
      fetch("/api/model-proof").then((r) => r.ok ? r.json() : null).then((d) => {
        if (!d) return;
        const ws = Object.entries(d.prior_residual.weights).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 6);
        const mx = Math.max(...ws.map((w) => Math.abs(w[1])));
        st.querySelector("#wbars").innerHTML = ws.map(([f, v]) => `<div class="wbar2"><span>${esc(cap(f))}</span>
          <span class="wtrack"><i style="--w:${(Math.abs(v) / mx * 50).toFixed(1)}%;${v < 0 ? "right:50%" : "left:50%"};background:${v < 0 ? "#d95926" : "#3987e5"}"></i></span>
          <b>${v > 0 ? "+" : ""}${v.toFixed(2)}</b></div>`).join("");
        st.querySelectorAll(".wtrack i").forEach((el, i) => el.animate?.([{ width: "0%" }, { width: el.style.getPropertyValue("--w") }], { duration: 800, delay: i * 80, fill: "both", easing: "cubic-bezier(.2,.8,.2,1)" }));
      });
    }
  }

  window.SystemView = SystemView;
})();
