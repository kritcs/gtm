// Peem's proposed architecture, made visible: the 8-step loop, the graph, and how learning works.
// Each step is mapped to what runs in this demo (live numbers) and marked honestly: in demo / partly / roadmap.
(function () {
  const RM = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const STATUS = { demo: ["#1baf7a", "in demo"], partly: ["#eda100", "partly in demo"], roadmap: ["#71717a", "roadmap"] };

  // From docs/peem-pipeline.txt. `live` names the /system keys that show real numbers for this step.
  const STEPS = [
    { n: 1, key: "intake", title: "Intake", q: "What do you sell, and who do you think buys it?", status: "demo", live: ["interview", "bottle"],
      what: "Founder beliefs per segment become Beta priors.", math: "“fintech, 70%, quite sure” → Beta(14, 6)", layers: ["Band"] },
    { n: 2, key: "graph", title: "Market graph", q: "Map the market.", status: "partly", live: ["posts", "review", "graph"],
      what: "Shared public graph; every fact cited. Founder data stays in a private tenant layer.",
      math: "Company · Person · Product · Competitor · Signal", layers: ["Flexprice graph_query"] },
    { n: 3, key: "score", title: "Score", q: "Who is most likely to become a big, lasting customer?", status: "partly", live: ["model", "rank"],
      what: "Expected retained value; Thompson sampling picks the batch, with exploration.",
      math: "E[value] = prior × graph similarity × contract size", layers: [] },
    { n: 4, key: "enrich", title: "Discover + enrich", q: "Find the real people.", status: "partly", live: ["credits"],
      what: "Cited leads and the buying committee per account. Spend is metered.", math: "wallet check before each batch",
      layers: ["Flexprice lead_enriched"] },
    { n: 5, key: "packet", title: "Signal packet", q: "Why reach out now?", status: "partly", live: ["hit", "disc", "pain"],
      what: "Ranked signals with sources, committee, warm-intro paths, persona rules.", math: "signals → packet → draft", layers: [] },
    { n: 6, key: "draft", title: "Draft + review", q: "Write it, then check it.", status: "partly", live: ["draft"],
      what: "Drafted in the founder's voice from the packet only; each sentence tagged to its signal; Reviewer bounces drafts that skip the top signal.",
      math: "sentence → signal → source", layers: ["Band @Builder @Reviewer"] },
    { n: 7, key: "send", title: "Approve + send", q: "Human says yes.", status: "demo", live: ["approve"],
      what: "Founder approves or edits; Dispatcher sends. Edits become rules and preference pairs.", math: "no send without approval",
      layers: ["Band @Dispatcher", "Flexprice outreach_sent"] },
    { n: 8, key: "learn", title: "Learn", q: "What actually worked?", status: "partly", live: ["reply"],
      what: "Replies, meetings, payments and retention return as value-weighted edges and update priors, signal ranks, persona rules and the voice set.",
      math: "retained = 1 · trial ≈ 0.3 · churned = 0", layers: ["Stripe", "Flexprice meeting_booked · deal_won"] },
  ];

  const W = 760, H = 540, CX = 380, CY = 265, R = 200;
  const pos = (i) => { const a = -Math.PI / 2 + (i / STEPS.length) * Math.PI * 2; return [CX + R * Math.cos(a), CY + R * Math.sin(a)]; };

  class SystemView {
    constructor(root) {
      this.root = root; this.live = {}; this.step = 0; this.running = false; this.sel = null; this.panel = "loop";
      root.innerHTML = `
        <div class="sys-head mono">
          <div class="seg" id="sys-seg"><button data-p="loop" class="on">loop</button><button data-p="graph">graph</button><button data-p="learn">learning</button></div>
          <span class="grow"></span>
          <span class="sys-legend">${Object.values(STATUS).map(([c, l]) => `<span><i style="background:${c}"></i>${l}</span>`).join("")}</span>
        </div>
        <div class="sys-body"><div class="sys-stage" id="sys-stage"></div><aside class="sys-drawer" id="sys-drawer"></aside></div>
        <p class="mono small muted sys-credit">architecture: Peem Puangmaha · docs/peem-pipeline.txt</p>`;
      root.querySelectorAll("#sys-seg button").forEach((b) => b.onclick = () => {
        root.querySelectorAll("#sys-seg button").forEach((x) => x.classList.toggle("on", x === b));
        this.show(b.dataset.p);
      });
    }

    setLive(live) { this.live = live || {}; if (this.panel === "loop" && this.sel != null) this.drawer(this.sel); }

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
      const nodes = STEPS.map((s, i) => {
        const [x, y] = pos(i), [col] = STATUS[s.status];
        const lx = x + (x > CX + 10 ? 30 : x < CX - 10 ? -30 : 0), anchor = x > CX + 10 ? "start" : x < CX - 10 ? "end" : "middle";
        const ly = y + (Math.abs(x - CX) < 10 ? (y < CY ? -32 : 42) : 5);
        return `<g class="step" data-i="${i}" tabindex="0">
          <circle class="halo" cx="${x}" cy="${y}" r="30"/>
          <circle class="dot" cx="${x}" cy="${y}" r="21"/>
          <text x="${x}" y="${y + 5}" class="num">${s.n}</text>
          <circle cx="${x + 15}" cy="${y - 15}" r="4.5" fill="${col}" stroke="#1e1e20" stroke-width="2"/>
          <text x="${lx}" y="${ly}" text-anchor="${anchor}" class="lbl">${esc(s.title)}</text></g>`;
      }).join("");
      const [lx, ly] = pos(7), [sx, sy] = pos(2);
      st.innerHTML = `<svg viewBox="0 0 ${W} ${H}" class="sys-svg">
        <defs><marker id="ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="rgba(255,255,255,.35)"/></marker>
              <marker id="ahg" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#1baf7a"/></marker></defs>
        <circle cx="${CX}" cy="${CY}" r="${R}" class="ring"/>
        <circle cx="${CX}" cy="${CY}" r="${R}" class="ring-flow"/>
        <path id="back" d="M${lx},${ly} Q${CX},${CY - 110} ${sx},${sy}" class="back" marker-end="url(#ahg)"/>
        <text x="${CX}" y="${CY + 40}" class="center-q" id="cq">forward pass</text>
        <text x="${CX}" y="${CY + 62}" class="center-sub" id="cs">find and contact the right accounts</text>
        <text x="${CX}" y="${CY - 66}" class="back-lbl">backward pass: Learn → Score</text>
        ${nodes}
        <circle r="7" class="token" id="tok" cx="${pos(0)[0]}" cy="${pos(0)[1]}"/>
        <circle r="7" class="token back-tok" id="btok" cx="${lx}" cy="${ly}" opacity="0"/>
      </svg>
      <div class="sys-controls"><button class="btn" id="sys-play"><i class="i" style="--i:url(/static/vendor/icons/play.svg)"></i>Run the loop</button></div>`;
      st.querySelectorAll(".step").forEach((g) => {
        g.onclick = () => { this.running = false; this.select(+g.dataset.i); };
        g.onkeydown = (e) => { if (e.key === "Enter") g.onclick(); };
      });
      st.querySelector("#sys-play").onclick = () => this.run();
      this.select(this.sel ?? 0);
    }

    select(i) {
      this.sel = i;
      this.root.querySelectorAll(".step").forEach((g) => g.classList.toggle("on", +g.dataset.i === i));
      const s = STEPS[i];
      const q = this.root.querySelector("#cq"), sub = this.root.querySelector("#cs");
      if (q) { q.textContent = `${s.n} · ${s.title}`; sub.textContent = s.q; }
      this.drawer(i);
    }

    drawer(i) {
      const s = STEPS[i], [col, lab] = STATUS[s.status];
      const live = s.live.flatMap((k) => Object.entries(this.live[k] || {})
        .filter(([, v]) => v !== null && v !== undefined && typeof v !== "object" && typeof v !== "boolean").slice(0, 3))
        .slice(0, 6).map(([a, b]) => `<tr><td>${esc(a)}</td><td>${esc(typeof b === "number" ? b.toLocaleString() : b)}</td></tr>`).join("");
      this.root.querySelector("#sys-drawer").innerHTML = `
        <p class="mono small"><i class="sdot" style="background:${col}"></i>${lab}</p>
        <h3>${s.n} · ${esc(s.title)}</h3>
        <p class="q">${esc(s.q)}</p>
        <p class="small">${esc(s.what)}</p>
        <p class="math mono">${esc(s.math)}</p>
        ${s.layers.length ? `<p class="mono small muted">${s.layers.map(esc).join(" · ")}</p>` : ""}
        ${live ? `<p class="mono small muted" style="margin-top:14px">live in this demo</p><table class="mini">${live}</table>` : ""}`;
      const d = this.root.querySelector("#sys-drawer");
      d.animate?.([{ opacity: 0, transform: "translateX(8px)" }, { opacity: 1, transform: "none" }], { duration: 300 });
    }

    async run() {
      if (this.running) return;
      this.running = true;
      const tok = this.root.querySelector("#tok"), btok = this.root.querySelector("#btok"), back = this.root.querySelector("#back");
      const wait = (ms) => new Promise((r) => setTimeout(r, RM ? 0 : ms));
      const move = (el, from, to, ms) => new Promise((res) => {
        if (RM) { el.setAttribute("cx", to[0]); el.setAttribute("cy", to[1]); return res(); }
        const t0 = performance.now(), a0 = Math.atan2(from[1] - CY, from[0] - CX);
        let a1 = Math.atan2(to[1] - CY, to[0] - CX); if (a1 < a0) a1 += Math.PI * 2;
        const f = (now) => {
          const t = Math.min(1, (now - t0) / ms), e = t < .5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2, a = a0 + (a1 - a0) * e;
          el.setAttribute("cx", CX + R * Math.cos(a)); el.setAttribute("cy", CY + R * Math.sin(a));
          t < 1 && this.running ? requestAnimationFrame(f) : res();
        };
        requestAnimationFrame(f);
      });
      tok.setAttribute("opacity", 1);
      for (let i = 0; i < STEPS.length && this.running; i++) {
        this.select(i);
        await wait(1500);
        if (i < STEPS.length - 1) await move(tok, pos(i), pos(i + 1), 650);
      }
      if (!this.running) return;
      // backward pass: Learn -> Score
      this.root.querySelector("#cq").textContent = "backward pass";
      this.root.querySelector("#cs").textContent = "outcomes rewrite who we target and what we say";
      back.classList.add("live"); btok.setAttribute("opacity", 1); tok.setAttribute("opacity", 0);
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
      this.root.querySelector("#cs").textContent = "priors updated · next batch re-ranked";
      await wait(1600);
      back.classList.remove("live"); btok.setAttribute("opacity", 0);
      tok.setAttribute("cx", pos(0)[0]); tok.setAttribute("cy", pos(0)[1]); tok.setAttribute("opacity", 1);
      this.running = false;
    }

    // ------------------------------------------------------------------ 2. the graph
    graph(st) {
      const live = this.live;
      const T = [ // [id, label, x, y, layer, status, count]
        ["company", "Company", 250, 170, "public", "demo", live.posts?.companies],
        ["product", "Product", 470, 120, "public", "demo", live.graph?.tools],
        ["signal", "Signal", 470, 260, "public", "demo", live.graph?.features],
        ["person", "Person", 90, 110, "public", "roadmap"],
        ["competitor", "Competitor", 640, 190, "public", "roadmap"],
        ["outcome", "Outcome", 250, 430, "tenant", "partly", live.reply?.replies],
        ["rule", "Persona rule", 470, 430, "tenant", "partly", live.bottle?.rules],
        ["email", "Founder email", 650, 430, "tenant", "partly", live.draft?.drafts],
      ];
      const E = [ // [from, to, label, status]
        ["company", "product", "uses", "demo"], ["company", "signal", "hiring_for", "demo"], ["product", "competitor", "competes_with", "roadmap"],
        ["person", "company", "works_at", "roadmap"], ["person", "company", "part_of_committee", "roadmap"],
        ["outcome", "company", "outcome_of", "partly"], ["outcome", "signal", "credits", "partly"], ["rule", "company", "targets", "partly"],
        ["email", "rule", "trains", "roadmap"],
      ];
      const P = Object.fromEntries(T.map((t) => [t[0], t]));
      const edges = E.map(([a, b, l, s], i) => {
        const A = P[a], B = P[b], off = (a === "person" && l === "part_of_committee") ? 40 : 0;
        const mx = (A[2] + B[2]) / 2, my = (A[3] + B[3]) / 2 + off;
        return `<path id="ge${i}" d="M${A[2]},${A[3]} Q${mx},${my} ${B[2]},${B[3]}" class="gedge ${s}"/>
          <text class="glbl" x="${mx}" y="${my - 6 + off / 2}">${l}</text>`;
      }).join("");
      const nodes = T.map(([id, l, x, y, layer, s, c]) => `<g class="gnode ${layer} ${s}">
          <rect x="${x - 62}" y="${y - 22}" width="124" height="44" rx="10"/>
          <text x="${x}" y="${y - 2}" class="gt">${l}</text>
          <text x="${x}" y="${y + 13}" class="gc">${c != null ? Number(c).toLocaleString() : STATUS[s][1]}</text></g>`).join("");
      st.innerHTML = `<svg viewBox="0 0 ${W} ${H}" class="sys-svg">
        <rect x="14" y="40" width="${W - 28}" height="290" rx="16" class="plane public"/>
        <text x="30" y="64" class="plane-lbl">public market graph · shared · every fact cited</text>
        <rect x="14" y="360" width="${W - 28}" height="140" rx="16" class="plane tenant"/>
        <text x="30" y="384" class="plane-lbl">tenant layer · this founder only</text>
        <line x1="30" x2="${W - 30}" y1="345" y2="345" class="boundary"/>
        <text x="${W - 30}" y="341" class="boundary-lbl" text-anchor="end">@Security keeps tenant data private · opt-in aggregates only</text>
        ${edges}${nodes}
        <circle r="4" class="gtok" id="gtok"/>
      </svg>`;
      this.root.querySelector("#sys-drawer").innerHTML = `
        <h3>The graph</h3>
        <p class="small">Kùzu for serving: cited facts, warm-intro paths, explanations. Relational deep learning on the same data, with time splits, for learning.</p>
        <p class="small">Use the graph only where tables fail: multi-hop paths, and linking outcomes back to the signals that caused them.</p>
        <p class="mono small muted" style="margin-top:14px">in this demo</p>
        <table class="mini"><tr><td>companies</td><td>${esc(live.posts?.companies)}</td></tr><tr><td>adoption edges</td><td>${esc(live.graph?.["adoption edges"])}</td></tr>
        <tr><td>triples kept</td><td>${esc(live.review?.kept)}</td></tr><tr><td>rejected by Reviewer</td><td>${esc(live.review?.rejected)}</td></tr></table>`;
      // a cited fact travels the multi-hop path: outcome -> signal -> company -> product
      const tok = st.querySelector("#gtok"), path = [5, 1, 0];
      let k = 0, t0 = performance.now();
      const f = (now) => {
        if (this.panel !== "graph") return;
        const el = st.querySelector("#ge" + [6, 1, 0][k % 3]);
        if (!el) return;
        const t = Math.min(1, (now - t0) / 1200), L = el.getTotalLength();
        const rev = k % 3 === 1; // hiring_for goes company->signal; walk it backwards
        const p = el.getPointAtLength(L * (rev ? 1 - t : t));
        tok.setAttribute("cx", p.x); tok.setAttribute("cy", p.y);
        if (t >= 1) { k++; t0 = now; }
        if (!RM) requestAnimationFrame(f);
      };
      void path; requestAnimationFrame(f);
    }

    // ------------------------------------------------------------------ 3. learning
    learn(st) {
      const segs = [{ name: "fintech", belief: 0.7, conf: "quite sure", k: 20 }, { name: "healthcare", belief: 0.5, conf: "a hunch", k: 4 }];
      st.innerHTML = `
        <div class="learn">
          <div class="learn-top">
            ${segs.map((s, i) => `<div class="beta-card" id="bc${i}">
              <p class="mono small muted">founder says</p>
              <p class="say">“${s.name}, ${Math.round(s.belief * 100)}%, ${s.conf}”</p>
              <svg viewBox="0 0 300 120" class="beta"><path class="bfill"/><path class="bline"/><line class="bmean" y1="8" y2="104"/>
                <line x1="10" x2="290" y1="104" y2="104" stroke="rgba(255,255,255,.15)"/></svg>
              <p class="mono small"><span class="ab"></span> <span class="muted">mean</span> <span class="mean"></span></p>
              <div class="chips-out"></div>
            </div>`).join("")}
          </div>
          <div class="sys-controls">
            <button class="btn" data-o="1">+ retained (1)</button><button class="btn" data-o="0.3">+ trial (0.3)</button><button class="btn" data-o="0">+ churned (0)</button>
            <button class="btn ghost" id="lreset">reset</button>
          </div>
          <div class="residual">
            <p class="math mono">score(account) = logit(prior) + correction(graph features)</p>
            <div id="wbars"></div>
            <p class="mono small muted">correction weights learned on the synthetic market (Proof tab) · a full GNN residual is the roadmap</p>
          </div>
          <div class="maturity mono">
            <span>0 customers<br><b>priors + graph similarity</b></span><span>tens<br><b>Beta updates</b></span>
            <span>hundreds<br><b>per-startup fine-tune</b></span><span>many tenants<br><b>shared model</b></span>
          </div>
        </div>`;
      const state = segs.map((s) => ({ a: s.belief * s.k, b: (1 - s.belief) * s.k, hist: [] }));
      const lgamma = (z) => { // Lanczos
        const g = 7, c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
          12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
        if (z < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * z)) - lgamma(1 - z);
        z -= 1; let x = c[0]; for (let i = 1; i < g + 2; i++) x += c[i] / (z + i);
        const t = z + g + 0.5; return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
      };
      const pdf = (x, a, b) => Math.exp((a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x) - (lgamma(a) + lgamma(b) - lgamma(a + b)));
      const draw = (i, from) => {
        const card = st.querySelector("#bc" + i), s = state[i];
        const target = Array.from({ length: 99 }, (_, j) => pdf((j + 1) / 100, s.a, s.b));
        const prev = from || target;
        const t0 = performance.now();
        const f = (now) => {
          const t = RM ? 1 : Math.min(1, (now - t0) / 700), e = 1 - Math.pow(1 - t, 3);
          const ys = target.map((v, j) => prev[j] + (v - prev[j]) * e), mx = Math.max(...ys, 1e-9);
          const pts = ys.map((v, j) => [10 + (j + 1) * 2.8, 104 - (v / mx) * 92]);
          const d = "M" + pts.map((p) => p.map((n) => n.toFixed(1)).join(",")).join(" L");
          card.querySelector(".bline").setAttribute("d", d);
          card.querySelector(".bfill").setAttribute("d", d + " L290,104 L10,104 Z");
          const mean = s.a / (s.a + s.b);
          card.querySelector(".bmean").setAttribute("x1", 10 + mean * 280); card.querySelector(".bmean").setAttribute("x2", 10 + mean * 280);
          card.querySelector(".ab").textContent = `Beta(${+s.a.toFixed(1)}, ${+s.b.toFixed(1)})`;
          card.querySelector(".mean").textContent = mean.toFixed(2);
          if (t < 1) requestAnimationFrame(f); else card._ys = target;
        };
        requestAnimationFrame(f);
      };
      state.forEach((_, i) => draw(i));
      st.querySelectorAll("[data-o]").forEach((b) => b.onclick = () => {
        const o = +b.dataset.o;
        state.forEach((s, i) => {
          const card = st.querySelector("#bc" + i), prev = card._ys;
          s.a += o; s.b += 1 - o;
          const chip = document.createElement("span"); chip.className = "out"; chip.textContent = o === 1 ? "retained" : o === 0 ? "churned" : "trial";
          card.querySelector(".chips-out").appendChild(chip);
          chip.animate?.([{ opacity: 0, transform: "translateY(-6px)" }, { opacity: 1, transform: "none" }], { duration: 300 });
          draw(i, prev);
        });
      });
      st.querySelector("#lreset").onclick = () => this.learn(st);
      // learned correction weights (synthetic run)
      fetch("/api/model-proof").then((r) => r.ok ? r.json() : null).then((d) => {
        if (!d) return;
        const ws = Object.entries(d.prior_residual.weights).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 6);
        const mx = Math.max(...ws.map((w) => Math.abs(w[1])));
        st.querySelector("#wbars").innerHTML = ws.map(([f, v]) => `<div class="wbar"><span class="wl">${esc(f)}</span>
          <span class="wtrack"><i style="--w:${(Math.abs(v) / mx * 50).toFixed(1)}%;${v < 0 ? "right:50%" : "left:50%"};background:${v < 0 ? "#eb6834" : "#3987e5"}"></i></span>
          <span class="wv">${v > 0 ? "+" : ""}${v.toFixed(2)}</span></div>`).join("");
        st.querySelectorAll(".wtrack i").forEach((el, i) => el.animate?.([{ width: "0%" }, { width: el.style.getPropertyValue("--w") }], { duration: 800, delay: i * 80, fill: "both", easing: "cubic-bezier(.2,.8,.2,1)" }));
      });
      this.root.querySelector("#sys-drawer").innerHTML = `
        <h3>Learning from outcomes</h3>
        <p class="small">A founder's belief is a Beta prior. Confidence sets its weight: “quite sure” counts like 20 past deals, “a hunch” like 4.</p>
        <p class="small">Each outcome adds to the counts: retained adds 1 success, a trial 0.3, churn a failure. Try the buttons.</p>
        <p class="small">No reply is not a no: uncontacted accounts stay unlabeled, and Thompson sampling keeps an exploration budget so the model doesn't only confirm itself.</p>`;
    }
  }

  window.SystemView = SystemView;
})();
