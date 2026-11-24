// The agent team, live: a relay strip (who hands off to whom) and one glowing card per agent.
(function () {
  const RM = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  const AGENTS = [
    { id: "Interviewer", role: "asks", col: 0, row: 1.5 },
    { id: "Extractor", role: "listens", col: 1, row: 1.5 },
    { id: "Market model", role: "scores the market", col: 2, row: 1.5 },
    { id: "Hit finder", role: "tests traits", col: 3, row: 0 },
    { id: "Discovery", role: "finds segments", col: 3, row: 1 },
    { id: "Pain matcher", role: "matches the problem", col: 3, row: 2 },
    { id: "Blind spot", role: "flags thin data", col: 3, row: 3 },
    { id: "Reviewer", role: "checks every claim", col: 4, row: 1.5 },
    { id: "Writer", role: "writes", col: 5, row: 1.5 },
    { id: "Flexprice", role: "meters credits", col: 6, row: 1.5 },
    { id: "Webhook", role: "hears replies", col: 7, row: 1.5 },
  ];
  const EDGES = [["Interviewer", "Extractor"], ["Extractor", "Market model"],
    ...["Hit finder", "Discovery", "Pain matcher", "Blind spot"].flatMap((a) => [["Market model", a], [a, "Reviewer"]]),
    ["Reviewer", "Writer"], ["Writer", "Flexprice"], ["Flexprice", "Webhook"], ["Webhook", "Interviewer"]];
  const W = 1000, H = 268, X = (c) => 50 + c * 128, Y = (r) => 32 + r * 56;
  const byId = Object.fromEntries(AGENTS.map((a) => [a.id, a]));

  class AgentsView {
    constructor(root) {
      this.root = root; this.state = {}; this.last = null; this.queue = Promise.resolve();
      for (const a of AGENTS) this.state[a.id] = { steps: 0, status: "idle", msg: "", err: 0 };
      const path = (a, b) => {
        const A = byId[a], B = byId[b];
        if (a === "Webhook") return `M${X(A.col)},${Y(A.row) + 22} C${X(A.col)},${H - 4} ${X(B.col)},${H - 4} ${X(B.col)},${Y(B.row) + 22}`;
        const x1 = X(A.col) + 18, y1 = Y(A.row), x2 = X(B.col) - 18, y2 = Y(B.row), mx = (x1 + x2) / 2;
        return `M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`;
      };
      root.innerHTML = `
        <div class="ag-top">
          <div class="ag-title"><span class="ag-live"><i></i>live</span><span class="muted" id="ag-count"></span></div>
          <button class="btn" id="ag-replay"><i class="i" style="--i:url(/static/vendor/icons/play.svg)"></i>Replay</button>
        </div>
        <svg viewBox="0 0 ${W} ${H}" class="relay">
          ${EDGES.map(([a, b]) => `<path class="redge ${a === "Webhook" ? "loop" : ""}" id="re-${a.replace(/ /g, "_")}-${b.replace(/ /g, "_")}" d="${path(a, b)}"/>`).join("")}
          ${AGENTS.map((a) => `<g class="rnode" id="rn-${a.id.replace(/ /g, "_")}" transform="translate(${X(a.col)},${Y(a.row)})">
              <circle class="rglow" r="22"/><circle class="rorb" r="14"/><text y="34">${esc(a.id)}</text></g>`).join("")}
          <circle r="4" class="rtok" id="rtok" opacity="0"/>
        </svg>
        <div class="ag-grid">${AGENTS.map((a) => `
          <div class="ag-card idle" id="ac-${a.id.replace(/ /g, "_")}">
            <div class="ag-head"><span class="orb"><i></i></span><div><b>${esc(a.id)}</b><span>${esc(a.role)}</span></div><em class="ag-n">0</em></div>
            <p class="ag-msg"></p>
          </div>`).join("")}</div>
        <details class="ag-log"><summary>log</summary><ol class="feed" id="feed-list"></ol></details>`;
      root.querySelector("#ag-replay").onclick = () => this.replay(this.events || []);
    }

    key(id) { return id.replace(/ /g, "_"); }

    // show final state instantly (no animation) for events that already happened
    load(events) {
      this.events = events;
      for (const a of AGENTS) this.state[a.id] = { steps: 0, status: "idle", msg: "", err: 0 };
      for (const e of events) this.apply(e, false);
      for (const a of AGENTS) if (this.state[a.id].status === "working") this.setStatus(a.id, "done");
      this.count();
    }

    count() {
      const n = Object.values(this.state).reduce((s, x) => s + x.steps, 0);
      const active = Object.values(this.state).filter((x) => x.steps).length;
      this.root.querySelector("#ag-count").textContent = `${active} agents · ${n} steps`;
    }

    setStatus(id, st) {
      this.state[id].status = st;
      const card = this.root.querySelector("#ac-" + this.key(id)), node = this.root.querySelector("#rn-" + this.key(id));
      if (card) card.className = "ag-card " + st + (this.state[id].err ? " has-err" : "");
      if (node) node.setAttribute("class", "rnode " + st);
    }

    apply(e, animate) {
      const id = byId[e.agent] ? e.agent : null;
      if (!id) return;
      const s = this.state[id];
      s.steps++; s.msg = e.content;
      if (e.type === "error") s.err++;
      const card = this.root.querySelector("#ac-" + this.key(id));
      card.querySelector(".ag-n").textContent = s.steps;
      const msg = card.querySelector(".ag-msg");
      if (animate && !RM) this.type(msg, e.content, e.type === "error"); else { msg.textContent = e.content; msg.classList.toggle("err", e.type === "error"); }
      if (this.last && this.last !== id) {
        for (const [a, st] of Object.entries(this.state)) if (a !== id && st.status === "working") this.setStatus(a, "done");
        if (animate) this.pulse(this.last, id);
      }
      this.setStatus(id, "working");
      this.last = id;
      this.count();
    }

    type(el, text, err) {
      el.classList.toggle("err", !!err);
      let i = 0;
      const step = () => { i += 2; el.textContent = text.slice(0, i); if (i < text.length) requestAnimationFrame(step); };
      requestAnimationFrame(step);
    }

    pulse(a, b) {
      const p = this.root.querySelector(`#re-${this.key(a)}-${this.key(b)}`), tok = this.root.querySelector("#rtok");
      if (!p || RM) return;
      p.classList.add("hot"); setTimeout(() => p.classList.remove("hot"), 700);
      const L = p.getTotalLength(), t0 = performance.now();
      tok.setAttribute("opacity", 1);
      const f = (now) => {
        const t = Math.min(1, (now - t0) / 450), pt = p.getPointAtLength(L * t);
        tok.setAttribute("cx", pt.x); tok.setAttribute("cy", pt.y);
        if (t < 1) requestAnimationFrame(f); else tok.setAttribute("opacity", 0);
      };
      requestAnimationFrame(f);
    }

    ingest(e) { this.events = [...(this.events || []), e]; this.apply(e, true); }

    async replay(events) {
      const btn = this.root.querySelector("#ag-replay");
      btn.disabled = true;
      for (const a of AGENTS) { this.state[a.id] = { steps: 0, status: "idle", msg: "", err: 0 }; this.setStatus(a.id, "idle");
        const c = this.root.querySelector("#ac-" + this.key(a.id)); c.querySelector(".ag-msg").textContent = ""; c.querySelector(".ag-n").textContent = 0; }
      this.last = null;
      for (const e of events) { this.apply(e, true); await new Promise((r) => setTimeout(r, RM ? 0 : 520)); }
      for (const a of AGENTS) if (this.state[a.id].status === "working") this.setStatus(a.id, "done");
      btn.disabled = false;
    }
  }
  window.AgentsView = AgentsView;
})();
