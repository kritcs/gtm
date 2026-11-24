// Pipeline as a force-directed graph, regrouped three ways: by industry, by shared signal, or by stage.
// Node colour = status (past customer, account stage, new segment). Edge colour = relation.
// Hand-rolled simulation (repulsion + springs + gravity), canvas 2D, hover to focus, drag to move.
(function () {
  const REDUCED = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const STATUS = {
    customer: { color: "#d55181", label: "past customer" },
    suggested: { color: "#e4e4e7", label: "suggested" },
    drafted: { color: "#eda100", label: "drafted" },
    sent: { color: "#3987e5", label: "sent" },
    replied: { color: "#1baf7a", label: "replied" },
    segment: { color: "#eb6834", label: "new segment" },
  };
  const REL = {
    industry: { color: "rgba(255,255,255,.16)", label: "same industry" },
    trigger: { color: "rgba(57,135,229,.55)", label: "same trigger" },
    tool: { color: "rgba(255,255,255,.22)", label: "same tool" },
    pain: { color: "rgba(235,104,52,.5)", label: "same problem" },
    stage: { color: "rgba(255,255,255,.16)", label: "stage" },
  };
  // which link kinds pull (and which hubs show) in each grouping
  const GROUP = {
    industry: { pull: ["industry"], hubs: ["industry"] },
    signal: { pull: ["trigger", "tool", "pain"], hubs: ["evidence"] },
    stage: { pull: ["stage"], hubs: ["stage"] },
  };

  class PipelineGraph {
    constructor(canvas) {
      this.c = canvas; this.ctx = canvas.getContext("2d");
      this.nodes = new Map(); this.links = []; this.group = "industry";
      this.alpha = 1; this.hover = null; this.drag = null; this.pulses = [];
      this.resize(); addEventListener("resize", () => this.resize());
      canvas.addEventListener("pointermove", (e) => this.onMove(e));
      canvas.addEventListener("pointerdown", (e) => this.onDown(e));
      addEventListener("pointerup", () => { if (this.drag) { this.drag.fixed = false; this.drag = null; } });
      canvas.addEventListener("pointerleave", () => { this.hover = null; });
      const loop = () => { this.step(); this.draw(); requestAnimationFrame(loop); };
      requestAnimationFrame(loop);
    }

    resize() {
      const r = this.c.getBoundingClientRect(), d = Math.min(2, devicePixelRatio);
      if (!r.width) return;
      this.w = r.width; this.h = r.height;
      this.c.width = r.width * d; this.c.height = r.height * d;
      this.ctx.setTransform(d, 0, 0, d, 0, 0);
    }

    node(id, o) {
      let n = this.nodes.get(id);
      if (!n) {
        const a = Math.random() * 6.28, rr = 60 + Math.random() * 120;
        n = { id, x: (this.w || 700) / 2 + Math.cos(a) * rr, y: (this.h || 500) / 2 + Math.sin(a) * rr, vx: 0, vy: 0,
              born: performance.now(), vis: 0 };
        this.nodes.set(id, n);
      }
      return Object.assign(n, o);
    }

    setGroup(g) { this.group = g; this.alpha = 1; }

    // data from /api/sessions/{id}/graph
    update(d, { pulse } = {}) {
      const keep = new Set();
      const add = (id, o) => { keep.add(id); return this.node(id, o); };
      this.links = [];
      const link = (a, b, kind) => this.links.push({ a, b, kind });
      add("p", { label: d.product, kind: "product", r: 12, color: "#9085e9" });

      const companies = [
        ...d.customers.map((c) => ({ ...c, status: "customer" })),
        ...d.accounts.map((c) => ({ ...c, status: c.stage })),
        ...d.segment.map((c) => ({ ...c, status: "segment" })),
      ];
      // hubs
      for (const ind of new Set(companies.map((c) => c.industry))) {
        add("i:" + ind, { label: ind, kind: "industry", r: 8, color: "#a1a1aa" });
        link("p", "i:" + ind, "hub");
      }
      const STAGE_HUB = { customer: "won", segment: "unexplored" };
      for (const st of ["customer", "suggested", "drafted", "sent", "replied", "segment"]) {
        add("s:" + st, { label: STAGE_HUB[st] || st, kind: "stage", r: 8, color: STATUS[st].color });
        link("p", "s:" + st, "hub");
      }
      const ev = (key, label, rel) => { add("e:" + key, { label, kind: "evidence", rel, r: 5, color: "#71717a" }); return "e:" + key; };
      const pain = ev("pain", d.founder_pain, "pain");
      link("p", pain, "hub");
      const term = d.segment_term ? ev("term", `“${d.segment_term}”`, "pain") : null;
      if (term) link(pain, term, "pain");
      const trait = d.trait ? ev("trait", d.trait, "trigger") : null;
      if (trait) link("p", trait, "hub");

      for (const c of companies) {
        const id = "c:" + c.id;
        add(id, { label: c.name, kind: "company", status: c.status, r: c.status === "customer" ? 6.5 : 5.5,
                  color: STATUS[c.status].color, industry: c.industry });
        link(id, "i:" + c.industry, "industry");
        if (c.status === "customer") { if (trait) link(id, trait, "trigger"); link(id, pain, "pain"); }
        if (c.status === "segment" && term) link(id, term, "pain");
        link(id, "s:" + c.status, "stage");
        for (const ch of c.chips || []) {
          if (ch.kind === "trigger" && trait) link(id, trait, "trigger");
          else if (ch.kind === "pain") link(id, pain, "pain");
          else if (ch.kind === "stack") {
            const tool = ch.label.split(" · ")[0].replace(/^Uses /, "");
            link(id, ev("t:" + tool, tool, "tool"), "tool");
          }
        }
      }
      for (const id of [...this.nodes.keys()]) if (!keep.has(id)) this.nodes.delete(id);
      if (pulse) this.pulses.push({ id: "c:" + pulse, t: performance.now() });
      this.alpha = 1;
    }

    shown(n) {
      const g = GROUP[this.group];
      if (n.kind === "industry") return g.hubs.includes("industry");
      if (n.kind === "stage") return g.hubs.includes("stage");
      if (n.kind === "evidence") return g.hubs.includes("evidence");
      return true;
    }
    strength(l) {
      const g = GROUP[this.group];
      if (l.kind === "hub") return { len: 170, k: 0.02 };
      if (!g.pull.includes(l.kind)) return null;
      return { len: l.kind === "industry" || l.kind === "stage" ? 60 : 85, k: 0.05 };
    }

    step() {
      if (this.alpha < 0.004 && !this.drag) return;
      const ns = [...this.nodes.values()].filter((n) => this.shown(n)), cx = this.w / 2, cy = this.h / 2;
      for (let i = 0; i < ns.length; i++) {
        const a = ns[i];
        for (let j = i + 1; j < ns.length; j++) {
          const b = ns[j];
          let dx = a.x - b.x, dy = a.y - b.y, d2 = dx * dx + dy * dy + 0.01;
          if (d2 > 250000) continue;
          const hub = (n) => n.kind !== "company";
          const f = (hub(a) && hub(b) ? 9000 : 1500) / d2;
          const d = Math.sqrt(d2); dx /= d; dy /= d;
          a.vx += dx * f; a.vy += dy * f; b.vx -= dx * f; b.vy -= dy * f;
        }
      }
      for (const l of this.links) {
        const a = this.nodes.get(l.a), b = this.nodes.get(l.b), s = this.strength(l);
        if (!a || !b || !s || !this.shown(a) || !this.shown(b)) continue;
        const dx = b.x - a.x, dy = b.y - a.y, d = Math.hypot(dx, dy) || 1, f = (d - s.len) * s.k;
        a.vx += dx / d * f; a.vy += dy / d * f; b.vx -= dx / d * f; b.vy -= dy / d * f;
      }
      for (const n of this.nodes.values()) {
        const g = n.kind === "product" ? 0.05 : 0.002;
        n.vx += (cx - n.x) * g; n.vy += (cy - n.y) * g;
        if (n.fixed) { n.vx = n.vy = 0; continue; }
        const s = REDUCED ? 1 : Math.max(this.alpha, 0.05);
        n.x += n.vx * s; n.y += n.vy * s; n.vx *= 0.6; n.vy *= 0.6;
        n.x = Math.max(30, Math.min(this.w - 30, n.x)); n.y = Math.max(60, Math.min(this.h - 80, n.y));
      }
      this.alpha *= 0.993;
    }

    neighbours(id) {
      const s = new Set([id]);
      for (const l of this.links) { if (l.a === id) s.add(l.b); if (l.b === id) s.add(l.a); }
      return s;
    }

    draw() {
      const x = this.ctx, now = performance.now();
      if (!this.w) return;
      x.clearRect(0, 0, this.w, this.h);
      for (const n of this.nodes.values()) n.vis += ((this.shown(n) ? 1 : 0) - n.vis) * (REDUCED ? 1 : 0.08);
      const focus = this.hover ? this.neighbours(this.hover.id) : null;
      const g = GROUP[this.group];
      for (const l of this.links) {
        const a = this.nodes.get(l.a), b = this.nodes.get(l.b);
        if (!a || !b) continue;
        const v = Math.min(a.vis, b.vis);
        if (v < 0.02) continue;
        const active = l.kind === "hub" || g.pull.includes(l.kind);
        const on = focus && focus.has(l.a) && focus.has(l.b);
        x.globalAlpha = v * (on ? 1 : focus ? 0.12 : active ? 1 : 0.35);
        x.strokeStyle = l.kind === "hub" ? "rgba(255,255,255,.08)" : REL[l.kind].color;
        x.lineWidth = on ? 1.6 : 1;
        x.beginPath(); x.moveTo(a.x, a.y); x.lineTo(b.x, b.y); x.stroke();
      }
      x.globalAlpha = 1;
      for (const p of this.pulses) {
        const t = (now - p.t) / 1600, n = this.nodes.get(p.id);
        if (t > 1 || !n) continue;
        x.strokeStyle = `rgba(27,175,122,${(1 - t).toFixed(3)})`; x.lineWidth = 2;
        x.beginPath(); x.arc(n.x, n.y, 8 + t * 50, 0, 6.283); x.stroke();
      }
      this.pulses = this.pulses.filter((p) => now - p.t < 1600);
      for (const n of this.nodes.values()) {
        if (n.vis < 0.02) continue;
        const grow = REDUCED ? 1 : Math.min(1, (now - n.born) / 500);
        const dim = focus && !focus.has(n.id);
        let r = n.r * (0.4 + 0.6 * grow);
        if (n.status === "replied") r *= 1 + 0.15 * Math.sin(now / 300);
        x.globalAlpha = n.vis * (dim ? 0.15 : 1);
        x.fillStyle = n.color;
        if (n.kind === "product" || n.status === "replied" || n.kind === "stage") { x.shadowColor = n.color; x.shadowBlur = 16; }
        x.beginPath(); x.arc(n.x, n.y, r, 0, 6.283); x.fill();
        x.shadowBlur = 0;
        if (n.kind === "industry") { x.strokeStyle = "rgba(255,255,255,.35)"; x.lineWidth = 1; x.beginPath(); x.arc(n.x, n.y, r + 4, 0, 6.283); x.stroke(); }
        const hub = n.kind !== "company";
        const max = n.kind === "evidence" ? 22 : 30;
        const lab = n.label.length > max ? n.label.slice(0, max - 1) + "…" : n.label;
        x.font = `${hub ? 600 : 400} ${hub ? 12 : 11}px Inter, ui-sans-serif, sans-serif`;
        x.fillStyle = hub ? "rgba(255,255,255,.9)" : "rgba(255,255,255,.6)";
        if (n.kind === "evidence") x.fillStyle = "rgba(255,255,255,.55)";
        x.textAlign = "center";
        x.fillText(lab, n.x, n.y + r + 14);
        x.globalAlpha = 1;
      }
    }

    at(e) {
      const r = this.c.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top;
      let best = null, bd = 18 * 18;
      for (const n of this.nodes.values()) {
        if (n.vis < 0.5) continue;
        const d = (n.x - px) ** 2 + (n.y - py) ** 2;
        if (d < bd) { bd = d; best = n; }
      }
      return { n: best, px, py };
    }
    onMove(e) {
      const { n, px, py } = this.at(e);
      if (this.drag) { this.drag.x = px; this.drag.y = py; this.alpha = Math.max(this.alpha, 0.3); return; }
      this.hover = n; this.c.style.cursor = n ? "grab" : "default";
    }
    onDown(e) { const { n } = this.at(e); if (n) { this.drag = n; n.fixed = true; } }
  }

  window.PipelineGraph = PipelineGraph;
  window.PIPELINE_STATUS = STATUS;
  window.PIPELINE_REL = REL;
})();
