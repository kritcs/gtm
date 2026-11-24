// Problem page diagram: a small heterogeneous graph that plays the three approach points in turn.
//   0  the founder is the prior: beliefs flow into company nodes along prior edges
//   1  heterogeneous graph: typed nodes (company, tool, signal, outcome) and typed edges appear
//   2  re-fit on feedback: a reply arrives and credits the path back to the signal that found it
(function () {
  const RM = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const NS = "http://www.w3.org/2000/svg";
  const el = (tag, a, p) => { const e = document.createElementNS(NS, tag); for (const [k, v] of Object.entries(a || {})) e.setAttribute(k, v); if (p) p.appendChild(e); return e; };
  const txt = (p, x, y, s, cls, anchor = "middle") => { const t = el("text", { x, y, class: cls, "text-anchor": anchor }, p); t.textContent = s; return t; };
  const sleep = (ms) => new Promise((r) => setTimeout(r, RM ? 0 : ms));
  const icon = (p, name, size = 16) => {
    const fo = el("foreignObject", { x: -size / 2, y: -size / 2, width: size, height: size }, p);
    fo.innerHTML = `<div xmlns="http://www.w3.org/1999/xhtml" class="pd-i" style="width:${size}px;height:${size}px;--i:url(/static/vendor/icons/${name}.svg)"></div>`;
  };

  const TYPE = {
    prior: { color: "#9085e9", icon: "wand-sparkles", label: "Prior" },
    company: { color: "#e4e4e7", icon: "building-2", label: "Company" },
    tool: { color: "#3987e5", icon: "package", label: "Tool" },
    signal: { color: "#eda100", icon: "radar", label: "Signal" },
    outcome: { color: "#22c55e", icon: "reply", label: "Outcome" },
  };
  // id: [label, type, x, y]
  const N = {
    prior: ["Your beliefs", "prior", 78, 168],
    acme: ["Acme", "company", 240, 92], borg: ["Borg", "company", 240, 244],
    stripe: ["Stripe", "tool", 420, 58], revops: ["RevOps hire", "signal", 430, 168],
    reply: ["Reply", "outcome", 420, 286],
  };
  const E = [ // [from, to, label, kind, bend]
    ["prior", "acme", "prior", "prior", -18], ["prior", "borg", "prior", "prior", 18],
    ["acme", "stripe", "uses", "typed", -10], ["acme", "revops", "hiring for", "typed", 14], ["borg", "revops", "hiring for", "typed", -14],
    ["reply", "borg", "outcome of", "feedback", 12],
  ];

  const curve = (a, b, bend) => {
    const [x1, y1, x2, y2] = [N[a][2], N[a][3], N[b][2], N[b][3]];
    const mx = (x1 + x2) / 2, my = (y1 + y2) / 2 + bend;
    return { d: `M${x1},${y1} Q${mx},${my} ${x2},${y2}`, mx, my: (y1 + y2) / 2 + bend / 2 };
  };

  function node(svg, id) {
    const [label, type, x, y] = N[id], T = TYPE[type];
    const g = el("g", { transform: `translate(${x},${y})`, class: `pd-node ${type}` }, svg);
    el("circle", { r: type === "prior" ? 46 : 30, class: "pd-halo", fill: T.color }, g);
    if (type === "prior") {
      el("circle", { r: 34, fill: "url(#pd-orb)", stroke: T.color, "stroke-width": 1.5 }, g);
      icon(g, T.icon, 18);
      txt(g, 0, 56, label, "pd-lbl");
    } else if (type === "company") {
      el("rect", { x: -42, y: -18, width: 84, height: 36, rx: 12, fill: "#1a1740", stroke: "rgba(255,255,255,.55)", "stroke-width": 1.2 }, g);
      const ig = el("g", { transform: "translate(-22,0)" }, g); icon(ig, T.icon, 15);
      txt(g, 8, 5, label, "pd-lbl");
    } else {
      el("circle", { r: 20, fill: "#1a1740", stroke: T.color, "stroke-width": 1.6 }, g);
      icon(g, T.icon, 16);
      txt(g, 0, 38, label, "pd-lbl");
    }
    return g;
  }

  let token = 0;
  async function play(root) {
    const my = ++token;
    const items = document.querySelectorAll("#apoints li");
    while (my === token && document.querySelector("#problem.active")) {
      root.innerHTML = "";
      const svg = el("svg", { viewBox: "0 0 500 360", class: "pd-svg" }, root);
      const defs = el("defs", {}, svg);
      const g = el("radialGradient", { id: "pd-orb", cx: "35%", cy: "30%" }, defs);
      el("stop", { offset: "0%", "stop-color": "#e9e5ff" }, g); el("stop", { offset: "55%", "stop-color": "#7c6fe0" }, g); el("stop", { offset: "100%", "stop-color": "#2b2470" }, g);
      for (let x = 20; x < 500; x += 28) for (let y = 20; y < 320; y += 28) el("circle", { cx: x, cy: y, r: 0.8, fill: "rgba(255,255,255,.08)" }, svg);

      const edges = {};
      for (const [a, b, label, kind, bend] of E) {
        const c = curve(a, b, bend);
        const path = el("path", { d: c.d, class: `pd-edge k-${kind}` }, svg);
        const chip = el("g", { transform: `translate(${c.mx},${c.my})`, class: `pd-chip k-${kind}` }, svg);
        const w = label.length * 5.6 + 14;
        el("rect", { x: -w / 2, y: -9, width: w, height: 18, rx: 9 }, chip);
        txt(chip, 0, 3.5, label, "pd-chipt");
        edges[a + ">" + b] = path;
      }
      const nodes = Object.fromEntries(Object.keys(N).map((id) => [id, node(svg, id)]));
      const lg = el("g", { transform: "translate(250,342)" }, svg);
      const types = Object.entries(TYPE);
      types.forEach(([, T], i) => {
        const x = (i - (types.length - 1) / 2) * 92;
        el("circle", { cx: x - 30, cy: 0, r: 4, fill: T.color }, lg);
        txt(lg, x - 22, 4, T.label, "pd-leg", "start");
      });
      const particles = [];
      const tok = el("circle", { r: 5, class: "pd-tok", opacity: 0 }, svg);
      const on = (i) => items.forEach((li) => li.classList.toggle("on", +li.dataset.p === i));
      const show = (sel, v = true) => svg.querySelectorAll(sel).forEach((x) => x.classList.toggle("vis", v));
      const focus = (ids) => Object.entries(nodes).forEach(([id, n]) => n.classList.toggle("hot", ids.includes(id)));

      // 0: the founder is the prior
      on(0);
      show(".pd-node.prior, .pd-node.company"); show(".k-prior"); focus(["prior"]);
      if (!RM) for (const k of ["prior>acme", "prior>borg"]) for (let j = 0; j < 3; j++) { const dot = el("circle", { r: 2.4, class: "pd-part" }); svg.insertBefore(dot, svg.querySelector(".pd-chip")); particles.push({ p: edges[k], s: j / 3, dot }); }
      let running = true;
      const t0 = performance.now();
      const flow = (now) => {
        if (!running || my !== token) return;
        for (const q of particles) {
          const L = q.p.getTotalLength(), u = ((now - t0) / 1600 + q.s) % 1, pt = q.p.getPointAtLength(L * u);
          q.dot.setAttribute("cx", pt.x); q.dot.setAttribute("cy", pt.y); q.dot.setAttribute("opacity", Math.sin(u * Math.PI));
        }
        requestAnimationFrame(flow);
      };
      requestAnimationFrame(flow);
      await sleep(3400); if (my !== token) return;

      // 1: heterogeneous graph
      on(1);
      show(".pd-node"); nodes.reply.classList.remove("vis"); show(".k-typed"); focus(["stripe", "revops", "acme", "borg"]);
      await sleep(3400); if (my !== token) return;

      // 2: a reply credits the path back to its signal
      on(2); focus(["reply"]);
      nodes.reply.classList.add("vis"); show(".k-feedback");
      await sleep(700);
      tok.setAttribute("opacity", 1);
      for (const [key, reverse] of [["reply>borg", false], ["borg>revops", false]]) {
        const p = edges[key], L = p.getTotalLength(), t1 = performance.now();
        await new Promise((res) => {
          const f = (now) => {
            const t = RM ? 1 : Math.min(1, (now - t1) / 750), pt = p.getPointAtLength(L * (reverse ? 1 - t : t));
            tok.setAttribute("cx", pt.x); tok.setAttribute("cy", pt.y);
            t < 1 && my === token ? requestAnimationFrame(f) : res();
          };
          requestAnimationFrame(f);
        });
      }
      tok.setAttribute("opacity", 0);
      edges["borg>revops"].classList.add("credited");
      focus(["revops", "prior"]);
      const plus = txt(svg, 462, 140, "trust +1", "pd-plus", "middle");
      plus.animate?.([{ opacity: 0, transform: "translateY(6px)" }, { opacity: 1, transform: "none" }], { duration: 400, fill: "both" });
      await sleep(3300);
      running = false;
    }
  }

  window.ProblemDiagram = { start() { const r = document.getElementById("adiagram"); if (r) play(r); }, stop() { token++; } };
})();
