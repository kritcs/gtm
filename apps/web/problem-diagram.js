// Problem page diagram: a small heterogeneous graph that plays the three approach points in turn.
//   0  the founder is the prior: beliefs flow into company nodes as prior edges
//   1  heterogeneous graph: typed nodes (company, tool, signal, person, outcome) and typed edges
//   2  re-fit on feedback: a reply arrives and credits the path back to the signal that found it
(function () {
  const RM = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const NS = "http://www.w3.org/2000/svg";
  const el = (tag, a, p) => { const e = document.createElementNS(NS, tag); for (const [k, v] of Object.entries(a || {})) e.setAttribute(k, v); if (p) p.appendChild(e); return e; };
  const txt = (p, x, y, s, cls, anchor = "middle") => { const t = el("text", { x, y, class: cls, "text-anchor": anchor }, p); t.textContent = s; return t; };
  const sleep = (ms) => new Promise((r) => setTimeout(r, RM ? 0 : ms));

  // id: [label, type, x, y]
  const N = {
    prior: ["Your beliefs", "prior", 66, 175],
    acme: ["Acme", "company", 220, 105], borg: ["Borg", "company", 220, 245],
    stripe: ["Stripe", "tool", 380, 55], revops: ["RevOps hire", "signal", 395, 175], vp: ["VP Finance", "person", 380, 295],
    reply: ["Reply", "outcome", 225, 340],
  };
  const E = [ // [from, to, label, kind]
    ["prior", "acme", "prior", "prior"], ["prior", "borg", "prior", "prior"],
    ["acme", "stripe", "uses", "typed"], ["acme", "revops", "hiring for", "typed"], ["borg", "revops", "hiring for", "typed"],
    ["vp", "borg", "works at", "typed"], ["reply", "borg", "outcome of", "feedback"],
  ];
  const TYPE = { prior: "#9085e9", company: "#e4e4e7", tool: "#3987e5", signal: "#eda100", person: "#d55181", outcome: "#22c55e" };

  function shape(g, type) {
    const c = TYPE[type];
    if (type === "company") return el("rect", { x: -34, y: -15, width: 68, height: 30, rx: 8, fill: "#1a1740", stroke: c, "stroke-width": 1.5 }, g);
    if (type === "signal") return el("rect", { x: -13, y: -13, width: 26, height: 26, rx: 4, transform: "rotate(45)", fill: "#2b220f", stroke: c, "stroke-width": 1.5 }, g);
    if (type === "outcome") return el("rect", { x: -26, y: -14, width: 52, height: 28, rx: 14, fill: "#0f2a1c", stroke: c, "stroke-width": 1.5 }, g);
    if (type === "prior") return el("circle", { r: 36, fill: "rgba(144,133,233,.18)", stroke: c, "stroke-width": 1.5 }, g);
    return el("circle", { r: 14, fill: "#1a1740", stroke: c, "stroke-width": 1.5 }, g);
  }

  let token = 0;
  async function play(root) {
    const my = ++token;
    const items = document.querySelectorAll("#apoints li");
    while (my === token && document.querySelector("#problem.active")) {
      root.innerHTML = "";
      const svg = el("svg", { viewBox: "0 0 500 380", class: "pd-svg" }, root);
      const edges = {}, nodes = {};
      for (const [a, b, label, kind] of E) {
        const A = N[a], B = N[b];
        const path = el("line", { x1: A[2], y1: A[3], x2: B[2], y2: B[3], class: `pd-edge k-${kind}` }, svg);
        const lab = txt(svg, (A[2] + B[2]) / 2, (A[3] + B[3]) / 2 - 6, label, `pd-elbl k-${kind}`);
        edges[a + ">" + b] = { path, lab };
      }
      for (const [id, [label, type, x, y]] of Object.entries(N)) {
        const g = el("g", { transform: `translate(${x},${y})`, class: `pd-node ${type}` }, svg);
        shape(g, type);
        txt(g, 0, type === "company" || type === "outcome" ? 4 : type === "prior" ? 4 : 30, label, type === "prior" ? "pd-plbl" : "pd-lbl");
        if (type !== "company" && type !== "prior" && type !== "outcome") txt(g, 0, -20, type, "pd-type");
        nodes[id] = g;
      }
      const tok = el("circle", { r: 5, class: "pd-tok", opacity: 0 }, svg);
      const on = (i) => items.forEach((li) => li.classList.toggle("on", +li.dataset.p === i));
      const show = (sel, v) => svg.querySelectorAll(sel).forEach((x) => x.classList.toggle("vis", v));

      // 0: prior
      on(0); svg.classList.add("p0");
      show(".pd-node.prior, .pd-node.company", true); show(".k-prior", true);
      await sleep(3200); if (my !== token) return;
      // 1: heterogeneous
      on(1); svg.classList.replace("p0", "p1");
      show(".pd-node", true); nodes.reply.classList.remove("vis"); show(".k-typed", true);
      await sleep(3400); if (my !== token) return;
      // 2: feedback credits the path back to its signal
      on(2); svg.classList.replace("p1", "p2");
      nodes.reply.classList.add("vis"); show(".k-feedback", true);
      await sleep(600);
      const path = [N.reply, N.borg, N.revops];
      tok.setAttribute("opacity", 1);
      for (let k = 0; k < path.length - 1 && my === token; k++) {
        const [x1, y1] = [path[k][2], path[k][3]], [x2, y2] = [path[k + 1][2], path[k + 1][3]], t0 = performance.now();
        await new Promise((res) => {
          const f = (now) => {
            const t = RM ? 1 : Math.min(1, (now - t0) / 700);
            tok.setAttribute("cx", x1 + (x2 - x1) * t); tok.setAttribute("cy", y1 + (y2 - y1) * t);
            t < 1 ? requestAnimationFrame(f) : res();
          };
          requestAnimationFrame(f);
        });
      }
      tok.setAttribute("opacity", 0);
      edges["borg>revops"].path.classList.add("credited");
      nodes.revops.classList.add("credited");
      const plus = txt(svg, 445, 160, "trust +1", "pd-plus", "start");
      plus.animate?.([{ opacity: 0, transform: "translateY(6px)" }, { opacity: 1, transform: "none" }], { duration: 400, fill: "both" });
      nodes.prior.classList.add("credited");
      await sleep(3200);
    }
  }

  window.ProblemDiagram = { start() { const r = document.getElementById("adiagram"); if (r) play(r); }, stop() { token++; } };
})();
