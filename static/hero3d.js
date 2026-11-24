// Front-flow backgrounds, drawn with canvas 2D. Two scenes, each with a meaning:
//   bottleneck: market signals on the left funnel through one glowing neck (your instinct) and fan out
//               to customers on the right; comets flow left to right through the neck.
//   fan:        rays burst up from one point (the founder) to companies across the market; pulses travel out.
// API: window.HeroBG.scene("bottleneck" | "fan"), .stop(), .start()
const REDUCED = matchMedia("(prefers-reduced-motion: reduce)").matches;
const host = document.getElementById("hero-bg");

if (host) {
  const cv = document.createElement("canvas");
  host.appendChild(cv);
  const ctx = cv.getContext("2d");
  let W = 0, H = 0, dpr = 1, running = true, scene = "bottleneck", mix = 1, prev = null;
  const rnd = (i, k = 1) => { const x = Math.sin(i * 127.1 + k * 311.7) * 43758.5453; return x - Math.floor(x); };

  const resize = () => {
    dpr = Math.min(2, devicePixelRatio);
    W = host.clientWidth; H = host.clientHeight;
    cv.width = W * dpr; cv.height = H * dpr; cv.style.width = W + "px"; cv.style.height = H + "px";
    build();
  };

  // --------------------------------------------------------------- geometry
  let funnel = [], rays = [];
  function build() {
    const neck = { x: W * 0.5, y: H * 0.74 };
    const NL = W < 700 ? 34 : 56, NR = W < 700 ? 28 : 46;
    const left = Array.from({ length: NL }, (_, i) => ({ x: W * (0.04 + rnd(i, 1) * 0.03), y: H * (0.38 + (i / (NL - 1)) * 0.6) + (rnd(i, 2) - .5) * 8 }));
    const right = Array.from({ length: NR }, (_, i) => ({ x: W * (0.93 + rnd(i, 3) * 0.03), y: H * (0.5 + (i / (NR - 1)) * 0.48) + (rnd(i, 4) - .5) * 8 }));
    funnel = { neck, left, right };
    const NRAY = W < 700 ? 70 : 120, o = { x: W * 0.5, y: H * 1.02 };
    rays = Array.from({ length: NRAY }, (_, i) => {
      const a = Math.PI + (0.04 + rnd(i, 5) * 0.92) * Math.PI; // upper half
      const len = Math.min(W * 0.5, H * 0.62) * (0.35 + Math.pow(rnd(i, 6), 0.8) * 0.65);
      return { o, a, len, dot: 0.8 + rnd(i, 7) * 1.6, phase: rnd(i, 8), speed: 0.12 + rnd(i, 9) * 0.18, alpha: 0.18 + rnd(i, 10) * 0.4 };
    });
  }

  // cubic bezier point
  const bz = (p0, c1, c2, p1, t) => {
    const u = 1 - t;
    return { x: u * u * u * p0.x + 3 * u * u * t * c1.x + 3 * u * t * t * c2.x + t * t * t * p1.x,
             y: u * u * u * p0.y + 3 * u * u * t * c1.y + 3 * u * t * t * c2.y + t * t * t * p1.y };
  };
  const lineL = (p, n) => [p, { x: W * 0.32, y: p.y }, { x: W * 0.42, y: n.y }, n];
  const lineR = (p, n) => [n, { x: W * 0.58, y: n.y }, { x: W * 0.68, y: p.y }, p];

  // --------------------------------------------------------------- draw
  function drawBottleneck(t, alpha) {
    const { neck, left, right } = funnel;
    ctx.globalAlpha = alpha;
    ctx.lineWidth = 1;
    const strokeCurve = (pts, a) => {
      ctx.strokeStyle = `rgba(190,200,255,${a})`;
      ctx.beginPath(); ctx.moveTo(pts[0].x, pts[0].y);
      ctx.bezierCurveTo(pts[1].x, pts[1].y, pts[2].x, pts[2].y, pts[3].x, pts[3].y); ctx.stroke();
    };
    left.forEach((p, i) => strokeCurve(lineL(p, neck), 0.16 + 0.12 * rnd(i, 11)));
    right.forEach((p, i) => strokeCurve(lineR(p, neck), 0.16 + 0.12 * rnd(i, 12)));
    // endpoint dots
    ctx.fillStyle = "rgba(220,228,255,.85)";
    [...left, ...right].forEach((p, i) => { ctx.beginPath(); ctx.arc(p.x, p.y, 1.6 + rnd(i, 13) * 1.4, 0, 6.283); ctx.fill(); });
    // the neck: your instinct
    const g = ctx.createRadialGradient(neck.x, neck.y, 0, neck.x, neck.y, 60);
    g.addColorStop(0, "rgba(255,255,255,.9)"); g.addColorStop(.25, "rgba(190,230,255,.35)"); g.addColorStop(1, "rgba(155,231,240,0)");
    ctx.fillStyle = g; ctx.beginPath(); ctx.arc(neck.x, neck.y, 60 + (REDUCED ? 0 : 4 * Math.sin(t * 2)), 0, 6.283); ctx.fill();
    // comets: travel a left curve into the neck, then out along a right curve
    if (!REDUCED) for (let k = 0; k < 26; k++) {
      const s = (t * (0.12 + rnd(k, 14) * 0.08) + rnd(k, 15)) % 1;
      const li = Math.floor(rnd(k, 16) * left.length), ri = Math.floor(rnd(k, 17) * right.length);
      const pts = s < 0.5 ? lineL(left[li], neck) : lineR(right[ri], neck);
      const u = s < 0.5 ? s * 2 : (s - 0.5) * 2;
      for (let j = 0; j < 7; j++) {
        const uu = Math.max(0, u - j * 0.012), p = bz(pts[0], pts[1], pts[2], pts[3], uu);
        ctx.fillStyle = `rgba(255,255,255,${(1 - j / 7) * 0.9})`;
        ctx.beginPath(); ctx.arc(p.x, p.y, 1.8 - j * 0.18, 0, 6.283); ctx.fill();
      }
    }
    ctx.globalAlpha = 1;
  }

  function drawFan(t, alpha) {
    ctx.globalAlpha = alpha;
    rays.forEach((r) => {
      const grow = REDUCED ? 1 : 0.92 + 0.08 * Math.sin(t * 0.6 + r.phase * 6.28);
      const ex = r.o.x + Math.cos(r.a) * r.len * grow, ey = r.o.y + Math.sin(r.a) * r.len * grow;
      const lg = ctx.createLinearGradient(r.o.x, r.o.y, ex, ey);
      lg.addColorStop(0, `rgba(155,231,240,${r.alpha * 0.9})`); lg.addColorStop(1, `rgba(190,200,255,${r.alpha * 0.35})`);
      ctx.strokeStyle = lg; ctx.lineWidth = 0.8;
      ctx.beginPath(); ctx.moveTo(r.o.x, r.o.y); ctx.lineTo(ex, ey); ctx.stroke();
      ctx.fillStyle = `rgba(225,232,255,${0.4 + r.alpha * 0.6})`;
      ctx.beginPath(); ctx.arc(ex, ey, r.dot, 0, 6.283); ctx.fill();
      if (!REDUCED) { // a pulse travelling outward
        const s = (t * r.speed + r.phase) % 1;
        const px = r.o.x + Math.cos(r.a) * r.len * grow * s, py = r.o.y + Math.sin(r.a) * r.len * grow * s;
        ctx.fillStyle = `rgba(255,255,255,${0.7 * (1 - s)})`;
        ctx.beginPath(); ctx.arc(px, py, 1.4, 0, 6.283); ctx.fill();
      }
    });
    const o = rays[0]?.o;
    if (o) {
      const g = ctx.createRadialGradient(o.x, o.y, 0, o.x, o.y, 110);
      g.addColorStop(0, "rgba(255,255,255,.55)"); g.addColorStop(1, "rgba(155,231,240,0)");
      ctx.fillStyle = g; ctx.beginPath(); ctx.arc(o.x, o.y, 110, 0, 6.283); ctx.fill();
    }
    ctx.globalAlpha = 1;
  }

  const DRAW = { bottleneck: drawBottleneck, fan: drawFan };
  const t0 = performance.now();
  function tick() {
    if (!running) return;
    const t = (performance.now() - t0) / 1000;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    mix = Math.min(1, mix + (REDUCED ? 1 : 0.04));
    if (prev && mix < 1) DRAW[prev](t, 1 - mix);
    DRAW[scene](t, mix);
    requestAnimationFrame(tick);
  }

  addEventListener("resize", resize);
  resize();
  requestAnimationFrame(tick);
  window.HeroBG = {
    scene(name) { if (name !== scene && DRAW[name]) { prev = scene; scene = name; mix = 0; } },
    stop() { running = false; },
    start() { if (!running) { running = true; requestAnimationFrame(tick); } },
  };
}
