// The market map: every company as a point, laid out by the model's own embeddings (PCA to 3D).
// Cards light up their evidence: customers, the new segment, the accounts, the one to start with.
import * as THREE from "./vendor/three.module.min.js";

const REDUCED = matchMedia("(prefers-reduced-motion: reduce)").matches;
const INK = new THREE.Color("#f5f5f4");
const ROLE = {
  customers: new THREE.Color("#3987e5"), // categorical slot 1 (dark step)
  segment: new THREE.Color("#d95926"),   // slot 2
  accounts: new THREE.Color("#199e70"),  // slot 3
  focus: new THREE.Color("#ffffff"),
};
const SCALE = 3.1;

const canvas = document.createElement("canvas");
canvas.id = "market-map";
document.body.prepend(canvas);
const labelLayer = document.createElement("div");
labelLayer.id = "map-labels";
document.body.prepend(labelLayer);

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(2, devicePixelRatio));
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 100);
camera.position.set(0, 0.4, 8);
const group = new THREE.Group();
scene.add(group);

let N = 0, names = [], base, pts, geo, mat;
let cur = {}, tgt = {};                // per-point eased state: size, alpha, r, g, b
let roles = {};                        // id -> role
let mode = "ambient", modeT = 0, scanT = -1;
let arc = null, arcT = 0, ring = null, labels = [];
const view = { offset: 0, offsetTgt: 0, dist: 8, distTgt: 8, look: new THREE.Vector3(), lookTgt: new THREE.Vector3(),
               fade: 0, fadeTgt: 0.0 };

function resize() {
  const w = innerWidth, h = innerHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
addEventListener("resize", resize);
resize();

const vert = `
  attribute float size; attribute float alpha; attribute vec3 tint;
  varying float vA; varying vec3 vC;
  void main() {
    vA = alpha; vC = tint;
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = size * (8.0 / -mv.z) * ${renderer.getPixelRatio().toFixed(1)};
    gl_Position = projectionMatrix * mv;
  }`;
const frag = `
  uniform float uFade; varying float vA; varying vec3 vC;
  void main() {
    float d = length(gl_PointCoord - 0.5);
    float a = smoothstep(0.5, 0.32, d) * vA * uFade;
    if (a < 0.01) discard;
    gl_FragColor = vec4(vC, a);
  }`;

async function init() {
  let data;
  for (let i = 0; i < 80 && !data; i++) {
    try { const r = await fetch("/api/world/map"); if (r.ok) data = await r.json(); } catch (e) {}
    if (!data) await new Promise((r) => setTimeout(r, 400));
  }
  N = data.points.length; names = data.names;
  base = new Float32Array(N * 3);
  data.points.forEach((p, i) => base.set([p[0] * SCALE, p[1] * SCALE * 0.8, p[2] * SCALE], i * 3));
  geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(base.slice(), 3));
  for (const k of ["size", "alpha"]) { cur[k] = new Float32Array(N); tgt[k] = new Float32Array(N); }
  cur.tint = new Float32Array(N * 3); tgt.tint = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    cur.size[i] = 0; tgt.size[i] = 6; cur.alpha[i] = 0; tgt.alpha[i] = 0.2;
    cur.tint.set([INK.r, INK.g, INK.b], i * 3); tgt.tint.set([INK.r, INK.g, INK.b], i * 3);
  }
  geo.setAttribute("size", new THREE.BufferAttribute(cur.size, 1));
  geo.setAttribute("alpha", new THREE.BufferAttribute(cur.alpha, 1));
  geo.setAttribute("tint", new THREE.BufferAttribute(cur.tint, 3));
  mat = new THREE.ShaderMaterial({ vertexShader: vert, fragmentShader: frag, transparent: true, depthWrite: false,
                                   uniforms: { uFade: { value: 0 } } });
  pts = new THREE.Points(geo, mat);
  group.add(pts);
  view.fadeTgt = document.querySelector("#enter.active") ? 0 : 1;
  requestAnimationFrame(tick);
}

function centroid(ids) {
  const v = new THREE.Vector3();
  if (!ids?.length) return v;
  for (const i of ids) v.add(new THREE.Vector3(base[i * 3], base[i * 3 + 1], base[i * 3 + 2]));
  return v.divideScalar(ids.length);
}

function clearArc() {
  if (arc) { group.remove(arc); arc.geometry.dispose(); arc = null; }
  if (ring) { group.remove(ring); ring = null; }
}

function setLabels(list) {
  labelLayer.innerHTML = "";
  labels = list.slice(0, 7).map(({ id, role }, i) => {
    const el = document.createElement("span");
    el.className = `map-label ${role}`;
    el.textContent = names[id];
    el.style.animationDelay = `${0.25 + i * 0.08}s`;
    labelLayer.appendChild(el);
    return { id, el };
  });
}

// ---------------------------------------------------------------- public API
const MarketMap = {
  mode(m) {
    mode = m; modeT = 0;
    const reading = m === "reading";
    view.offsetTgt = reading ? 0.22 : 0;
    view.distTgt = reading ? 6.4 : m === "scan" ? 6.8 : 8;
    view.fadeTgt = m === "dash" ? 0 : 1;
    if (m === "scan") scanT = 0;
    if (m !== "reading") { this.show({}); }
    const baseA = m === "scan" ? 0.22 : m === "reading" ? 0.14 : 0.2;
    for (let i = 0; i < N; i++) if (!roles[i]) tgt.alpha[i] = baseA;
  },

  // map: {customers:[], segment:[], accounts:[], focus:[], removed:[]}
  show(map = {}, { arcFrom = "customers", arcTo = "segment" } = {}) {
    if (!N) return;
    roles = {};
    const order = ["accounts", "segment", "customers", "focus"];
    for (const r of order) for (const id of map[r] || []) roles[id] = r;
    const removed = new Set(map.removed || []);
    const dim = Object.keys(roles).length ? 0.1 : 0.16;
    for (let i = 0; i < N; i++) {
      const r = roles[i];
      const c = r ? ROLE[r] : INK;
      tgt.tint.set([c.r, c.g, c.b], i * 3);
      tgt.size[i] = removed.has(i) ? 3 : r === "focus" ? 26 : r === "customers" ? 20 : r ? 14 : 6;
      tgt.alpha[i] = removed.has(i) ? 0.06 : r ? 0.95 : dim;
      if (removed.has(i)) { tgt.tint.set([0.35, 0.35, 0.36], i * 3); }
    }
    clearArc();
    const from = map[arcFrom], to = map[arcTo];
    if (from?.length && to?.length && arcTo === "segment") {
      const a = centroid(from), b = centroid(to.filter((x) => !removed.has(x)));
      const mid = a.clone().add(b).multiplyScalar(0.5);
      mid.add(new THREE.Vector3(0, a.distanceTo(b) * 0.45, 0.4));
      const curve = new THREE.QuadraticBezierCurve3(a, mid, b);
      const g = new THREE.BufferGeometry().setFromPoints(curve.getPoints(80));
      g.setDrawRange(0, 0);
      arc = new THREE.Line(g, new THREE.LineBasicMaterial({ color: ROLE.segment, transparent: true, opacity: 0.75 }));
      group.add(arc); arcT = 0;
    }
    const focusIds = map.focus || [];
    if (focusIds.length) {
      ring = new THREE.Mesh(new THREE.RingGeometry(0.16, 0.19, 48),
                            new THREE.MeshBasicMaterial({ color: INK, transparent: true, opacity: 0.8, side: THREE.DoubleSide }));
      ring.position.copy(centroid(focusIds));
      group.add(ring);
    }
    // aim the camera between the things that matter
    const lit = [...(map.customers || []), ...(map.segment || []), ...focusIds];
    view.lookTgt.copy(lit.length ? centroid(lit).multiplyScalar(0.45) : new THREE.Vector3());
    const lab = [...focusIds.map((id) => ({ id, role: "focus" })),
                 ...(map.customers || []).map((id) => ({ id, role: "customers" })),
                 ...(map.segment || []).filter((x) => !removed.has(x)).slice(0, 3).map((id) => ({ id, role: "segment" }))];
    setLabels(lab);
  },

  // a company named in the interview lights up as you say it
  light(id) {
    if (!N) return;
    roles[id] = "customers";
    tgt.tint.set([ROLE.customers.r, ROLE.customers.g, ROLE.customers.b], id * 3);
    tgt.size[id] = 22; tgt.alpha[id] = 1; cur.size[id] = 60;
    setLabels(Object.keys(roles).map((x) => ({ id: +x, role: "customers" })));
  },
};
window.MarketMap = MarketMap;
init();

// ---------------------------------------------------------------- frame loop
const clock = new THREE.Clock();
const tmp = new THREE.Vector3();
function tick() {
  const dt = Math.min(0.05, clock.getDelta());
  const t = clock.elapsedTime;
  modeT += dt;
  const k = REDUCED ? 1 : 1 - Math.pow(0.002, dt);            // frame-rate independent easing
  view.offset += (view.offsetTgt - view.offset) * k;
  view.dist += (view.distTgt - view.dist) * k * 0.6;
  view.look.lerp(view.lookTgt, k * 0.5);
  view.fade += (view.fadeTgt - view.fade) * k;
  mat.uniforms.uFade.value = view.fade;
  canvas.style.opacity = view.fade.toFixed(3);
  labelLayer.style.opacity = view.fade.toFixed(3);

  if (!REDUCED) group.rotation.y += dt * (mode === "scan" ? 0.35 : mode === "reading" ? 0.05 : 0.08);
  group.rotation.x = Math.sin(t * 0.15) * 0.08;
  camera.position.set(view.look.x, view.look.y + 0.4, view.look.z + view.dist);
  camera.lookAt(view.look);
  const w = innerWidth, h = innerHeight;
  camera.setViewOffset(w, h, view.offset * w, 0, w, h);

  // scan wave: a shell expanding from the centre; points flash as it passes
  if (scanT >= 0) {
    scanT += dt;
    const r = (scanT % 1.6) / 1.6 * 4.2;
    for (let i = 0; i < N; i++) {
      const d = Math.hypot(base[i * 3], base[i * 3 + 1], base[i * 3 + 2]);
      if (Math.abs(d - r) < 0.12 && !REDUCED) { cur.alpha[i] = 0.6; cur.size[i] = 10; }
    }
    if (mode !== "scan") scanT = -1;
  }

  const pulse = 1 + 0.18 * Math.sin(t * 3);
  for (let i = 0; i < N; i++) {
    const role = roles[i];
    const sz = role ? tgt.size[i] * (role === "customers" || role === "focus" ? pulse : 1) : tgt.size[i];
    cur.size[i] += (sz - cur.size[i]) * k;
    cur.alpha[i] += (tgt.alpha[i] - cur.alpha[i]) * k * 0.8;
    for (let j = 0; j < 3; j++) cur.tint[i * 3 + j] += (tgt.tint[i * 3 + j] - cur.tint[i * 3 + j]) * k;
  }
  geo.attributes.size.needsUpdate = true;
  geo.attributes.alpha.needsUpdate = true;
  geo.attributes.tint.needsUpdate = true;

  if (arc) {
    arcT = Math.min(1, arcT + dt / (REDUCED ? 0.01 : 1.4));
    const e = 1 - Math.pow(1 - arcT, 3);
    arc.geometry.setDrawRange(0, Math.floor(e * 81));
  }
  if (ring) {
    ring.quaternion.copy(camera.quaternion);
    ring.quaternion.premultiply(group.quaternion.clone().invert());
    const s = 1 + 0.25 * Math.sin(t * 2.4);
    ring.scale.setScalar(s);
    ring.material.opacity = 0.85 - 0.35 * (s - 1) * 4;
  }

  renderer.render(scene, camera);

  // HTML labels follow their points
  for (const { id, el } of labels) {
    tmp.set(base[id * 3], base[id * 3 + 1], base[id * 3 + 2]).applyMatrix4(group.matrixWorld).project(camera);
    const x = (tmp.x * 0.5 + 0.5) * w, y = (-tmp.y * 0.5 + 0.5) * h;
    el.style.transform = `translate(${x.toFixed(1)}px, ${y.toFixed(1)}px)`;
    el.style.visibility = tmp.z < 1 ? "visible" : "hidden";
  }
  requestAnimationFrame(tick);
}
