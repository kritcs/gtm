// How-it-works background: a dome of light streaks. Every arc starts at one point (your intuition)
// and lands on one company in the market. Comet heads run along the arcs; additive blending; slow drift.
import * as THREE from "./vendor/three.module.min.js";

const REDUCED = matchMedia("(prefers-reduced-motion: reduce)").matches;
const host = document.getElementById("dome-bg");
if (host) {
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(2, devicePixelRatio));
  host.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100);
  camera.position.set(0, 1.6, 11);
  camera.lookAt(0, 1.4, 0);
  const group = new THREE.Group();
  group.position.y = -2.6;
  scene.add(group);

  const R = 4.2, N = 900, SEG = 64;
  // origin on the dome, upper left: the founder's intuition
  const origin = new THREE.Vector3(-0.72, 0.42, 0.55).normalize().multiplyScalar(R);
  // endpoints: Fibonacci points on the upper hemisphere, one per company
  const ends = [];
  const golden = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < N * 2 && ends.length < N; i++) {
    const y = 1 - (i / (N * 2 - 1)) * 2, r = Math.sqrt(1 - y * y), th = golden * i;
    if (y < 0.02) continue;
    ends.push(new THREE.Vector3(Math.cos(th) * r, y, Math.sin(th) * r).multiplyScalar(R));
  }

  // arcs: slerp on the sphere, lifted slightly mid-way
  const pos = new Float32Array(ends.length * SEG * 2 * 3), aT = new Float32Array(ends.length * SEG * 2), aSeed = new Float32Array(ends.length * SEG * 2);
  const o = origin.clone().normalize();
  let k = 0;
  ends.forEach((e, idx) => {
    const n = e.clone().normalize(), seed = Math.random();
    const ang = o.angleTo(n), axis = new THREE.Vector3().crossVectors(o, n).normalize();
    const pt = (t) => {
      const v = o.clone().applyAxisAngle(axis, ang * t);
      return v.multiplyScalar(R * (1 + 0.06 * Math.sin(Math.PI * t)));
    };
    let prev = pt(0);
    for (let s = 1; s <= SEG; s++) {
      const t0 = (s - 1) / SEG, t1 = s / SEG, cur = pt(t1);
      pos.set([prev.x, prev.y, prev.z, cur.x, cur.y, cur.z], k * 3);
      aT[k] = t0; aT[k + 1] = t1; aSeed[k] = aSeed[k + 1] = seed;
      k += 2; prev = cur;
    }
  });
  const lg = new THREE.BufferGeometry();
  lg.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  lg.setAttribute("aT", new THREE.BufferAttribute(aT, 1));
  lg.setAttribute("aSeed", new THREE.BufferAttribute(aSeed, 1));
  const lm = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    uniforms: { uTime: { value: 0 }, uReveal: { value: 0 } },
    vertexShader: `attribute float aT; attribute float aSeed; varying float vT; varying float vSeed;
      void main() { vT = aT; vSeed = aSeed; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
    fragmentShader: `uniform float uTime; uniform float uReveal; varying float vT; varying float vSeed;
      void main() {
        if (vT > uReveal + vSeed * 0.15) discard;
        float head = fract(uTime * (0.08 + vSeed * 0.07) + vSeed);
        float d = head - vT;
        float comet = d > 0.0 ? exp(-d * 9.0) : 0.0;
        float base = 0.025 + 0.025 * (1.0 - vT);
        vec3 col = mix(vec3(0.75, 0.78, 1.0), vec3(0.75, 0.98, 1.0), vT);
        gl_FragColor = vec4(col, base + comet * 0.55);
      }`,
  });
  group.add(new THREE.LineSegments(lg, lm));

  // endpoint dots that light up when a comet arrives
  const pg = new THREE.BufferGeometry();
  pg.setAttribute("position", new THREE.BufferAttribute(new Float32Array(ends.flatMap((e) => [e.x, e.y, e.z])), 3));
  pg.setAttribute("aSeed", new THREE.BufferAttribute(aSeed.filter((_, i) => i % (SEG * 2) === 0), 1));
  const pm = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    uniforms: { uTime: { value: 0 }, uReveal: { value: 0 }, uDpr: { value: renderer.getPixelRatio() } },
    vertexShader: `attribute float aSeed; uniform float uTime; uniform float uDpr; varying float vA;
      void main() {
        float head = fract(uTime * (0.08 + aSeed * 0.07) + aSeed);
        vA = 0.25 + 0.75 * smoothstep(0.9, 1.0, head);
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        gl_PointSize = (2.0 + 3.0 * vA) * uDpr;
        gl_Position = projectionMatrix * mv;
      }`,
    fragmentShader: `uniform float uReveal; varying float vA;
      void main() { float d = length(gl_PointCoord - 0.5); if (d > 0.5 || uReveal < 0.9) discard; gl_FragColor = vec4(1.0, 1.0, 1.0, vA * (1.0 - d * 2.0)); }`,
  });
  group.add(new THREE.Points(pg, pm));

  // the origin: a soft bright core
  const core = new THREE.Mesh(new THREE.SphereGeometry(0.035, 24, 24), new THREE.MeshBasicMaterial({ color: 0xffffff }));
  core.position.copy(origin);
  group.add(core);
  const halo = new THREE.Sprite(new THREE.SpriteMaterial({
    map: (() => { const c = document.createElement("canvas"); c.width = c.height = 64; const x = c.getContext("2d");
      const g = x.createRadialGradient(32, 32, 0, 32, 32, 32); g.addColorStop(0, "rgba(255,255,255,.55)"); g.addColorStop(.4, "rgba(200,220,255,.18)"); g.addColorStop(1, "rgba(255,255,255,0)");
      x.fillStyle = g; x.fillRect(0, 0, 64, 64); return new THREE.CanvasTexture(c); })(),
    blending: THREE.AdditiveBlending, depthWrite: false, transparent: true }));
  halo.position.copy(origin); halo.scale.set(0.5, 0.5, 1);
  group.add(halo);

  const resize = () => {
    const w = host.clientWidth, h = host.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.position.z = w / h < 1 ? 16 : 11;
    camera.updateProjectionMatrix();
  };
  addEventListener("resize", resize);
  resize();

  const clock = new THREE.Clock();
  let reveal = REDUCED ? 1 : 0, running = false;
  const tick = () => {
    if (!running) return;
    const t = clock.getElapsedTime();
    reveal = Math.min(1.2, reveal + (REDUCED ? 1 : 0.008));
    lm.uniforms.uTime.value = pm.uniforms.uTime.value = REDUCED ? 3 : t;
    lm.uniforms.uReveal.value = pm.uniforms.uReveal.value = reveal;
    if (!REDUCED) group.rotation.y = Math.sin(t * 0.06) * 0.18;
    halo.scale.setScalar(0.5 + 0.06 * Math.sin(t * 2));
    renderer.render(scene, camera);
    requestAnimationFrame(tick);
  };
  // the How it works page turns this on; it renders only while visible
  window.DomeBG = { stop() { running = false; }, start() { if (!running) { running = true; resize(); requestAnimationFrame(tick); } } };
}
