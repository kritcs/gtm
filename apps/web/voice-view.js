// Your voice: drop in documents, watch a LoRA fine-tune run. A simulation: files stay in the browser.
(function () {
  const RM = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const I = (n) => `<i class="i" style="--i:url(/static/vendor/icons/${n}.svg)"></i>`;
  const SAMPLES = [["Sent emails.mbox", 2_400_000], ["Investor update Q3.md", 18_000], ["Sales call notes.txt", 64_000], ["Launch post.md", 9_000]];
  const st = { files: [], trained: false, running: false };
  const kb = (b) => (b > 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${Math.max(1, Math.round(b / 1e3))} KB`);
  const tokens = () => Math.round(st.files.reduce((a, f) => a + f.size, 0) / 4.2);

  function render(root) {
    const ready = st.files.length > 0;
    root.innerHTML = `<div class="voice">
      <div class="vhead"><h3>Teach it your voice</h3>
        <p class="muted">Drop in emails and docs you wrote. We fine-tune a small model on them so every draft sounds like you.</p></div>
      <label class="vdrop" id="vdrop">${I("upload")}<span>Drop files or click to choose</span>
        <input type="file" id="vfile" multiple hidden></label>
      <div class="vfiles">${st.files.map((f) => `<span class="vfile">${I("file-text")}${esc(f.name)}<b>${kb(f.size)}</b></span>`).join("")}
        ${ready ? "" : `<button class="btn" id="vsample">Use sample docs</button>`}</div>
      <div class="vtrain" id="vtrain" ${ready ? "" : "hidden"}>
        <div class="vrow"><button class="btn primary" id="vgo" ${st.running ? "disabled" : ""}>${I("play")}${st.trained ? "Train again" : "Train"}</button>
          <span class="mono small muted">${tokens().toLocaleString()} tokens · Llama 3.1 8B · LoRA r=16</span></div>
        <div class="vstages" id="vstages"></div>
        <svg class="vloss" id="vloss" viewBox="0 0 600 180"></svg>
        <p class="vdone" id="vdone">${st.trained ? "Your model is ready. Drafts now use it." : ""}</p>
      </div></div>`;
    const add = (list) => { st.files.push(...[...list].map((f) => ({ name: f.name, size: f.size }))); st.trained = false; render(root); };
    root.querySelector("#vfile").onchange = (e) => add(e.target.files);
    const drop = root.querySelector("#vdrop");
    drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("over"); };
    drop.ondragleave = () => drop.classList.remove("over");
    drop.ondrop = (e) => { e.preventDefault(); add(e.dataTransfer.files); };
    const s = root.querySelector("#vsample");
    if (s) s.onclick = () => add(SAMPLES.map(([name, size]) => ({ name, size })));
    if (ready) root.querySelector("#vgo").onclick = () => train(root);
  }

  async function train(root) {
    st.running = true; st.trained = false; render(root);
    const stages = root.querySelector("#vstages"), svg = root.querySelector("#vloss"), done = root.querySelector("#vdone");
    const STEPS = [["Tokenize", 900], ["LoRA fine-tune", 5200], ["DPO on your edits", 1600]];
    stages.innerHTML = STEPS.map(([n]) => `<div class="vstage"><span>${n}</span><div class="vbar"><i></i></div><b class="mono">0%</b></div>`).join("");
    const rows = [...stages.children];
    // loss falls like a real fine-tune: fast drop, slow tail, a little noise
    const N = 300, loss = Array.from({ length: N }, (_, i) => 0.62 + 1.9 * Math.exp(-i / 55) + 0.07 * Math.sin(i * 1.7) * Math.exp(-i / 200) + 0.04 * (Math.sin(i * 12.9898) * 43758.5453 % 1));
    const X = (i) => (i / (N - 1)) * 600, Y = (v) => 170 - ((v - 0.5) / 2.2) * 160;
    svg.innerHTML = `<line x1="0" x2="600" y1="${Y(0.62)}" y2="${Y(0.62)}" class="vbase"/><path class="vline"/><text x="596" y="14" class="vlbl" text-anchor="end">Training loss</text>`;
    const path = svg.querySelector(".vline");
    for (const [k, [, ms]] of STEPS.entries()) {
      const row = rows[k], bar = row.querySelector("i"), pct = row.querySelector("b"), t0 = performance.now();
      row.classList.add("on");
      await new Promise((res) => {
        const f = (now) => {
          const t = RM ? 1 : Math.min(1, (now - t0) / ms);
          bar.style.width = `${t * 100}%`; pct.textContent = `${Math.round(t * 100)}%`;
          if (k === 1) {
            const n = Math.max(2, Math.round(t * N));
            path.setAttribute("d", "M" + loss.slice(0, n).map((v, i) => `${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join("L"));
            pct.textContent = `step ${Math.round(t * N)}/${N} · loss ${loss[n - 1].toFixed(2)}`;
          }
          t < 1 ? requestAnimationFrame(f) : res();
        };
        requestAnimationFrame(f);
      });
      row.classList.add("ok");
    }
    st.running = false; st.trained = true;
    done.textContent = "Your model is ready. Drafts now use it.";
    root.querySelector("#vgo").disabled = false;
  }

  window.VoiceView = { render, state: st };
})();
