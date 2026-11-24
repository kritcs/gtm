// Static demo (GitHub Pages): replays recorded API responses and agent events, so app.js runs unchanged.
// Loaded first by dist/index.html only; scripts/build_static.py records static/demo-data.json.
(function () {
  window.STATIC_DEMO = true;
  const realFetch = window.fetch.bind(window);
  const data = realFetch("static/demo-data.json").then((r) => r.json());
  const st = { fixed: false, acct: null, stage: null, answer: 0 };
  const drafts = {};          // draft id -> account id
  const log = [], sources = new Set();
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const fk = () => (st.fixed ? "fixed" : "base");

  const deliver = (ev) => { log.push(ev); for (const s of sources) s._emit(ev); };
  async function play(rec) {          // events arrive while the request "runs", then the response
    if (!rec) return null;
    let prev = null;
    for (const ev of rec.ev || []) {
      const gap = prev == null ? 120 : Math.min(450, Math.max(60, (ev.ts - prev) * 1000));
      prev = ev.ts;
      await sleep(gap);
      deliver(ev);
    }
    return rec.r;
  }

  class FakeEventSource {
    constructor() { this.onmessage = null; sources.add(this); setTimeout(() => log.forEach((ev) => this._emit(ev)), 0); }
    _emit(ev) { this.onmessage?.({ data: JSON.stringify(ev) }); }
    addEventListener(t, f) { if (t === "message") this.onmessage = f; }
    close() { sources.delete(this); }
  }
  window.EventSource = FakeEventSource;

  async function route(method, path, body) {
    const D = await data;
    if (method === "GET" && D.static[path]) return D.static[path];
    if (method === "POST" && path === "/api/sessions") return play(D.start);
    if (method === "POST" && path === "/api/webhooks/reply") {
      const id = body?.draft_id || (body?.text || "").match(/ref ([0-9a-f]{8})/)?.[1];
      return reply(D, drafts[id]);
    }
    const m = path.match(/^\/api\/sessions\/[^/]+\/(.+)$/);
    if (!m) return null;
    const [, rest] = m;
    if (method === "GET" && D.states.base[rest]) {
      const key = st.acct ? `${fk()}|${st.acct}|${st.stage}` : fk();
      return (D.states[key] || D.states[fk()] || D.states.base)[rest];
    }
    if (rest === "answer") return play(D.answers[Math.min(st.answer++, D.answers.length - 1)]);
    if (rest === "reading") return play(D.reading);
    if (rest === "enrich") return play(D.enrich[fk()]);
    let f = rest.match(/^cards\/([^/]+)\/feedback$/);
    if (f) {
      const yes = /^\s*yes\b/i.test(body?.text || "");
      if (f[1] === "discovery" && !yes) st.fixed = true;
      return play(D.feedback[f[1]]?.[yes ? "yes" : "no"]);
    }
    if (rest === "drafts") {
      const aid = String(body?.account_id);
      const rec = D.drafts[fk()][aid] || D.drafts.base[aid] || D.drafts.fixed[aid];
      if (!rec) return null;
      drafts[rec.r.id] = aid;
      if (D.states[`${fk()}|${aid}|drafted`]) { st.acct = aid; st.stage = "drafted"; }
      return play(rec);
    }
    f = rest.match(/^drafts\/([^/]+)\/approve$/);
    if (f) {
      const aid = drafts[f[1]], rec = D.approve[fk()][aid];
      if (rec && st.acct === aid) st.stage = "sent";
      // nobody can reply to a static page, so the prospect "replies" a few seconds after sending
      setTimeout(() => route("POST", "/api/webhooks/reply", { draft_id: f[1] }), 6000);
      return rec ? play(rec) : { id: f[1], status: "sent", body: body?.body };
    }
    return null;
  }

  async function reply(D, aid) {
    const rec = aid && D.replies[fk()][aid];
    if (!rec || st.replied?.has(aid)) return null;
    (st.replied ||= new Set()).add(aid);
    if (st.acct === aid) st.stage = "replied";
    return play(rec);
  }

  window.fetch = async (input, opts = {}) => {
    const url = typeof input === "string" ? input : input.url;
    const m = url.match(/\/api\/[^?#]*/);
    if (!m || url.includes("api/posts/")) return realFetch(input, opts);
    const method = (opts.method || "GET").toUpperCase();
    const body = opts.body ? JSON.parse(opts.body) : null;
    const out = await route(method, m[0], body);
    return out == null
      ? new Response(JSON.stringify({ detail: "not in the recorded demo" }), { status: 404 })
      : new Response(JSON.stringify(out), { status: 200, headers: { "Content-Type": "application/json" } });
  };
})();
