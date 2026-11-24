// Shared display helpers: sentence case, industry names, plain-English odds.
(function () {
  const IND = { "saas-billing": "SaaS billing", fintech: "Fintech", devtools: "Devtools", martech: "Martech", healthtech: "Healthtech",
    logistics: "Logistics", ecommerce: "E-commerce", edtech: "Edtech", insurtech: "Insurtech", "ml-platform": "ML platform", "hr-tech": "HR tech" };
  const STAGE = { seed: "Seed", series_a: "Series A", series_b: "Series B", series_c: "Series C", public: "Public" };
  window.cap = (s) => (s ? String(s)[0].toUpperCase() + String(s).slice(1) : s);
  window.prettyInd = (s) => IND[s] || window.cap(s);
  window.prettyStage = (s) => STAGE[s] || window.cap(s);
  // q is the corrected chance a pattern is luck; say it as odds
  window.oddsText = (q) => {
    q = Number(q);
    if (!isFinite(q) || q <= 0) return "";
    if (q >= 0.05) return "could be luck";
    const n = 1 / q, mag = Math.pow(10, Math.floor(Math.log10(n)) - 1);
    return `1 in ${(Math.round(n / mag) * mag).toLocaleString()} odds of luck`;
  };
})();
