"""Simulated market: a seeded generator of companies, tools and HN-style posts.

We have no time to pull and extract the real HN "Who is hiring" corpus during
the hackathon, so this module generates a market with the same *shape*:

- ~900 fictional companies (domains end in `.example`) with industry, stage,
  size and a latent profile.
- ~100 real-world tool names grouped in categories, each with a description.
- Monthly hiring posts (2019-01 .. 2026-10) whose text names roles, the stack,
  and sometimes a pain sentence, exactly like an HN hiring comment.
- "Ask HN"-style posts written by strangers describing pains.

Only the post *text* leaves this module as evidence. Adoption edges and
features are recovered later by `api.market.extract`, the same way they would
be from the real corpus, so every downstream claim cites a post.

Planted structure (ground truth, kept for evaluation only):
- The founder's product, Ledgerlight (revenue reconciliation), launches
  2022-06. Its adoption hazard rises with (a) billing + CRM in the stack,
  (b) a RevOps hire in the previous six months (the trigger the founder
  never mentions -> "the hit"), (c) a finance-complexity latent.
- Logistics and healthtech companies with heavy ops describe the same pain in
  different words ("freight audit", "remittance matching") and start
  adopting in 2025 ("the one you didn't see").
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

START_YEAR = 2019
N_MONTHS = (2026 - START_YEAR) * 12 + 10  # 2019-01 .. 2026-10 inclusive
LATENT = ["data", "finance", "sales", "infra", "ml", "consumer", "regulated", "ops"]
PRODUCT_NAME = "Ledgerlight"
PRODUCT_LAUNCH = "2022-06"


def month_label(m: int) -> str:
    return f"{START_YEAR + m // 12}-{m % 12 + 1:02d}"


def month_index(label: str) -> int:
    y, mm = map(int, label.split("-")[:2])
    return (y - START_YEAR) * 12 + mm - 1


# --------------------------------------------------------------------------- data


@dataclass
class Company:
    id: int
    name: str
    domain: str
    industry: str
    stage: str
    size: str
    region: str
    description: str
    latent: np.ndarray = field(repr=False)


@dataclass
class Tool:
    id: int
    name: str
    category: str
    description: str


@dataclass
class Post:
    id: int
    month: int
    kind: str  # "hiring" | "ask"
    text: str
    company_id: int | None = None
    author: str = ""
    title: str = ""
    day: int = 1

    @property
    def date(self) -> str:
        return f"{month_label(self.month)}-{self.day:02d}"

    @property
    def url(self) -> str:
        return f"/api/posts/{self.id}"


@dataclass
class World:
    companies: list[Company]
    tools: list[Tool]
    posts: list[Post]
    product_tool_id: int
    truth: dict  # ground truth, only for evaluation and assertions
    seed: int

    def company_by_name(self) -> dict[str, Company]:
        return {c.name.lower(): c for c in self.companies}


# ------------------------------------------------------------------- vocabularies

INDUSTRIES = {
    # industry: latent mean over LATENT dims, customer phrase
    "fintech": ([0.4, 1.2, 0.6, 0.3, 0.1, 0.2, 0.8, 0.2], "payments and lending infrastructure"),
    "devtools": ([0.8, 0.1, 0.4, 1.1, 0.3, 0.0, 0.0, 0.0], "developer tooling"),
    "martech": ([0.6, 0.6, 1.2, 0.2, 0.2, 0.4, 0.0, 0.1], "marketing automation"),
    "saas-billing": ([0.3, 1.3, 1.0, 0.2, 0.0, 0.0, 0.1, 0.2], "billing and subscription software"),
    "ecommerce": ([0.5, 0.5, 0.3, 0.4, 0.3, 1.2, 0.0, 0.6], "online retail"),
    "healthtech": ([0.4, 0.7, 0.1, 0.2, 0.3, 0.1, 1.3, 1.1], "clinic and payer operations"),
    "logistics": ([0.4, 0.8, 0.1, 0.3, 0.1, 0.0, 0.2, 1.4], "freight and fulfilment"),
    "edtech": ([0.3, 0.2, 0.3, 0.2, 0.3, 1.0, 0.2, 0.2], "online learning"),
    "insurtech": ([0.5, 0.9, 0.3, 0.2, 0.5, 0.2, 1.1, 0.5], "insurance distribution"),
    "ml-platform": ([1.1, 0.1, 0.3, 0.9, 1.4, 0.0, 0.0, 0.0], "machine learning infrastructure"),
    "hr-tech": ([0.3, 0.5, 0.7, 0.2, 0.1, 0.2, 0.4, 0.3], "payroll and people operations"),
}
INDUSTRY_WEIGHTS = [0.12, 0.12, 0.11, 0.08, 0.09, 0.09, 0.09, 0.07, 0.06, 0.08, 0.09]

STAGES = ["seed", "series_a", "series_b", "series_c", "public"]
STAGE_P = [0.30, 0.28, 0.22, 0.13, 0.07]
STAGE_LABEL = {"seed": "Seed", "series_a": "Series A", "series_b": "Series B", "series_c": "Series C", "public": "Public"}
SIZE_BY_STAGE = {
    "seed": (["1-50"], [1.0]),
    "series_a": (["1-50", "51-200"], [0.6, 0.4]),
    "series_b": (["51-200", "201-1000"], [0.6, 0.4]),
    "series_c": (["201-1000", "1000+"], [0.7, 0.3]),
    "public": (["201-1000", "1000+"], [0.2, 0.8]),
}
SIZE_ORDER = ["1-50", "51-200", "201-1000", "1000+"]
INDUSTRY_NAME = {"saas-billing": "SaaS billing", "fintech": "fintech", "devtools": "devtools", "martech": "martech",
                 "healthtech": "healthtech", "logistics": "logistics", "ecommerce": "e-commerce", "edtech": "edtech",
                 "insurtech": "insurtech", "ml-platform": "ML platform", "hr-tech": "HR tech"}
POST_RATE = {"1-50": 0.05, "51-200": 0.09, "201-1000": 0.14, "1000+": 0.2}
REGIONS = ["San Francisco", "New York", "REMOTE", "London", "Berlin", "Austin", "Toronto"]

PREFIX = ("Quill Bright Copper Juniper Lumen Harbor Fable Cinder Marble Oak Pebble Tandem Vesper "
          "Willow Arbor Beacon Cobalt Drift Ember Fern Glade Hollow Indigo Kestrel Lattice Meadow "
          "Nimbus Orchid Prism Quarry Rook Sable Thistle Umber Verdant Wren Yarrow Zephyr Alder "
          "Basalt Calyx Dapple Egret Flint Garnet Heron Iris Jasper Kelp Larch").split()
SUFFIX = ("stack loop works ly base grid forge path nest yard hub labs flow line craft bridge field "
          "shift gate port mark sync wise dock tide").split()
INDUSTRY_WORD = {
    "healthtech": ["Health", "Care", "Clinic"], "logistics": ["Freight", "Logistics", "Cargo"],
    "fintech": ["Pay", "Capital", "Ledger"], "insurtech": ["Insure", "Cover", "Mutual"],
}

TOOLS = {
    "warehouse": (["Snowflake", "BigQuery", "Redshift", "ClickHouse", "Databricks"],
                  "cloud data warehouse for analytics queries over large tables",
                  [1.2, 0.2, 0.2, 0.4, 0.4, 0, 0, 0]),
    "orchestration": (["Airflow", "Dagster", "Prefect", "Temporal"],
                      "workflow orchestration for scheduled data pipelines and jobs",
                      [1.0, 0, 0, 0.6, 0.3, 0, 0, 0.1]),
    "transformation": (["dbt", "Fivetran", "Airbyte"],
                       "data transformation and ingestion into the warehouse",
                       [1.1, 0.3, 0.2, 0.1, 0, 0, 0, 0]),
    "crm": (["Salesforce", "HubSpot", "Pipedrive", "Attio"],
            "customer relationship management for sales pipeline and accounts",
            [0, 0.3, 1.3, 0, 0, 0.1, 0, 0]),
    "billing": (["Stripe", "Chargebee", "Recurly", "Zuora", "Metronome", "Orb"],
                "subscription billing invoicing and usage based pricing",
                [0, 1.2, 0.5, 0, 0, 0.3, 0, 0.2]),
    "accounting": (["NetSuite", "QuickBooks", "Xero", "Sage Intacct"],
                   "general ledger accounting and financial close",
                   [0, 1.3, 0.1, 0, 0, 0, 0.3, 0.5]),
    "observability": (["Datadog", "Grafana", "Sentry", "Honeycomb"],
                      "monitoring logs traces and alerting for production systems",
                      [0.2, 0, 0, 1.2, 0, 0, 0, 0]),
    "streaming": (["Kafka", "Kinesis", "Redpanda"],
                  "event streaming and real time data pipelines",
                  [0.9, 0, 0, 0.8, 0.2, 0, 0, 0.2]),
    "database": (["Postgres", "MySQL", "MongoDB", "Redis", "DynamoDB"],
                 "operational database for application data",
                 [0.3, 0, 0, 0.5, 0, 0.3, 0, 0]),
    "infra": (["Kubernetes", "Terraform", "AWS", "GCP", "Pulumi"],
              "cloud infrastructure provisioning and container orchestration",
              [0.2, 0, 0, 1.2, 0.2, 0, 0, 0]),
    "frontend": (["React", "TypeScript", "Next.js", "Vue"],
                 "frontend web application framework",
                 [0, 0, 0.1, 0.1, 0, 0.8, 0, 0]),
    "backend": (["Python", "Go", "Rust", "Node.js", "Ruby on Rails", "Elixir"],
                "backend programming language and web framework",
                [0.2, 0, 0, 0.4, 0.2, 0.2, 0, 0]),
    "ml": (["PyTorch", "Ray", "MLflow", "Weights & Biases"],
           "machine learning training experiment tracking and serving",
           [0.6, 0, 0, 0.3, 1.5, 0, 0, 0]),
    "product-analytics": (["Amplitude", "Mixpanel", "Segment", "PostHog"],
                          "product analytics events funnels and retention",
                          [0.3, 0, 0.3, 0, 0, 1.1, 0, 0]),
    "support": (["Zendesk", "Intercom"], "customer support ticketing and chat",
                [0, 0, 0.3, 0, 0, 0.9, 0, 0.4]),
    "revenue-intel": (["Gong", "Clari", "Outreach"],
                      "revenue intelligence forecasting and sales engagement",
                      [0, 0.4, 1.4, 0, 0, 0, 0, 0]),
    "compliance": (["Vanta", "Drata"], "security compliance automation for SOC 2 audits",
                   [0, 0.2, 0, 0.3, 0, 0, 1.3, 0]),
    "health-systems": (["Epic", "athenahealth", "Availity"],
                       "electronic health records claims and payer connectivity",
                       [0, 0.5, 0, 0, 0, 0, 1.2, 1.0]),
    "logistics-systems": (["Descartes", "MercuryGate", "ShipBob"],
                          "transportation management carrier invoices and fulfilment",
                          [0, 0.5, 0, 0, 0, 0.2, 0, 1.5]),
}
PRODUCT_DESCRIPTION = (
    "Ledgerlight reconciles billing, CRM and the general ledger automatically so finance and "
    "revenue operations teams stop matching invoices to deals in spreadsheets at month end close."
)

ROLES = {
    # role text: latent weights (log-rate boost), base log-rate
    "Senior Backend Engineer": ([0, 0, 0, 0.6, 0, 0.2, 0, 0], -0.4),
    "Frontend Engineer": ([0, 0, 0, 0, 0, 0.8, 0, 0], -1.0),
    "Data Engineer": ([1.0, 0, 0, 0.2, 0.3, 0, 0, 0], -1.4),
    "ML Engineer": ([0.3, 0, 0, 0, 1.4, 0, 0, 0], -2.2),
    "SRE": ([0, 0, 0, 1.0, 0, 0, 0, 0], -1.8),
    "Product Designer": ([0, 0, 0, 0, 0, 0.8, 0, 0], -1.6),
    "Product Manager": ([0, 0, 0.2, 0, 0, 0.4, 0, 0], -1.5),
    "Account Executive": ([0, 0, 1.2, 0, 0, 0, 0, 0], -1.6),
    "SDR": ([0, 0, 1.0, 0, 0, 0, 0, 0], -2.0),
    "Sales Ops Manager": ([0, 0.2, 0.8, 0, 0, 0, 0, 0], -3.0),
    "Controller": ([0, 1.0, 0, 0, 0, 0, 0.2, 0.2], -2.9),
    "FinOps Analyst": ([0, 0.8, 0, 0.3, 0, 0, 0, 0], -3.1),
    "Customer Success Manager": ([0, 0, 0.7, 0, 0, 0.3, 0, 0], -1.9),
    "Security Engineer": ([0, 0, 0, 0.4, 0, 0, 0.9, 0], -2.4),
    "Growth Marketer": ([0, 0, 0.4, 0, 0, 0.8, 0, 0], -2.2),
    "Clinical Ops Lead": ([0, 0, 0, 0, 0, 0, 1.2, 0.6], -3.4),
    "Logistics Analyst": ([0, 0, 0, 0, 0, 0, 0, 1.3], -3.0),
    "Billing Specialist": ([0, 0.9, 0, 0, 0, 0, 0.2, 0.7], -3.3),
}
REVOPS_ROLES = ["Head of RevOps", "Revenue Operations Manager", "RevOps Analyst"]

PAINS = {
    "recon_saas": [
        "Month-end close takes our finance team two weeks of spreadsheet work.",
        "Nobody trusts the revenue numbers because Stripe and Salesforce never match.",
        "We reconcile invoices against the CRM by hand every month.",
        "Our billing data and our pipeline data disagree and someone fixes it manually.",
        "Finance exports invoices and deals into spreadsheets to reconcile revenue every close.",
    ],
    "recon_ops": [
        "Freight audit is still a manual invoice matching exercise in spreadsheets every month end.",
        "Our finance ops team spends days matching carrier invoices to shipments by hand in spreadsheets.",
        "Remittance matching between payer files and our invoices is done by hand at every month end.",
        "Claims reconciliation against remittance advice eats a week of spreadsheet work every close.",
        "We reconcile carrier invoices against purchase orders in spreadsheets and still miss overbilling.",
    ],
    "data_infra": [
        "Our pipelines break every time a schema changes upstream.",
        "Dashboards are slow because the warehouse is a mess of duplicated tables.",
    ],
    "scale_infra": [
        "We are outgrowing our monolith and on-call is painful.",
        "Deploys take an hour and rollbacks are scary.",
    ],
    "ml_ops": [
        "Getting models from notebooks to production takes months.",
        "We retrain models by hand and nobody knows which version is live.",
    ],
    "growth": [
        "We need to understand why users churn after the first week.",
        "Activation is flat and we cannot tell which onboarding step loses people.",
    ],
    "support": [
        "Our support queue doubled and response times slipped.",
    ],
    "compliance": [
        "Every audit means weeks of collecting evidence from ten systems.",
    ],
}
ATTR_SENTENCES = {
    "usage_based_pricing": "We recently moved to usage-based pricing.",
    "multi_entity": "We now operate multiple legal entities across three countries.",
    "intl_expansion": "We are expanding into Europe this year.",
    "recently_raised": "We just raised our {stage}.",
}
ASK_TITLES = {
    "recon_saas": ["Ask HN: How do you reconcile Stripe with your CRM?",
                   "Ask HN: Is a two week month-end close normal at 80 people?"],
    "recon_ops": ["Ask HN: How do logistics companies do freight audit without spreadsheets?",
                  "Ask HN: Anyone automated remittance matching at a clinic group?"],
    "data_infra": ["Ask HN: How do you stop upstream schema changes breaking pipelines?"],
    "scale_infra": ["Ask HN: When did you split your monolith?"],
    "ml_ops": ["Ask HN: How do small teams ship models to production?"],
    "growth": ["Ask HN: How do you find where onboarding loses users?"],
    "support": ["Ask HN: Support queue doubled, what helped you?"],
    "compliance": ["Ask HN: Is SOC 2 evidence collection always this painful?"],
}
ASK_BODIES = {
    "recon_saas": "I run finance at a {size} person SaaS company. {p1} {p2} What tools or processes actually fixed this for you?",
    "recon_ops": "I'm in finance ops at a mid-size {ind} company. {p1} {p2} Is there anything better than more spreadsheets and more people?",
    "default": "{p1} {p2} Curious how others handle this.",
}
AUTHORS = ("quiet_ledger tired_controller opsgremlin throwaway_close mkwan_dev h_ferreira "
           "pallet_math remit_rae ntz flatfile_fan ana_k jvdb carrier_cal sre_sam "
           "deltalake dkoh frieda_ops gbranch_ops ledger_lou").split()


# ---------------------------------------------------------------------- generator


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _pain_theme(rng, industry: str, z: np.ndarray) -> str | None:
    if industry in ("logistics", "healthtech") and z[7] > 1.0 and rng.random() < 0.8:
        return "recon_ops"
    scores = {
        "recon_saas": 1.6 * z[1] + 0.9 * z[2] - 1.8 - (1.5 if industry in ("logistics", "healthtech") else 0),
        "data_infra": 1.2 * z[0] - 1.2,
        "scale_infra": 1.2 * z[3] - 1.2,
        "ml_ops": 1.5 * z[4] - 1.5,
        "growth": 1.3 * z[5] - 1.0,
        "support": 0.8 * z[5] - 1.2,
        "compliance": 1.2 * z[6] - 1.5,
    }
    keys = list(scores)
    w = np.exp(np.array([scores[k] for k in keys]))
    return keys[rng.choice(len(keys), p=w / w.sum())]


def generate(seed: int = 7, n_companies: int = 900) -> World:
    rng = np.random.default_rng(seed)

    # tools -------------------------------------------------------------------
    tools: list[Tool] = []
    tool_w: list[np.ndarray] = []
    tool_b: list[float] = []
    tool_launch: list[int] = []
    for cat, (names, desc, w) in TOOLS.items():
        for name in names:
            flavor = rng.choice(["popular", "growing", "niche"], p=[0.35, 0.35, 0.3])
            tools.append(Tool(len(tools), name, cat, f"{name}: {desc}."))
            tool_w.append(np.array(w, float) + rng.normal(0, 0.15, len(LATENT)))
            tool_b.append({"popular": -1.2, "growing": -2.0, "niche": -2.8}[flavor])
            tool_launch.append(0 if flavor != "growing" else int(rng.integers(0, 40)))
    product_id = len(tools)
    tools.append(Tool(product_id, PRODUCT_NAME, "revenue-reconciliation", PRODUCT_DESCRIPTION))
    tool_by_name = {t.name: t.id for t in tools}
    W = np.stack(tool_w)
    B = np.array(tool_b)

    # companies ---------------------------------------------------------------
    names: set[str] = set()
    companies: list[Company] = []
    inds = list(INDUSTRIES)
    while len(companies) < n_companies:
        ind = inds[rng.choice(len(inds), p=np.array(INDUSTRY_WEIGHTS) / sum(INDUSTRY_WEIGHTS))]
        if ind in INDUSTRY_WORD and rng.random() < 0.5:
            name = f"{rng.choice(PREFIX)} {rng.choice(INDUSTRY_WORD[ind])}"
        else:
            name = f"{rng.choice(PREFIX)}{rng.choice(SUFFIX)}"
        if name in names:
            continue
        names.add(name)
        stage = STAGES[rng.choice(len(STAGES), p=STAGE_P)]
        sizes, sp = SIZE_BY_STAGE[stage]
        size = sizes[rng.choice(len(sizes), p=sp)]
        mean, phrase = INDUSTRIES[ind]
        z = np.clip(np.array(mean) + rng.normal(0, 0.35, len(LATENT)), 0, None)
        slug = name.lower().replace(" ", "")
        companies.append(Company(
            id=len(companies), name=name, domain=f"{slug}.example", industry=ind, stage=stage,
            size=size, region=str(rng.choice(REGIONS)),
            description=f"{name} builds {phrase} software.", latent=z))

    # adoption (ground truth) -------------------------------------------------
    adopt: dict[tuple[int, int], int] = {}
    for c in companies:
        p = _sigmoid(W @ c.latent + B - 0.6)
        for t in range(len(tools) - 1):
            if rng.random() < p[t]:
                lo = tool_launch[t]
                adopt[(c.id, t)] = int(rng.integers(lo, N_MONTHS)) if rng.random() < 0.6 else max(lo, 0)

    # hiring posts (roles first, because the product hazard reads them) --------
    posts: list[Post] = []
    role_names = list(ROLES)
    RW = np.stack([np.array(ROLES[r][0], float) for r in role_names])
    RB = np.array([ROLES[r][1] for r in role_names])
    revops_months: dict[int, list[int]] = {c.id: [] for c in companies}
    company_posts: dict[int, list[tuple]] = {c.id: [] for c in companies}
    stage_change = {c.id: int(rng.integers(20, N_MONTHS)) for c in companies if c.stage in ("series_a", "series_b", "series_c")}
    for c in companies:
        theme = _pain_theme(rng, c.industry, c.latent)
        role_p = _sigmoid(RW @ c.latent + RB)
        revops_p = 0.012
        if c.stage in ("series_b", "series_c") and c.latent[2] + c.latent[1] > 1.4:
            revops_p = 0.07
        elif c.stage in ("series_a", "public"):
            revops_p = 0.02
        rate = POST_RATE[c.size]
        for m in range(N_MONTHS):
            if rng.random() >= rate:
                continue
            roles = [r for r, p in zip(role_names, role_p) if rng.random() < p][:3]
            if rng.random() < revops_p:
                roles.append(str(rng.choice(REVOPS_ROLES)))
                revops_months[c.id].append(m)
            if not roles:
                roles = ["Senior Backend Engineer"]
            pain = theme if (theme and rng.random() < 0.55) else None
            attr = None
            if rng.random() < 0.15:
                attr = str(rng.choice(list(ATTR_SENTENCES)))
            company_posts[c.id].append((m, roles, pain, attr, False))

    # product (Ledgerlight) adoption hazard -----------------------------------
    launch = month_index(PRODUCT_LAUNCH)
    has = lambda cid, cat: any((cid, tool_by_name[n]) in adopt for n in TOOLS[cat][0])  # noqa: E731
    product_adopt: dict[int, int] = {}
    driver: dict[int, str] = {}
    for c in companies:
        stack = 1.0 if (has(c.id, "billing") and has(c.id, "crm")) else 0.0
        ops_seg = c.industry in ("logistics", "healthtech") and c.latent[7] > 1.0
        for m in range(launch, N_MONTHS):
            ramp = min(1.0, (m - launch + 1) / 30)
            recent = any(m - 6 <= r < m for r in revops_months[c.id])
            logit = -8.1 + 1.0 * stack + 3.8 * recent + 0.8 * c.latent[1] + (0.6 if c.size == "1000+" else 0)
            if ops_seg and m >= month_index("2024-11"):
                logit = max(logit, -5.4 + 0.9 * c.latent[7] + 0.5 * c.latent[1])
            if rng.random() < ramp * _sigmoid(logit):
                product_adopt[c.id] = m
                adopt[(c.id, product_id)] = m
                driver[c.id] = ("ops_segment" if ops_seg and m >= month_index("2024-11")
                                else "revops_trigger" if recent else "base")
                # rolling out a finance tool usually comes with a finance hire post
                if rng.random() < 0.75 and m + 2 < N_MONTHS:
                    fm = m + int(rng.integers(1, 3))
                    role = str(rng.choice(["Billing Specialist", "Controller", "FinOps Analyst"]))
                    company_posts[c.id].append((fm, [role], _pain_theme(rng, c.industry, c.latent), None, True))
                break

    # render hiring posts -----------------------------------------------------
    for c in companies:
        adopted = sorted((t for (cid, t), m0 in adopt.items() if cid == c.id), key=lambda t: adopt[(c.id, t)])
        for (m, roles, pain, attr, force) in sorted(company_posts[c.id], key=lambda x: x[0]):
            live = [t for t in adopted if adopt[(c.id, t)] <= m]
            mention = [t for t in live if t != product_id and rng.random() < 0.45][:6]
            if product_id in live and (force or rng.random() < 0.8):
                mention.append(product_id)
            rng.shuffle(mention)
            stage_lbl = STAGE_LABEL[c.stage]
            remote = "REMOTE" if c.region == "REMOTE" else "ONSITE"
            header = f"{c.name} | {', '.join(roles)} | {c.region} | {remote} | {stage_lbl}"
            body = [c.description]
            if mention:
                body.append("Stack: " + ", ".join(tools[t].name for t in mention) + ".")
            if pain:
                body.append(str(rng.choice(PAINS[pain])))
            if attr:
                s = ATTR_SENTENCES[attr]
                if attr == "recently_raised":
                    if c.id not in stage_change or abs(m - stage_change[c.id]) > 2:
                        s = ""
                    s = s.format(stage=stage_lbl)
                if s:
                    body.append(s)
            if rng.random() < 0.06:
                neg = tools[int(rng.integers(0, product_id))].name
                body.append(rng.choice([f"No {neg} experience needed.", f"We are migrating off {neg}."]))
            body.append(f"https://{c.domain}/careers")
            posts.append(Post(id=0, month=m, kind="hiring", company_id=c.id,
                              text=header + "\n" + " ".join(body), author=f"{c.name.split()[0].lower()}_hiring",
                              day=int(rng.integers(1, 6))))

    # stranger posts (Ask HN) -------------------------------------------------
    themes = list(ASK_TITLES)
    tw = np.array([4, 3, 1.2, 1, 1, 1, 0.8, 0.8])
    for _ in range(420):
        th = themes[rng.choice(len(themes), p=tw / tw.sum())]
        m = int(rng.integers(12, N_MONTHS))
        p1, p2 = rng.choice(PAINS[th], 2, replace=len(PAINS[th]) < 2)
        tmpl = ASK_BODIES.get(th, ASK_BODIES["default"])
        body = tmpl.format(size=int(rng.choice([40, 80, 150, 300])), ind=rng.choice(["freight", "healthcare", "3PL"]),
                           p1=p1, p2=p2 if p2 != p1 else "")
        posts.append(Post(id=0, month=m, kind="ask", author=str(rng.choice(AUTHORS)),
                          title=str(rng.choice(ASK_TITLES[th])), text=body, day=int(rng.integers(1, 29))))

    posts.sort(key=lambda p: (p.month, p.day, p.kind, p.company_id or -1))
    for i, p in enumerate(posts):
        p.id = 100000 + i

    truth = {
        "adopt": adopt, "product_adopt": product_adopt, "product_driver": driver,
        "revops_months": revops_months, "tool_launch": tool_launch,
    }
    return World(companies, tools, posts, product_id, truth, seed)
