"""Real-outcome tier 1: UCI Bank Marketing (Moro, Cortez & Rita 2014; CC BY 4.0).

41,188 real telemarketing contacts with real outcomes (term-deposit subscription, ~11%).
Only the founder is simulated. Segments = 8 job groups x 4 age bands (32, the same shape as
fprior.sim, so PC-RFM v0 runs unchanged).

Leakage control: the pool is split once at random into
  A — the "founder's experience": segment rates estimated here become the founder's beliefs;
  B — the market pool: each seed samples a market from B and reveals B's real outcomes.
Fields used: age, job, y. Dropped: duration (leaks the outcome), call timing, macro indicators.
The original calls were targeted, not randomized; every row was contacted, so every account in
our markets has a real outcome.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import logit

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "data/raw/bank/bank-additional/bank-additional-full.csv"

JOB_GROUP = {
    "admin.": 0, "blue-collar": 1, "technician": 2, "services": 3, "housemaid": 3,
    "management": 4, "entrepreneur": 4, "self-employed": 4, "retired": 5,
    "student": 6, "unemployed": 6, "unknown": 7,
}
N_GROUPS, N_BANDS = 8, 4


def age_band(age: int) -> int:
    return 0 if age < 30 else 1 if age < 45 else 2 if age < 60 else 3


@dataclass
class BankPool:
    group: np.ndarray
    band: np.ndarray
    y: np.ndarray

    @property
    def segment(self) -> np.ndarray:
        return self.group * N_BANDS + self.band


def load_bank(path: Path = DEFAULT_CSV) -> BankPool:
    g, b, y = [], [], []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh, delimiter=";"):
            g.append(JOB_GROUP[row["job"]])
            b.append(age_band(int(row["age"])))
            y.append(1.0 if row["y"] == "yes" else 0.0)
    return BankPool(np.array(g), np.array(b), np.array(y))


HILLSTROM_CSV = ROOT / "data/raw/hillstrom/hillstrom.csv"


def load_hillstrom(path: Path = HILLSTROM_CSV) -> BankPool:
    """Real-outcome tier 2: Hillstrom MineThatData e-mail test (2008; public, no licence stated — do not
    redistribute the raw file). Customers who were randomly e-mailed (two thirds); outcome = website visit
    within two weeks (16.7%). Segments in the same 8 x 4 shape as Bank: "group" = recency quartile (1-3,
    4-6, 7-9, 10-12 months) x newbie flag; "band" = past-year spend (<100, 100-200, 200-500, 500+ USD).
    Channel, product line and zip code are not used."""
    g, b, y = [], [], []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["segment"] == "No E-Mail":
                continue
            rec, hist = int(row["recency"]), float(row["history"])
            g.append(min((rec - 1) // 3, 3) * 2 + int(row["newbie"]))
            b.append(0 if hist < 100 else 1 if hist < 200 else 2 if hist < 500 else 3)
            y.append(float(row["visit"]))
    return BankPool(np.array(g), np.array(b), np.array(y))


def load_pool(name: str = "bank") -> BankPool:
    return {"bank": load_bank, "hillstrom": load_hillstrom}[name]()


@dataclass
class BankMarket:
    """Duck-types fprior.sim.Market for make_founder / run_outreach / CPR-LP / PC-RFM."""
    industry: np.ndarray
    size: np.ndarray
    segment: np.ndarray
    n_segments: int
    intercept: float
    segment_logit: np.ndarray      # founder's best possible knowledge: rates estimated on split A
    y: np.ndarray                  # real outcomes
    graph_cov: np.ndarray

    @property
    def n(self) -> int:
        return len(self.segment)

    @property
    def p(self) -> np.ndarray:     # metrics take "p"; here it is the realized outcome
        return self.y


def split_pool(pool: BankPool, seed: int = 12345):
    idx = np.random.default_rng(seed).permutation(len(pool.y))
    return idx[: len(idx) // 2], idx[len(idx) // 2:]


def experience_logits(pool: BankPool, A: np.ndarray, prior_n: float = 2.0) -> tuple[np.ndarray, float]:
    """Segment logits estimated on split A, shrunk toward the overall rate."""
    K = N_GROUPS * N_BANDS
    seg, y = pool.segment[A], pool.y[A]
    base = y.mean()
    pos = np.bincount(seg, weights=y, minlength=K)
    cnt = np.bincount(seg, minlength=K)
    rate = (pos + prior_n * base) / (cnt + prior_n)
    return logit(rate), float(logit(base))


def make_bank_market(pool: BankPool, A: np.ndarray, B: np.ndarray, n_accounts: int,
                     rng: np.random.Generator) -> BankMarket:
    """Category indices shown to models are randomly relabelled per market (industry / size
    one-hots), so a model cannot score well by an accidental match between our numbering of
    job groups / age bands and its training-time habits. Segment identity is unchanged."""
    seg_logit, c = experience_logits(pool, A)
    idx = rng.choice(B, size=n_accounts, replace=False)
    pg, pb = rng.permutation(N_GROUPS), rng.permutation(N_BANDS)
    g, b = pg[pool.group[idx]], pb[pool.band[idx]]
    seg = g * N_BANDS + b
    # re-index the founder's segment logits to the relabelled segments
    old = np.arange(N_GROUPS * N_BANDS)
    new = pg[old // N_BANDS] * N_BANDS + pb[old % N_BANDS]
    relabelled = np.empty_like(seg_logit)
    relabelled[new] = seg_logit
    return BankMarket(g, b, seg, N_GROUPS * N_BANDS, c, relabelled, pool.y[idx], np.eye(n_accounts))
