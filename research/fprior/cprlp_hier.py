"""Hierarchical CPR-LP: segment = industry x size, with shared main effects.

Adds random effects for industry and size on top of CPR-LP's per-cell segment deviation, so evidence
about one cell informs cells that share its industry or size (results/diag_learner_gap.md). Implemented
through CPR-LP's exact random-effect mechanism (one-hot extra_features with N(0, s^2) coefficients, i.e.
kernel s^2 ([same industry] + [same size])). The main-effect scale s is evidence-averaged over a grid.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .cprlp import CPRConfig, CPRResult, fit_predict

MAIN_SD_GRID = (0.0, 0.25, 0.5, 1.0)   # 0 = flat model; evidence chooses whether to pool


def main_effect_features(industry: np.ndarray, size: np.ndarray, n_ind: int = 8, n_size: int = 4) -> np.ndarray:
    return np.column_stack([np.eye(n_ind)[industry], np.eye(n_size)[size]])


def fit_predict_hier(segment, n_segments, graph_cov, founder_segment_logit, contacted, outcome,
                     industry, size, cfg: CPRConfig = CPRConfig(), grid=MAIN_SD_GRID) -> CPRResult:
    """Evidence-weighted average over main-effect scales. With no labels, all scales give the founder
    prior mean (coefficients sit at 0), so the first fit is returned."""
    X = main_effect_features(industry, size)
    fits = [fit_predict(segment, n_segments, graph_cov, founder_segment_logit, contacted, outcome,
                        replace(cfg, extra_sd=s), extra_features=X) for s in grid]
    if len(contacted) == 0:
        return fits[0]
    le = np.array([f.log_evidence for f in fits])
    w = np.exp(le - le.max())
    w /= w.sum()
    out = fits[int(np.argmax(w))]
    out.logit = sum(wi * f.logit for wi, f in zip(w, fits))
    if out.prob is not None:
        out.prob = sum(wi * f.prob for wi, f in zip(w, fits))
    return out
