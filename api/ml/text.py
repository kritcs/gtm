"""Text embeddings: TF-IDF over unigrams and bigrams, then LSA (truncated SVD).

The production plan uses bge-small. LSA needs no model download, trains in
about a second on the simulated corpus, and still links different vocabularies
that co-occur ("freight audit" ~ "invoice matching"), which is what pain
matching needs.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np

TOKEN = re.compile(r"[a-z][a-z\-]+")
STOP = set("""a an the and or of to in on for with is are was were be been our we you your it its this that
these those at by from as into than then there their they them i my me he she his her us not no so if
but about all any can could do does did done has have had how just more most much some such too very
what when where which who why will would also every each other per one two three year month
week weeks after before first need needs still new get use used work day days time times even same""".split())


def stem(w: str) -> str:
    """Tiny suffix stripper: invoices/invoice -> invoic, matching/matched -> match."""
    if w.endswith("ies") and len(w) > 5:
        return w[:-3] + "y"
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and not w.endswith("ss") and len(w) - len(suf) >= 4:
            w = w[: -len(suf)]
            break
    return w[:-1] if w.endswith("e") and len(w) > 4 else w


def tokens(text: str) -> list[str]:
    words = [stem(w) for w in TOKEN.findall(text.lower()) if w not in STOP and len(w) > 2]
    return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]


class LSA:
    def __init__(self, dim: int = 64, max_vocab: int = 3000, min_df: int = 2):
        self.dim, self.max_vocab, self.min_df = dim, max_vocab, min_df

    def fit(self, docs: list[str]) -> "LSA":
        df = Counter()
        toks = [tokens(d) for d in docs]
        for t in toks:
            df.update(set(t))
        # sort by (-df, word): Counter ties would otherwise follow set order, which varies per process
        vocab = [w for w, n in sorted(df.items(), key=lambda x: (-x[1], x[0])) if n >= self.min_df][: self.max_vocab]
        self.vocab = {w: i for i, w in enumerate(vocab)}
        n = len(docs)
        self.idf = np.array([np.log((1 + n) / (1 + df[w])) + 1 for w in vocab], np.float32)
        X = self._tfidf(toks)
        # eigendecomposition of the (vocab x vocab) Gram matrix: cheap truncated SVD
        G = X.T @ X
        vals, vecs = np.linalg.eigh(G)
        order = np.argsort(vals)[::-1][: self.dim]
        self.components = vecs[:, order].astype(np.float32)  # (V, dim)
        # random projection of raw TF-IDF (Johnson-Lindenstrauss): keeps lexical cosine in 256 dims
        rng = np.random.default_rng(0)
        self.rp = (rng.normal(0, 1, (len(vocab), 256)) / np.sqrt(256)).astype(np.float32)
        return self

    def _tfidf(self, toks: list[list[str]]) -> np.ndarray:
        X = np.zeros((len(toks), len(self.vocab)), np.float32)
        for i, t in enumerate(toks):
            for w, c in Counter(t).items():
                j = self.vocab.get(w)
                if j is not None:
                    X[i, j] = (1 + np.log(c)) * self.idf[j]
        X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
        return X

    def transform(self, docs: list[str]) -> np.ndarray:
        Z = self._tfidf([tokens(d) for d in docs]) @ self.components
        return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)


    def transform_hybrid(self, docs: list[str], dense_weight: float = 0.5) -> np.ndarray:
        """Dense (LSA) + lexical (projected TF-IDF), like blending embeddings with BM25.

        cos(hybrid_a, hybrid_b) ~= w * cos_lsa + (1 - w) * cos_tfidf.
        """
        X = self._tfidf([tokens(d) for d in docs])
        z1 = X @ self.components
        z1 /= np.linalg.norm(z1, axis=1, keepdims=True) + 1e-9
        z2 = X @ self.rp
        z2 /= np.linalg.norm(z2, axis=1, keepdims=True) + 1e-9
        return np.hstack([np.sqrt(dense_weight) * z1, np.sqrt(1 - dense_weight) * z2]).astype(np.float32)


def free_text(post) -> str:
    """The parts of a post a person wrote: no header, stack list, company blurb or URL."""
    if post.kind == "ask":
        return f"{post.title} {post.text}"
    body = post.text.split("\n", 1)[-1]
    keep = [x.strip() for x in re.split(r"(?<=\.)\s", body)
            if x.strip() and not x.startswith("Stack:") and "http" not in x and " builds " not in x]
    return " ".join(keep)


def cos(a: np.ndarray, B: np.ndarray) -> np.ndarray:
    return B @ a / ((np.linalg.norm(B, axis=-1) + 1e-9) * (np.linalg.norm(a) + 1e-9))
