"""Shared test helpers: a tiny synthetic pool and a scaled-down Config."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kys3b.config import Config  # noqa: E402
from kys3b.pool import Pool  # noqa: E402

TOPICS = [f"T{i:02d}" for i in range(24)]


# Scale 5,000 keeps the tests fast while leaving every block large enough to be meaningful:
# B_s = 4M train tokens ~= 2,600 documents, so DIVERSITY gets ~110 documents per topic and the
# DISAGREEMENT-AWARE per-scorer sets are thousands of documents rather than a handful.  At a much
# larger scale the per-topic quotas would be one document wide and "last document kept whole"
# would dominate every distribution check.
DEFAULT_SCALE = 5_000


def scaled_config(scale: int = DEFAULT_SCALE, tmp_root: Path | None = None) -> Config:
    """The real config with every token budget divided by `scale`, for fast tests.

    Ratios -- B_s = 2B, rewire = kappa*B_s, da_per_scorer = B_s -- are preserved exactly, so a
    test that passes here exercises the same relations the production budgets do.
    """
    c = Config.load()
    c = Config(
        budgets=copy.deepcopy(c.budgets),
        paths=copy.deepcopy(c.paths),
        vllm=copy.deepcopy(c.vllm),
        cluster=copy.deepcopy(c.cluster),
    )
    for k in list(c.budgets["budgets"]):
        c.budgets["budgets"][k] = int(c.budgets["budgets"][k] // scale)
    t = c.budgets["training"]
    t["epoch_tokens"] = int(t["epoch_tokens"] // scale)
    t["horizons"] = [t["epoch_tokens"] * i for i in (1, 2, 3)]
    c.budgets["pool"]["val_size"] = 50
    if tmp_root is not None:
        c.paths["roots"]["data"] = str(tmp_root)
        c.paths["roots"]["pool"] = str(tmp_root / "00_pool")
        c.paths["roots"]["dataset"] = str(tmp_root / "dataset")
    c.check_invariants()
    return c


def synthetic_pool(n: int = 60_000, seed: int = 7) -> Pool:
    """A pool with realistic structure: percentile-like scores, a tie floor, 24 topics.

    The fastText tie floor is deliberate -- the real pool has an 8.6M-document tie at its
    minimum percentile, which is why the seeded tie-break in `order_desc` is load-bearing.
    """
    rng = np.random.default_rng(seed)
    tok = rng.integers(60, 3000, size=n).astype(np.int64) + 1  # TRAIN length
    ft = rng.random(n).astype(np.float32)
    ft[ft < 0.09] = np.float32(0.0433)  # the tie floor
    fw = np.clip(0.45 * ft + 0.55 * rng.random(n), 1e-6, 1.0).astype(np.float32)
    mb = np.clip(0.40 * ft + 0.60 * rng.random(n), 1e-6, 1.0).astype(np.float32)
    ft = np.clip(ft, 1e-6, 1.0).astype(np.float32)
    topic = rng.integers(0, 24, size=n).astype(np.int8)
    return Pool(n=n, tok=tok, ft=ft, fw=fw, mb=mb, topic=topic, vocab=list(TOPICS))


def pass_arrays(doc_ids, status, tokens, key=None, topic=None):
    d = dict(
        doc_id=np.asarray(doc_ids, np.int64),
        status=np.asarray(status, np.int8),
        rewritten_tokens=np.asarray(tokens, np.int64),
    )
    d["len"] = d["rewritten_tokens"] + 1
    d["key"] = np.asarray(key if key is not None else np.zeros(d["doc_id"].size), np.float32)
    if topic is not None:
        d["topic"] = np.asarray(topic, np.int8)
    d["source_tokens"] = np.full(d["doc_id"].size, 1000, np.int64)
    return d
