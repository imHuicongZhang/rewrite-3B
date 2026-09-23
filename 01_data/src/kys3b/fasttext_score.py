"""DCLM fastText scoring of REWRITTEN text, reproduced exactly from the 1.5B pipeline.

Used only by the REWIRE arm.  Verified against 01_explore/score_fasttext.py and
10_postprocess/03_fasttext_score_rewrite.py:

  preprocessing  text.replace("\\n"," ").replace("\\r"," ")[:100000]
                 -- no lowercasing, no whitespace collapse, no HTML/URL stripping
  raw score      labels, probs = model.predict(cleaned, k=1)
                 raw = p if labels[0] == "__label__hq" else 1 - p
                 empty text -> 0.0  (no predict call)
  model          models/5m/external/fasttext_oh_eli5.bin
                 (mlfoundations/fasttext-oh-eli5 @ cd8b714a...)

The v2 percentile against the original 99,949,162-document raw distribution is provided for
readability only; the retained set is identical whether sorted by the raw score or the
percentile, because the map is monotone.  SORTING USES THE RAW SCORE.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .io import check

CLEAN_MAX_CHARS = 100_000
HQ_LABEL = "__label__hq"


def clean(text: str) -> str:
    """The DCLM-official-style minimal pass.  Do not add normalisation here."""
    if not text:
        return ""
    return text.replace("\n", " ").replace("\r", " ")[:CLEAN_MAX_CHARS]


class FastTextScorer:
    def __init__(self, model_path: Path):
        import fasttext

        model_path = Path(model_path)
        check(model_path.exists(), f"missing fastText model {model_path}")
        self.model = fasttext.load_model(str(model_path))
        check(
            HQ_LABEL in self.model.get_labels(),
            f"{model_path}: {HQ_LABEL} not in labels {self.model.get_labels()}",
        )

    def score_one(self, text: str) -> float:
        c = clean(text)
        if not c:
            return 0.0
        labels, probs = self.model.predict(c, k=1)
        p = float(probs[0])
        return p if labels[0] == HQ_LABEL else 1.0 - p

    def score_many(self, texts) -> np.ndarray:
        # one at a time, exactly as the 1.5B pipeline did
        return np.array([self.score_one(t) for t in texts], dtype=np.float32)


def v2_percentile(raw: np.ndarray, reference_sorted: np.ndarray, reference_n: int) -> np.ndarray:
    """Tie-aware percentile of `raw` against the ORIGINAL pool's raw fastText distribution.

    Matches `fasttext-ranking-v2`'s scale: rankdata(., 'average') / N, evaluated by
    searchsorted against the sorted reference so it is O(log N) per value.
    """
    lo = np.searchsorted(reference_sorted, raw, side="left")
    hi = np.searchsorted(reference_sorted, raw, side="right")
    avg_rank = (lo + hi + 1) / 2.0  # 1-based average rank of the tie group
    return (avg_rank / float(reference_n)).astype(np.float32)
