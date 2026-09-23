"""Config loading + the invariant gates.

Every budget, path and engine kwarg lives in configs/*.yaml.  Nothing in the pipeline
hard-codes a budget.  `Config.check_invariants()` is what plan.md promises: the 3B budget
relations are asserted, not assumed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from .io import check

CODE_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = CODE_ROOT / "configs"

# The exact five kwargs the 1.5B run passed to vllm.LLM().  Anything else is a config error.
ALLOWED_ENGINE_KWARGS = {
    "tensor_parallel_size",
    "dtype",
    "gpu_memory_utilization",
    "max_model_len",
}
ALLOWED_SAMPLING_KWARGS = {"temperature", "top_p", "max_tokens"}

SETTINGS = (
    "quality-base",
    "quality-first",
    "wrap-inspired",
    "rewire-inspired",
    "diversity-oriented",
    "disagreement-aware",
)
REWRITE_SETTINGS = tuple(s for s in SETTINGS if s != "quality-base")
WRAP_STYLES = ["easy", "hard", "wiki", "qa"]  # index order is part of the reproducible seed
PASSES = ("p1", "distill")


def _load(name: str) -> dict:
    p = CONFIG_DIR / name
    check(p.exists(), f"missing config {p}")
    return yaml.safe_load(p.read_text())


@dataclass
class Config:
    budgets: dict
    paths: dict
    vllm: dict
    cluster: dict

    # ---------------------------------------------------------------- loading
    @classmethod
    def load(cls) -> "Config":
        c = cls(
            budgets=_load("budgets.yaml"),
            paths=_load("paths.yaml"),
            vllm=_load("vllm.yaml"),
            cluster=_load("cluster.yaml"),
        )
        c.check_invariants()
        return c

    # ---------------------------------------------------------------- budgets
    @property
    def seed(self) -> int:
        return int(self.budgets["seed"])

    @property
    def lam(self) -> float:
        return float(self.budgets["lambda_da"])

    @property
    def kappa(self) -> int:
        return int(self.budgets["kappa"])

    @property
    def pool(self) -> dict:
        return self.budgets["pool"]

    @property
    def B(self) -> int:
        """Rewritten-output budget."""
        return int(self.budgets["budgets"]["rewritten_target"])

    @property
    def Bs(self) -> int:
        """Source budget for the four non-REWIRE rewriting arms."""
        return int(self.budgets["budgets"]["source"])

    def budget(self, name: str) -> int:
        return int(self.budgets["budgets"][name])

    def setting(self, name: str) -> dict:
        check(name in self.budgets["settings"], f"unknown setting {name!r}")
        return self.budgets["settings"][name]

    def source_budget(self, name: str) -> int | None:
        s = self.setting(name)
        key = s.get("source_budget")
        return None if key is None else self.budget(key)

    # ---------------------------------------------------------------- paths
    def root(self, key: str) -> Path:
        return Path(self.paths["roots"][key])

    def stage(self, key: str) -> Path:
        return self.root("dataset") / self.paths["stages"][key]

    def model(self, key: str) -> Path:
        return Path(self.paths["models"][key])

    def pool_shard(self, i: int) -> Path:
        return self.root("pool") / f"merged_clean_{i:05d}.parquet"

    # ---------------------------------------------------------------- vllm
    def engine_kwargs(self) -> dict:
        e = dict(self.vllm["engine"])
        extra = set(e) - ALLOWED_ENGINE_KWARGS
        check(not extra, f"configs/vllm.yaml engine has non-1.5B kwargs: {sorted(extra)}")
        missing = ALLOWED_ENGINE_KWARGS - set(e)
        check(not missing, f"configs/vllm.yaml engine missing kwargs: {sorted(missing)}")
        return e

    def sampling_kwargs(self) -> dict:
        s = dict(self.vllm["sampling"])
        extra = set(s) - ALLOWED_SAMPLING_KWARGS
        check(not extra, f"configs/vllm.yaml sampling has non-1.5B kwargs: {sorted(extra)}")
        return s

    @property
    def heartbeat_seconds(self) -> int:
        return int(self.budgets["sharding"]["heartbeat_seconds"])

    @property
    def claim_stale_seconds(self) -> int:
        return int(self.budgets["sharding"]["claim_stale_seconds"])

    def input_drop(self, pass_name: str) -> int:
        check(pass_name in PASSES, f"unknown pass {pass_name!r}")
        return int(self.vllm["input_drop"][pass_name])

    # ---------------------------------------------------------------- invariants
    def check_invariants(self) -> None:
        b = self.budgets["budgets"]
        inv = self.budgets.get("invariants", {})
        if inv.get("source_eq_2x_rewritten"):
            check(
                b["source"] == 2 * b["rewritten_target"],
                f"B_s ({b['source']:,}) != 2*B ({2*b['rewritten_target']:,})",
            )
        if inv.get("rewire_eq_kappa_x_source"):
            check(
                b["rewire_source"] == self.kappa * b["source"],
                f"REWIRE source ({b['rewire_source']:,}) != kappa*B_s "
                f"({self.kappa * b['source']:,})",
            )
        if inv.get("da_per_scorer_eq_source"):
            check(
                b["da_per_scorer"] == b["source"],
                f"DA per-scorer budget ({b['da_per_scorer']:,}) != B_s ({b['source']:,}) "
                "-- Option C is locked in plan.md section 20 D-2",
            )
        if inv.get("base_eq_quality_base"):
            check(
                b["shared_base"] == b["quality_base"],
                f"shared_base ({b['shared_base']:,}) != quality_base ({b['quality_base']:,})",
            )
        check(self.lam == 0.5, f"lambda must be 0.5 (Decision 12), got {self.lam}")
        check(self.kappa == 2, f"kappa must be 2, got {self.kappa}")
        check(
            set(self.budgets["settings"]) == set(SETTINGS),
            f"configs/budgets.yaml settings != the six locked settings: "
            f"{sorted(set(self.budgets['settings']) ^ set(SETTINGS))}",
        )
        # every rewriting arm must generate BOTH passes (Decision 3)
        for s in REWRITE_SETTINGS:
            rw = self.setting(s)["rewrite"]
            check(rw is not None, f"{s}: rewrite must not be null")
            check(
                rw.get("distill") is True,
                f"{s}: distill pass must be generated (Decision 3 -- no adaptive skip)",
            )
            check(
                rw.get("pass1") in ("grounded", "wrap"),
                f"{s}: pass1 must be 'grounded' or 'wrap', got {rw.get('pass1')!r}",
            )
        check(self.setting("quality-base")["rewrite"] is None, "quality-base must not be rewritten")
        # training horizons must be matched multiples of the epoch budget, never corpus-derived
        t = self.budgets["training"]
        ep = int(t["epoch_tokens"])
        check(
            list(t["horizons"]) == [ep, 2 * ep, 3 * ep],
            f"training.horizons must be [1,2,3]*epoch_tokens, got {t['horizons']}",
        )
        check(len(t["seeds"]) == 3, f"expected 3 seeds, got {t['seeds']}")
        sh = self.budgets["sharding"]
        hb, stale = int(sh["heartbeat_seconds"]), int(sh["claim_stale_seconds"])
        check(hb > 0, f"heartbeat_seconds must be positive, got {hb}")
        check(
            hb * 3 <= stale,
            f"heartbeat_seconds ({hb}) * 3 must be <= claim_stale_seconds ({stale}) so a claim "
            "survives a few missed refreshes before it is judged dead",
        )


def env_offline() -> None:
    """Never reach the HF hub from a compute node (the 1.5B convention)."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
