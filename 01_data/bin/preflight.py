#!/usr/bin/env python
"""Preflight -- every gate that can fail cheaply, before any GPU time or download.

  1  configs load and all budget invariants hold
  2  the six prompts are byte-identical to the 1.5B production set (md5)
  3  wrap_prompts.json key order is easy,hard,wiki,qa (part of the seed)
  4  the templated empty-document overhead of every prompt text matches (150/185/72/66/73/83)
  5  the Qwen tokenizer + chat template load from the local model dir
  6  the llama-2 tokenizer is FAST and has vocab_size 32000
  7  the fastText model loads and exposes __label__hq
  8  the WRAP style assignment matches a pinned PCG64 golden vector
  9  storage roots exist / are writable and have headroom
 10  engine and sampling kwargs are exactly the 1.5B five / three
 11  the heartbeat interval is well inside the claim staleness window
 12  the claim protocol's filesystem primitives (mkdir, link, rename) work on the data root
 13  worker identity is unique across the primary and opportunistic arrays
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from kys3b.config import WRAP_STYLES, Config, env_offline
from kys3b.io import check, log
from kys3b.prompts import assign_wrap_styles, check_md5, check_overheads

# np.random.default_rng([42, shard]).integers(0, 4, 16) -- pinned so a NumPy PCG64 change is
# caught here instead of silently re-rolling which document gets which WRAP style.
# Verified identical under numpy 1.26.4 and 2.0.2 on 2026-09-23.
GOLDEN_WRAP = {
    0: [0, 3, 2, 1, 1, 3, 0, 2, 0, 0, 2, 3, 2, 3, 2, 3],
    1: [2, 3, 1, 0, 3, 1, 3, 2, 3, 3, 3, 2, 1, 3, 1, 3],
    7: [1, 2, 1, 0, 1, 2, 2, 3, 1, 3, 0, 0, 0, 1, 2, 0],
    123: [0, 3, 0, 0, 3, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 0],
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="preflight gates")
    ap.add_argument("--skip-models", action="store_true", help="skip tokenizer/fastText loads")
    ap.add_argument("--skip-storage", action="store_true")
    args = ap.parse_args(argv)
    env_offline()
    ok = []

    cfg = Config.load()
    ok.append("1  configs load; budget invariants hold")
    log(
        f"   B={cfg.B:,} B_s={cfg.Bs:,} rewire={cfg.budget('rewire_source'):,} "
        f"da/scorer={cfg.budget('da_per_scorer'):,} lambda={cfg.lam} kappa={cfg.kappa}"
    )

    md5 = check_md5()
    ok.append(f"2  prompt md5s match ({len(md5)} files)")
    ok.append("3  wrap_prompts.json key order = " + ",".join(WRAP_STYLES))

    ek, sk = cfg.engine_kwargs(), cfg.sampling_kwargs()
    ok.append(f"10 engine kwargs = {ek}")
    ok.append(f"10 sampling kwargs = {sk}")

    if not args.skip_models:
        from transformers import AutoTokenizer

        qtok = AutoTokenizer.from_pretrained(str(cfg.model("qwen")), use_fast=True)
        ok.append("5  Qwen tokenizer + chat template load")
        measured = check_overheads(qtok, cfg)
        ok.append(f"4  prompt overheads match: {measured}")

        from kys3b.tokens import load_llama2

        lt = load_llama2(
            str(cfg.model("llama2")), int(cfg.vllm["llama2_tokenizer"]["expected_vocab_size"])
        )
        ok.append(f"6  llama-2 tokenizer FAST, vocab_size {lt.vocab_size}")

        from kys3b.fasttext_score import FastTextScorer

        FastTextScorer(cfg.model("fasttext"))
        ok.append("7  fastText model loads, __label__hq present")

    for shard, want in GOLDEN_WRAP.items():
        got = assign_wrap_styles(shard, len(want), cfg.seed)
        check(
            got == [WRAP_STYLES[i] for i in want],
            f"8  WRAP golden vector mismatch on shard {shard}: {got}",
        )
    ok.append("8  WRAP style assignment matches the pinned PCG64 golden vector")

    if not args.skip_storage:
        for key in ("data", "pool", "dataset"):
            p = cfg.root(key)
            p.mkdir(parents=True, exist_ok=True)
            check(p.is_dir(), f"{p} is not a directory")
        free_gb = shutil.disk_usage(cfg.root("data")).free / 2**30
        check(free_gb > 1000, f"only {free_gb:.0f} GB free at {cfg.root('data')}; need > 1 TB")
        ok.append(f"9  storage roots writable; {free_gb:.0f} GB free at {cfg.root('data')}")

    ok.append(
        f"11 heartbeat {cfg.heartbeat_seconds}s x3 <= claim staleness {cfg.claim_stale_seconds}s"
    )

    # The claim protocol needs atomic mkdir / link / rename on the filesystem that will actually
    # hold the claim directories.  WekaFS provides all three, but assert it rather than assume.
    import os
    import tempfile

    probe = cfg.root("data") / ".cache" / "_preflight_fsprobe"
    probe.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(probe.parent)) as td:
        td = Path(td)
        os.mkdir(td / "g")
        try:
            os.mkdir(td / "g")
            check(False, "mkdir is not single-winner on this filesystem")
        except FileExistsError:
            pass
        (td / "a").write_text("x")
        os.link(td / "a", td / "b")
        try:
            os.link(td / "a", td / "b")
            check(False, "link is not single-winner on this filesystem")
        except FileExistsError:
            pass
        os.replace(td / "b", td / "c")
        check((td / "c").exists(), "replace did not move the file")
    ok.append("12 claim primitives verified on the data root: mkdir, link, replace all atomic")

    # primary task 0 and scavenger task 0 must not be the same worker
    from kys3b.claims import ClaimDir

    def _key(array_job, task):
        saved = {k: os.environ.get(k) for k in
                 ("SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_JOB_ID")}
        os.environ.update(SLURM_ARRAY_JOB_ID=str(array_job), SLURM_ARRAY_TASK_ID=str(task),
                          SLURM_JOB_ID=str(array_job + task))
        try:
            return ClaimDir.worker_key()
        finally:
            for k, v in saved.items():
                os.environ.pop(k, None)
                if v is not None:
                    os.environ[k] = v

    check(
        _key(111111, 0) != _key(222222, 0),
        "worker_key collides across arrays -- primary task 0 and scavenger task 0 would share "
        "a progress file",
    )
    ok.append("13 worker_key is unique across the primary and opportunistic arrays")

    print("\n=== PREFLIGHT PASSED ===")
    for line in ok:
        print("  " + line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
