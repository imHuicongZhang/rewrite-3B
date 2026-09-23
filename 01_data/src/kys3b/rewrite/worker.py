"""One data-parallel vLLM worker: one GPU, one vLLM process, tensor_parallel_size=1.

This is the merge of the 1.5B `07_rewrite/rewrite_worker.py` and `09_Distill/rewrite_worker.py`.
Those two files differed only in prompt, output subdirectory, drop threshold and cache
namespace; here those are `--pass {p1,distill}` plus a per-pass table.  Every behavioural
detail is preserved:

  * ONE llm.generate() call per shard (continuous batching)
  * per-doc SamplingParams with max_tokens = min(4096, max_model_len - n_in)
  * status 0 = templated input over the pass's drop threshold (NOT rewritten)
           1 = finish_reason == 'length'   (truncated)
           2 = finish_reason == 'stop'     (complete)
  * rewritten_tokens counted with the LLAMA-2 tokenizer, add_special_tokens=False
  * atomic .tmp + os.replace per shard; an existing output shard is skipped (resume)
  * distill reads the RAW SOURCE shards, never pass-1 output (asserted)
  * the wrap styled pass records `wrap_style`; other passes do not

Deviation from 1.5B, approved as Decision 8: shard ownership is a claim directory rather
than `shard_idx % num_workers`.  Per-document output is unaffected -- it depends only on
(shard_index, row_index).
"""
from __future__ import annotations

import argparse
import fcntl
import json
import re
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ..claims import ClaimDir
from ..config import PASSES, Config, env_offline
from ..io import atomic_write_table, check, log, parquet_rows
from ..prompts import (
    assign_wrap_styles,
    build_content,
    check_md5,
    check_overheads,
    load_text,
    wrap_prompts,
)
from ..shards import load_index, shard_path
from ..tokens import count_batch, load_llama2

# per-pass behaviour table -- the only thing that differed between 07_rewrite and 09_Distill
PASS_SPEC = {
    "p1": dict(out_subdir="p1", overhead_modes=("grounded_p1", "wrap")),
    "distill": dict(out_subdir="distill", overhead_modes=("distill",)),
}


# --------------------------------------------------------------------------- monitoring
def _word_set(s: str) -> set:
    return {w for w in re.split(r"\W+", (s or "").lower()) if w}


def flag_format_only(src: str, out: str) -> bool:
    """>90% of the output's word types also appear in the input (1.5B heuristic)."""
    o = _word_set(out)
    if not o:
        return False
    return (len(o & _word_set(src)) / len(o)) > 0.90


def flag_repetition(out: str, win: int = 50, reps: int = 5) -> bool:
    if not out or len(out) < win:
        return False
    seen: dict[str, int] = {}
    for k in range(0, len(out) - win + 1, 10):
        sub = out[k : k + win]
        c = seen.get(sub, 0) + 1
        seen[sub] = c
        if c >= reps:
            return True
    return False


def flag_short(out_tokens: int, in_tokens: int) -> bool:
    return out_tokens < 20 and in_tokens > 200


def append_monitor(monitor_file: Path, setting: str, pass_name: str, worker: str, samples: list) -> None:
    lines = []
    for s in samples:
        warns = []
        if s["status"] == 2:
            if flag_format_only(s["src"], s["out"]):
                warns.append("FORMAT-ONLY (>90% token overlap)")
            if flag_repetition(s["out"]):
                warns.append("DEGENERATE REPETITION")
            if flag_short(s["out_tokens"], s["in_tokens"]):
                warns.append("SUSPICIOUSLY SHORT")
        hdr = (
            f"- **{setting}/{pass_name}** | worker {worker} | doc_id={s['doc_id']} "
            f"| status={s['status']} | in_tok={s['in_tokens']} out_tok={s['out_tokens']}"
        )
        if warns:
            hdr += "  WARNING: " + "; ".join(warns)
        lines.append(hdr)
        lines.append(f"  - input[:500]: {s['src'][:500]!r}")
        lines.append(f"  - output: {s['out'][:2000]!r}")
    block = (
        f"\n### {setting}/{pass_name} -- worker {worker} -- sample @ "
        f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n" + "\n".join(lines) + "\n"
    )
    monitor_file = Path(monitor_file)
    monitor_file.parent.mkdir(parents=True, exist_ok=True)
    with open(monitor_file, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(block)
        fcntl.flock(f, fcntl.LOCK_UN)


# --------------------------------------------------------------------------- worker
def run(args) -> int:
    env_offline()
    cfg = Config.load()
    check(args.pass_name in PASSES, f"--pass must be one of {PASSES}")
    spec = PASS_SPEC[args.pass_name]
    setting = args.setting
    sdef = cfg.setting(setting)
    check(sdef["rewrite"] is not None, f"{setting} is a raw setting and is never rewritten")

    # prompt mode for this (setting, pass)
    if args.pass_name == "distill":
        mode = "grounded"           # every arm, including wrap, uses the distill prompt here
        template = load_text("p2_distill")
        wp = None
    else:
        p1 = sdef["rewrite"]["pass1"]
        if p1 == "wrap":
            mode, template, wp = "wrap", None, wrap_prompts()
        else:
            mode, template, wp = "grounded", load_text("p1_wiki"), None

    src_dir = cfg.stage("sources") / setting
    out_dir = cfg.stage("rewritten") / setting / spec["out_subdir"]
    # Raw-source guarantee (the 1.5B 09_Distill assertion): never read an output dir.
    check(
        cfg.stage("rewritten") not in src_dir.parents and src_dir.name != "p1",
        f"refusing to read rewritten output as input: {src_dir}",
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # --dry-run must work BEFORE stage 02 has run, so the prompt gates can be checked first
    if args.dry_run:
        return _dry_run(cfg, args, setting, mode, template, wp, src_dir, None)

    index = load_index(src_dir)
    n_shards = int(index["n_shards"])
    claims = ClaimDir(
        cfg.stage("rewritten") / setting / "_claims" / spec["out_subdir"],
        stale_seconds=int(cfg.budgets["sharding"]["claim_stale_seconds"]),
    )

    def done(s: int) -> bool:
        p = shard_path(out_dir, s)
        if not p.exists():
            return False
        try:
            return parquet_rows(p) == int(index["shards"][s]["rows"])
        except Exception:
            return False

    todo_now = [s for s in range(n_shards) if not done(s)]
    log(
        f"{setting}/{args.pass_name}: {n_shards} shards, {n_shards-len(todo_now)} already done, "
        f"{len(todo_now)} outstanding"
    )
    if not todo_now:
        log("nothing to do")
        return 0

    # ---- tokenizers + gates BEFORE the engine loads (fail cheap) ----
    from transformers import AutoTokenizer

    check_md5()
    qtok = AutoTokenizer.from_pretrained(str(cfg.model("qwen")), use_fast=True)
    modes = ("wrap",) if mode == "wrap" else spec["overhead_modes"][:1]
    if args.pass_name == "distill":
        modes = ("distill",)
    measured = check_overheads(qtok, cfg, modes=modes)
    log(f"prompt overhead gate OK: {measured}")
    ltok = load_llama2(
        str(cfg.model("llama2")), int(cfg.vllm["llama2_tokenizer"]["expected_vocab_size"])
    )

    # ---- engine ----
    from vllm import LLM, SamplingParams

    ek = cfg.engine_kwargs()
    llm = LLM(model=str(cfg.model("qwen")), **ek)
    max_model_len = int(ek["max_model_len"])
    smp = cfg.sampling_kwargs()
    drop = cfg.input_drop(args.pass_name)
    chat = cfg.vllm["chat"]
    log(f"engine up: {ek}; sampling {smp}; input_drop {drop}")

    prog_dir = cfg.stage("rewritten") / setting / "_progress" / spec["out_subdir"]
    prog_dir.mkdir(parents=True, exist_ok=True)
    wid = claims.identity()["worker"]
    prog = dict(
        setting=setting, pass_name=args.pass_name, worker=wid,
        shards_completed=0, docs_completed=0,
        docs_status_0=0, docs_status_1=0, docs_status_2=0,
        total_input_tokens=0, total_output_tokens=0, elapsed_seconds=0.0,
    )
    t0 = time.time()
    since_monitor = 0
    monitor_every = int(cfg.vllm["monitor_every"])
    monitor_file = cfg.stage("logs") / f"monitor_{setting}_{args.pass_name}.md"

    for shard in claims.iter_available(range(n_shards), done):
        try:
            t = pq.read_table(shard_path(src_dir, shard), use_threads=False)
            doc_ids = t.column("doc_id").to_numpy(zero_copy_only=False)
            texts = t.column("text").to_pylist()
            n_rows = len(texts)
            check(
                n_rows == int(index["shards"][shard]["rows"]),
                f"shard {shard}: {n_rows} rows != index {index['shards'][shard]['rows']}",
            )

            styles = assign_wrap_styles(shard, n_rows, cfg.seed) if mode == "wrap" else [None] * n_rows

            prompts, params, keep_pos, n_in_all = [], [], [], [0] * n_rows
            status = [0] * n_rows
            for j in range(n_rows):
                content = build_content(mode, texts[j], template, wp, styles[j])
                final = qtok.apply_chat_template(
                    [{"role": chat["role"], "content": content}],
                    add_generation_prompt=chat["add_generation_prompt"],
                    tokenize=False,
                )
                n_in = len(qtok(final, add_special_tokens=False).input_ids)
                n_in_all[j] = n_in
                if n_in > drop:
                    continue  # status stays 0
                max_new = min(int(smp["max_tokens"]), max_model_len - n_in)
                prompts.append(final)
                params.append(
                    SamplingParams(
                        temperature=smp["temperature"],
                        top_p=smp["top_p"],
                        max_tokens=max(1, max_new),
                    )
                )
                keep_pos.append(j)

            rewritten = [""] * n_rows
            finish = [""] * n_rows
            if prompts:
                claims.heartbeat(shard)
                outs = llm.generate(prompts, params)
                for k, g in enumerate(outs):
                    j = keep_pos[k]
                    o = g.outputs[0]
                    rewritten[j] = o.text
                    finish[j] = o.finish_reason or ""
                    status[j] = 1 if o.finish_reason == "length" else 2

            rtok = count_batch(ltok, rewritten)
            cols = {
                "doc_id": pa.array(doc_ids, type=pa.int64()),
                "rewritten": pa.array(rewritten, type=pa.large_string()),
                "rewritten_tokens": pa.array(rtok, type=pa.int32()),
                "status": pa.array(status, type=pa.int8()),
                "finish_reason": pa.array(finish, type=pa.large_string()),
                "input_tokens_qwen": pa.array(n_in_all, type=pa.int32()),
            }
            if mode == "wrap":
                cols["wrap_style"] = pa.array(styles, type=pa.large_string())
            atomic_write_table(pa.table(cols), shard_path(out_dir, shard))

            prog["shards_completed"] += 1
            prog["docs_completed"] += n_rows
            prog["docs_status_0"] += status.count(0)
            prog["docs_status_1"] += status.count(1)
            prog["docs_status_2"] += status.count(2)
            prog["total_input_tokens"] += int(sum(n_in_all[j] for j in keep_pos))
            prog["total_output_tokens"] += int(sum(rtok))
            prog["elapsed_seconds"] = round(time.time() - t0, 1)
            (prog_dir / f"worker_{wid}.json").write_text(json.dumps(prog, indent=2))

            since_monitor += n_rows
            if since_monitor >= monitor_every:
                since_monitor = 0
                pick = list(range(0, n_rows, max(1, n_rows // 10)))[:10]
                append_monitor(
                    monitor_file, setting, args.pass_name, str(wid),
                    [
                        dict(
                            doc_id=int(doc_ids[j]), status=status[j], src=texts[j] or "",
                            out=rewritten[j], in_tokens=n_in_all[j], out_tokens=rtok[j],
                        )
                        for j in pick
                    ],
                )
            log(
                f"  shard {shard:05d} done: {n_rows} docs "
                f"(s0={status.count(0)} s1={status.count(1)} s2={status.count(2)}) "
                f"out_tok={sum(rtok):,}"
            )
        finally:
            claims.release(shard)

    log(f"worker {wid} finished: {prog['shards_completed']} shards in {prog['elapsed_seconds']}s")
    return 0


def _dry_run(cfg, args, setting, mode, template, wp, src_dir, index) -> int:
    """Build prompts and token counts for the first rows of the first shard. No GPU."""
    from transformers import AutoTokenizer

    check_md5()
    qtok = AutoTokenizer.from_pretrained(str(cfg.model("qwen")), use_fast=True)
    modes = ("wrap",) if mode == "wrap" else (("distill",) if args.pass_name == "distill" else ("grounded_p1",))
    log(f"prompt overhead gate: {check_overheads(qtok, cfg, modes=modes)}")
    p = shard_path(src_dir, 0)
    if not p.exists():
        log(f"(no materialized shard at {p}; prompt gates passed, nothing else to show)")
        return 0
    t = pq.read_table(p, use_threads=False).slice(0, args.dry_rows)
    texts = t.column("text").to_pylist()
    styles = assign_wrap_styles(0, len(texts), cfg.seed) if mode == "wrap" else [None] * len(texts)
    drop = cfg.input_drop(args.pass_name)
    s0 = 0
    for j, tx in enumerate(texts):
        content = build_content(mode, tx, template, wp, styles[j])
        final = qtok.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False
        )
        n_in = len(qtok(final, add_special_tokens=False).input_ids)
        s0 += n_in > drop
        if j == 0:
            print("\n----- sample templated prompt (truncated) -----")
            print(final[:1200])
            print("----- end -----\n")
    log(f"dry-run: {len(texts)} rows, status0={s0}, style[:8]={styles[:8]}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="kys3b rewriting worker (one GPU)")
    ap.add_argument("--setting", required=True)
    ap.add_argument("--pass", dest="pass_name", required=True, choices=list(PASSES))
    ap.add_argument("--dry-run", action="store_true", help="build prompts only, no GPU")
    ap.add_argument("--dry-rows", type=int, default=20)
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
