"""Manifests are the inter-stage contract.

Every stage writes `_manifest.json` and validates the previous stage's manifest before
doing anything.  Each manifest records the code commit so an artifact can always be traced
to the code that produced it.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import subprocess
from pathlib import Path

import numpy as np

from .io import atomic_write_json, check, log

MANIFEST_NAME = "_manifest.json"


def code_commit() -> str:
    """Short git sha of the implementation, or 'uncommitted'."""
    try:
        root = Path(__file__).resolve().parents[3]
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            sha = out.stdout.strip()
            dirty = subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain"],
                capture_output=True, text=True, timeout=10,
            ).stdout.strip()
            return sha + ("-dirty" if dirty else "")
    except Exception:
        pass
    return "uncommitted"


def now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_ids(ids: np.ndarray) -> str:
    """Stable digest of a doc_id set: sorted, int64, little-endian.

    This is the identity check plan.md promises for "the shared base is identical across
    settings" -- proven by digest, not by count.
    """
    a = np.sort(np.asarray(ids, dtype=np.int64))
    return hashlib.sha256(a.tobytes(order="C")).hexdigest()


def write(directory: Path, stage: str, payload: dict) -> Path:
    directory = Path(directory)
    m = dict(stage=stage, code_commit=code_commit(), written_at=now(), **payload)
    dest = directory / MANIFEST_NAME
    atomic_write_json(m, dest)
    log(f"manifest: {dest}")
    return dest


def read(directory: Path, stage: str | None = None) -> dict:
    import json

    p = Path(directory) / MANIFEST_NAME
    check(p.exists(), f"missing manifest {p} -- run the previous stage first")
    m = json.loads(p.read_text())
    if stage is not None:
        check(m.get("stage") == stage, f"{p}: stage {m.get('stage')!r} != expected {stage!r}")
    return m


def exists(directory: Path) -> bool:
    return (Path(directory) / MANIFEST_NAME).exists()
