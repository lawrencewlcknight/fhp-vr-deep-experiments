"""Stable JSON, CSV, hashing, and summary helpers for experiment artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
from typing import Iterable, Mapping

import numpy as np


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialise {type(value).__name__}")


def write_json(path: str | Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            sort_keys=True,
            default=json_default,
            allow_nan=True,
        )


def read_json(path: str | Path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: str | Path, rows: Iterable[Mapping[str, object]]) -> None:
    path = Path(path)
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def canonical_sha256(payload) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=json_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def peak_rss_mib() -> float:
    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return peak / (1024.0 * 1024.0)
    return peak / 1024.0


def repository_commit(root: str | Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(root),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def stats(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    count = int(array.size)
    if count == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "se": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if count > 1 else 0.0
    se = std / np.sqrt(count) if count > 1 else 0.0
    return {
        "count": count,
        "mean": mean,
        "std": std,
        "se": float(se),
        "ci95_low": float(mean - 1.96 * se),
        "ci95_high": float(mean + 1.96 * se),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }
