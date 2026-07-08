"""Dataset loading utilities for evaluation and demos."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_DATASET = Path(__file__).resolve().parent.parent / "data" / "final_dataset_projext.csv"


def load_dataset(path: Optional[str | Path] = None, encoding: str = "latin-1") -> List[Dict[str, str]]:
    csv_path = Path(path) if path else DEFAULT_DATASET
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    with open(csv_path, encoding=encoding, newline="") as f:
        return list(csv.DictReader(f))


def sample_context(
    rows: Optional[List[Dict[str, str]]] = None,
    min_len: int = 80,
    seed: Optional[int] = None,
) -> Dict[str, str]:
    if rows is None:
        rows = load_dataset()
    filtered = [r for r in rows if len(str(r.get("context", ""))) > min_len]
    if not filtered:
        raise ValueError("No rows with sufficient context length.")
    rng = random.Random(seed)
    return rng.choice(filtered)


def dataset_stats(rows: Optional[List[Dict[str, str]]] = None) -> Dict:
    if rows is None:
        rows = load_dataset()
    from collections import Counter

    return {
        "total_rows": len(rows),
        "question_types": dict(Counter(r.get("question_type", "") for r in rows).most_common()),
        "difficulties": dict(Counter(r.get("difficulty", "") for r in rows).most_common()),
        "subjects": dict(Counter(r.get("subject", "") for r in rows).most_common(10)),
    }
