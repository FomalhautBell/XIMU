"""Protein-level rescoring for XIMU candidates."""

from __future__ import annotations

from pathlib import Path
import math
import sqlite3

import numpy as np

from .features import compute_features, compute_max_consecutive_Q
from .utils import iter_windows


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def score_candidate(
    query_vec: np.ndarray,
    candidate_sequence: str,
    window_size: int = 48,
    stride: int = 24,
) -> dict:
    """Score a full candidate protein by its best matching local window."""

    best = {
        "score": -1.0,
        "best_window_start": 0,
        "best_window_end": min(len(candidate_sequence), window_size),
        "best_cosine_sim": -1.0,
    }

    windows = list(iter_windows(candidate_sequence, window_size, stride))
    if not windows and candidate_sequence:
        windows = [(0, len(candidate_sequence), candidate_sequence)]

    for start, end, window_seq in windows:
        candidate_vec = compute_features(window_seq, window_start=start, protein_length=len(candidate_sequence))
        similarity = cosine_similarity(query_vec, candidate_vec)
        if similarity > best["score"]:
            best = {
                "score": similarity,
                "best_window_start": start,
                "best_window_end": end,
                "best_cosine_sim": similarity,
            }
    return best


def rescore_results(query_sequence: str, candidates: list[dict], db_dir: str | None = None) -> list[dict]:
    """Rescore candidate proteins and sort by descending protein-level score."""

    del db_dir
    query_vec = compute_features(query_sequence)
    rescored = []
    for candidate in candidates:
        sequence = candidate.get("sequence", "")
        score = score_candidate(query_vec, sequence)
        merged = dict(candidate)
        merged.update(score)
        rescored.append(merged)
    return sorted(rescored, key=lambda item: item["score"], reverse=True)


def _require_polyq_dependencies() -> object:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "polyQ scoring requires pandas and pyarrow; install them with `pip install -r requirements.txt`"
        ) from exc
    return pd


def _best_polyq_reference_window(sequence: str, window_size: int = 48, stride: int = 24) -> tuple[int, int, str]:
    windows = list(iter_windows(sequence, window_size, stride))
    if not windows:
        return 0, len(sequence), sequence
    return max(windows, key=lambda item: compute_max_consecutive_Q(item[2]))


def _load_meta(db_dir: Path, protein_ids: list[str]) -> dict[str, dict]:
    if not protein_ids:
        return {}
    placeholders = ",".join("?" for _ in protein_ids)
    connection = sqlite3.connect(db_dir / "ximu_meta.db")
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            f"""
            SELECT protein_id, description, prot_len, genome, sequence
            FROM proteins
            WHERE protein_id IN ({placeholders})
            """,
            protein_ids,
        ).fetchall()
    finally:
        connection.close()
    return {row["protein_id"]: dict(row) for row in rows}


def score_polyq(
    query_sequence: str,
    db_dir: str,
    reference_window: str | None = None,
    top_n: int = 50,
) -> list[dict]:
    """Rank proteins using the Q5 inverted index and a polyQ-weighted score."""

    pd = _require_polyq_dependencies()
    database_dir = Path(db_dir)

    if reference_window is None:
        _, _, reference_window = _best_polyq_reference_window(query_sequence)
    reference_vec = compute_features(reference_window)

    q5_path = database_dir / "ximu_q5_index.npy"
    if not q5_path.exists():
        return []
    q5_ids = np.load(q5_path).astype(np.int64)
    if len(q5_ids) == 0:
        return []

    windows = pd.read_parquet(database_dir / "ximu_windows.parquet")
    q5_windows = windows.iloc[q5_ids]
    best_by_protein: dict[str, dict] = {}
    feature_cols = [f"feat_{idx:02d}" for idx in range(26)]

    for row in q5_windows.to_dict("records"):
        candidate_vec = np.array([row[col] for col in feature_cols], dtype=np.float32)
        similarity = cosine_similarity(reference_vec, candidate_vec)
        score = similarity * math.log1p(int(row["max_consec_q"]))
        protein_id = str(row["protein_id"])
        current = best_by_protein.get(protein_id)
        if current is None or score > current["score"]:
            best_by_protein[protein_id] = {
                "protein_id": protein_id,
                "score": float(score),
                "best_window_start": int(row["start"]),
                "best_window_end": int(row["end"]),
                "best_cosine_sim": float(similarity),
                "prot_len": int(row["prot_len"]),
            }

    ranked = sorted(best_by_protein.values(), key=lambda item: item["score"], reverse=True)[:top_n]
    meta = _load_meta(database_dir, [item["protein_id"] for item in ranked])
    for item in ranked:
        item.update(meta.get(item["protein_id"], {}))
    return ranked
