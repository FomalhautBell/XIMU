"""Protein-level rescoring for XIMU candidates."""

from __future__ import annotations

import numpy as np

from .features import compute_features
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
