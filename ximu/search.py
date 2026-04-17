"""Faiss-backed candidate retrieval."""

from __future__ import annotations

from pathlib import Path
import sqlite3

import numpy as np

from .features import compute_features


def _require_search_dependencies() -> tuple[object, object]:
    try:
        import faiss
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "search requires faiss-cpu, pandas, and pyarrow; "
            "install them with `pip install -r requirements.txt`"
        ) from exc
    return faiss, pd


def _load_protein_meta(db_path: Path, protein_ids: list[str]) -> dict[str, dict]:
    if not protein_ids:
        return {}
    placeholders = ",".join("?" for _ in protein_ids)
    connection = sqlite3.connect(db_path)
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


def search(
    query_sequence: str,
    db_dir: str,
    top_k: int = 200,
    top_n: int = 50,
) -> list[dict]:
    """Return candidate proteins ranked by nearest indexed window distance."""

    faiss, pd = _require_search_dependencies()
    database_dir = Path(db_dir)
    index = faiss.read_index(str(database_dir / "ximu.faiss"))
    if hasattr(index, "nprobe"):
        index.nprobe = min(10, getattr(index, "nlist", 10))

    query_vec = compute_features(query_sequence).reshape(1, -1).astype(np.float32)
    distances, indices = index.search(query_vec, top_k)
    hit_ids = [int(idx) for idx in indices[0] if idx >= 0]
    if not hit_ids:
        return []

    windows = pd.read_parquet(database_dir / "ximu_windows.parquet")
    hit_rows = windows.iloc[hit_ids].copy()
    hit_rows["faiss_distance"] = [float(dist) for idx, dist in zip(indices[0], distances[0]) if idx >= 0]
    best_rows = (
        hit_rows.sort_values("faiss_distance", ascending=True)
        .drop_duplicates("protein_id", keep="first")
        .head(top_n)
    )

    protein_ids = best_rows["protein_id"].astype(str).tolist()
    meta = _load_protein_meta(database_dir / "ximu_meta.db", protein_ids)
    results: list[dict] = []
    for row in best_rows.to_dict("records"):
        protein_id = str(row["protein_id"])
        item = {
            "protein_id": protein_id,
            "description": "",
            "genome": "",
            "prot_len": int(row["prot_len"]),
            "best_window_start": int(row["start"]),
            "best_window_end": int(row["end"]),
            "faiss_distance": float(row["faiss_distance"]),
            "sequence": "",
        }
        item.update(meta.get(protein_id, {}))
        results.append(item)
    return results
