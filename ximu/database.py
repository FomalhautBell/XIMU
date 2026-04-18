"""Database construction for XIMU."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sqlite3
from typing import Iterable, Iterator

import numpy as np

from .features import compute_features, compute_max_consecutive_Q
from .utils import FastaRecord, is_standard_sequence, iter_windows, parse_fasta

WINDOW_SIZE = 48
STRIDE = 24
MIN_PROT_LEN = 48
VECTOR_DIM = 26
DEFAULT_CHUNK_SIZE = 500


def _require_database_dependencies() -> tuple[object, object, object]:
    try:
        import faiss
        import pandas as pd
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "database building/search requires faiss-cpu, pandas, and pyarrow; "
            "install them with `pip install -r requirements.txt`"
        ) from exc
    return faiss, pd, pyarrow


def build_faiss_index(vectors: np.ndarray):
    """Build a Faiss index, using FlatL2 for small data and IVF-PQ for larger data."""

    faiss, _, _ = _require_database_dependencies()
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[1] != VECTOR_DIM:
        raise ValueError(f"vectors must have shape (N, {VECTOR_DIM})")
    if len(vectors) == 0:
        raise ValueError("cannot build a Faiss index with zero vectors")

    if len(vectors) < 4096:
        index = faiss.IndexFlatL2(VECTOR_DIM)
        index.add(vectors)
        return index

    nlist = max(100, int(np.sqrt(len(vectors))))
    nlist = min(nlist, max(1, len(vectors) // 39))
    if nlist < 2:
        index = faiss.IndexFlatL2(VECTOR_DIM)
        index.add(vectors)
        return index

    quantizer = faiss.IndexFlatL2(VECTOR_DIM)
    # 26 dimensions are not divisible by 8; M=2 is compatible with Faiss PQ.
    index = faiss.IndexIVFPQ(quantizer, VECTOR_DIM, nlist, 2, 8)
    index.train(vectors)
    index.add(vectors)
    index.nprobe = min(10, nlist)
    return index


def _load_processed_ids(progress_path: Path) -> set[str]:
    if not progress_path.exists():
        return set()
    return {line.strip() for line in progress_path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _initial_state(db_dir: Path, append: bool):
    _, pd, _ = _require_database_dependencies()
    parquet_path = db_dir / "ximu_windows.parquet"
    vectors_path = db_dir / "ximu_vectors.npy"

    if append:
        old_rows = pd.read_parquet(parquet_path) if parquet_path.exists() else None
        old_vectors = np.load(vectors_path) if vectors_path.exists() else None
        next_window_id = 0 if old_rows is None else len(old_rows)
        return old_rows, old_vectors, next_window_id

    existing_outputs = [db_dir / "ximu.faiss", parquet_path, db_dir / "ximu_meta.db", vectors_path]
    present = [path.name for path in existing_outputs if path.exists()]
    if present:
        names = ", ".join(present)
        raise FileExistsError(f"{db_dir} already contains database outputs: {names}; use append=True")
    return None, None, 0


def _write_sqlite(db_path: Path, proteins: list[dict], append: bool) -> None:
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS proteins (
                protein_id   TEXT PRIMARY KEY,
                description  TEXT,
                prot_len     INTEGER,
                genome       TEXT,
                sequence     TEXT
            )
            """
        )
        if not append:
            connection.execute("DELETE FROM proteins")
        connection.executemany(
            """
            INSERT OR REPLACE INTO proteins
                (protein_id, description, prot_len, genome, sequence)
            VALUES
                (:protein_id, :description, :prot_len, :genome, :sequence)
            """,
            proteins,
        )
        connection.commit()
    finally:
        connection.close()


def _batched(records: Iterable[FastaRecord], chunk_size: int) -> Iterator[list[FastaRecord]]:
    batch: list[FastaRecord] = []
    for record in records:
        batch.append(record)
        if len(batch) >= chunk_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _process_record(record: FastaRecord, genome_name: str) -> dict:
    seq = record.sequence
    if len(seq) < MIN_PROT_LEN:
        return {"status": "short", "protein_id": record.protein_id}
    if not is_standard_sequence(seq):
        return {"status": "nonstandard", "protein_id": record.protein_id}

    windows = []
    vectors = []
    for start, end, window_seq in iter_windows(seq, WINDOW_SIZE, STRIDE):
        features = compute_features(window_seq, window_start=start, protein_length=len(seq))
        windows.append(
            {
                "protein_id": record.protein_id,
                "start": start,
                "end": end,
                "prot_len": len(seq),
                "max_consec_q": compute_max_consecutive_Q(window_seq),
            }
        )
        vectors.append(features)

    return {
        "status": "ok",
        "protein_id": record.protein_id,
        "protein": {
            "protein_id": record.protein_id,
            "description": record.description,
            "prot_len": len(seq),
            "genome": genome_name,
            "sequence": seq,
        },
        "windows": windows,
        "vectors": vectors,
    }


def _process_chunk(args: tuple[list[FastaRecord], str]) -> list[dict]:
    records, genome_name = args
    return [_process_record(record, genome_name) for record in records]


def _iter_processed_chunks(
    records: Iterable[FastaRecord],
    genome_name: str,
    jobs: int,
    chunk_size: int,
) -> Iterator[list[dict]]:
    chunks = ((chunk, genome_name) for chunk in _batched(records, chunk_size))
    if jobs <= 1:
        for chunk in chunks:
            yield _process_chunk(chunk)
        return

    with ProcessPoolExecutor(max_workers=jobs) as executor:
        yield from executor.map(_process_chunk, chunks)


def build_database(
    fasta_file: str,
    db_dir: str,
    genome_name: str,
    append: bool = False,
    jobs: int = 1,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> dict:
    """Build or append a XIMU vector database from a protein FASTA file."""

    faiss, pd, _ = _require_database_dependencies()
    if jobs < 1:
        raise ValueError("jobs must be >= 1")
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")

    output_dir = Path(db_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    old_rows, old_vectors, next_window_id = _initial_state(output_dir, append)

    progress_path = output_dir / "progress.txt"
    processed = _load_processed_ids(progress_path) if append else set()
    new_rows: list[dict] = []
    new_vectors: list[np.ndarray] = []
    proteins: list[dict] = []
    skipped_short = 0
    skipped_nonstandard = 0
    processed_count = 0

    with progress_path.open("a", encoding="utf-8") as progress:
        record_iter = (record for record in parse_fasta(fasta_file) if record.protein_id not in processed)
        for results in _iter_processed_chunks(record_iter, genome_name, jobs, chunk_size):
            for result in results:
                processed_count += 1
                protein_id = result["protein_id"]
                status = result["status"]
                if status == "short":
                    skipped_short += 1
                    progress.write(protein_id + "\n")
                    continue
                if status == "nonstandard":
                    skipped_nonstandard += 1
                    progress.write(protein_id + "\n")
                    continue

                proteins.append(result["protein"])
                for window, features in zip(result["windows"], result["vectors"]):
                    row = {
                        "window_id": next_window_id,
                        "protein_id": window["protein_id"],
                        "start": window["start"],
                        "end": window["end"],
                        "prot_len": window["prot_len"],
                        "max_consec_q": window["max_consec_q"],
                    }
                    for idx, value in enumerate(features):
                        row[f"feat_{idx:02d}"] = np.float16(value)
                    new_rows.append(row)
                    new_vectors.append(features)
                    next_window_id += 1

                processed.add(protein_id)
                progress.write(protein_id + "\n")
            progress.flush()

    if not new_rows and old_rows is None:
        raise ValueError(
            "no valid protein windows were produced "
            f"(skipped_short={skipped_short}, skipped_nonstandard={skipped_nonstandard})"
        )

    new_frame = pd.DataFrame(new_rows)
    if old_rows is not None and not old_rows.empty:
        windows = pd.concat([old_rows, new_frame], ignore_index=True)
    else:
        windows = new_frame

    vectors = np.vstack(new_vectors).astype(np.float32) if new_vectors else np.empty((0, VECTOR_DIM), dtype=np.float32)
    if old_vectors is not None:
        vectors = np.vstack([old_vectors.astype(np.float32), vectors]).astype(np.float32)

    parquet_path = output_dir / "ximu_windows.parquet"
    windows.to_parquet(parquet_path, index=False)
    np.save(output_dir / "ximu_vectors.npy", vectors)
    _write_sqlite(output_dir / "ximu_meta.db", proteins, append=append)

    index = build_faiss_index(vectors)
    faiss.write_index(index, str(output_dir / "ximu.faiss"))
    return {
        "processed": processed_count,
        "proteins": len(proteins),
        "windows": len(new_rows),
        "skipped_short": skipped_short,
        "skipped_nonstandard": skipped_nonstandard,
        "total_windows": len(windows),
        "jobs": jobs,
        "chunk_size": chunk_size,
    }
