"""Utility functions for FASTA input and sliding windows."""

from __future__ import annotations

from dataclasses import dataclass
import gzip
from pathlib import Path
from typing import Iterator, TextIO

from .features import STANDARD_AA


@dataclass(frozen=True)
class FastaRecord:
    protein_id: str
    description: str
    sequence: str


def _open_text(path: str | Path) -> TextIO:
    file_path = Path(path)
    if file_path.suffix == ".gz":
        return gzip.open(file_path, "rt", encoding="utf-8")
    return file_path.open("rt", encoding="utf-8")


def parse_fasta(path: str | Path) -> Iterator[FastaRecord]:
    """Yield FASTA records without requiring Biopython."""

    header: str | None = None
    chunks: list[str] = []

    with _open_text(path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield _record_from_parts(header, chunks)
                header = line[1:].strip()
                chunks = []
            else:
                chunks.append(line)

    if header is not None:
        yield _record_from_parts(header, chunks)


def _record_from_parts(header: str, chunks: list[str]) -> FastaRecord:
    protein_id = header.split(maxsplit=1)[0]
    sequence = "".join(chunks).upper()
    return FastaRecord(protein_id=protein_id, description=header, sequence=sequence)


def read_single_sequence(path: str | Path) -> str:
    """Read one protein sequence from a FASTA file."""

    records = parse_fasta(path)
    try:
        return next(records).sequence
    except StopIteration as exc:
        raise ValueError(f"no FASTA records found in {path}") from exc


def is_standard_sequence(sequence: str) -> bool:
    return bool(sequence) and set(sequence).issubset(STANDARD_AA)


def iter_windows(sequence: str, window_size: int = 48, stride: int = 24) -> Iterator[tuple[int, int, str]]:
    """Yield fixed-size windows as (start, end, sequence)."""

    seq_len = len(sequence)
    if seq_len < window_size:
        return
    for start in range(0, seq_len - window_size + 1, stride):
        end = start + window_size
        yield start, end, sequence[start:end]
