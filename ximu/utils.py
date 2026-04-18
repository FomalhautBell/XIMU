"""Utility functions for FASTA input and sliding windows."""

from __future__ import annotations

from dataclasses import dataclass
import gzip
from pathlib import Path
import re
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


def clean_genome_name(genome: str) -> str:
    """Return a compact display name inferred from a stored genome label."""

    skip_words = {
        "pep",
        "all",
        "filter",
        "longest",
        "proteins",
        "protein",
        "genecatalog",
        "assembly",
        "pseudomolecules",
        "genome",
        "with",
        "organelles",
        "models",
        "aa",
        "gca",
        "reference",
        "nam",
        "cds",
    }
    words: list[str] = []
    for part in re.split(r"[_\W]+", genome):
        if not part:
            continue
        lowered = part.lower()
        if lowered in skip_words or lowered.isdigit():
            continue
        if (lowered.startswith("v") or lowered.startswith("r")) and lowered[1:].isdigit():
            continue
        if re.fullmatch(r"[A-Z]*\d+[A-Z0-9]*", part):
            continue
        words.append(part)
        if len(words) == 2:
            break
    if len(words) >= 2:
        return f"{words[0]} {words[1]}"
    return genome.replace("_", " ")
