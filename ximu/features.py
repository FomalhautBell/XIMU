"""Feature extraction for low-complexity protein tracts."""

from __future__ import annotations

from collections import Counter
import math

import numpy as np

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

GROUPS: tuple[tuple[str, set[str]], ...] = (
    ("polar_disordered", set("QNST")),
    ("positive", set("KRH")),
    ("negative", set("DE")),
    ("hydrophobic", set("AVLIMFW")),
    ("special", set("GPCY")),
)

HYDROPHOBICITY = {
    "A": 0.700,
    "R": 0.000,
    "N": 0.111,
    "D": 0.111,
    "C": 0.778,
    "Q": 0.111,
    "E": 0.111,
    "G": 0.456,
    "H": 0.144,
    "I": 1.000,
    "L": 0.922,
    "K": 0.067,
    "M": 0.711,
    "F": 0.811,
    "P": 0.322,
    "S": 0.411,
    "T": 0.422,
    "W": 0.400,
    "Y": 0.356,
    "V": 0.967,
}

DISORDER_PROMOTING = set("ARGQSEKP")
STRUCTURE_PROMOTING = set("VILMFWC")
LLPS_PROMOTING = set("FYWQNSRK")
MAX_ENTROPY_20_AA = math.log2(20)


def _clean_sequence(sequence: str) -> str:
    return "".join(str(sequence).upper().split())


def validate_standard_sequence(sequence: str) -> str:
    """Return an uppercase sequence or raise ValueError for non-standard amino acids."""

    seq = _clean_sequence(sequence)
    if not seq:
        raise ValueError("protein sequence is empty")
    invalid = set(seq) - STANDARD_AA
    if invalid:
        bad = "".join(sorted(invalid))
        raise ValueError(f"sequence contains non-standard amino acids: {bad}")
    return seq


def compute_max_consecutive_Q(sequence: str) -> int:
    """Return the longest consecutive glutamine run length."""

    longest = 0
    current = 0
    for aa in _clean_sequence(sequence):
        if aa == "Q":
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _mutation_percentage(sequence: str, max_unit_len: int = 3) -> float:
    """Approximate distance to the best mono/di/tri-peptide repeat."""

    seq_len = len(sequence)
    if seq_len < 2:
        return 1.0

    max_k = min(max_unit_len, seq_len // 2)
    if max_k < 1:
        return 1.0

    best_mismatches = seq_len
    for k in range(1, max_k + 1):
        unit = []
        for offset in range(k):
            residues = sequence[offset::k]
            unit.append(Counter(residues).most_common(1)[0][0])

        mismatches = 0
        for idx, aa in enumerate(sequence):
            if aa != unit[idx % k]:
                mismatches += 1
        best_mismatches = min(best_mismatches, mismatches)

    return best_mismatches / seq_len


def _homopolymer_stats(sequence: str) -> tuple[int, int, int]:
    """Return longest run, number of >=2 runs, and total length of >=3 runs."""

    longest = 0
    run_count_ge2 = 0
    total_len_ge3 = 0
    current_aa = None
    current_len = 0

    def close_run(length: int) -> None:
        nonlocal longest, run_count_ge2, total_len_ge3
        if length <= 0:
            return
        longest = max(longest, length)
        if length >= 2:
            run_count_ge2 += 1
        if length >= 3:
            total_len_ge3 += length

    for aa in sequence:
        if aa == current_aa:
            current_len += 1
        else:
            close_run(current_len)
            current_aa = aa
            current_len = 1
    close_run(current_len)
    return longest, run_count_ge2, total_len_ge3


def _interruption_density(sequence: str, dominant_aa: str) -> float:
    """Estimate interruptions inside the broadest tract containing >=80% dominant AA."""

    n = len(sequence)
    best_start = -1
    best_end = -1
    best_len = 0
    left = 0
    dominant_count = 0

    for right, aa in enumerate(sequence):
        if aa == dominant_aa:
            dominant_count += 1

        while left <= right and dominant_count / (right - left + 1) < 0.8:
            if sequence[left] == dominant_aa:
                dominant_count -= 1
            left += 1

        span = right - left + 1
        if span > best_len and dominant_count >= 2:
            best_start = left
            best_end = right + 1
            best_len = span

    if best_len <= 0:
        return 1.0

    interruptions = 0
    in_interruption = False
    for aa in sequence[best_start:best_end]:
        if aa == dominant_aa:
            in_interruption = False
        elif not in_interruption:
            interruptions += 1
            in_interruption = True

    return interruptions / best_len


def _relative_position(window_start: int, protein_length: int | None, sequence_length: int) -> float:
    if protein_length is None or protein_length == sequence_length:
        return 0.5
    denominator = protein_length - sequence_length
    if denominator <= 0:
        return 0.5
    return window_start / denominator


def compute_features(
    sequence: str,
    window_start: int = 0,
    protein_length: int | None = None,
) -> np.ndarray:
    """Compute the normalized 26-dimensional XIMU feature vector."""

    seq = validate_standard_sequence(sequence)
    seq_len = len(seq)
    counts = Counter(seq)
    values = np.zeros(26, dtype=np.float32)

    for idx, (_, group) in enumerate(GROUPS):
        group_counts = [counts[aa] for aa in group]
        group_total = sum(group_counts)
        values[idx] = group_total / seq_len
        values[idx + 5] = max(group_counts) / group_total if group_total else 0.0

    aa_counts = sorted(counts.values(), reverse=True)
    top1 = aa_counts[0]
    top2 = aa_counts[1] if len(aa_counts) > 1 else 0
    dominant_aa = counts.most_common(1)[0][0]
    longest_run, run_count_ge2, total_len_ge3 = _homopolymer_stats(seq)

    values[10] = top1 / seq_len
    values[11] = _mutation_percentage(seq)
    values[12] = values[10]
    values[13] = (top1 - top2) / seq_len

    entropy = -sum((count / seq_len) * math.log2(count / seq_len) for count in counts.values())
    values[14] = 1.0 - (entropy / MAX_ENTROPY_20_AA)
    values[15] = total_len_ge3 / seq_len
    values[16] = longest_run / seq_len
    values[17] = run_count_ge2 / seq_len
    values[18] = _interruption_density(seq, dominant_aa)
    values[19] = sum(HYDROPHOBICITY[aa] for aa in seq) / seq_len

    positive = sum(counts[aa] for aa in "KRH")
    negative = sum(counts[aa] for aa in "DE")
    values[20] = ((positive - negative) / seq_len + 1.0) / 2.0
    values[21] = (positive + negative) / seq_len
    values[22] = sum(counts[aa] for aa in DISORDER_PROMOTING) / seq_len
    values[23] = sum(counts[aa] for aa in STRUCTURE_PROMOTING) / seq_len
    values[24] = sum(counts[aa] for aa in LLPS_PROMOTING) / seq_len
    values[25] = _relative_position(window_start, protein_length, seq_len)

    return np.clip(values, 0.0, 1.0).astype(np.float32)
