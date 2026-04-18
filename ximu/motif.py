"""Motif-wide database scans and distance heatmaps."""

from __future__ import annotations

from pathlib import Path
import sqlite3

import numpy as np

from .features import STANDARD_AA, compute_features
from .utils import clean_genome_name

FEATURE_COLUMNS = [f"feat_{idx:02d}" for idx in range(26)]
MAX_MOTIF_LENGTH = 24


def normalize_motif(motif: str) -> str:
    """Validate and normalize a short amino-acid motif."""

    cleaned = "".join(str(motif).upper().split())
    if not cleaned:
        raise ValueError("Motif must not be empty.")
    if len(cleaned) > MAX_MOTIF_LENGTH:
        raise ValueError(
            f"Motif too long (max {MAX_MOTIF_LENGTH}aa, got {len(cleaned)}aa). "
            "Use general mode for longer queries."
        )
    if not set(cleaned).issubset(STANDARD_AA):
        raise ValueError("Motif contains non-standard amino acids.")
    return cleaned


def _motif_count_exact(sequence: str, motif: str) -> int:
    count = 0
    start = sequence.find(motif)
    while start != -1:
        count += 1
        start = sequence.find(motif, start + 1)
    return count


def _motif_count_with_mismatch(sequence: str, motif: str, max_mismatches: int) -> int:
    motif_len = len(motif)
    if len(sequence) < motif_len:
        return 0
    count = 0
    for start in range(len(sequence) - motif_len + 1):
        mismatches = 0
        for idx, motif_aa in enumerate(motif):
            if sequence[start + idx] != motif_aa:
                mismatches += 1
                if mismatches > max_mismatches:
                    break
        if mismatches <= max_mismatches:
            count += 1
    return count


def find_motif_windows(motif: str, db_dir: str | Path, mismatch: int = 0):
    """Scan all indexed windows and return those containing a motif."""

    motif = normalize_motif(motif)
    if mismatch < 0:
        raise ValueError("Mismatch must be >= 0.")
    if mismatch > 3:
        raise ValueError("Mismatch values above 3 are not recommended for short motif scans.")

    try:
        import pandas as pd
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise ImportError("Motif scans require pandas, pyarrow, and tqdm.") from exc

    database_dir = Path(db_dir)
    windows = pd.read_parquet(
        database_dir / "ximu_windows.parquet",
        columns=["window_id", "protein_id", "start", "end", "prot_len", "max_consec_q"] + FEATURE_COLUMNS,
    )
    connection = sqlite3.connect(database_dir / "ximu_meta.db")
    try:
        proteins = pd.read_sql("SELECT protein_id, genome, sequence FROM proteins", connection)
    finally:
        connection.close()

    windows = windows.merge(proteins, on="protein_id", how="left", validate="many_to_one")
    matcher = _motif_count_exact if mismatch == 0 else None
    motif_counts: list[int] = []
    iterator = zip(windows["sequence"], windows["start"], windows["end"])
    for sequence, start, end in tqdm(iterator, total=len(windows), desc=f"Scanning motif {motif}"):
        window_sequence = sequence[int(start) : int(end)]
        if matcher is not None:
            motif_counts.append(matcher(window_sequence, motif))
        else:
            motif_counts.append(_motif_count_with_mismatch(window_sequence, motif, mismatch))

    windows["motif_count"] = motif_counts
    matched = windows.loc[windows["motif_count"] > 0].copy()
    return matched.drop(columns=["sequence"])


def score_motif_windows(query_sequence: str, motif_windows):
    """Add cosine similarity and distance columns for motif-containing windows."""

    query_vec = compute_features(query_sequence).astype(np.float32)
    query_norm = float(np.linalg.norm(query_vec))
    if query_norm == 0.0:
        raise ValueError("Query feature vector has zero norm.")
    query_vec = query_vec / query_norm

    vectors = motif_windows[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    cosine_sim = (vectors / norms) @ query_vec
    scored = motif_windows.copy()
    scored["cosine_sim"] = cosine_sim
    scored["cosine_dist"] = 1.0 - cosine_sim
    return scored


def _species_group(name: str) -> tuple[str, int]:
    group_rules = {
        "Chlamydomonas reinhardtii": ("green algae", 0),
        "ChloA99 1": ("green algae", 0),
        "Cyapar1 GeneCatalog": ("green algae", 0),
        "Cyccr1 GeneCatalog": ("green algae", 0),
        "Cyamer1 GeneCatalog": ("green algae", 0),
        "Chabra1 GeneCatalog": ("green algae", 0),
        "Nanoce1779 2": ("green algae", 0),
        "Amborella trichopoda": ("plants", 1),
        "Brassica napus": ("plants", 1),
        "Gossypium raimondii": ("plants", 1),
        "Hordeum vulgare": ("plants", 1),
        "Oryza sativa": ("plants", 1),
        "Physcomitrium patens": ("plants", 1),
        "Solanum lycopersicum": ("plants", 1),
        "Triticum aestivum": ("plants", 1),
        "Triticum timopheevii": ("plants", 1),
        "Zea mays": ("plants", 1),
        "Danio rerio": ("animals", 2),
        "Drosophila melanogaster": ("animals", 2),
        "Homo sapiens": ("animals", 2),
        "Mus musculus": ("animals", 2),
        "Xenopus tropicalis": ("animals", 2),
        "Saccharomyces cerevisiae": ("fungi", 3),
        "Schizosaccharomyces pombe": ("fungi", 3),
        "Bradyrhizobium japonicum": ("bacteria", 4),
        "Escherichia coli": ("bacteria", 4),
        "Synechocystis sp": ("bacteria", 4),
    }
    for prefix, group in group_rules.items():
        if name.startswith(prefix):
            return group
    return "other", 5


def plot_motif_heatmap(
    motif_windows,
    motif: str,
    query_id: str,
    output_path: str | Path,
    n_bins: int = 20,
) -> tuple[Path, Path | None]:
    """Plot raw motif-window counts by species and cosine-distance bin."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.colors as colors
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError as exc:
        raise ImportError("Heatmap generation requires matplotlib and pandas.") from exc

    if motif_windows.empty:
        raise ValueError("Cannot plot a heatmap for an empty motif window set.")

    data = motif_windows.copy()
    data["genome_clean"] = data["genome"].map(clean_genome_name)
    dist_min = float(data["cosine_dist"].min())
    dist_max = float(data["cosine_dist"].max())
    if np.isclose(dist_min, dist_max):
        dist_max = dist_min + 1e-6
    bins = np.linspace(dist_min, dist_max, n_bins + 1)
    ascending_labels = [f"{(bins[idx] + bins[idx + 1]) / 2:.3f}" for idx in range(n_bins)]
    data["bin"] = pd.cut(data["cosine_dist"], bins=bins, labels=ascending_labels, include_lowest=True)

    raw_ascending = data.groupby(["genome_clean", "bin"], observed=True).size().unstack(fill_value=0)
    raw_ascending = raw_ascending.reindex(columns=ascending_labels, fill_value=0)
    distances = np.array([float(label) for label in ascending_labels])

    def weighted_median(row) -> float:
        counts = row.to_numpy(dtype=float)
        total = counts.sum()
        if total <= 0:
            return np.inf
        return float(distances[np.searchsorted(np.cumsum(counts), total / 2.0)])

    median_distance = raw_ascending.apply(weighted_median, axis=1)
    ordered_index = sorted(
        raw_ascending.index,
        key=lambda name: (_species_group(name)[1], median_distance.get(name, np.inf), name),
    )
    raw = raw_ascending.loc[ordered_index, list(reversed(ascending_labels))]
    values = raw.to_numpy(dtype=float)
    masked = np.ma.masked_where(values <= 0, values)

    cmap = plt.get_cmap("OrRd").copy()
    cmap.set_bad(color="white")
    norm = colors.LogNorm(vmin=1, vmax=max(1, values.max()))

    fig, ax = plt.subplots(figsize=(18, max(10, 0.42 * len(raw.index))))
    mesh = ax.pcolormesh(
        np.arange(values.shape[1] + 1),
        np.arange(values.shape[0] + 1),
        masked,
        cmap=cmap,
        norm=norm,
        edgecolors="white",
        linewidth=0.75,
        shading="flat",
    )
    ax.set_xlim(0, values.shape[1])
    ax.set_ylim(values.shape[0], 0)
    ax.set_yticks(np.arange(values.shape[0]) + 0.5)
    ax.set_yticklabels(raw.index, fontsize=8)
    xtick_idx = list(range(0, values.shape[1], 4))
    if values.shape[1] - 1 not in xtick_idx:
        xtick_idx.append(values.shape[1] - 1)
    labels = list(raw.columns)
    ax.set_xticks([idx + 0.5 for idx in xtick_idx])
    ax.set_xticklabels([labels[idx] for idx in xtick_idx], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Cosine distance to query (larger to smaller; most similar on the right)", fontsize=11)
    ax.set_ylabel("Species grouped by broad taxonomy", fontsize=11)
    ax.set_title(
        f"Motif [{motif}] windows: raw cosine-distance counts by species\n"
        f"Query: {query_id}; color uses log2-scaled window counts",
        fontsize=12,
    )

    current_group = None
    start = 0
    for idx, name in enumerate(list(raw.index) + [None]):
        group = _species_group(name)[0] if name is not None else None
        if current_group is None:
            current_group = group
            start = idx
        elif group != current_group:
            ax.axhline(idx, color="black", linewidth=0.9)
            ax.text(-1.15, (start + idx) / 2.0, current_group, ha="right", va="center", fontsize=8, fontweight="bold")
            current_group = group
            start = idx

    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = int(values[row_idx, col_idx])
            text_color = "white" if value >= 128 else "black"
            ax.text(
                col_idx + 0.5,
                row_idx + 0.5,
                str(value),
                ha="center",
                va="center",
                fontsize=8.5,
                fontweight="bold",
                color=text_color,
            )

    colorbar = plt.colorbar(mesh, ax=ax)
    ticks = [tick for tick in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] if tick <= values.max()]
    if int(values.max()) not in ticks:
        ticks.append(int(values.max()))
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels([str(tick) for tick in ticks])
    colorbar.set_label("Number of windows (log2 color scale)")
    plt.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, bbox_inches="tight", dpi=300)
    paired_output = None
    if output.suffix.lower() == ".pdf":
        paired_output = output.with_suffix(".png")
        plt.savefig(paired_output, bbox_inches="tight", dpi=300)
    plt.close(fig)
    raw.to_csv(output.with_suffix(".raw_counts.csv"))
    return output, paired_output
