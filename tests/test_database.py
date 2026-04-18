from pathlib import Path

from ximu.database import build_database
from ximu.motif import find_motif_windows, normalize_motif, score_motif_windows


def test_parallel_build_matches_serial_window_count(tmp_path: Path):
    fasta = tmp_path / "mini.fa"
    fasta.write_text(
        ">p1 synthetic low-complexity\n"
        + "M"
        + "S" * 70
        + "AAAAAAA\n"
        + ">p2 synthetic mixed\n"
        + "ACDEFGHIKLMNPQRSTVWY" * 4
        + "\n",
        encoding="utf-8",
    )

    serial = build_database(str(fasta), str(tmp_path / "serial_db"), "mini", jobs=1, chunk_size=1)
    parallel = build_database(str(fasta), str(tmp_path / "parallel_db"), "mini", jobs=2, chunk_size=1)

    assert serial["proteins"] == parallel["proteins"] == 2
    assert serial["windows"] == parallel["windows"]
    assert serial["total_windows"] == parallel["total_windows"]


def test_motif_scan_and_scoring(tmp_path: Path):
    fasta = tmp_path / "mini.fa"
    fasta.write_text(
        ">p1 synthetic low-complexity\n"
        + "M"
        + "S" * 70
        + "AAAAAAA\n"
        + ">p2 synthetic mixed\n"
        + "ACDEFGHIKLMNPQRSTVWY" * 4
        + "\n",
        encoding="utf-8",
    )
    db_dir = tmp_path / "motif_db"
    build_database(str(fasta), str(db_dir), "mini_species", jobs=1, chunk_size=1)

    assert normalize_motif(" ssssss ") == "SSSSSS"
    motif_windows = find_motif_windows("SSSSSS", db_dir)
    assert not motif_windows.empty
    assert set(motif_windows["protein_id"]) == {"p1"}
    assert motif_windows["motif_count"].min() > 0

    scored = score_motif_windows("M" + "S" * 70 + "AAAAAAA", motif_windows)
    assert "cosine_dist" in scored.columns
    assert scored["cosine_dist"].between(0, 2).all()


def test_motif_validation_rejects_long_motif():
    try:
        normalize_motif("S" * 25)
    except ValueError as exc:
        assert "Motif too long" in str(exc)
    else:
        raise AssertionError("expected long motif to be rejected")
