#!/usr/bin/env python
"""Query a XIMU database."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ximu.motif import find_motif_windows, normalize_motif, plot_motif_heatmap, score_motif_windows
from ximu.scoring import rescore_results
from ximu.search import search
from ximu.utils import read_single_sequence


def _read_query_sequence(seq: str | None, fasta: str | None) -> str:
    if seq:
        return "".join(seq.upper().split())
    if fasta:
        return read_single_sequence(fasta)
    raise SystemExit("必须提供 --seq 或 --fasta")


def _print_results(results: list[dict]) -> None:
    print(f"{'排名':<5} {'蛋白ID':<20} {'得分':<10} {'物种':<20} {'最佳窗口':<15} {'蛋白长度'}")
    for idx, result in enumerate(results, 1):
        window = f"{result['best_window_start']}-{result['best_window_end']}"
        score = result.get("score", -result.get("faiss_distance", 0.0))
        print(
            f"{idx:<5} {result['protein_id']:<20} {score:<10.4f} "
            f"{result.get('genome', ''):<20} {window:<15} {result.get('prot_len', '')}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="XIMU检索")
    parser.add_argument("--seq", help="直接输入序列字符串")
    parser.add_argument("--fasta", help="输入单条蛋白的FASTA文件")
    parser.add_argument("--db", required=True, help="数据库目录")
    parser.add_argument("--top", type=int, default=50, help="返回蛋白数")
    parser.add_argument("--mode", choices=["general", "motif"], default="general")
    parser.add_argument("--motif", help="motif模式：长度不超过24aa的标准氨基酸短序列")
    parser.add_argument("--mismatch", type=int, default=0, help="motif匹配允许错配数，默认0")
    parser.add_argument("--heatmap", action="store_true", default=True, help="motif模式：输出热图")
    parser.add_argument("--no-heatmap", dest="heatmap", action="store_false", help="motif模式：不输出热图")
    parser.add_argument("--heatmap-out", default="ximu_motif_heatmap.pdf", help="motif热图输出路径")
    args = parser.parse_args()

    sequence = _read_query_sequence(args.seq, args.fasta)
    candidates = search(sequence, args.db, top_k=100, top_n=args.top * 2)
    results = rescore_results(sequence, candidates, args.db)[: args.top]
    _print_results(results)

    if args.mode == "motif":
        if not args.motif:
            parser.error("--motif is required for motif mode")
        try:
            motif = normalize_motif(args.motif)
        except ValueError as exc:
            parser.error(str(exc))

        print(f"Scanning database for motif [{motif}] with mismatch={args.mismatch}...")
        try:
            motif_windows = find_motif_windows(motif, args.db, args.mismatch)
        except ValueError as exc:
            parser.error(str(exc))
        genome_count = motif_windows["genome"].nunique() if len(motif_windows) else 0
        print(f"Found {len(motif_windows)} windows containing motif across {genome_count} genomes.")
        if motif_windows.empty:
            raise SystemExit("No motif windows found. Try a different motif or increase --mismatch.")

        scored_windows = score_motif_windows(sequence, motif_windows)
        if args.heatmap:
            query_id = args.seq[:20] if args.seq else Path(args.fasta).stem
            heatmap, paired = plot_motif_heatmap(scored_windows, motif, query_id, args.heatmap_out)
            print(f"Heatmap saved: {heatmap}")
            if paired is not None:
                print(f"Heatmap saved: {paired}")


if __name__ == "__main__":
    main()
