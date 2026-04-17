#!/usr/bin/env python
"""Query a XIMU database."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ximu.scoring import rescore_results, score_polyq
from ximu.search import search
from ximu.utils import read_single_sequence


def _read_query_sequence(seq: str | None, fasta: str | None) -> str:
    if seq:
        return "".join(seq.upper().split())
    if fasta:
        return read_single_sequence(fasta)
    raise SystemExit("必须提供 --seq 或 --fasta")


def main() -> None:
    parser = argparse.ArgumentParser(description="XIMU检索")
    parser.add_argument("--seq", help="直接输入序列字符串")
    parser.add_argument("--fasta", help="输入单条蛋白的FASTA文件")
    parser.add_argument("--db", required=True, help="数据库目录")
    parser.add_argument("--top", type=int, default=50, help="返回蛋白数")
    parser.add_argument("--mode", choices=["general", "polyq"], default="general")
    args = parser.parse_args()

    sequence = _read_query_sequence(args.seq, args.fasta)
    if args.mode == "general":
        candidates = search(sequence, args.db, top_k=200, top_n=args.top * 2)
        results = rescore_results(sequence, candidates, args.db)[: args.top]
    else:
        results = score_polyq(sequence, args.db, top_n=args.top)

    print(f"{'排名':<5} {'蛋白ID':<20} {'得分':<10} {'物种':<20} {'最佳窗口':<15} {'蛋白长度'}")
    for idx, result in enumerate(results, 1):
        window = f"{result['best_window_start']}-{result['best_window_end']}"
        score = result.get("score", -result.get("faiss_distance", 0.0))
        print(
            f"{idx:<5} {result['protein_id']:<20} {score:<10.4f} "
            f"{result.get('genome', ''):<20} {window:<15} {result.get('prot_len', '')}"
        )


if __name__ == "__main__":
    main()
