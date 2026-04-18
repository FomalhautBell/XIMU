#!/usr/bin/env python
"""Build a XIMU protein-window database."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ximu.database import build_database


def main() -> None:
    parser = argparse.ArgumentParser(description="XIMU建库")
    parser.add_argument("--fasta", required=True, help="输入pep.fa文件")
    parser.add_argument("--db", required=True, help="数据库输出目录")
    parser.add_argument("--genome", required=True, help="物种/基因组名称")
    parser.add_argument("--append", action="store_true", help="追加到已有数据库")
    parser.add_argument("--jobs", type=int, default=1, help="并行worker数，默认1为串行")
    parser.add_argument("--chunk-size", type=int, default=500, help="每个worker批次处理的蛋白数")
    args = parser.parse_args()
    stats = build_database(args.fasta, args.db, args.genome, args.append, jobs=args.jobs, chunk_size=args.chunk_size)
    print(
        "建库完成: "
        f"processed={stats['processed']} proteins={stats['proteins']} windows={stats['windows']} "
        f"skipped_short={stats['skipped_short']} skipped_nonstandard={stats['skipped_nonstandard']} "
        f"total_windows={stats['total_windows']} jobs={stats['jobs']}"
    )


if __name__ == "__main__":
    main()
