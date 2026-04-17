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
    args = parser.parse_args()
    build_database(args.fasta, args.db, args.genome, args.append)


if __name__ == "__main__":
    main()
