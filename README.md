# XIMU

eXpressive Inference-based Motif Utilities

XIMU 是一个面向低复杂度区域（LCR）蛋白片段的向量检索工具。它把蛋白窗口编码为 26 维特征，使用 Faiss 做近邻召回，再按整蛋白窗口相似度重排。

## 环境

当前项目按 Python 3.12 编写，建议在你的 `XIMUENV` conda 环境中安装依赖：

```bash
conda activate XIMUENV
python -m pip install -r requirements.txt
```

依赖包括：

- `numpy`：26 维特征计算与向量操作
- `faiss-cpu`：向量索引与检索
- `pandas` / `pyarrow`：窗口表 Parquet 读写
- `biopython`：兼容后续 FASTA 生态，当前代码内置了轻量 FASTA 解析器
- `tqdm`：后续长任务进度显示预留
- `pytest`：测试

本轮 smoke test 使用的 `XIMUENV` 实际安装版本：

```text
biopython==1.87
faiss-cpu==1.13.2
numpy==2.4.4
pandas==3.0.2
pyarrow==23.0.1
pytest==9.0.3
tqdm==4.67.3
```

## 代码结构

```text
ximu/
  features.py      # 26维特征、polyQ长度、序列校验
  utils.py         # FASTA读取、滑窗
  database.py      # 建库、Faiss索引、Parquet/SQLite/Q5索引输出
  search.py        # Faiss召回与蛋白元数据合并
  scoring.py       # 整蛋白重排与polyQ模式打分
scripts/
  build_db.py      # 建库入口
  query.py         # 查询入口
tests/
  test_features.py # 特征计算 smoke/unit tests
```

## 建库

数据库输入是蛋白 FASTA 文件。建库会跳过短于 48 aa 或含非标准氨基酸的蛋白。

```bash
python scripts/build_db.py \
  --fasta Pep_Databases/Chlamydomonas_reinhardtii.Chlamydomonas_reinhardtii_v5.5.pep.all.filter.fa \
  --db ximu_db/chlamy \
  --genome Chlamydomonas_reinhardtii_v5.5
```

追加新物种：

```bash
python scripts/build_db.py \
  --fasta Pep_Databases/Homo_sapiens.GRCh38.pep.all.filter.fa \
  --db ximu_db/all \
  --genome Homo_sapiens_GRCh38 \
  --append
```

输出目录中会生成：

- `ximu.faiss`
- `ximu_windows.parquet`
- `ximu_meta.db`
- `ximu_q5_index.npy`
- `ximu_vectors.npy`：额外保存的原始 float32 向量，用于追加建库时重建 Faiss 索引
- `progress.txt`

## 查询

实验输入蛋白序列可以直接使用仓库里的 `.faa` 文件：

```bash
python scripts/query.py --fasta CHLRE_15g640203v5.faa --db ximu_db/chlamy --top 50
```

polyQ 专用模式：

```bash
python scripts/query.py --fasta CHLRE_15g640203v5.faa --db ximu_db/chlamy --mode polyq --top 50
```

## 测试

```bash
python -m pytest tests
```

## Git 注意

本地需求文档和数据库输入/产物不会进入 git：

- `codex.md`
- `databases/`
- `Pep_Databases/`
- `ximu_db/`
- Faiss、Parquet、SQLite、NumPy 索引产物
