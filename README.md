# XIMU

eXpressive Inference-based Motif Utilities

XIMU is a vector-search utility for protein low-complexity regions (LCRs). It encodes fixed protein windows as 26-dimensional composition and physicochemical feature vectors, retrieves similar local windows with Faiss, and re-ranks candidate proteins with full-protein window rescoring.

## Installation

XIMU is developed and tested with Python 3.12.

```bash
conda create -n XIMUENV python=3.12
conda activate XIMUENV
python -m pip install -r requirements.txt
```

Core dependencies:

- `numpy` for feature vectors and numerical operations
- `faiss-cpu` for vector indexing and nearest-neighbor search
- `pandas` and `pyarrow` for Parquet window tables
- `biopython` for compatibility with common FASTA workflows
- `matplotlib` for optional analysis plots
- `pytest` for tests

## Package Layout

```text
ximu/
  features.py      # 26-dimensional feature extraction
  utils.py         # FASTA parsing and sliding windows
  database.py      # database construction, Parquet, SQLite, and Faiss index writing
  search.py        # Faiss retrieval and metadata joins
  scoring.py       # full-protein candidate rescoring
scripts/
  build_db.py      # database builder CLI
  query.py         # query CLI
tests/
  test_features.py
  test_database.py
```

## Build a Database

Input files are protein FASTA files. Proteins shorter than 48 amino acids or containing non-standard amino acids are skipped.

```bash
python scripts/build_db.py \
  --fasta proteins.fa \
  --db ximu_db/example \
  --genome example_genome \
  --jobs 15
```

Append another FASTA file to the same database:

```bash
python scripts/build_db.py \
  --fasta another_species.fa \
  --db ximu_db/example \
  --genome another_species \
  --append \
  --jobs 15
```

`--jobs 1` uses the serial path. On multi-core servers, increase `--jobs` near the available CPU count. `--chunk-size` controls how many protein records are sent to each worker task; the default is `500`.

Database outputs:

- `ximu.faiss`: Faiss vector index
- `ximu_windows.parquet`: indexed window metadata and feature table
- `ximu_meta.db`: protein metadata and full protein sequences
- `ximu_vectors.npy`: float32 feature matrix used to support later append operations
- `progress.txt`: resumable build progress log

For public database distribution, only `ximu.faiss`, `ximu_windows.parquet`, and `ximu_meta.db` are required for querying.

## Query

Query from a FASTA file:

```bash
python scripts/query.py --fasta query.faa --db ximu_db/example --top 50
```

Query from a sequence string:

```bash
python scripts/query.py --seq "MSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS" --db ximu_db/example --top 20
```

The CLI returns candidate proteins sorted by full-protein rescoring. The initial Faiss retrieval uses `top_k=100` windows by default.

## Motif Scan Mode

XIMU can also scan the full indexed database for windows containing a user-specified short amino-acid motif, compare those windows to the query protein in feature space, and generate a species-level heatmap of cosine-distance counts.

Motif constraints:

- Motif length must be from 1 to 24 amino acids.
- Motifs must use the standard 20 amino-acid alphabet.
- `--mismatch` allows approximate matching; values from 0 to 3 are recommended.

Example with an exact serine-rich motif:

```bash
python scripts/query.py \
  --fasta query.faa \
  --db ximu_db/example \
  --mode motif \
  --motif SSSSSS \
  --heatmap-out motif_heatmap.pdf
```

Example allowing one mismatch:

```bash
python scripts/query.py \
  --seq "MSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS" \
  --db ximu_db/example \
  --mode motif \
  --motif RGRGRG \
  --mismatch 1 \
  --heatmap-out motif_scan.pdf
```

Disable heatmap output:

```bash
python scripts/query.py \
  --fasta query.faa \
  --db ximu_db/example \
  --mode motif \
  --motif SSSSSS \
  --no-heatmap
```

Motif heatmaps use raw motif-window counts per species and cosine-distance bin. The x-axis is ordered from larger to smaller cosine distance, so the right side contains windows closest to the query. Cell colors use a log-scaled `OrRd` count scale, and each cell is labeled with the raw window count.

## Analysis Output

The repository includes a generated example heatmap from the full database:

- [Motif cosine-distance heatmap](motif_cosine_dist_heatmap.pdf)

This PDF shows raw window counts by species and cosine-distance bin for the repository's reference query sequence. The x-axis is ordered from larger to smaller cosine distance, so the right side contains windows closest to the query.

## Database

The full database built on 2026-04-18 contains the following indexed species.

| Species | Genome label | Proteins |
|---|---|---:|
| Triticum timopheevii | `Triticum_timopheevii_WRC_timopheevii_genome_with_organelles_pep_all` | 215,467 |
| Triticum aestivum | `Triticum_aestivum_IWGSC_pep_all` | 106,402 |
| Brassica napus | `Brassica_napus_AST_PRJEB5043_v1_pep_all` | 97,939 |
| Oryza sativa | `Oryza_sativa_all_models_pep_longest` | 55,745 |
| Zea mays | `Zea_mays_Zm_B73_REFERENCE_NAM_5_0_pep_all` | 39,756 |
| Gossypium raimondii | `Gossypium_raimondii_Graimondii2_0_v6_pep_all` | 37,852 |
| Hordeum vulgare | `Hordeum_vulgare_MorexV3_pseudomolecules_assembly_pep_all` | 35,825 |
| Solanum lycopersicum | `Solanum_lycopersicum_SL3_0_pep_all` | 33,810 |
| Physcomitrium patens | `Physcomitrium_patens_Phypa_V3_pep_all` | 31,172 |
| Danio rerio | `Danio_rerio_GRCz11_pep_all` | 28,893 |
| Amborella trichopoda | `Amborella_trichopoda_AMTR1_0_pep_all` | 27,302 |
| Homo sapiens | `Homo_sapiens_GRCh38_pep_all` | 22,868 |
| Xenopus tropicalis | `Xenopus_tropicalis_UCB_Xtro_10_0_pep_all` | 22,078 |
| Mus musculus | `Mus_musculus_GRCm39_pep_all` | 21,980 |
| Chlamydomonas reinhardtii | `Chlamydomonas_reinhardtii_Chlamydomonas_reinhardtii_v5_5_pep_all` | 17,731 |
| Drosophila melanogaster | `Drosophila_melanogaster_BDGP6_32_pep_all` | 13,506 |
| Bradyrhizobium japonicum | `Bradyrhizobium_japonicum_gca_000773865_ASM77386v1_pep_all_longest` | 9,121 |
| Saccharomyces cerevisiae | `Saccharomyces_cerevisiae_R64_1_1_pep_all` | 6,485 |
| Escherichia coli | `Escherichia_coli_110957_gca_000485615_ASM48561v1_pep_all` | 5,300 |
| Schizosaccharomyces pombe | `Schizosaccharomyces_pombe_ASM294v2_pep_longsest` | 5,128 |
| Synechocystis sp | `Synechocystis_sp_pcc_6803_gca_000009725_ASM972v1_longest_pep` | 3,543 |
| Cyccr1 GeneCatalog proteins 20200805 aa | `Cyccr1_GeneCatalog_proteins_20200805_aa` | 2,629 |
| Nanoce1779 2 GeneCatalog proteins 20180119 aa | `Nanoce1779_2_GeneCatalog_proteins_20180119_aa` | 397 |
| Chabra1 GeneCatalog proteins 20200807 aa | `Chabra1_GeneCatalog_proteins_20200807_aa` | 363 |
| Cyapar1 GeneCatalog proteins 20200807 aa | `Cyapar1_GeneCatalog_proteins_20200807_aa` | 202 |
| ChloA99 1 GeneCatalog proteins 20200807 aa | `ChloA99_1_GeneCatalog_proteins_20200807_aa` | 30 |

Summary:

- Total source species processed: 27
- Indexed species: 26
- Source species with no standard matching proteins: `Cyamer1_GeneCatalog_proteins_20180616_aa`
- Total indexed proteins: 841,524
- Total indexed windows: 13,287,321
- Build date: 2026-04-18

## Tests

```bash
python -m pytest tests
```

## Please Cite

If you use XIMU, please also cite the following LCR benchmarking work:

"A Benchmarking Framework for Comparative Evaluation of Low-Complexity Region Detection Tools in the Human Proteome." bioRxiv (2026). DOI: 10.64898/2026.01.24.701293v1

This work provides a systematic benchmarking framework for LCR detection methods. XIMU's low-complexity space coordinates, mutation percentage and dominant residue fraction, follow its theoretical framework.
