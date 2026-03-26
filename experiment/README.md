# Experiment: Qdrant ANN vs OpenSearch KNN Comparative Analysis

**Experiment ID:** EXP-VSC-2026-001

## Purpose

Direct comparative evaluation of vector search quality between:
- **Control:** Extant OpenSearch cluster with KNN (HNSW via k-NN plugin)
- **Treatment:** Qdrant cluster with ANN (HNSW + INT8 quantization + rescore)

## Hypotheses

| ID | Tests | Metric |
|----|-------|--------|
| H1 | Qdrant recall > OpenSearch recall | Recall@10 |
| H2 | Qdrant latency < OpenSearch latency | p99 latency (ms) |
| H3 | Qdrant cost < OpenSearch cost by >40% | Cost per 1M queries |
| H4 | Qdrant exact recall >= 0.90 | Exact Recall@10 |
| S1 | Near-duplicate reduction | Dup rate in top-10 |
| S2 | RRF > single-vector search | Card ID Recall@10 |
| S3 | Catalogue boost improves results | Catalogue in top-10 |

## Directory Structure

- `dataset/` — Query set construction, ground truth annotation, brute-force reference
- `runner/` — Experiment orchestrator, latency harness, recall evaluator
- `analysis/` — Statistical tests, cost model, report generator
- `results/` — Output artefacts (git-ignored)
