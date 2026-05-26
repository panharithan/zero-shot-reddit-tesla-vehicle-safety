# Monitoring Tesla Vehicle Safety Reputation from Reddit Using Zero-Shot Classification

**Authors:** Panharith An, Joel Isaria Mwende, Rifqi Ardia Ramadhan  
**Course:** Open-Source Intelligence — SRH Berlin University of Applied Sciences  
**Instructor:** Prof. Dr. Klaus Schwarz

---

## Overview

This repository contains the Jupyter notebook for a three-stage OSINT pipeline that extracts safety intelligence from Tesla-related Reddit discourse. The pipeline embeds posts, detects semantic communities, and labels each community using a grammar-constrained LLM. Weekly cluster activity is tracked and evaluated against NHTSA recall records.

---

## Pipeline

**Stage 1 — Preprocessing**  
163,857 raw Reddit posts are cleaned and deduplicated, reducing the corpus to 29,245 posts.

**Stage 2 — Embedding and Community Detection**  
Posts are embedded using `all-MiniLM-L6-v2` and grouped into 34 semantic communities via cosine similarity clustering at a threshold of 0.75.

**Stage 3 — LLM Labelling and Intelligence Extraction**  
Each community is labelled by Mistral Nemo Instruct 12B (Q5_K_M) via `llama-cpp-python` with GBNF grammar-constrained JSON output. Weekly post counts per cluster are tracked and alert thresholds are set at mean plus one standard deviation.

---

## Datasets

| Dataset | Source |
|---|---|
| Tesla Reddit posts | [Kaggle — aravindrajpalepu/teslaissues-reddit](https://www.kaggle.com/datasets/aravindrajpalepu/teslaissues-reddit) |
| NHTSA recall records | [data.transportation.gov](https://catalog.data.gov/dataset/recalls-data) |
| Google Trends | [trends.google.com](https://trends.google.com) — keyword: "Tesla recalls", US, Sep 2023 – Sep 2024 |

---

## Models

| Model | Purpose |
|---|---|
| `all-MiniLM-L6-v2` | Sentence embedding (384-dim) |
| `Mistral-Nemo-Instruct-2407-Q5_K_M` | Community labelling via GBNF-constrained JSON |

---

## Requirements

```
kagglehub
pandas
numpy
torch
scipy
matplotlib
sentence-transformers
pytrends
huggingface_hub
hf_transfer
llama-cpp-python (CUDA build)
```

The notebook was run on Google Colab with a Tesla T4 GPU (15.6 GB VRAM). Mistral Nemo Q5_K_M requires at least 10 GB VRAM.

---

## Reproducibility

All random seeds are fixed at 42. Model version and quantization are pinned. Community detection uses deterministic cosine similarity clustering with fixed threshold and minimum community size parameters.

---

## Key Results

- Clean posts: 29,245 (82.2% noise removed)
- Communities detected: 34 (3.9% corpus coverage)
- Top alert clusters: Autopilot Safety Concerns (14 weeks), Lane Detection Malfunction (13 weeks)
- SCI vs NHTSA recalls: r = 0.090 (not significant)

---

## Citation

If you use this work, please cite:

> P. An, J. I. Mwende, and R. A. Ramadhan, "Monitoring Tesla Vehicle Safety Reputation from Reddit Using Zero-Shot Classification," SRH Berlin University of Applied Sciences, 2026.
