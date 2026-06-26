# JEPA-WMS — From Scratch

A personal learning project: reimplementing a **Joint Embedding Predictive Architecture (JEPA)** data, built from scratch in Python.

> Built to understand self-supervised learning deeply — not to use a library, but to write every component by hand.

---

## What this project covers

- Vision Transformer (ViT) backbone — implemented from scratch in `models/vit.py`
- DINO-style self-supervised training — `models/dino.py`
- Custom dataset pipeline on PushT-style WMS sequences — `data/dataset_PushT.py`
- Training loop with EMA target encoder — `src/training.py`
- Benchmarking downstream task performance — `src/benchmark.py`

---

## Structure

```
data/
  dataset_PushT.py    # Dataset class for WMS event sequences
  utils.py            # Data preprocessing helpers
models/
  vit.py              # Vision Transformer backbone
  dino.py             # DINO self-supervised objective
  utils.py            # EMA update, weight init
src/
  main.py             # Entry point
  training.py         # Training loop
  benchmark.py        # Downstream evaluation
```

---

## Installation

```bash
git clone https://github.com/loumpare/JEPA-WMS_From-Scratch.git
cd JEPA-WMS_From-Scratch

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Requires:** Python ≥ 3.10, PyTorch ≥ 2.1

---

## Run

```bash
# Train
python src/main.py

# Benchmark
python src/benchmark.py
```

---

## Why JEPA for WMS?

Standard supervised approaches for warehouse systems (demand forecasting, slot assignment, order flow) require large labeled datasets. JEPA learns useful representations from raw sequences **without labels**, by predicting latent representations of future states — making it well suited to WMS data where annotations are scarce.

---

## Status

Work in progress — built for learning and portfolio purposes.