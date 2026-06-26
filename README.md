# JEPA-WMS — From Scratch

A personal learning project: reimplementing a **Joint Embedding Predictive Architecture (JEPA)** for **World Model** learning, built from scratch in Python.

> Built to understand self-supervised world modeling deeply — every component written by hand, no high-level SSL library.

---

## What this project covers

- Vision Transformer (ViT) backbone — `models/vit.py`
- DINO-style self-supervised objective — `models/dino.py`
- World model training on PushT environment sequences — `data/dataset_PushT.py`
- Training loop with EMA target encoder — `src/training.py`
- Downstream task benchmarking — `src/benchmark.py`

---

## Core idea

Rather than predicting future observations in pixel space (like a video model), JEPA learns to predict **future latent representations** given a context. Applied to world modeling, this means the model learns a structured embedding space that captures the dynamics of an environment — without ever reconstructing raw observations.

```
context state ──► context encoder ──► predictor ──► predicted latent
                                                           ▲
target state  ──► target encoder (EMA) ──────────────────── (loss)
```

---

## Structure

```
data/
  dataset_PushT.py    # PushT environment episode dataset
  utils.py            # Preprocessing, normalization
models/
  vit.py              # Vision Transformer backbone (from scratch)
  dino.py             # DINO self-supervised objective
  utils.py            # EMA update, weight initialization
src/
  main.py             # Entry point
  training.py         # Training loop
  benchmark.py        # Evaluation on downstream tasks
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
python src/main.py        # train
python src/benchmark.py   # evaluate
```

---

## Status

Work in progress — built for learning and portfolio purposes.