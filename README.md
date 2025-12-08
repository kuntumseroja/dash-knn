

---
# Proximity Finder (Corp/SME ↔ Prospects)

Two-stage “proximity” system to surface relationships between companies:

1. **ANN candidate search** (Nearest Neighbors on fused embeddings `Z`)
2. **Supervised re-ranker** (XGBoost) over rich pair features (similarity, Jaccard, geo, directors, industry…)

The Streamlit app visualizes the top-K neighbors with explainability.

---

## 1) Prerequisites

* **Python 3.11** (recommended). Install with one of:

  * macOS (Homebrew): `brew install python@3.11`
  * Windows: install Python 3.11 from python.org
  * Linux: `pyenv install 3.11.9` (or system package)
* **Git**
* (macOS) Command line tools: `xcode-select --install`

> **Note:** The code uses `scikit-learn`'s `NearestNeighbors` (cosine, brute) for ANN search, not FAISS. Re-ranking uses `xgboost`. Some packages in `requirements.txt` (faiss-cpu, lightgbm, shap) are included but not currently used in the codebase.

---

## 2) Project layout (key files)

```
data/
  raw/
    entities.csv             # entity_id,name,industry_code,about_text,lat,lon,is_customer
    directors.csv            # entity_id,person_name
    transactions.csv         # src_entity_id,dst_entity_id,yyyymm,amount,category
    financials.csv           # entity_id,year,revenue,margin,utilization,delinq_flag
    suppliers.csv            # buyer_entity_id,supplier_entity_id
    labels_links.csv         # left_entity_id,right_entity_id,link_type  (positives for training)

src/
  features/
    build_text.py            # make text embeddings
    build_txn.py             # txn/counterparty features
    build_graph.py           # suppliers/directors block
    fuse.py                  # fuse all → Z.npy
  models/
    ann_index.py             # sklearn NearestNeighbors (cosine)
    train_reranker.py        # XGBoost (supervised)
  eval/
    report_metrics.py        # Precision@K / Recall@K

app/
  app.py                     # Streamlit dashboard (graph + table)
config.yaml                  # paths + ANN defaults (K0, K)
```

---

## 3) Create a Python environment

```bash
# From repo root
python3.11 -m venv .venv311
source .venv311/bin/activate         # Windows: .venv311\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
```

> **Note:** Ensure your Python environment is activated before installing dependencies and running scripts. You should see `(.venv311)` in your terminal prompt.

---

## 4) Install dependencies

Install from `requirements.txt`:

```bash
pip install -r requirements.txt
```

**Key dependencies:**
- **Core:** numpy, pandas, scikit-learn, pyyaml, joblib
- **Modeling:** xgboost (for re-ranker)
- **Embeddings:** sentence-transformers (requires torch)
- **Graph:** networkx (for graph embeddings)
- **App:** streamlit, st-cytoscape

**Note:** `requirements.txt` includes some optional packages (faiss-cpu, lightgbm, shap) that are not currently used by the codebase. These can be removed if you encounter installation issues.

> If `torch` or `sentence-transformers` installation fails on your platform, try:
>
> * macOS (Apple Silicon): `pip install torch --index-url https://download.pytorch.org/whl/cpu`
> * Or re-run `pip install torch` once more (wheels are available for M-series Macs).

---

## 5) Configure paths (optional)

Check `config.yaml`:

```yaml
paths:
  raw_dir: data/raw
  processed_dir: data/processed
  models_dir: models
  entities_csv: data/raw/entities.csv
  directors_csv: data/raw/directors.csv
  transactions_csv: data/raw/transactions.csv
  financials_csv: data/raw/financials.csv
  suppliers_csv: data/raw/suppliers.csv
  labels_links_csv: data/raw/labels_links.csv

features:
  txn_embed_dim: 32
  graph_embed_dim: 32
  weights:
    text: 1.0
    txn: 1.0
    graph: 1.0

ann:
  hnsw_M: 64          # Not used (sklearn uses brute-force)
  efConstruction: 200 # Not used (sklearn uses brute-force)
  K0: 400             # ANN candidates (pre-rerank)
  K:  20              # Top-K to display/evaluate
```

> **Note:** The `hnsw_M` and `efConstruction` parameters in `config.yaml` are for FAISS HNSW, but the code currently uses sklearn's brute-force NearestNeighbors. These parameters are ignored but kept for potential future migration.

---

## 6) End-to-end: build → train → evaluate → run

> Run each step from the repo root (env activated).
> 
> **Alternative:** Use the `Makefile` for convenience:
> ```bash
> make featurize  # Build all features
> make index      # Build ANN index
> make train      # Train reranker + evaluate
> make app        # Launch Streamlit app
> make all        # Run featurize + index + train
> ```

### 6.1 Build features & embeddings

```bash
python src/features/build_text.py
python src/features/build_txn.py
python src/features/build_graph.py
python src/features/fuse.py
```

**Outputs** (in `data/processed/`):

* `Z.npy` — fused embedding matrix (one row per entity)
* `index_to_entity.json` — mapping between matrix row and `entity_id`

### 6.2 Build ANN index (Nearest Neighbors, cosine)

```bash
python src/models/ann_index.py
```

**Output** (in `models/`):

* `ann_sklearn.joblib`

### 6.3 Train the supervised re-ranker (XGBoost)

```bash
python src/models/train_reranker.py
```

* Uses `data/raw/labels_links.csv` as **positives**; automatically samples **negatives**
* Computes pair features: `cos_sim, jacc_ctp (Jaccard), geo_km, industry_match, director_overlap, diff_mean, had_mean`

**Output** (in `models/`):

* `reranker_xgb.json`

### 6.4 Evaluate (precision\@K / recall\@K)

```bash
python src/eval/report_metrics.py
```

**Output**:

* `data/processed/metrics.json` (e.g., `{"precision_at_20": ..., "recall_at_20": ..., "queries_evaluated": 100}`)

### 6.5 Launch the dashboard

```bash
streamlit run app/app.py
```

* Choose an entity in the sidebar
* **K0** = ANN candidates; **Top-K** = final neighbors after re-ranking
* **Edge thickness** ∝ score; colors: **green** same industry, **orange** common directors, **blue** otherwise
* Table includes tooltips + **Download results (CSV)**

> The app hot-reloads on save. If needed, click **🔄 Reload models/data** in the sidebar to clear caches.

---

## 7) Updating data

When you edit/append CSVs in `data/raw/`, re-run only what’s needed:

* Changed **about\_text/industry** → `build_text.py` → `fuse.py` → `ann_index.py`
* Changed **transactions** → `build_txn.py` → `fuse.py` → `ann_index.py`
* Changed **directors/suppliers** → `build_graph.py` → `fuse.py` → `ann_index.py`
* Added/edited **labels\_links.csv** → `train_reranker.py` → `report_metrics.py`

Then refresh the app (or use the **Reload** button).

---

## 8) Troubleshooting

* **"No matching distribution for st-cytoscape==..."**
  The version in `requirements.txt` should work. If issues persist, try installing without version pin: `pip install st-cytoscape`.

* **"faiss / lightgbm build failed"**
  These packages are in `requirements.txt` but not used by the code. You can safely remove them if they cause installation issues:
  ```bash
  pip uninstall faiss-cpu lightgbm shap
  ```

* **"KNeighbors: n\_neighbors > n\_samples"**
  Fixed in code; we **clamp K0 and K** to safe values automatically based on dataset size.

* **Torch / sentence-transformers install issues**
  Try:
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  pip install sentence-transformers
  ```
  Or re-run `pip install torch` (wheels are available for most platforms including Apple Silicon).

* **App doesn't reflect new models/data**
  Use the sidebar **🔄 Reload models/data**, or restart:
  `Ctrl+C` then `streamlit run app/app.py`.

* **Missing dependencies (xgboost, joblib, networkx)**
  Ensure all packages from `requirements.txt` are installed. If `xgboost` is missing, the re-ranker training will fail.

---

## 9) What’s supervised vs unsupervised?

* **Unsupervised**: building embeddings `Z`, ANN/K-NN search (no labels needed).
* **Supervised**: XGBoost re-ranker trained on `labels_links.csv` (positives) vs sampled negatives.
* **Self-supervised (indirect)**: the sentence-transformer used for text embeddings was pretrained elsewhere; we don’t fine-tune it here.

---

## 10) One-liner to run everything (optional)

Create a simple script (bash) and run `bash run_all.sh`:

```bash
#!/usr/bin/env bash
set -e
python src/features/build_text.py
python src/features/build_txn.py
python src/features/build_graph.py
python src/features/fuse.py
python src/models/ann_index.py
python src/models/train_reranker.py
python src/eval/report_metrics.py
streamlit run app/app.py
```

---


