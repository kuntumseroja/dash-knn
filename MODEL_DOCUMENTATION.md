# Model Documentation - Proximity Finder System

## Table of Contents
1. [Overview](#overview)
2. [Models Used](#models-used)
3. [Training Mode](#training-mode)
4. [Inference Mode](#inference-mode)
5. [Feature Engineering](#feature-engineering)
6. [Model Architecture Details](#model-architecture-details)

---

## Overview

This **Proximity Finder** system uses a **two-stage retrieval and re-ranking pipeline** to discover relationships between companies:

1. **Stage 1: ANN (Approximate Nearest Neighbors) Search**  
   Fast candidate retrieval using cosine similarity on fused embeddings
   
2. **Stage 2: Supervised Re-ranking**  
   XGBoost model scores candidates using rich pair features for final ranking

This architecture balances efficiency (ANN for speed) with accuracy (supervised learning for precision).

---

## Models Used

### 1. Sentence-Transformer (Text Embeddings)
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Purpose**: Generate semantic embeddings from company text descriptions
- **Type**: Pre-trained transformer model (unsupervised/self-supervised)
- **Output**: 384-dimensional dense vectors
- **Framework**: HuggingFace Sentence-Transformers library

### 2. sklearn NearestNeighbors (ANN Index)
- **Model**: `sklearn.neighbors.NearestNeighbors`
- **Purpose**: Fast similarity search for candidate retrieval
- **Algorithm**: Brute-force with cosine metric
- **Type**: Unsupervised K-NN search
- **Framework**: scikit-learn 1.5.2

### 3. XGBoost (Re-ranker)
- **Model**: `xgboost.Booster`
- **Purpose**: Supervised re-ranking of ANN candidates
- **Objective**: Binary classification (binary:logistic)
- **Metric**: Area Under Precision-Recall Curve (aucpr)
- **Framework**: XGBoost 2.0.3

---

## Training Mode

Training consists of four phases: **feature building → embedding fusion → ANN index creation → re-ranker training**.

### Phase 1: Feature Building

#### 1.1 Text Embeddings (`src/features/build_text.py`)

**Input Data:**
- `data/raw/entities.csv` → `about_text`, `industry_code` columns

**Process:**
```python
# Concatenate text description with industry code
texts = about_text + ' Industry:' + industry_code

# Generate embeddings using sentence-transformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(texts, normalize_embeddings=True)
```

**Output:**
- `data/processed/text_embed.npy` → (N entities × 384 dimensions)
- `data/processed/entity_ids.parquet` → entity ID ordering

**Why this works:**
- Pre-trained BERT-based model captures semantic similarity
- Normalized embeddings enable cosine similarity comparison
- Industry code augmentation adds categorical context

#### 1.2 Transaction Embeddings (`src/features/build_txn.py`)

**Input Data:**
- `data/raw/transactions.csv` → transaction patterns between entities

**Process:**
- Aggregates counterparty relationships
- Builds behavior-based features (e.g., transaction counts, categories)
- Reduces to fixed-dimension representation

**Output:**
- `data/processed/txn_embed.npy` → (N entities × 32 dimensions)
- `data/processed/txn_feats.parquet` → raw transaction features

**Why this works:**
- Captures "who transacts with whom" patterns
- Companies with similar counterparty networks are likely related

#### 1.3 Graph Embeddings (`src/features/build_graph.py`)

**Input Data:**
- `data/raw/directors.csv` → shared directors between companies
- `data/raw/suppliers.csv` → supplier-buyer relationships

**Process:**
- Constructs graph structure from director/supplier links
- Generates structural embeddings (e.g., node2vec-style or simple aggregation)

**Output:**
- `data/processed/graph_embed.npy` → (N entities × 32 dimensions)

**Why this works:**
- Director overlap indicates ownership/control relationships
- Supplier networks reveal business ecosystem connections

### Phase 2: Embedding Fusion (`src/features/fuse.py`)

**Purpose:** Combine all embeddings into a single representation for ANN search

**Process:**
```python
# L2 normalize each embedding type
text_norm = text / ||text||
txn_norm = txn / ||txn||
graph_norm = graph / ||graph||

# Weighted concatenation
Z = [w_text × text_norm | w_txn × txn_norm | w_graph × graph_norm]
```

**Configuration** (`config.yaml`):
```yaml
features:
  weights:
    text: 1.0    # Equal weighting
    txn: 1.0
    graph: 1.0
```

**Output:**
- `data/processed/Z.npy` → (N entities × 448 dimensions)
  - 384 (text) + 32 (txn) + 32 (graph) = 448 total dims
- `data/processed/index_to_entity.json` → row index to entity_id mapping

**Why this works:**
- Normalization ensures each modality contributes equally
- Concatenation preserves all information without conflicts
- Weights can be tuned to prioritize certain signals

### Phase 3: ANN Index Building (`src/models/ann_index.py`)

**Input:**
- `data/processed/Z.npy` → fused embeddings

**Process:**
```python
Z = np.load('Z.npy')
index = NearestNeighbors(metric='cosine', algorithm='brute')
index.fit(Z)
joblib.dump(index, 'ann_sklearn.joblib')
```

**Output:**
- `models/ann_sklearn.joblib` → trained NearestNeighbors index

**Why brute-force?**
- Simplicity: No complex index building parameters
- Accuracy: Exact K-NN search (no approximation errors)
- Speed: Sufficient for datasets < 100K entities
- Portability: Avoids FAISS/native library dependencies

**Alternative:** For larger datasets (>100K entities), consider FAISS HNSW index for faster search.

### Phase 4: Re-ranker Training (`src/models/train_reranker.py`)

**Input Data:**
- `data/raw/labels_links.csv` → positive examples (supplier_of, common_owner, co_customer)
- Sampled negative pairs from random entity combinations

**Sampling Strategy:**
```python
# Positive pairs from labeled data
pos = [(left_id, right_id, link_type) for each label]

# Negative sampling: random pairs not in positive set
neg = []
while len(neg) < len(pos):
    a, b = random.choice(entities), random.choice(entities)
    if (a,b) not in positive_set and a != b:
        neg.append((a, b, 'no_link'))

pairs = pos + neg  # Balanced dataset
```

**Feature Engineering for Pairs:**

For each candidate pair `(A, B)`, compute:

| Feature | Description | Formula |
|---------|-------------|---------|
| `cos_sim` | Cosine similarity of embeddings | `(Z_a · Z_b) / (||Z_a|| × ||Z_b||)` |
| `jacc_ctp` | Jaccard index of counterparties | `|ctp_A ∩ ctp_B| / |ctp_A ∪ ctp_B|` |
| `geo_km` | Geographic distance (km) | `haversine(lat_A, lon_A, lat_B, lon_B)` |
| `ind_match` | Industry code match (binary) | `1 if industry_A == industry_B else 0` |
| `dir_overlap` | Shared directors count | `|directors_A ∩ directors_B|` |
| `diff_mean` | Mean absolute difference | `mean(|Z_a - Z_b|)` |
| `had_mean` | Hadamard product mean | `mean(Z_a * Z_b)` |

**XGBoost Training:**
```python
params = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",          # Optimize precision-recall
    "eta": 0.05,                      # Learning rate
    "max_depth": 6,                   # Tree depth
    "subsample": 0.9,                 # Row sampling
    "colsample_bytree": 0.9,          # Column sampling
    "tree_method": "hist"             # Fast histogram-based
}

model = xgb.train(params, dtrain, 
                  num_boost_round=600,
                  early_stopping_rounds=50)
```

**Output:**
- `models/reranker_xgb.json` → trained XGBoost model

**Why XGBoost?**
- Handles mixed feature types (continuous, binary, counts)
- Built-in feature importance for explainability
- Fast inference (critical for real-time apps)
- Strong performance on tabular data
- Robust to feature scaling differences

**Why AUCPR metric?**
- Precision-recall is better for imbalanced datasets
- Emphasizes correct positive predictions
- More relevant than accuracy when true negatives dominate

---

## Inference Mode

Inference runs inside the Streamlit app (`app/app.py`) or evaluation scripts.

### Step 1: Load Models and Data

```python
# Load artifacts (cached by Streamlit)
Z = np.load('Z.npy')                              # Fused embeddings
index = joblib.load('ann_sklearn.joblib')         # ANN index
reranker = xgb.Booster()                          # XGBoost model
reranker.load_model('reranker_xgb.json')
entities = pd.read_csv('entities.csv')            # Metadata
```

### Step 2: ANN Candidate Retrieval

**Goal:** Find K0 nearest neighbors (default K0=400) using cosine similarity

```python
# Get query entity index
query_id = selected_entity_id
query_idx = entity_id_to_index[query_id]

# Search ANN index
distances, indices = index.kneighbors(
    Z[query_idx:query_idx+1],
    n_neighbors=K0,
    return_distance=True
)

# Convert indices back to entity IDs
candidate_ids = [index_to_entity[i] for i in indices[0]]
```

**Output:** ~400 candidate entities (adjustable via UI slider)

**Performance:** 
- O(N) time complexity (brute-force cosine)
- ~10-50ms for 10K entities on modern CPU
- Dominated by vector dot products (highly optimized in NumPy/BLAS)

### Step 3: Feature Computation for Pairs

For each candidate pair `(query_entity, candidate)`:

```python
for candidate_id in candidate_ids:
    # Embedding similarity
    cos_sim = cosine_similarity(Z[query_idx], Z[cand_idx])
    
    # Counterparty overlap
    query_counterparties = get_counterparties(query_id)
    cand_counterparties = get_counterparties(candidate_id)
    jacc = jaccard(query_counterparties, cand_counterparties)
    
    # Geographic distance
    geo_km = haversine(query_lat, query_lon, cand_lat, cand_lon)
    
    # Industry match
    ind_match = (query_industry == cand_industry)
    
    # Director overlap
    dir_overlap = len(query_directors ∩ cand_directors)
    
    # Embedding difference features
    diff_mean = mean(|Z_query - Z_cand|)
    had_mean = mean(Z_query * Z_cand)
    
    feature_vector = [cos_sim, jacc, geo_km, ind_match, 
                      dir_overlap, diff_mean, had_mean]
```

**Output:** Feature matrix X of shape `(K0, 7)`

### Step 4: XGBoost Re-ranking

```python
# Create DMatrix with feature names
dmatrix = xgb.DMatrix(X, feature_names=[
    'cos_sim', 'jacc_ctp', 'geo_km', 'ind_match',
    'dir_overlap', 'diff_mean', 'had_mean'
])

# Predict link probabilities
scores = reranker.predict(dmatrix)  # Shape: (K0,)
```

**Output:** Probability scores [0, 1] for each candidate

**What the score means:**
- `score > 0.7` → Strong relationship signal (high confidence)
- `0.5 < score < 0.7` → Moderate relationship signal
- `score < 0.5` → Weak or no relationship

### Step 5: Filtering and Final Top-K

```python
# Filter by user criteria
filtered = candidates[
    (scores >= min_score) &           # Score threshold
    (geo_km <= max_distance) &        # Geographic radius
    (industry_match if only_industry) # Industry filter
]

# Sort by score and take top K
top_k = filtered.sort_values('score', ascending=False).head(K)
```

**Output:** Final Top-K neighbors (default K=20)

**UI Controls:**
- **K0 slider**: Breadth of search (more candidates = higher recall)
- **Top-K slider**: Depth of results (final list size)
- **Filters**: Industry, geography, customer status, director overlap

### Step 6: Explainability

For each selected neighbor, the system provides:

```python
def explain_candidate(row):
    reasons = []
    if row['industry_match']:
        reasons.append("Same industry")
    if row['director_overlap'] > 0:
        reasons.append(f"Shared directors: {row['director_overlap']}")
    if row['shared_counterparties'] > 0:
        reasons.append(f"Shared counterparties: {row['shared_counterparties']}")
    if row['similarity'] >= 0.5:
        reasons.append(f"Strong semantic similarity: {row['similarity']:.2f}")
    if row['geo_km'] <= 250:
        reasons.append(f"Nearby: {row['geo_km']:.0f} km")
    return reasons
```

**CTA (Call-to-Action) Logic:**
```python
if score >= 0.70:
    return "Upsell / Cross-sell" if is_customer else "Prospect outreach"
elif director_overlap > 0 or jaccard >= 0.30:
    return "Warm intro"
elif geo_km <= 250 and similarity >= 0.50:
    return "Local lead"
else:
    return "Research"
```

---

## Feature Engineering

### Why These Features?

| Feature | Business Rationale |
|---------|-------------------|
| **cos_sim** | Semantic similarity captures "what they do" (text descriptions, industry) |
| **jacc_ctp** | Companies sharing counterparties likely operate in same ecosystem |
| **geo_km** | Proximity enables local partnerships, shared clients, supply chains |
| **ind_match** | Same industry → likely competitors or collaborators |
| **dir_overlap** | Shared directors indicate ownership ties or strategic relationships |
| **diff_mean** | Embedding distance captures dissimilarity in business model |
| **had_mean** | Element-wise product captures co-activation patterns |

### Feature Importance (Example)

After training, XGBoost provides feature importance scores:

```python
importance = reranker.get_score(importance_type='gain')
# Example output:
# cos_sim: 0.35        (most important)
# jacc_ctp: 0.22
# dir_overlap: 0.18
# ind_match: 0.12
# geo_km: 0.08
# diff_mean: 0.03
# had_mean: 0.02
```

This tells us:
- Embedding similarity is the strongest signal
- Counterparty overlap and director connections are highly predictive
- Geographic distance is less critical (unless explicitly filtered)

---

## Model Architecture Details

### 1. Embedding Fusion Architecture

```
┌─────────────────┐
│  Raw Data       │
└────┬────────────┘
     │
     ├──> Text (about_text + industry) ──> SentenceTransformer ──> 384-d
     ├──> Transactions (patterns)      ──> Aggregation        ──> 32-d
     └──> Graph (directors, suppliers) ──> Graph Embedding    ──> 32-d
            │
            v
       L2 Normalize each
            │
            v
       Weighted Concat
            │
            v
       Z (448-d fused embedding)
```

### 2. Two-Stage Pipeline

```
┌──────────────┐
│ Query Entity │
└──────┬───────┘
       │
       v
┌──────────────────────────────┐
│ Stage 1: ANN Search          │
│ - Load Z matrix              │
│ - Cosine similarity K-NN     │
│ - Return K0 candidates       │
│   (e.g., 400)                │
└──────┬───────────────────────┘
       │
       v
┌──────────────────────────────┐
│ Stage 2: Feature Engineering │
│ - For each pair:             │
│   - cos_sim                  │
│   - jacc_ctp                 │
│   - geo_km                   │
│   - ind_match                │
│   - dir_overlap              │
│   - diff_mean, had_mean      │
└──────┬───────────────────────┘
       │
       v
┌──────────────────────────────┐
│ Stage 3: XGBoost Re-ranking  │
│ - Predict(feature_matrix)    │
│ - Score = P(link | features) │
└──────┬───────────────────────┘
       │
       v
┌──────────────────────────────┐
│ Stage 4: Filtering & Top-K   │
│ - Apply user filters         │
│ - Sort by score              │
│ - Return top K (e.g., 20)    │
└──────┬───────────────────────┘
       │
       v
┌──────────────────────────────┐
│ Visualization & Explainability│
│ - Network graph              │
│ - Feature table              │
│ - Explanation bullets        │
│ - CTA recommendations        │
└──────────────────────────────┘
```

### 3. XGBoost Model Details

**Hyperparameters:**
- `num_boost_round`: 600 (with early stopping at 50 rounds)
- `max_depth`: 6 (moderate tree depth prevents overfitting)
- `eta`: 0.05 (slow learning for better generalization)
- `subsample`: 0.9 (row sampling reduces overfitting)
- `colsample_bytree`: 0.9 (column sampling)
- `tree_method`: hist (faster histogram-based algorithm)

**Why these choices?**
- Low `eta` with high `num_boost_round` → more stable convergence
- Moderate `max_depth` → captures interactions without memorizing noise
- Subsampling → improves generalization and speed

**Training Process:**
1. Split data 80/20 train/test (stratified by label)
2. Train on train set with validation on test set
3. Use AUCPR as eval metric (prioritizes precision-recall balance)
4. Early stopping prevents overfitting (best iteration saved)
5. Save model to JSON format (portable, human-readable)

---

## Performance Characteristics

### Training Time (on typical hardware)

| Step | Time (10K entities) | Time (100K entities) |
|------|---------------------|----------------------|
| Text embeddings | 2-5 min | 20-30 min |
| Transaction features | 10-30 sec | 2-5 min |
| Graph features | 10-30 sec | 2-5 min |
| Embedding fusion | <1 sec | 1-2 sec |
| ANN index build | <1 sec | 2-5 sec |
| XGBoost training | 10-30 sec | 1-2 min |
| **Total** | **5-10 min** | **30-45 min** |

### Inference Time (per query)

| Step | Time |
|------|------|
| ANN search (K0=400) | 10-50 ms |
| Feature computation (400 pairs) | 50-100 ms |
| XGBoost scoring | 5-10 ms |
| **Total query time** | **70-160 ms** |

**Scalability Notes:**
- ANN search is O(N) with brute-force (linear in dataset size)
- For datasets >100K, use FAISS HNSW (O(log N) search)
- Feature computation parallelizable across pairs
- XGBoost inference is extremely fast (~0.02ms per prediction)

---

## Model Evaluation

Evaluation script: `src/eval/report_metrics.py`

**Metrics Computed:**
- **Precision@K**: Of the top-K predictions, how many are true links?
- **Recall@K**: Of all true links, how many are in the top-K?

**Example Output** (`data/processed/metrics.json`):
```json
{
  "precision_at_20": 0.78,
  "recall_at_20": 0.45,
  "queries_evaluated": 100
}
```

**Interpretation:**
- `precision_at_20 = 0.78` → 78% of top-20 results are relevant
- `recall_at_20 = 0.45` → The top-20 captures 45% of all true relationships

**Trade-off:**
- Increase K0 → higher recall, slower inference
- Increase K → more results, potentially lower precision
- Adjust score threshold → precision/recall trade-off

---

## Comparison: Training vs Inference

| Aspect | Training Mode | Inference Mode |
|--------|---------------|----------------|
| **Input** | All entities + labels | Single query entity |
| **Output** | Trained models | Top-K neighbors |
| **Time** | Minutes to hours | Milliseconds |
| **Models Used** | SentenceTransformer, XGBoost | sklearn K-NN, XGBoost |
| **Data Required** | Full dataset + ground truth | Embeddings + metadata |
| **Goal** | Learn relationship patterns | Predict relationships |
| **Frequency** | Batch (daily/weekly) | Real-time (per user query) |

**Offline (Training):**
- Build embeddings for all entities
- Train supervised re-ranker on labeled pairs
- Optimize for accuracy and generalization

**Online (Inference):**
- Fast candidate retrieval (ANN)
- Quick feature computation (vectorized)
- Real-time scoring (optimized XGBoost)
- Interactive UI with filters and explainability

---

## Best Practices

### Training
1. **Regularly retrain** when new entities or transactions are added
2. **Tune feature weights** in `config.yaml` based on domain knowledge
3. **Balance positive/negative sampling** to match real-world prevalence
4. **Use cross-validation** if dataset is small (<1000 labeled pairs)
5. **Monitor AUCPR** on validation set to detect overfitting

### Inference
1. **Cache models** in Streamlit (`@st.cache_resource`) for speed
2. **Adjust K0 dynamically** based on dataset size (K0 = min(400, N-1))
3. **Set reasonable defaults** for score threshold (e.g., 0.3-0.5)
4. **Provide fallback** to ANN-only results if filters are too strict
5. **Log queries** for offline analysis and retraining data

### Deployment
1. **Separate training/inference** pipelines (different compute needs)
2. **Version models** with timestamps (e.g., `reranker_xgb_20240115.json`)
3. **Monitor latency** with percentile metrics (p50, p95, p99)
4. **A/B test** model versions to validate improvements
5. **Document features** for compliance and explainability

---

## Relationship Parameters for Account Anchors

This section documents all possible parameters that can be identified and used to detect relationships across account anchors (central entities like corporate HQs, major subsidiaries, or hub organizations in business ecosystems).

### Currently Used Parameters (7 Core Features)

The XGBoost re-ranker currently uses these 7 parameters to identify relationships:

| Parameter | Type | Description | Data Source |
|-----------|------|-------------|-------------|
| **cos_sim** | Continuous (0-1) | Cosine similarity of fused embeddings (text + transaction + graph) | `Z.npy` embeddings |
| **jacc_ctp** | Continuous (0-1) | Jaccard index of shared counterparties | `transactions.csv` |
| **geo_km** | Continuous (km) | Haversine distance between entity coordinates | `entities.csv` (lat, lon) |
| **ind_match** | Binary (0/1) | Industry code match between entities | `entities.csv` (industry_code) |
| **dir_overlap** | Integer (count) | Number of shared directors/executives | `directors.csv` |
| **diff_mean** | Continuous | Mean absolute difference of embedding vectors | `Z.npy` embeddings |
| **had_mean** | Continuous | Mean Hadamard product (element-wise multiplication) of embeddings | `Z.npy` embeddings |

### Additional Parameters from Existing Data Sources

The following parameters are available in the data but not currently used in the re-ranker:

#### Financial Similarity Parameters

| Parameter | Type | Description | Calculation Method |
|-----------|------|-------------|-------------------|
| **revenue_ratio** | Continuous | Ratio of smaller to larger revenue | `min(revenue_A, revenue_B) / max(revenue_A, revenue_B)` |
| **revenue_diff** | Continuous | Absolute difference in revenue | `|revenue_A - revenue_B|` |
| **revenue_log_diff** | Continuous | Log-scale difference (handles wide scales) | `|log(revenue_A + 1) - log(revenue_B + 1)|` |
| **margin_similarity** | Continuous (0-1) | Similarity of profit margins | `1 - |margin_A - margin_B|` |
| **utilization_similarity** | Continuous (0-1) | Similarity of utilization rates | `1 - |utilization_A - utilization_B|` |
| **delinq_match** | Binary (0/1) | Both entities have delinquency flags | `(delinq_flag_A == 1) AND (delinq_flag_B == 1)` |
| **financial_health_match** | Binary (0/1) | Both entities have similar financial profiles | Combination of margin/utilization similarity thresholds |

**Data Source**: `financials.csv` (revenue, margin, utilization, delinq_flag)

#### Supplier Network Parameters

| Parameter | Type | Description | Calculation Method |
|-----------|------|-------------|-------------------|
| **supplier_overlap** | Integer (count) | Number of shared suppliers | `|suppliers_A ∩ suppliers_B|` |
| **supplier_jaccard** | Continuous (0-1) | Jaccard index of supplier sets | `|suppliers_A ∩ suppliers_B| / |suppliers_A ∪ suppliers_B|` |
| **buyer_overlap** | Integer (count) | Number of shared buyers (reverse supplier) | Count of entities that buy from both |
| **direct_supplier_link** | Binary (0/1) | Direct supplier-buyer relationship exists | `(A supplies B) OR (B supplies A)` |
| **supply_chain_distance** | Integer (hops) | Shortest path in supplier graph | Graph traversal distance |
| **common_supplier_ratio** | Continuous (0-1) | Percentage of suppliers in common | Normalized overlap metric |

**Data Source**: `suppliers.csv` (buyer_entity_id, supplier_entity_id)

#### Transaction Pattern Parameters

| Parameter | Type | Description | Calculation Method |
|-----------|------|-------------|-------------------|
| **txn_amount_ratio** | Continuous | Ratio of transaction volumes | Compare total transaction amounts |
| **txn_category_overlap** | Continuous (0-1) | Jaccard index of transaction categories | `|categories_A ∩ categories_B| / |categories_A ∪ categories_B|` |
| **txn_frequency_similarity** | Continuous (0-1) | Similarity in transaction frequency patterns | Compare monthly transaction counts |
| **seasonality_correlation** | Continuous (-1 to 1) | Correlation of seasonal patterns | Pearson correlation of monthly patterns |
| **avg_txn_amount_similarity** | Continuous (0-1) | Similarity in average transaction amounts | `1 - |mean_amount_A - mean_amount_B| / max(mean_amount_A, mean_amount_B)` |
| **txn_category_diversity_match** | Continuous (0-1) | Similarity in category diversity | Compare entropy/distribution of categories |
| **direct_txn_exists** | Binary (0/1) | Direct transactions between entities exist | `(A → B) OR (B → A)` |
| **txn_volume_total** | Continuous | Total transaction volume between entities | Sum of all A↔B transactions |

**Data Source**: `transactions.csv` (src_entity_id, dst_entity_id, amount, category, yyyymm)

#### Geographic and Demographics Parameters

| Parameter | Type | Description | Calculation Method |
|-----------|------|-------------|-------------------|
| **region_match** | Binary (0/1) | Same region/province/city | Extract from coordinates or entity metadata |
| **timezone_match** | Binary (0/1) | Same timezone | Derived from longitude |
| **geo_cluster** | Binary (0/1) | Within geographic cluster threshold | Density-based clustering |
| **customer_status_match** | Binary (0/1) | Both ETB or both NTB | `is_customer_A == is_customer_B` |

**Data Source**: `entities.csv` (lat, lon, is_customer)

#### Network Structure Parameters

| Parameter | Type | Description | Calculation Method |
|-----------|------|-------------|-------------------|
| **common_neighbors** | Integer (count) | Number of shared neighbors in transaction graph | Count entities connected to both |
| **adamic_adar_score** | Continuous | Weighted common neighbors score | Graph-based similarity metric |
| **preferential_attachment** | Continuous | Product of neighbor counts | `|neighbors_A| × |neighbors_B|` |
| **transitive_triangles** | Integer (count) | Number of triangles in graph (A→C, B→C, A↔B) | Graph motif counting |

**Data Source**: `transactions.csv` + `suppliers.csv` (graph structure)

### Derived/Computed Parameters

These parameters can be computed from existing data but require additional feature engineering:

#### Temporal Pattern Parameters

| Parameter | Type | Description | Calculation Method |
|-----------|------|-------------|-------------------|
| **relationship_age** | Continuous (months) | Duration of relationship (if direct transactions exist) | `max(date) - min(date)` from transactions |
| **txn_trend_correlation** | Continuous (-1 to 1) | Correlation of transaction trends over time | Time series correlation |
| **relationship_stability** | Continuous (0-1) | Consistency of transactions over time | Coefficient of variation of monthly counts |
| **recent_txn_flag** | Binary (0/1) | Transactions in last N months | Recent activity indicator |

#### Business Model Similarity Parameters

| Parameter | Type | Description | Calculation Method |
|-----------|------|-------------|-------------------|
| **text_similarity** | Continuous (0-1) | Cosine similarity of text embeddings only | `text_embed_A · text_embed_B` |
| **txn_embed_similarity** | Continuous (0-1) | Cosine similarity of transaction embeddings only | Transaction pattern similarity |
| **graph_embed_similarity** | Continuous (0-1) | Cosine similarity of graph embeddings only | Graph structure similarity |
| **business_model_match** | Continuous (0-1) | Composite score of business characteristics | Weighted combination of embeddings |

**Data Source**: `data/processed/text_embed.npy`, `txn_embed.npy`, `graph_embed.npy`

#### Hierarchical Relationship Parameters (Anchor-Specific)

| Parameter | Type | Description | Calculation Method |
|-----------|------|-------------|-------------------|
| **ancestor_descendant** | Binary (0/1) | One entity is ancestor/descendant via supplier chain | Path traversal in supplier graph |
| **sibling_entities** | Binary (0/1) | Entities share common supplier/buyer (siblings in supply chain) | Common parent in supplier tree |
| **ecosystem_distance** | Integer (hops) | Distance in business ecosystem graph | Shortest path in combined graph |
| **hub_centrality_diff** | Continuous | Difference in network centrality (PageRank, betweenness) | Centrality metric comparison |
| **anchor_proximity** | Continuous (0-1) | Distance to nearest anchor entity | For non-anchors, distance to ecosystem hub |

**Use Case**: Identifying relationships within corporate groups or business ecosystems

### Relationship Types Identifiable Across Account Anchors

Based on the available parameters, the following relationship types can be identified:

#### 1. **Corporate Relationships**
- **Parent-Subsidiary**: Shared directors + supplier links + high embedding similarity
- **Sibling Companies**: Common parent in supplier graph + shared directors
- **Holding Structure**: Network centrality + director overlap + transaction patterns

**Key Parameters**: `dir_overlap`, `supplier_chain_distance`, `ecosystem_distance`

#### 2. **Supply Chain Relationships**
- **Supplier-Buyer**: Direct supplier links or transaction patterns
- **Supply Chain Partners**: Shared suppliers/buyers + geographic proximity
- **Competitors**: Same industry + similar transaction patterns + geographic overlap

**Key Parameters**: `direct_supplier_link`, `supplier_jaccard`, `ind_match`, `txn_category_overlap`

#### 3. **Business Ecosystem Relationships**
- **Ecosystem Partners**: Shared counterparties + similar transaction categories
- **Co-Customers**: Entities that transact with same counterparties
- **Value Chain Participants**: Sequential transactions (A→B→C patterns)

**Key Parameters**: `jacc_ctp`, `common_neighbors`, `txn_category_overlap`, `supply_chain_distance`

#### 4. **Financial Relationships**
- **Similar Financial Profile**: Comparable revenue/margin/utilization
- **Risk Correlated**: Both have delinquency flags or similar risk patterns
- **Growth Pattern Match**: Similar revenue trends over time

**Key Parameters**: `revenue_ratio`, `margin_similarity`, `delinq_match`, `txn_trend_correlation`

#### 5. **Geographic Relationships**
- **Regional Partners**: Same region + industry match + transaction patterns
- **Local Ecosystem**: Geographic clustering + shared counterparties
- **Market Participants**: Same geographic market + similar business models

**Key Parameters**: `geo_km`, `region_match`, `ind_match`, `jacc_ctp`

#### 6. **Behavioral Relationships**
- **Transaction Pattern Match**: Similar seasonality, frequency, category mix
- **Business Model Similarity**: Similar text descriptions + transaction patterns
- **Operational Alignment**: Similar utilization rates + transaction behaviors

**Key Parameters**: `seasonality_correlation`, `text_similarity`, `txn_frequency_similarity`, `utilization_similarity`

### Implementation Recommendations

To leverage these additional parameters:

1. **Immediate Addition (Low Effort, High Impact)**:
   - Add `direct_supplier_link` (binary flag if direct supplier relationship exists)
   - Add `customer_status_match` (both ETB or both NTB)
   - Add `revenue_ratio` (financial size similarity)

2. **Medium Priority (Moderate Effort)**:
   - Compute `supplier_jaccard` for supplier network overlap
   - Add `txn_category_overlap` for transaction pattern matching
   - Implement `seasonality_correlation` for temporal pattern matching

3. **Advanced Features (Higher Effort)**:
   - Graph-based metrics (`adamic_adar_score`, `ecosystem_distance`)
   - Time-series features (`relationship_age`, `txn_trend_correlation`)
   - Financial similarity composite scores

4. **Anchor-Specific Features**:
   - Identify anchor entities (high network centrality, multiple subsidiaries)
   - Compute `anchor_proximity` for non-anchor entities
   - Detect hierarchical relationships (parent-subsidiary-sibling patterns)

### Feature Engineering Code Example

To add new parameters to `build_pair_features()`:

```python
# Financial similarity
if 'financials' in available_data:
    rev_a = get_latest_revenue(a_eid)
    rev_b = get_latest_revenue(b_eid)
    revenue_ratio = min(rev_a, rev_b) / max(rev_a, rev_b) if max(rev_a, rev_b) > 0 else 0
    margin_sim = 1 - abs(margin_a - margin_b)

# Supplier network
suppliers_a = get_suppliers(a_eid)
suppliers_b = get_suppliers(b_eid)
supplier_jaccard = len(suppliers_a & suppliers_b) / len(suppliers_a | suppliers_b) if (suppliers_a or suppliers_b) else 0
direct_supplier_link = (a_eid in suppliers_b) or (b_eid in suppliers_a)

# Transaction patterns
txn_cats_a = set(get_transaction_categories(a_eid))
txn_cats_b = set(get_transaction_categories(b_eid))
txn_category_overlap = len(txn_cats_a & txn_cats_b) / len(txn_cats_a | txn_cats_b) if (txn_cats_a or txn_cats_b) else 0

# Add to feature vector
features.extend([revenue_ratio, margin_sim, supplier_jaccard, direct_supplier_link, txn_category_overlap])
```

### Summary

**Currently Used**: 7 parameters  
**Available but Unused**: 25+ parameters  
**Derivable**: 15+ additional parameters  

Total potential parameters: **47+ relationship parameters** can be identified across account anchors, spanning financial, network, geographic, temporal, and behavioral dimensions.

---

## Future Enhancements

### Model Improvements
- **Fine-tune Sentence-Transformer** on domain-specific text (e.g., financial documents)
- **Upgrade to FAISS HNSW** for sub-millisecond ANN search on large datasets
- **Try LightGBM** or CatBoost for re-ranking (may handle categorical features better)
- **Ensemble models** (XGBoost + Neural Network) for higher accuracy
- **Deep learning re-ranker** using pair embeddings as input

### Feature Engineering
- **Time-series features** from transaction history (trends, seasonality)
- **Network centrality** metrics (PageRank, betweenness)
- **Financial ratios** from `financials.csv` (revenue, margin, utilization)
- **Text similarity** on specific fields (industry descriptions, product categories)
- **Behavioral patterns** (transaction frequency, amount distributions)

### System Enhancements
- **Real-time updates** (incremental index updates as new entities arrive)
- **Multi-modal embeddings** (images, logos, document attachments)
- **Personalized ranking** (user-specific weights based on feedback)
- **Active learning** (suggest which pairs to label next)
- **Graph neural networks** (end-to-end learning on relationship graph)

---

## Conclusion

The **Proximity Finder** system demonstrates a production-ready **two-stage retrieval-reranking architecture**:

✅ **Fast**: ANN search handles large candidate sets efficiently  
✅ **Accurate**: XGBoost re-ranker captures complex relationship patterns  
✅ **Explainable**: Feature-based approach enables transparent decisions  
✅ **Flexible**: Modular design allows easy updates to embeddings or re-ranker  
✅ **Scalable**: Handles 10K-100K entities with sub-second latency  

By combining **unsupervised embeddings** (text, transactions, graph) with **supervised learning** (XGBoost on pairs), the system achieves both broad coverage and high precision in relationship discovery.

For questions or improvements, refer to the main `README.md` and source code in `src/`.


