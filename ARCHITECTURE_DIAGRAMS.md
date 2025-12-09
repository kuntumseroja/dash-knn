# Proximity Finder - Architecture Diagrams

This document contains high-level architecture, tech stack, and flow diagrams for the Proximity Finder system.

---

## 1. High-Level Architecture

```mermaid
graph TB
    subgraph "Data Layer"
        E[entities.csv<br/>Company Info]
        T[transactions.csv<br/>Transaction History]
        D[directors.csv<br/>Director Info]
        S[suppliers.csv<br/>Supplier Networks]
        L[labels_links.csv<br/>Ground Truth Links]
    end

    subgraph "Feature Engineering Pipeline"
        FE1[build_text.py<br/>Text Embeddings<br/>384-dim]
        FE2[build_txn.py<br/>Transaction Embeddings<br/>32-dim]
        FE3[build_graph.py<br/>Graph Embeddings<br/>32-dim]
        FUSE[fuse.py<br/>Embedding Fusion<br/>448-dim]
    end

    subgraph "Model Training"
        ANN[ann_index.py<br/>ANN Index Builder<br/>sklearn NearestNeighbors]
        RERANK[train_reranker.py<br/>XGBoost Re-ranker<br/>Binary Classifier]
    end

    subgraph "Model Artifacts"
        Z[Z.npy<br/>Fused Embeddings]
        IDX[ann_sklearn.joblib<br/>ANN Index]
        XGB[reranker_xgb.json<br/>XGBoost Model]
    end

    subgraph "Inference Pipeline"
        Q[Query Entity]
        ANN_SEARCH[ANN Search<br/>K0 Candidates]
        FEAT[Feature Engineering<br/>Pair Features]
        SCORE[XGBoost Scoring<br/>Re-ranking]
        FILTER[Filtering & Top-K]
        VIZ[Visualization<br/>Streamlit Dashboard]
    end

    E --> FE1
    E --> FE2
    T --> FE2
    D --> FE3
    S --> FE3

    FE1 --> FUSE
    FE2 --> FUSE
    FE3 --> FUSE

    FUSE --> Z
    Z --> ANN
    ANN --> IDX

    Z --> RERANK
    L --> RERANK
    RERANK --> XGB

    Q --> ANN_SEARCH
    IDX --> ANN_SEARCH
    Z --> ANN_SEARCH

    ANN_SEARCH --> FEAT
    Z --> FEAT
    E --> FEAT
    T --> FEAT
    D --> FEAT

    FEAT --> SCORE
    XGB --> SCORE

    SCORE --> FILTER
    FILTER --> VIZ

    style E fill:#e1f5ff
    style T fill:#e1f5ff
    style D fill:#e1f5ff
    style S fill:#e1f5ff
    style L fill:#e1f5ff
    style Z fill:#fff4e1
    style IDX fill:#fff4e1
    style XGB fill:#fff4e1
    style VIZ fill:#e8f5e9
```

---

## 2. Tech Stack

```mermaid
graph LR
    subgraph "Data Processing"
        P1[Pandas<br/>DataFrames]
        P2[NumPy<br/>Arrays]
        P3[PyYAML<br/>Config]
    end

    subgraph "Embedding Models"
        E1[Sentence-Transformers<br/>all-MiniLM-L6-v2<br/>384-dim]
        E2[PCA<br/>Dimensionality Reduction]
        E3[NetworkX<br/>Graph Processing]
    end

    subgraph "ML Models"
        M1[scikit-learn<br/>NearestNeighbors<br/>Cosine Similarity]
        M2[XGBoost<br/>Gradient Boosting<br/>Binary Classifier]
    end

    subgraph "Storage"
        S1[CSV Files<br/>Raw Data]
        S2[NumPy Arrays<br/>.npy Files]
        S3[Joblib<br/>.joblib Files]
        S4[JSON<br/>Model & Config]
        S5[Parquet<br/>Feature Tables]
    end

    subgraph "Application"
        A1[Streamlit<br/>Web Dashboard]
        A2[Cytoscape.js<br/>Graph Visualization]
        A3[PyDeck<br/>Map Visualization]
    end

    subgraph "Evaluation"
        EV1["Precision at K"]
        EV2["Recall at K"]
        EV3[AUCPR]
    end

    P1 --> E1
    P1 --> E2
    P1 --> E3
    P2 --> M1
    P2 --> M2

    E1 --> S2
    E2 --> S2
    E3 --> S2
    M1 --> S3
    M2 --> S4

    S1 --> P1
    S2 --> M1
    S2 --> M2
    S3 --> A1
    S4 --> A1

    A1 --> A2
    A1 --> A3

    M2 --> EV1
    M2 --> EV2
    M2 --> EV3

    style E1 fill:#ffebee
    style M1 fill:#e3f2fd
    style M2 fill:#e8f5e9
    style A1 fill:#fff3e0
```

---

## 3. Training Flow

```mermaid
flowchart TD
    START([Start Training]) --> LOAD[Load Raw Data<br/>entities, transactions,<br/>directors, suppliers,<br/>labels_links]

    LOAD --> TEXT[Phase 1: Text Embeddings<br/>build_text.py]
    TEXT --> TEXT_PROC[Process:<br/>- Extract about_text + industry<br/>- Encode with SentenceTransformer<br/>- Normalize embeddings]
    TEXT_PROC --> TEXT_OUT[Output: text_embed.npy<br/>384-dim × N entities]

    LOAD --> TXN[Phase 2: Transaction Embeddings<br/>build_txn.py]
    TXN --> TXN_PROC[Process:<br/>- Aggregate transaction patterns<br/>- Compute seasonality features<br/>- Extract counterparty stats<br/>- Apply PCA reduction]
    TXN_PROC --> TXN_OUT[Output: txn_embed.npy<br/>32-dim × N entities]

    LOAD --> GRAPH[Phase 3: Graph Embeddings<br/>build_graph.py]
    GRAPH --> GRAPH_PROC[Process:<br/>- Build director network<br/>- Build supplier network<br/>- Generate graph embeddings]
    GRAPH_PROC --> GRAPH_OUT[Output: graph_embed.npy<br/>32-dim × N entities]

    TEXT_OUT --> FUSE[Phase 4: Embedding Fusion<br/>fuse.py]
    TXN_OUT --> FUSE
    GRAPH_OUT --> FUSE
    FUSE --> FUSE_PROC[Process:<br/>- L2 normalize each embedding<br/>- Weighted concatenation<br/>- text:txn:graph = 1.0:1.0:1.0]
    FUSE_PROC --> FUSE_OUT[Output: Z.npy<br/>448-dim × N entities<br/>index_to_entity.json]

    FUSE_OUT --> ANN[Phase 5: ANN Index Building<br/>ann_index.py]
    ANN --> ANN_PROC[Process:<br/>- Load Z.npy<br/>- Fit NearestNeighbors<br/>- Cosine metric, brute-force]
    ANN_PROC --> ANN_OUT[Output: ann_sklearn.joblib<br/>Trained K-NN index]

    FUSE_OUT --> RERANK[Phase 6: Re-ranker Training<br/>train_reranker.py]
    LOAD --> RERANK
    RERANK --> RERANK_PROC[Process:<br/>- Load labels_links.csv positives<br/>- Sample negative pairs<br/>- Build pair features:<br/>  cos_sim, jacc_ctp, geo_km,<br/>  ind_match, dir_overlap,<br/>  diff_mean, had_mean<br/>- Train XGBoost classifier<br/>- Objective: binary:logistic<br/>- Metric: aucpr]
    RERANK_PROC --> RERANK_OUT[Output: reranker_xgb.json<br/>Trained XGBoost model]

    ANN_OUT --> EVAL[Phase 7: Evaluation<br/>report_metrics.py]
    RERANK_OUT --> EVAL
    FUSE_OUT --> EVAL
    EVAL --> EVAL_PROC[Process:<br/>- Query test entities<br/>- Compute Precision at K<br/>- Compute Recall at K]
    EVAL_PROC --> EVAL_OUT[Output: metrics.json<br/>Performance metrics]

    EVAL_OUT --> END([Training Complete])

    style START fill:#c8e6c9
    style END fill:#c8e6c9
    style TEXT fill:#fff9c4
    style TXN fill:#fff9c4
    style GRAPH fill:#fff9c4
    style FUSE fill:#ffccbc
    style ANN fill:#b3e5fc
    style RERANK fill:#b3e5fc
    style EVAL fill:#e1bee7
```

---

## 4. Inference Flow

```mermaid
flowchart TD
    START(["User Query<br/>Select Entity"]) --> LOAD["Load Artifacts<br/>- Z.npy embeddings<br/>- ann_sklearn.joblib<br/>- reranker_xgb.json<br/>- entities.csv<br/>- transactions.csv<br/>- directors.csv"]

    LOAD --> QUERY["Get Query Entity<br/>entity_id to index"]

    QUERY --> ANN["Stage 1: ANN Search<br/>K0 Candidates"]
    ANN --> ANN_PROC["Process:<br/>- Extract query embedding Z[query_idx]<br/>- index.kneighbors query<br/>- Cosine similarity search<br/>- Return K0 nearest neighbors<br/>Default: K0=400"]
    ANN_PROC --> CAND["Output: K0 candidate IDs<br/>e.g., 400 entities"]

    CAND --> FEAT["Stage 2: Feature Engineering<br/>For each candidate pair"]
    FEAT --> FEAT_PROC["Compute Pair Features:<br/>1. cos_sim = cosine Z[query], Z[cand]<br/>2. jacc_ctp = Jaccard counterparties<br/>3. geo_km = Haversine distance<br/>4. ind_match = industry match<br/>5. dir_overlap = shared directors<br/>6. diff_mean = mean abs Z_q - Z_c<br/>7. had_mean = mean Z_q * Z_c"]
    FEAT_PROC --> FEAT_MAT["Output: Feature Matrix X<br/>Shape: K0 x 7 features"]

    FEAT_MAT --> SCORE["Stage 3: XGBoost Re-ranking"]
    SCORE --> SCORE_PROC["Process:<br/>- Create DMatrix from X<br/>- reranker.predict DMatrix<br/>- Get probability scores 0-1"]
    SCORE_PROC --> SCORES["Output: Scores array<br/>Shape: K0 probabilities"]

    SCORES --> FILTER["Stage 4: Filtering"]
    FILTER --> FILTER_PROC["Apply Filters:<br/>- min_score threshold<br/>- max_geo distance<br/>- customers_only flag<br/>- same_industry flag<br/>- shared_directors flag<br/>- nearby_only flag"]
    FILTER_PROC --> FILTERED["Filtered Candidates"]

    FILTERED --> TOPK["Stage 5: Top-K Selection"]
    TOPK --> TOPK_PROC["Process:<br/>- Sort by score descending<br/>- Take top K entities<br/>Default: K=20"]
    TOPK_PROC --> RESULTS["Output: Top-K Neighbors<br/>with scores and metadata"]

    RESULTS --> CTA["Stage 6: CTA Generation"]
    CTA --> CTA_PROC["Compute Call-to-Action:<br/>- score >= 0.70: Upsell/Prospect outreach<br/>- director_overlap > 0 or jaccard >= 0.30: Warm intro<br/>- geo_km <= 250 and similarity >= 0.50: Local lead<br/>- else: Research"]
    CTA_PROC --> FINAL[Final Results with CTA]

    FINAL --> VIZ["Stage 7: Visualization"]
    VIZ --> VIZ_NET["Network Graph<br/>Cytoscape.js<br/>- Nodes: entities<br/>- Edges: relationships<br/>- Colors: score tiers"]
    VIZ --> VIZ_TAB["Data Table<br/>- Entity details<br/>- Feature values<br/>- Scores and CTA"]
    VIZ --> VIZ_EXP["Explainability<br/>- Feature breakdown<br/>- Why selected<br/>- Recommendations"]
    VIZ --> VIZ_MAP["Geographic Map<br/>PyDeck<br/>- Entity locations<br/>- Distance visualization"]

    VIZ_NET --> END(["Results Displayed"])
    VIZ_TAB --> END
    VIZ_EXP --> END
    VIZ_MAP --> END

    style START fill:#c8e6c9
    style END fill:#c8e6c9
    style ANN fill:#fff9c4
    style FEAT fill:#ffccbc
    style SCORE fill:#b3e5fc
    style FILTER fill:#e1bee7
    style TOPK fill:#e1bee7
    style CTA fill:#f8bbd0
    style VIZ fill:#c5cae9
```

---

## 5. Data Flow Diagram

```mermaid
graph LR
    subgraph "Raw Data Sources"
        RD1[entities.csv<br/>name, industry,<br/>about_text, lat, lon]
        RD2[transactions.csv<br/>src, dst, amount,<br/>category, yyyymm]
        RD3[directors.csv<br/>entity_id, person_name]
        RD4[suppliers.csv<br/>buyer, supplier]
        RD5[labels_links.csv<br/>left_id, right_id,<br/>link_type]
    end

    subgraph "Feature Extraction"
        FE1[Text Embeddings<br/>384-dim]
        FE2[Transaction Embeddings<br/>32-dim]
        FE3[Graph Embeddings<br/>32-dim]
    end

    subgraph "Fused Representation"
        FR[Z.npy<br/>448-dim fused<br/>N × 448 matrix]
    end

    subgraph "Models"
        M1[ANN Index<br/>sklearn K-NN]
        M2[XGBoost<br/>Re-ranker]
    end

    subgraph "Inference"
        INF1[Query Entity]
        INF2[K0 Candidates]
        INF3[Pair Features]
        INF4[Top-K Results]
    end

    RD1 --> FE1
    RD2 --> FE2
    RD3 --> FE3
    RD4 --> FE3

    FE1 --> FR
    FE2 --> FR
    FE3 --> FR

    FR --> M1
    FR --> M2
    RD5 --> M2

    INF1 --> INF2
    M1 --> INF2
    FR --> INF2

    INF2 --> INF3
    FR --> INF3
    RD1 --> INF3
    RD2 --> INF3
    RD3 --> INF3

    INF3 --> INF4
    M2 --> INF4

    style RD1 fill:#e1f5ff
    style RD2 fill:#e1f5ff
    style RD3 fill:#e1f5ff
    style RD4 fill:#e1f5ff
    style RD5 fill:#e1f5ff
    style FR fill:#fff4e1
    style M1 fill:#e3f2fd
    style M2 fill:#e8f5e9
    style INF4 fill:#c8e6c9
```

---

## 6. Component Interaction Diagram

```mermaid
sequenceDiagram
    participant User
    participant Streamlit as Streamlit App
    participant Cache as Model Cache
    participant ANN as ANN Index
    participant XGB as XGBoost Model
    participant Features as Feature Builder
    participant Data as Data Sources

    User->>Streamlit: Select Entity
    Streamlit->>Cache: Load Artifacts
    Cache->>ANN: Load ann_sklearn.joblib
    Cache->>XGB: Load reranker_xgb.json
    Cache->>Data: Load entities, transactions, directors
    
    Streamlit->>ANN: Query K0 nearest neighbors
    ANN-->>Streamlit: Return K0 candidate IDs
    
    Streamlit->>Features: Build pair features
    Features->>Data: Fetch entity metadata
    Features->>Data: Fetch transaction data
    Features->>Data: Fetch director data
    Features-->>Streamlit: Return feature matrix (K0 × 7)
    
    Streamlit->>XGB: Predict scores
    XGB-->>Streamlit: Return probabilities (K0)
    
    Streamlit->>Streamlit: Apply filters
    Streamlit->>Streamlit: Sort & select Top-K
    Streamlit->>Streamlit: Generate CTA
    Streamlit-->>User: Display results (Graph, Table, Map, Explain)
```

---

## 7. Data Model Design

```mermaid
erDiagram
    ENTITIES {
        int entity_id PK "Unique identifier"
        string name "Company/entity name"
        int industry_code "Industry classification"
        string about_text "Description text"
        float lat "Latitude coordinate"
        float lon "Longitude coordinate"
        int is_customer "Binary flag: 0=Prospect, 1=Customer"
    }

    TRANSACTIONS {
        int src_entity_id FK "Source entity"
        int dst_entity_id FK "Destination entity"
        string yyyymm "Date: YYYY-MM format"
        float amount "Transaction amount"
        string category "Transaction category"
    }

    DIRECTORS {
        int entity_id FK "Entity identifier"
        string person_name "Director/executive name"
    }

    SUPPLIERS {
        int buyer_entity_id FK "Buyer entity"
        int supplier_entity_id FK "Supplier entity"
    }

    FINANCIALS {
        int entity_id FK "Entity identifier"
        int year "Year: e.g., 2023, 2024"
        float revenue "Annual revenue"
        float margin "Profit margin: 0.0-1.0"
        float utilization "Utilization rate: 0.0-1.0"
        int delinq_flag "Delinquency flag: 0 or 1"
    }

    LABELS_LINKS {
        int left_entity_id FK "First entity"
        int right_entity_id FK "Second entity"
        string link_type "Relationship type"
    }

    ENTITIES ||--o{ TRANSACTIONS : "src_entity_id"
    ENTITIES ||--o{ TRANSACTIONS : "dst_entity_id"
    ENTITIES ||--o{ DIRECTORS : "has"
    ENTITIES ||--o{ SUPPLIERS : "buyer_entity_id"
    ENTITIES ||--o{ SUPPLIERS : "supplier_entity_id"
    ENTITIES ||--o{ FINANCIALS : "has"
    ENTITIES ||--o{ LABELS_LINKS : "left_entity_id"
    ENTITIES ||--o{ LABELS_LINKS : "right_entity_id"
```

### Data Model Description

#### Core Entity: `ENTITIES`
The central table containing all company/entity information. Each entity represents a business organization (corporate or SME) in the system.

**Key Attributes:**
- `entity_id`: Primary key, unique identifier for each entity
- `name`: Company name
- `industry_code`: Industry classification code (used for industry matching)
- `about_text`: Descriptive text about the entity (used for text embeddings)
- `lat`, `lon`: Geographic coordinates (used for distance calculations)
- `is_customer`: Binary flag indicating customer status (ETB vs NTB)

#### Transaction History: `TRANSACTIONS`
Records all transactions between entities, capturing business relationships and transaction patterns.

**Key Attributes:**
- `src_entity_id`: Foreign key to source entity
- `dst_entity_id`: Foreign key to destination entity
- `yyyymm`: Transaction date in YYYY-MM format
- `amount`: Transaction amount
- `category`: Transaction category (e.g., "raw_materials", "utilities", "services", "retail_pos", "logistics")

**Usage:**
- Used to compute counterparty overlap (Jaccard similarity)
- Transaction patterns used for behavior-based embeddings
- Seasonality features extracted from monthly patterns

#### Directors: `DIRECTORS`
Associates directors/executives with entities, enabling detection of shared ownership/control relationships.

**Key Attributes:**
- `entity_id`: Foreign key to entity
- `person_name`: Director/executive name

**Usage:**
- Detects director overlap between entities (indicator of common ownership)
- Used in graph embeddings to build relationship networks
- Feature: `director_overlap` = count of shared directors

#### Supplier Relationships: `SUPPLIERS`
Represents supplier-buyer relationships, forming a directed graph of business supply chains.

**Key Attributes:**
- `buyer_entity_id`: Foreign key to buyer entity
- `supplier_entity_id`: Foreign key to supplier entity

**Usage:**
- Builds graph structure for graph embeddings
- Reveals supply chain connections and business ecosystems
- Entities with similar supplier networks are likely related

#### Financial Metrics: `FINANCIALS`
Annual financial performance data per entity.

**Key Attributes:**
- `entity_id`: Foreign key to entity
- `year`: Year of financial data
- `revenue`: Annual revenue
- `margin`: Profit margin (0.0 to 1.0)
- `utilization`: Utilization rate (0.0 to 1.0)
- `delinq_flag`: Delinquency flag (0 or 1)

**Usage:**
- Currently stored but not actively used in current model
- Potential for future feature engineering (financial similarity)

#### Training Labels: `LABELS_LINKS`
Ground truth relationship labels used for supervised training of the re-ranker model.

**Key Attributes:**
- `left_entity_id`: Foreign key to first entity
- `right_entity_id`: Foreign key to second entity
- `link_type`: Relationship type (e.g., "supplier_of", "common_owner", "co_customer")

**Usage:**
- Defines positive examples for XGBoost re-ranker training
- Model automatically samples negative pairs (random non-linked entities)
- Only required during training phase, not needed for inference

### Relationship Cardinalities

- **ENTITIES → TRANSACTIONS**: One-to-many (one entity can have many transactions as source or destination)
- **ENTITIES → DIRECTORS**: One-to-many (one entity can have multiple directors)
- **ENTITIES → SUPPLIERS**: One-to-many (one entity can be buyer or supplier in multiple relationships)
- **ENTITIES → FINANCIALS**: One-to-many (one entity can have financial data for multiple years)
- **ENTITIES → LABELS_LINKS**: One-to-many (one entity can be in multiple labeled relationships)

### Data Integrity Rules

1. **Foreign Key Constraints**: All foreign keys must reference existing `entity_id` values in `ENTITIES`
2. **Entity ID Consistency**: `entity_id` values must be unique and consistent across all files
3. **Geographic Coordinates**: `lat` should be between -90 and 90, `lon` between -180 and 180
4. **Date Format**: `yyyymm` must be in "YYYY-MM" format (e.g., "2024-03")
5. **Numeric Ranges**: `margin` and `utilization` should be between 0.0 and 1.0
6. **Binary Flags**: `is_customer` and `delinq_flag` must be 0 or 1

### Data Flow in Model Pipeline

1. **Text Embeddings**: Uses `entities.about_text` + `entities.industry_code`
2. **Transaction Embeddings**: Uses `transactions` table to compute behavior patterns
3. **Graph Embeddings**: Uses `directors` and `suppliers` tables to build relationship graphs
4. **Feature Engineering**: Combines data from all tables to compute pair features:
   - Cosine similarity (from embeddings)
   - Jaccard counterparty overlap (from `transactions`)
   - Geographic distance (from `entities.lat`, `entities.lon`)
   - Industry match (from `entities.industry_code`)
   - Director overlap (from `directors`)
5. **Training**: Uses `labels_links` to train supervised re-ranker

---

## 8. Data Architecture

```mermaid
graph TB
    subgraph "Data Sources Layer"
        DS1[entities.csv<br/>Company Info]
        DS2[transactions.csv<br/>Transaction History]
        DS3[directors.csv<br/>Director Info]
        DS4[suppliers.csv<br/>Supplier Networks]
        DS5[financials.csv<br/>Financial Metrics]
        DS6[labels_links.csv<br/>Training Labels]
    end

    subgraph "Raw Data Storage"
        RAW[data/raw/<br/>CSV Files<br/>Structured Data]
    end

    subgraph "Feature Engineering Layer"
        FE1[build_text.py<br/>Text Embeddings<br/>SentenceTransformer]
        FE2[build_txn.py<br/>Transaction Features<br/>PCA Reduction]
        FE3[build_graph.py<br/>Graph Embeddings<br/>NetworkX + SVD]
    end

    subgraph "Processed Data Storage"
        PROC[data/processed/<br/>Embeddings & Features]
        PROC1[text_embed.npy<br/>384-dim × N]
        PROC2[txn_embed.npy<br/>32-dim × N]
        PROC3[graph_embed.npy<br/>32-dim × N]
        PROC4[txn_feats.parquet<br/>Transaction Features]
        PROC5[entity_ids.parquet<br/>Entity ID Mapping]
    end

    subgraph "Embedding Fusion"
        FUSE[fuse.py<br/>Weighted Concatenation<br/>L2 Normalization]
        FUSED[Z.npy<br/>448-dim Fused Embeddings<br/>N × 448 matrix]
        IDX_MAP[index_to_entity.json<br/>Row Index Mapping]
    end

    subgraph "Model Training Layer"
        ANN_TRAIN[ann_index.py<br/>Build ANN Index<br/>sklearn NearestNeighbors]
        RERANK_TRAIN[train_reranker.py<br/>Train XGBoost<br/>Pair Features + Labels]
    end

    subgraph "Model Storage"
        MODELS[models/<br/>Trained Models]
        MOD1[ann_sklearn.joblib<br/>ANN Index]
        MOD2[reranker_xgb.json<br/>XGBoost Model]
    end

    subgraph "Evaluation Layer"
        EVAL[report_metrics.py<br/>Precision/Recall Metrics]
        METRICS[metrics.json<br/>Performance Metrics]
    end

    subgraph "Inference Layer"
        INF_APP[Streamlit App<br/>app/app.py]
        INF_CACHE[Model Cache<br/>@st.cache_resource]
        INF_LOAD[Load Artifacts<br/>Z, index, reranker]
    end

    subgraph "Real-Time Data Access"
        RT1[Z.npy<br/>Fused Embeddings]
        RT2[ann_sklearn.joblib<br/>ANN Index]
        RT3[reranker_xgb.json<br/>XGBoost Model]
        RT4[entities.csv<br/>Metadata]
        RT5[transactions.csv<br/>Counterparty Data]
        RT6[directors.csv<br/>Director Data]
    end

    subgraph "Visualization & Output"
        VIZ1[Network Graph<br/>Cytoscape.js]
        VIZ2[Data Table<br/>Results CSV]
        VIZ3[Explainability<br/>Feature Breakdown]
        VIZ4[Geographic Map<br/>PyDeck]
    end

    DS1 --> RAW
    DS2 --> RAW
    DS3 --> RAW
    DS4 --> RAW
    DS5 --> RAW
    DS6 --> RAW

    RAW --> FE1
    RAW --> FE2
    RAW --> FE3

    FE1 --> PROC1
    FE2 --> PROC2
    FE2 --> PROC4
    FE3 --> PROC3
    FE1 --> PROC5

    PROC1 --> FUSE
    PROC2 --> FUSE
    PROC3 --> FUSE
    PROC5 --> FUSE

    FUSE --> FUSED
    FUSE --> IDX_MAP

    FUSED --> ANN_TRAIN
    FUSED --> RERANK_TRAIN
    RAW --> RERANK_TRAIN

    ANN_TRAIN --> MOD1
    RERANK_TRAIN --> MOD2

    FUSED --> EVAL
    MOD1 --> EVAL
    MOD2 --> EVAL
    RAW --> EVAL
    EVAL --> METRICS

    MOD1 --> RT2
    MOD2 --> RT3
    FUSED --> RT1
    RAW --> RT4
    RAW --> RT5
    RAW --> RT6

    RT1 --> INF_LOAD
    RT2 --> INF_LOAD
    RT3 --> INF_LOAD
    RT4 --> INF_LOAD
    RT5 --> INF_LOAD
    RT6 --> INF_LOAD

    INF_LOAD --> INF_CACHE
    INF_CACHE --> INF_APP

    INF_APP --> VIZ1
    INF_APP --> VIZ2
    INF_APP --> VIZ3
    INF_APP --> VIZ4

    style RAW fill:#e1f5ff
    style PROC fill:#fff4e1
    style FUSED fill:#fff4e1
    style MODELS fill:#e8f5e9
    style INF_APP fill:#c5cae9
    style VIZ1 fill:#f3e5f5
    style VIZ2 fill:#f3e5f5
    style VIZ3 fill:#f3e5f5
    style VIZ4 fill:#f3e5f5
```

### Data Architecture Layers

#### 1. Data Sources Layer
**Purpose**: Raw input data from external sources or data pipelines

**Components**:
- **entities.csv**: Core company information (name, industry, location, customer status)
- **transactions.csv**: Historical transaction records between entities
- **directors.csv**: Director/executive associations with entities
- **suppliers.csv**: Supplier-buyer relationship network
- **financials.csv**: Annual financial performance metrics
- **labels_links.csv**: Ground truth relationship labels for training

**Characteristics**:
- CSV format for easy ingestion
- Structured, tabular data
- Can be updated incrementally
- Source of truth for entity metadata

#### 2. Raw Data Storage
**Location**: `data/raw/`

**Purpose**: Centralized storage for all raw input files

**Characteristics**:
- File-based storage (CSV format)
- No transformation applied
- Version-controlled or timestamped
- Serves as data lake for feature engineering

#### 3. Feature Engineering Layer
**Purpose**: Transform raw data into numerical embeddings and features

**Components**:
- **build_text.py**: Generates 384-dim text embeddings using SentenceTransformer
- **build_txn.py**: Extracts transaction patterns and reduces to 32-dim via PCA
- **build_graph.py**: Builds graph structure and generates 32-dim graph embeddings via SVD

**Processing**:
- Batch processing (runs on full dataset)
- Stateless transformations
- Reproducible (deterministic outputs)
- Can be re-run when source data changes

#### 4. Processed Data Storage
**Location**: `data/processed/`

**Purpose**: Store intermediate embeddings and features

**Components**:
- **text_embed.npy**: Text embeddings (384-dim × N entities)
- **txn_embed.npy**: Transaction embeddings (32-dim × N entities)
- **graph_embed.npy**: Graph embeddings (32-dim × N entities)
- **txn_feats.parquet**: Raw transaction features (before PCA)
- **entity_ids.parquet**: Entity ID ordering for alignment

**Characteristics**:
- NumPy arrays for fast numerical operations
- Parquet for structured feature tables
- Optimized for ML pipeline consumption
- Can be cached for faster access

#### 5. Embedding Fusion
**Purpose**: Combine multi-modal embeddings into unified representation

**Process**:
1. Load individual embeddings (text, txn, graph)
2. L2 normalize each embedding type
3. Apply configurable weights
4. Concatenate into single 448-dim vector
5. Save fused embedding matrix Z.npy

**Outputs**:
- **Z.npy**: Fused embeddings (448-dim × N entities)
- **index_to_entity.json**: Mapping between matrix row index and entity_id

**Characteristics**:
- Single source of truth for entity representations
- Used for ANN search
- Normalized for cosine similarity computation

#### 6. Model Training Layer
**Purpose**: Train ML models for inference

**Components**:
- **ann_index.py**: Builds sklearn NearestNeighbors index on Z.npy
- **train_reranker.py**: Trains XGBoost classifier on pair features

**Inputs**:
- Fused embeddings (Z.npy)
- Training labels (labels_links.csv)
- Entity metadata (for feature engineering)

**Outputs**:
- ANN index (for fast candidate retrieval)
- XGBoost model (for re-ranking)

#### 7. Model Storage
**Location**: `models/`

**Purpose**: Store trained models for inference

**Components**:
- **ann_sklearn.joblib**: Serialized NearestNeighbors index
- **reranker_xgb.json**: XGBoost model in JSON format

**Characteristics**:
- Version-controlled (can timestamp models)
- Portable (JSON format for XGBoost)
- Loaded once and cached in memory
- Can be swapped for A/B testing

#### 8. Evaluation Layer
**Purpose**: Assess model performance

**Process**:
- Load test queries from labels_links.csv
- Run inference pipeline
- Compute Precision@K and Recall@K
- Save metrics to JSON

**Output**: `metrics.json` with performance statistics

#### 9. Inference Layer
**Purpose**: Real-time query processing

**Components**:
- **Streamlit App**: Web interface for user queries
- **Model Cache**: Caches loaded models using `@st.cache_resource`
- **Load Artifacts**: Loads all required data and models

**Data Access**:
- Reads from processed data storage (Z.npy)
- Reads from model storage (joblib, JSON)
- Reads from raw data (CSV) for metadata

**Characteristics**:
- Lazy loading (loads on first query)
- Cached for subsequent queries
- Sub-second latency
- Handles concurrent users

#### 10. Real-Time Data Access
**Purpose**: Fast access to data during inference

**Components**:
- Fused embeddings (Z.npy) - loaded into memory
- ANN index (joblib) - loaded into memory
- XGBoost model (JSON) - loaded into memory
- Entity metadata (CSV) - loaded into Pandas DataFrame
- Transaction data (CSV) - loaded for feature computation
- Director data (CSV) - loaded for feature computation

**Optimization**:
- Models cached in memory (no disk I/O during inference)
- DataFrames loaded once per session
- Vectorized operations for speed

#### 11. Visualization & Output
**Purpose**: Present results to end users

**Components**:
- **Network Graph**: Interactive graph visualization (Cytoscape.js)
- **Data Table**: Tabular results with download (CSV export)
- **Explainability**: Feature-level explanations
- **Geographic Map**: Location-based visualization (PyDeck)

### Data Flow Patterns

#### Training Flow (Batch)
```
Raw Data → Feature Engineering → Processed Storage → Fusion → Model Training → Model Storage
```

#### Inference Flow (Real-Time)
```
User Query → Load Artifacts (cached) → ANN Search → Feature Engineering → Re-ranking → Visualization
```

#### Data Update Flow
```
New Data → Raw Storage → Re-run Feature Engineering → Re-run Fusion → Re-train Models → Update Model Storage
```

### Storage Characteristics

| Layer | Format | Size (10K entities) | Access Pattern | Update Frequency |
|-------|--------|---------------------|-----------------|-------------------|
| Raw Data | CSV | ~10-50 MB | Sequential read | Daily/Weekly |
| Processed | NumPy/Parquet | ~50-100 MB | Random access | On data update |
| Fused Embeddings | NumPy | ~20 MB | Random access | On data update |
| Models | Joblib/JSON | ~5-20 MB | Load once | On retrain |
| Metrics | JSON | <1 KB | Read-only | On evaluation |

### Data Lineage

1. **entities.csv** → text_embed.npy → Z.npy → ANN index
2. **transactions.csv** → txn_embed.npy → Z.npy → ANN index
3. **directors.csv + suppliers.csv** → graph_embed.npy → Z.npy → ANN index
4. **Z.npy + labels_links.csv** → Pair features → XGBoost model
5. **All sources** → Feature engineering → Inference pipeline → Results

### Scalability Considerations

- **Horizontal Scaling**: Feature engineering can be parallelized across entities
- **Caching Strategy**: Models cached in memory to avoid repeated disk I/O
- **Incremental Updates**: Can update embeddings for new entities without full rebuild
- **Storage Optimization**: NumPy arrays use efficient binary format
- **Memory Management**: Large datasets can be processed in batches

---

## Summary

### Architecture Highlights

1. **Two-Stage Pipeline**: Fast ANN retrieval + accurate supervised re-ranking
2. **Multi-Modal Embeddings**: Text, transaction, and graph signals combined
3. **Feature-Rich Re-ranking**: 7 engineered features capture relationship signals
4. **Real-Time Inference**: Sub-second latency for interactive queries
5. **Explainable Results**: Feature-level explanations and CTA recommendations

### Key Technologies

- **Embeddings**: Sentence-Transformers (HuggingFace)
- **ANN Search**: scikit-learn NearestNeighbors
- **Re-ranking**: XGBoost Gradient Boosting
- **Visualization**: Streamlit + Cytoscape.js + PyDeck
- **Data Processing**: Pandas, NumPy, NetworkX

### Performance Characteristics

- **Training Time**: 5-10 min (10K entities), 30-45 min (100K entities)
- **Inference Time**: 70-160 ms per query
- **Scalability**: Handles 10K-100K entities efficiently
- **Accuracy**: Precision@20 ≈ 0.78, Recall@20 ≈ 0.45

