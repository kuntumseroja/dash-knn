import os
import json
import yaml
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
from st_cytoscape import cytoscape
import joblib
import pydeck as pdk  # bundled with Streamlit


# -------------------------------
# Config / Paths
# -------------------------------
CFG = yaml.safe_load(open("config.yaml"))
PROC_DIR = CFG["paths"]["processed_dir"]
MODELS_DIR = CFG["paths"]["models_dir"]
RAW = CFG["paths"]


# -------------------------------
# Small UI helpers (cards/css)
# -------------------------------
CARD_CSS = """
<style>
.block-container {padding-top: 1.2rem; background: linear-gradient(135deg, #f0f7ff 0%, #e6f2ff 50%, #ddeeff 100%);}
.main {background: linear-gradient(135deg, #f0f7ff 0%, #e6f2ff 50%, #ddeeff 100%);}
.card { padding: 1rem 1.2rem; border-radius: 16px;
  background: linear-gradient(135deg, #ffffff 0%, #f5f9ff 100%);
  border: 1px solid rgba(100, 149, 237, 0.2);
  box-shadow: 0 4px 20px rgba(100, 149, 237, 0.1), 0 2px 8px rgba(100, 149, 237, 0.05);}
.kpi {font-size: 28px; font-weight: 700; margin-bottom: 0.15rem;
  background: linear-gradient(135deg, #4a90e2 0%, #357abd 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;}
.kpi-small {color: #6b8db8; font-size: 12px; font-weight: 500;}
.badge { display:inline-block; padding: .25rem .55rem; border-radius: 999px;
  font-size: 12px; font-weight: 600; margin-right:.35rem;
  background: linear-gradient(135deg, #e8f2ff 0%, #d6e8ff 100%);
  color: #2c5aa0; border: 1px solid rgba(100, 149, 237, 0.2);}
.badge-green { background: linear-gradient(135deg, #d4f0e8 0%, #c0e8d8 100%); color: #1a7f37; border: 1px solid rgba(26, 127, 55, 0.2); }
.badge-blue  { background: linear-gradient(135deg, #d6e8ff 0%, #c4ddff 100%); color: #0b63c7; border: 1px solid rgba(11, 99, 199, 0.2); }
.badge-amber { background: linear-gradient(135deg, #fff4d6 0%, #ffecc0 100%); color: #9a6b00; border: 1px solid rgba(154, 107, 0, 0.2); }
.legend-chip { display:inline-flex; align-items:center; gap:.45rem;
  padding:.35rem .6rem; border-radius:999px;
  background: linear-gradient(135deg, #ffffff 0%, #f5f9ff 100%);
  border: 1px solid rgba(100, 149, 237, 0.3);
  margin-right:.4rem;
  box-shadow: 0 2px 6px rgba(100, 149, 237, 0.1);}
.legend-dot { width:10px; height:10px; border-radius:999px; display:inline-block;}
.score-tier-high { background: linear-gradient(135deg, #4a90e2 0%, #357abd 100%); }
.score-tier-medium { background: linear-gradient(135deg, #7bb3e8 0%, #5a9dd4 100%); }
.score-tier-low { background: linear-gradient(135deg, #a8c8e8 0%, #8bb0d4 100%); }
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: #f0f7ff;
    padding: 8px;
    border-radius: 12px;
}
.stTabs [data-baseweb="tab"] {
    background: linear-gradient(135deg, #ffffff 0%, #f5f9ff 100%);
    border: 1px solid rgba(100, 149, 237, 0.2);
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-weight: 500;
    color: #2c5aa0;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #4a90e2 0%, #357abd 100%);
    color: white;
    border-color: #2c5aa0;
}
[data-testid="stSidebar"] {
    background: linear-gradient(135deg, #f0f7ff 0%, #e6f2ff 100%);
}
</style>
"""


def kpi_card(label, value, suffix=""):
    st.markdown(
        f"""
        <div class="card"><div class="kpi">{value}{suffix}</div>
        <div class="kpi-small">{label}</div></div>
        """,
        unsafe_allow_html=True,
    )


# -------------------------------
# Load artifacts (cached)
# -------------------------------
@st.cache_resource
def load_artifacts():
    Z = np.load(os.path.join(PROC_DIR, "Z.npy")).astype("float32")

    # sklearn NearestNeighbors (cosine) saved by src/models/ann_index.py
    index = joblib.load(os.path.join(MODELS_DIR, "ann_sklearn.joblib"))

    # XGBoost re-ranker saved by src/models/train_reranker.py
    reranker = xgb.Booster()
    reranker.load_model(os.path.join(MODELS_DIR, "reranker_xgb.json"))

    with open(os.path.join(PROC_DIR, "index_to_entity.json"), "r") as f:
        idx2eid = {int(k): int(v) for k, v in json.load(f).items()}
    eid2idx = {v: k for k, v in idx2eid.items()}

    entities = pd.read_csv(RAW["entities_csv"])
    transactions = pd.read_csv(RAW["transactions_csv"])
    directors = pd.read_csv(RAW["directors_csv"])

    if "is_customer" not in entities.columns:
        entities["is_customer"] = 0

    return Z, index, reranker, idx2eid, eid2idx, entities, transactions, directors


# -------------------------------
# Pair features (must match training)
# -------------------------------
def build_pair_features_for_query(
    a_eid, cand_eids, Z, entities, directors, transactions, idx2eid, eid2idx
):
    from math import radians, sin, cos, atan2, sqrt

    def haversine_km(lat1, lon1, lat2, lon2):
       R = 6371.0  # Earth radius in km
       phi1, phi2 = radians(lat1), radians(lat2)
       dphi = radians(lat2 - lat1)
       dl = radians(lon2 - lon1)
       a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dl / 2) ** 2
       c = 2 * atan2(sqrt(a), sqrt(1 - a))   # <-- correct order
       return R * c
    # def haversine_km(lat1, lon1, lat2, lon2):
    #     R = 6371.0
    #     phi1, phi2 = radians(lat1), radians(lat2)
    #     dphi = radians(lat2 - lat1)
    #     dl = radians(lon2 - lon1)
    #     a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dl / 2) ** 2
    #     return 2 * R * atan2(sqrt(1 - a), sqrt(a))


    ent = entities.set_index("entity_id")
    ctp = transactions.groupby("src_entity_id")["dst_entity_id"].apply(set).to_dict()
    dset = directors.groupby("entity_id")["person_name"].apply(set).to_dict()

    ia = eid2idx[a_eid]
    za = Z[ia]
    X, meta, reasons = [], [], []
    for b in cand_eids:
        ib = eid2idx[b]
        zb = Z[ib]
        cos_sim = float((za @ zb) / (np.linalg.norm(za) * np.linalg.norm(zb) + 1e-9))
        sa, sb = ctp.get(a_eid, set()), ctp.get(b, set())
        inter = len(sa & sb)
        union = len(sa | sb) if (sa or sb) else 1
        jacc = inter / union
        try:
            gkm = haversine_km(
                ent.loc[a_eid, "lat"],
                ent.loc[a_eid, "lon"],
                ent.loc[b, "lat"],
                ent.loc[b, "lon"],
            )
        except KeyError:
            gkm = 1e3
        ind_match = float(ent.loc[a_eid, "industry_code"] == ent.loc[b, "industry_code"])
        da, db = dset.get(a_eid, set()), dset.get(b, set())
        d_olap = len(da & db)
        diff_mean = float(np.mean(np.abs(za - zb)))
        had_mean = float(np.mean(za * zb))
        X.append([cos_sim, jacc, gkm, ind_match, d_olap, diff_mean, had_mean])
        meta.append((a_eid, b))
        reasons.append(
            {
                "shared_counterparties": int(inter),
                "jaccard": float(round(jacc, 3)),
                "geo_km": float(round(gkm, 1)),
                "industry_match": bool(ind_match),
                "director_overlap": int(d_olap),
                "similarity": float(round(cos_sim, 3)),
            }
        )
    return np.array(X, dtype="float32"), meta, reasons


# -------------------------------
# CTA rules
# -------------------------------
def compute_cta(row, is_customer):
    """
    CTA rules (priority):
      1) score >= 0.70  -> Customer: 'Upsell / Cross-sell' mod, Prospect: 'Prospect outreach'
      2) director_overlap > 0 or jaccard >= 0.30 -> 'Warm intro'
      3) geo_km <= 250 and similarity >= 0.50 -> 'Local lead'
      4) else -> 'Research'
    """
    score = float(row.get("score", 0))
    if score >= 0.70:
        return "Upsell / Cross-sell" if is_customer else "Prospect outreach"
    if int(row.get("director_overlap", 0)) > 0 or float(row.get("jaccard", 0)) >= 0.30:
        return "Warm intro"
    if float(row.get("geo_km", 1e9)) <= 250 and float(row.get("similarity", 0)) >= 0.50:
        return "Local lead"
    return "Research"


# -------------------------------
# Graph builder (emoji labels, classes)
# -------------------------------
def build_graph(focus_row, neighbors_df, entities, show_edge_labels=True):
    def node_meta(eid, name):
        is_cust = (
            int(entities.loc[entities["entity_id"] == eid, "is_customer"].fillna(0).values[0])
            == 1
        )
        clazz = "cust" if is_cust else "prospect"
        emoji = "💼" if is_cust else "🧭"
        return clazz, f"{emoji} {name}"

    # Focus node
    focus_class, _ = node_meta(int(focus_row["entity_id"]), focus_row["name"])
    nodes = [
        {
            "data": {"id": str(focus_row["entity_id"]), "label": f"🎯 {focus_row['name']}"},
            "classes": f"focus {focus_class}",
        }
    ]

    edges = []
    for _, row in neighbors_df.iterrows():
        score = float(np.clip(row["score"], 0.0, 1.0))
        
        # Score-based tiering: High (>=0.7), Medium (0.4-0.7), Low (<0.4)
        if score >= 0.7:
            tier = "high"
            base_color = "#2c5aa0"  # Deep blue
            opacity = 0.9
        elif score >= 0.4:
            tier = "medium"
            base_color = "#5a9dd4"  # Medium blue
            opacity = 0.7
        else:
            tier = "low"
            base_color = "#8bb0d4"  # Light blue
            opacity = 0.5
        
        # Override with relationship type colors if applicable
        if row["industry_match"]:
            color = "#2ecc71"  # Green for same industry
        elif row["director_overlap"] > 0:
            color = "#e67e22"  # Orange for shared directors
        else:
            color = base_color  # Use tier-based blue
        
        width = 2 + 8 * score
        edges.append(
            {
                "data": {
                    "source": str(int(row["focus_id"])),
                    "target": str(int(row["entity_id"])),
                    "score": float(round(row["score"], 3)),
                    "label": f'{row["score"]:.2f}',
                    "color": color,
                    "width": width,
                    "opacity": opacity,
                },
                "classes": f"tier-{tier}",
            }
        )
        clazz, lbl = node_meta(int(row["entity_id"]), row["name"])
        nodes.append({"data": {"id": str(int(row["entity_id"])), "label": lbl}, "classes": clazz})

    stylesheet = [
        {
            "selector": "node",
            "style": {
                "background-color": "#a8c8e8",
                "label": "data(label)",
                "font-size": 10,
                "border-color": "#7bb3e8",
                "border-width": 2,
            },
        },  # prospects - soft blue
        {
            "selector": ".cust",
            "style": {
                "background-color": "#5a9dd4",
                "border-color": "#357abd",
                "border-width": 2,
            },
        },
        {
            "selector": ".focus",
            "style": {
                "background-color": "#4a90e2",
                "border-width": 4,
                "border-color": "#2c5aa0",
                "font-weight": "bold",
                "width": 40,
                "height": 40,
            },
        },
        {
            "selector": "edge",
            "style": {
                "line-color": "data(color)",
                "width": "data(width)",
                "curve-style": "bezier",
                "opacity": "data(opacity)",
            },
        },
        {
            "selector": ".tier-high",
            "style": {
                "opacity": 0.9,
                "line-color": "data(color)",
            },
        },
        {
            "selector": ".tier-medium",
            "style": {
                "opacity": 0.7,
                "line-color": "data(color)",
            },
        },
        {
            "selector": ".tier-low",
            "style": {
                "opacity": 0.5,
                "line-color": "data(color)",
            },
        },
    ]
    if show_edge_labels:
        stylesheet.append(
            {"selector": "edge", "style": {"label": "data(label)", "font-size": 8, "color": "#bdc3c7"}}
        )
    return nodes, edges, stylesheet


# -------------------------------
# App
# -------------------------------
def main():
    st.set_page_config(page_title="RM Proximity Dashboard", layout="wide", page_icon="🔗")
    st.markdown(CARD_CSS, unsafe_allow_html=True)
    
    # Add gradient header background
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #4a90e2 0%, #357abd 50%, #2c5aa0 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(74, 144, 226, 0.2);
    }
    .main-header h2 {
        color: white;
        margin: 0;
        font-weight: 600;
    }
    .main-header p {
        color: rgba(255, 255, 255, 0.9);
        margin: 0.5rem 0 0 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    top = st.columns([1, 6, 2], vertical_alignment="center")
    with top[0]:
        if st.button("🔄 Reload models/data", use_container_width=True):
            st.cache_resource.clear()
            st.cache_data.clear()
            st.rerun()
    with top[1]:
        st.markdown("""
        <div class="main-header">
            <h2>🔗 Proximity Finder — Relationship Manager View</h2>
            <p>Discover relevant corporate/SME neighbors & prospects with explainability.</p>
        </div>
        """, unsafe_allow_html=True)
    with top[2]:
        show_edge_labels = st.toggle("Edge labels", value=True)

    # Load
    Z, index, reranker, idx2eid, eid2idx, entities, transactions, directors = load_artifacts()

    # Search & scenario bar
    left, right = st.columns([3, 7], vertical_alignment="center")
    with left:
        names = entities[["entity_id", "name"]].drop_duplicates().sort_values("name")
        chosen = st.selectbox("Choose an entity", names["name"].tolist())
        focus = entities.loc[entities["name"] == chosen].iloc[0]
        a_eid = int(focus["entity_id"])

        # Entity chips
        is_cust = bool(int(focus.get("is_customer", 0)))
        industry = str(focus.get("industry_code", "?"))
        loc = f"{focus.get('lat','?')}, {focus.get('lon','?')}"
        st.markdown(
            f"""
            <div style="margin-top:.25rem;">
              <span class="badge {'badge-green' if is_cust else 'badge-blue'}">{'Customer' if is_cust else 'Prospect'}</span>
              <span class="badge badge-amber">Industry: {industry}</span>
              <span class="badge">Loc: {loc}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        K0_default = max(100, min(400, max(1, Z.shape[0] - 1)))
        K_default = min(20, max(5, Z.shape[0] // 10))
        cols = st.columns(4)
        with cols[0]:
            K0 = st.slider(
                "Candidates (K0)",
                10,
                max(10, max(1, Z.shape[0] - 1)),
                K0_default,
                10,
                help="Nearest neighbors taken from ANN before re-ranking.",
            )
        with cols[1]:
            K = st.slider(
                "Top-K",
                1,
                100,
                K_default,
                1,
                help="Final neighbors shown after XGBoost re-ranking.",
            )
        with cols[2]:
            min_score = st.slider(
                "Min score", 0.0, 1.0, 0.0, 0.01, help="Hide links with predicted probability below this value."
            )
        with cols[3]:
            max_geo = st.slider(
                "Max geo (km)", 0, 5000, 5000, 50, help="Filter by distance between coordinates."
            )
        sc = st.columns(4)
        with sc[0]:
            customers_only = st.checkbox("ETB only", value=False)
        with sc[1]:
            only_industry = st.checkbox("Same industry", value=False)
        with sc[2]:
            only_director = st.checkbox("Shared directors", value=False)
        with sc[3]:
            nearby_only = st.checkbox("Nearby only (< 250km)", value=False)
            if nearby_only:
                max_geo = min(max_geo, 250)

    # ANN shortlist
    ia = eid2idx[a_eid]
    k0 = min(int(K0), max(1, Z.shape[0] - 1))
    distances, indices = index.kneighbors(Z[ia : ia + 1], n_neighbors=k0, return_distance=True)
    cand_idx = indices[0].tolist()
    cand_eids = [idx2eid[i] for i in cand_idx if idx2eid[i] != a_eid]

    # Features & XGB scores
    X_pairs, meta, reasons = build_pair_features_for_query(
        a_eid, cand_eids, Z, entities, directors, transactions, idx2eid, eid2idx
    )
    dm = xgb.DMatrix(
        X_pairs,
        feature_names=["cos_sim", "jacc_ctp", "geo_km", "ind_match", "dir_overlap", "diff_mean", "had_mean"],
    )
    scores = reranker.predict(dm)

    rows = []
    for i, ((src, tgt), r) in enumerate(zip(meta, reasons)):
        name = entities.loc[entities["entity_id"] == tgt, "name"].values[0]
        rows.append({"focus_id": src, "entity_id": tgt, "name": name, "score": float(scores[i]), **r})
    df = pd.DataFrame(rows)

    # Merge in is_customer
    df = df.merge(entities[["entity_id", "is_customer"]], on="entity_id", how="left")

    # ---------- Normalize score, then apply filters ----------
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0.0)

    if customers_only:
        df = df[df["is_customer"] == 1]
    if only_industry:
        df = df[df["industry_match"] == True]
    if only_director:
        df = df[df["director_overlap"] > 0]
    df = df[df["score"] >= float(min_score)]
    df = df[df["geo_km"] <= float(max_geo)]

    # ---------- Fallback: if empty after filters, show ANN-only top-K ----------
    if df.empty:
        # Cosine similarity from same shortlist
        za = Z[ia]
        sims = []
        for eid in cand_eids:
            ib = eid2idx[eid]
            zb = Z[ib]
            denom = np.linalg.norm(za) * np.linalg.norm(zb) + 1e-9
            cos_sim = float((za @ zb) / denom) if denom > 0 else 0.0
            sims.append((eid, cos_sim))
        sims = sorted(sims, key=lambda t: t[1], reverse=True)[: int(K)]
        fallback_ids = {eid for eid, _ in sims}

        # Rebuild df from rows for the chosen fallback IDs
        df = pd.DataFrame([r for r in rows if r["entity_id"] in fallback_ids]).copy()

        if not df.empty:
            # (1) make score the cosine similarity so the UI can scale edges
            cs_map = {eid: cs for eid, cs in sims}
            df["score"] = df["entity_id"].map(cs_map).fillna(0.0)

            # (2) re-attach is_customer (this fixes the KeyError in the Table)
            df = df.merge(entities[["entity_id", "is_customer"]], on="entity_id", how="left")

            st.info("Showing ANN-only fallback (no XGBoost matches after current filters).")

    # ---------- Final Top-K + CTA (empty-safe) ----------
    df = df.sort_values("score", ascending=False).head(int(K)).reset_index(drop=True)
    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    if df.empty:
        df["CTA"] = pd.Series(dtype="object")
    else:
        tmp = df.apply(lambda r: compute_cta(r, bool(int(r.get("is_customer", 0)))), axis=1)
        if isinstance(tmp, pd.DataFrame):
            tmp = tmp.iloc[:, 0]
        cta_series = pd.Series(tmp, index=df.index, name="CTA").astype(str)
        df = df.drop(columns=[c for c in df.columns if c == "CTA"], errors="ignore")
        df["CTA"] = cta_series

    # KPI cards
    kpi = st.columns(4)
    with kpi[0]:
        kpi_card("Top-K shown", len(df))
    with kpi[1]:
        kpi_card("Median score", f"{df['score'].median():.2f}" if len(df) else "—")
    with kpi[2]:
        pct_ind = (df["industry_match"].mean() * 100) if len(df) else 0
        kpi_card("Same industry", f"{pct_ind:.0f}", suffix="%")
    with kpi[3]:
        avg_geo = df["geo_km"].mean() if len(df) else 0
        kpi_card("Avg geo distance", f"{avg_geo:.0f}", suffix=" km")

    # Tabs
    t1, t2, t3, t4 = st.tabs(["🌐 Network", "📋 Table", "🔎 Explain", "🗺️ Map"])

    with t1:
        nodes, edges, stylesheet = build_graph(focus, df, entities, show_edge_labels)
        cytoscape(
            elements={"nodes": nodes, "edges": edges},
            layout={"name": "cose"},
            stylesheet=stylesheet,
            height="640px",
        )
        st.markdown(
            """
            <div style="margin-top:.6rem;">
              <span class="legend-chip"><span class="legend-dot" style="background:#4a90e2;"></span>🎯 Focus company</span>
              <span class="legend-chip"><span class="legend-dot" style="background:#5a9dd4;"></span>💼 ETB node</span>
              <span class="legend-chip"><span class="legend-dot" style="background:#a8c8e8;"></span>🧭 NTB node</span>
              <span class="legend-chip"><span class="legend-dot" style="background:#2ecc71;"></span>Edge: same industry</span>
              <span class="legend-chip"><span class="legend-dot" style="background:#e67e22;"></span>Edge: shared directors</span>
              <span class="legend-chip"><span class="legend-dot" style="background:#2c5aa0;"></span>Edge: High score (≥0.7)</span>
              <span class="legend-chip"><span class="legend-dot" style="background:#5a9dd4;"></span>Edge: Medium score (0.4-0.7)</span>
              <span class="legend-chip"><span class="legend-dot" style="background:#8bb0d4;"></span>Edge: Low score (&lt;0.4)</span>
              <span class="legend-chip">Edge width & opacity = model score</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with t2:
        st.subheader("Top candidates (with CTA)")

        # --- Column guard: ensure table cols exist (prevents KeyErrors) ---
        required_defaults = {
            "is_customer": 0,
            "CTA": "",
            "similarity": 0.0,
            "jaccard": 0.0,
            "shared_counterparties": 0,
            "director_overlap": 0,
            "industry_match": False,
            "geo_km": 0.0,
        }
        for col, default in required_defaults.items():
            if col not in df.columns:
                df[col] = default

        view_cols = [
            "entity_id",
            "name",
            "is_customer",
            "CTA",
            "score",
            "similarity",
            "jaccard",
            "shared_counterparties",
            "director_overlap",
            "industry_match",
            "geo_km",
        ]
        # keep only those that exist + order
        view_cols = [c for c in view_cols if c in df.columns]
        view_df = df[["focus_id"] + view_cols] if "focus_id" in df.columns else df[view_cols]

        st.dataframe(
            view_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "focus_id": st.column_config.NumberColumn("focus_id", help="Selected (center) entity ID."),
                "entity_id": st.column_config.NumberColumn("entity_id", help="Neighbor entity ID."),
                "name": st.column_config.TextColumn("name", help="Neighbor company name."),
                "is_customer": st.column_config.CheckboxColumn("is_customer", help="1 if neighbor is an existing customer."),
                "CTA": st.column_config.TextColumn("CTA", help="Next best action based on score, overlap, and proximity."),
                "score": st.column_config.NumberColumn("score", format="%.3f", help="Re-ranker probability (0–1)."),
                "similarity": st.column_config.NumberColumn(
                    "similarity", format="%.3f", help="Cosine similarity of embeddings (text/semantic likeness)."
                ),
                "jaccard": st.column_config.NumberColumn("jaccard", format="%.3f", help="|A∩B| / |A∪B| based on counterparties."),
                "shared_counterparties": st.column_config.NumberColumn(
                    "shared_counterparties", help="How many counterparties both companies transact with."
                ),
                "director_overlap": st.column_config.NumberColumn("director_overlap", help="Shared director names count."),
                "industry_match": st.column_config.CheckboxColumn("industry_match", help="Industry_code matches."),
                "geo_km": st.column_config.NumberColumn("geo_km", format="%.1f", help="Haversine distance (km)."),
            },
        )
        csv = view_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download results (CSV)", csv, "proximity_results.csv", mime="text/csv")

    with t3:
        st.subheader("Explain a candidate")
        if len(df) == 0:
            st.info("No candidates after filters. Loosen filters or increase K0/Top-K.")
        else:
            label_map = {int(r.entity_id): f"{r.name} (score {r.score:.2f})" for r in df.itertuples()}
            pick_id = st.selectbox("Choose a candidate", list(label_map.keys()), format_func=lambda x: label_map[x])
            row = df.loc[df["entity_id"] == pick_id].iloc[0]
            is_cust_row = bool(int(row.get("is_customer", 0)))
            cta = compute_cta(row, is_cust_row)

            st.markdown("### Why was this selected?")
            bullets = []
            if row["industry_match"]:
                bullets.append("Same industry.")
            if row["director_overlap"] > 0:
                bullets.append(f"Shared directors: **{int(row['director_overlap'])}**.")
            if row["shared_counterparties"] > 0:
                bullets.append(
                    f"Shared counterparties: **{int(row['shared_counterparties'])}** (Jaccard **{row['jaccard']:.2f}**)."
                )
            if row["similarity"] >= 0.5:
                bullets.append(f"Strong text/semantic similarity (**{row['similarity']:.2f}**).")
            if row["geo_km"] <= 250:
                bullets.append(f"Geographically close (**{row['geo_km']:.0f} km**).")
            if not bullets:
                bullets.append("Model score is a blend of weaker signals (text/behavior/geo).")
            st.write("- " + "\n- ".join(bullets))

            st.markdown("#### Recommended action (CTA)")
            st.success(cta)

            st.markdown("#### Feature snapshot")
            st.write(
                {
                    "Customer?": is_cust_row,
                    "score": float(row["score"]),
                    "similarity": float(row["similarity"]),
                    "jaccard": float(row["jaccard"]),
                    "shared_counterparties": int(row["shared_counterparties"]),
                    "director_overlap": int(row["director_overlap"]),
                    "industry_match": bool(row["industry_match"]),
                    "geo_km": float(row["geo_km"]),
                }
            )

    with t4:
        st.subheader("Mini map")
        map_rows = [
            {
                "name": f"🎯 {focus['name']}",
                "lat": float(focus.get("lat", np.nan)),
                "lon": float(focus.get("lon", np.nan)),
                "type": "Focus",
                "color": [74, 144, 226],  # Soft blue #4a90e2
                "size": 14,
            }
        ]
        for _, r in df.iterrows():
            ent = entities.loc[entities["entity_id"] == int(r["entity_id"])].iloc[0]
            is_cust = int(ent.get("is_customer", 0)) == 1
            score = float(np.clip(r["score"], 0, 1))
            # Use soft blue tones: darker for customers, lighter for prospects
            if is_cust:
                color = [90, 157, 212]  # Medium blue #5a9dd4 for customers
            else:
                color = [168, 200, 232]  # Light blue #a8c8e8 for prospects
            emoji = "💼" if is_cust else "🧭"
            map_rows.append(
                {
                    "name": f"{emoji} {r['name']}",
                    "lat": float(ent.get("lat", np.nan)),
                    "lon": float(ent.get("lon", np.nan)),
                    "type": "Customer" if is_cust else "Prospect",
                    "color": color,
                    "size": int(8 + 12 * score),
                }
            )
        mdf = pd.DataFrame(map_rows).dropna(subset=["lat", "lon"])
        if len(mdf) == 0:
            st.info("No coordinates available to plot. Ensure lat/lon are present in entities.csv.")
        else:
            center = [mdf["lat"].mean(), mdf["lon"].mean()]
            layer = pdk.Layer(
                "ScatterplotLayer",
                data=mdf,
                get_position="[lon, lat]",
                get_fill_color="color",
                get_radius="size*100",
                pickable=True,
            )
            tooltip = {"text": "{name}\n{type}"}
            view = pdk.ViewState(latitude=center[0], longitude=center[1], zoom=5)
            st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, tooltip=tooltip))

    st.caption("Tip: K0 controls breadth (ANN candidates). Top-K controls depth (final re-ranked list).")


if __name__ == "__main__":
    main()