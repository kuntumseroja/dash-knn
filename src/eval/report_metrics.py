import os, yaml, json, numpy as np, pandas as pd
import xgboost as xgb
import joblib  # sklearn ANN is saved with joblib

CFG = yaml.safe_load(open('config.yaml'))
PROC_DIR = CFG['paths']['processed_dir']
MODELS_DIR = CFG['paths']['models_dir']
RAW = CFG['paths']
K0 = CFG['ann']['K0']; K = CFG['ann']['K']

def build_pair_features_for_query(q_idx, cands_idx, Z, entities, directors, transactions):
    from math import radians, sin, cos, atan2, sqrt
    def haversine_km(lat1, lon1, lat2, lon2):
        R=6371.0
        phi1,phi2=radians(lat1),radians(lat2)
        dphi=radians(lat2-lat1); dl=radians(lon2-lon1)
        a=sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dl/2)**2
        return 2*R*atan2(sqrt(1-a), sqrt(a))

    with open(os.path.join(PROC_DIR, 'index_to_entity.json'),'r') as f:
        idx2eid = {int(k):int(v) for k,v in json.load(f).items()}
    ent = entities.set_index('entity_id')
    ctp = transactions.groupby('src_entity_id')['dst_entity_id'].apply(set).to_dict()
    dset = directors.groupby('entity_id')['person_name'].apply(set).to_dict()

    a = idx2eid[q_idx]; za = Z[q_idx]
    X, meta = [], []
    for ci in cands_idx:
        b = idx2eid[ci]; zb = Z[ci]
        cos_sim = float((za @ zb) / (np.linalg.norm(za)*np.linalg.norm(zb) + 1e-9))
        sa, sb = ctp.get(a,set()), ctp.get(b,set())
        inter = len(sa & sb); union = len(sa | sb) if (sa or sb) else 1
        jacc = inter / union
        try:
            gkm = haversine_km(ent.loc[a,'lat'], ent.loc[a,'lon'], ent.loc[b,'lat'], ent.loc[b,'lon'])
        except KeyError:
            gkm = 1e3
        ind_match = float(ent.loc[a,'industry_code'] == ent.loc[b,'industry_code'])
        da, db = dset.get(a,set()), dset.get(b,set())
        d_olap = len(da & db)
        diff_mean = float(np.mean(np.abs(za - zb)))
        had_mean = float(np.mean(za * zb))
        X.append([cos_sim, jacc, gkm, ind_match, d_olap, diff_mean, had_mean])
        meta.append((a,b))
    return np.array(X, dtype='float32'), meta

def main():
    Z = np.load(os.path.join(PROC_DIR, 'Z.npy')).astype('float32')
    index = joblib.load(os.path.join(MODELS_DIR, 'ann_sklearn.joblib'))

    entities = pd.read_csv(RAW['entities_csv'])
    directors = pd.read_csv(RAW['directors_csv'])
    transactions = pd.read_csv(RAW['transactions_csv'])
    labels = pd.read_csv(RAW['labels_links_csv'])
    pos = set(tuple(sorted((int(a),int(b)))) for a,b,_ in
              labels[['left_entity_id','right_entity_id','link_type']].values.tolist())

    reranker = xgb.Booster()
    reranker.load_model(os.path.join(MODELS_DIR, 'reranker_xgb.json'))

    nq = min(100, Z.shape[0])

    # SAFE K0
    k0 = min(K0, max(1, Z.shape[0] - 1))
    distances, indices = index.kneighbors(Z[:nq], n_neighbors=k0, return_distance=True)

    hits_at_k = 0
    total_pos_in_topk = 0
    total_topk = nq * min(K, indices.shape[1])

    for qi in range(nq):
        X_pairs, meta = build_pair_features_for_query(qi, indices[qi], Z, entities, directors, transactions)

        dm = xgb.DMatrix(
            X_pairs,
            feature_names=['cos_sim','jacc_ctp','geo_km','ind_match','dir_overlap','diff_mean','had_mean']
        )
        scores = reranker.predict(dm)

        K_eff = min(K, len(scores))
        order = np.argsort(scores)[::-1][:K_eff]
        top_pairs = [meta[i] for i in order]

        hits_at_k += sum(1 for a,b in top_pairs if tuple(sorted((a,b))) in pos)
        total_pos_in_topk += sum(1 for a,b in top_pairs if tuple(sorted((a,b))) in pos)

    precision_at_k = total_pos_in_topk / total_topk if total_topk else 0.0
    recall_at_k = hits_at_k / len(pos) if len(pos) > 0 else 0.0

    metrics = {f'precision_at_{K}': precision_at_k,
               f'recall_at_{K}': recall_at_k,
               'queries_evaluated': nq}
    print('Metrics:', metrics)
    with open(os.path.join(PROC_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

if __name__ == '__main__':
    main()