import os, yaml, json, numpy as np, pandas as pd
from math import radians, sin, cos, atan2, sqrt
from sklearn.model_selection import train_test_split
import xgboost as xgb

CFG = yaml.safe_load(open('config.yaml'))
PROC_DIR = CFG['paths']['processed_dir']
MODELS_DIR = CFG['paths']['models_dir']
RAW = CFG['paths']

def haversine_km(lat1, lon1, lat2, lon2):
    R=6371.0
    phi1,phi2=radians(lat1),radians(lat2)
    dphi=radians(lat2-lat1); dl=radians(lon2-lon1)
    a=sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dl/2)**2
    return 2*R*atan2(sqrt(a), sqrt(1-a))

def build_pair_features(pairs, Z, entities, directors, transactions):
    with open(os.path.join(PROC_DIR, 'index_to_entity.json'),'r') as f:
        idx2eid = {int(k):int(v) for k,v in json.load(f).items()}
    eid2idx = {v:k for k,v in idx2eid.items()}
    ent = entities.set_index('entity_id')
    ctp = transactions.groupby('src_entity_id')['dst_entity_id'].apply(set).to_dict()
    dset = directors.groupby('entity_id')['person_name'].apply(set).to_dict()
    X, y = [], []
    for (a,b,label) in pairs:
        if a not in eid2idx or b not in eid2idx: continue
        ia, ib = eid2idx[a], eid2idx[b]
        za, zb = Z[ia], Z[ib]
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
        y.append(1 if label in ('supplier_of','common_owner','co_customer') else 0)
    cols = ['cos_sim','jacc_ctp','geo_km','ind_match','dir_overlap','diff_mean','had_mean']
    return np.array(X, dtype='float32'), np.array(y, dtype='int32'), cols

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    Z = np.load(os.path.join(PROC_DIR, 'Z.npy')).astype('float32')
    entities = pd.read_csv(RAW['entities_csv'])
    directors = pd.read_csv(RAW['directors_csv'])
    transactions = pd.read_csv(RAW['transactions_csv'])
    labels = pd.read_csv(RAW['labels_links_csv'])

    pos = labels[['left_entity_id','right_entity_id','link_type']].values.tolist()
    pos_set = set((int(a),int(b)) for a,b,_ in pos) | set((int(b),int(a)) for a,b,_ in pos)
    eids = entities['entity_id'].tolist()
    rng = np.random.default_rng(42)
    neg = []
    while len(neg) < len(pos):
        a,b = int(rng.choice(eids)), int(rng.choice(eids))
        if a==b or (a,b) in pos_set or (b,a) in pos_set: continue
        neg.append([a,b,'no_link'])
    pairs = pos + neg

    X, y, cols = build_pair_features(pairs, Z, entities, directors, transactions)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    dtr = xgb.DMatrix(Xtr, label=ytr, feature_names=cols)
    dte = xgb.DMatrix(Xte, label=yte, feature_names=cols)

    params = {"objective":"binary:logistic","eval_metric":"aucpr","eta":0.05,
              "max_depth":6,"subsample":0.9,"colsample_bytree":0.9,"tree_method":"hist"}
    bst = xgb.train(params, dtr, num_boost_round=600, evals=[(dte,"valid")], early_stopping_rounds=50)
    bst.save_model(os.path.join(MODELS_DIR, "reranker_xgb.json"))
    print("Saved models/reranker_xgb.json")

if __name__ == '__main__':
    main()