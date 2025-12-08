import os, yaml, pandas as pd, numpy as np
from sklearn.decomposition import PCA

CFG = yaml.safe_load(open('config.yaml'))
TRANSACTIONS = CFG['paths']['transactions_csv']
PROC_DIR = CFG['paths']['processed_dir']
EMBED_DIM = CFG['features']['txn_embed_dim']

os.makedirs(PROC_DIR, exist_ok=True)

def month_to_idx(m):
    # m is 'YYYY-MM'
    y, mm = m.split('-')
    return int(mm) - 1

def main():
    tx = pd.read_csv(TRANSACTIONS)
    # Basic per-entity aggregates
    # Seasonality: 12-dim sums by month over outgoing transactions
    tx['midx'] = tx['yyyymm'].astype(str).apply(month_to_idx)
    by_e = tx.groupby(['src_entity_id','midx'])['amount'].sum().unstack(fill_value=0).reindex(columns=range(12), fill_value=0)
    by_e = by_e.div(by_e.sum(axis=1).replace(0,1), axis=0)  # normalize
    by_e.columns = [f'season_{i+1:02d}' for i in range(12)]

    # Counterparty stats (diversity)
    ccount = tx.groupby(['src_entity_id'])['dst_entity_id'].nunique().rename('n_unique_ctp')
    total_tx = tx.groupby(['src_entity_id'])['yyyymm'].count().rename('n_tx')
    cats = tx.pivot_table(index='src_entity_id', columns='category', values='amount', aggfunc='sum').fillna(0.0)
    cats = cats.div(cats.sum(axis=1).replace(0,1), axis=0)
    cats.columns = [f'cat_{c}' for c in cats.columns]

    feats = by_e.join([ccount, total_tx, cats], how='outer').fillna(0.0)
    feats.index.name = 'entity_id'
    feats.reset_index(inplace=True)

    # Build a compact embedding via PCA
    X = feats.drop(columns=['entity_id']).values.astype('float32')
    pca = PCA(n_components=min(EMBED_DIM, X.shape[1]))
    Xp = pca.fit_transform(X)
    np.save(os.path.join(PROC_DIR, 'txn_embed.npy'), Xp)
    feats.to_parquet(os.path.join(PROC_DIR, 'txn_feats.parquet'), index=False)
    print('Saved txn_embed:', Xp.shape)

if __name__ == '__main__':
    main()
