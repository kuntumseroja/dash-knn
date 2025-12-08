import os, yaml, numpy as np, pandas as pd, json

CFG = yaml.safe_load(open('config.yaml'))
PROC_DIR = CFG['paths']['processed_dir']

def main():
    ent = pd.read_parquet(os.path.join(PROC_DIR, 'entity_ids.parquet'))  # entity_id column
    text = np.load(os.path.join(PROC_DIR, 'text_embed.npy'))
    txn  = np.load(os.path.join(PROC_DIR, 'txn_embed.npy'))
    graph= np.load(os.path.join(PROC_DIR, 'graph_embed.npy'))

    W = CFG['features']['weights']
    # ensure same ordering by entity_id (assumes CSVs and previous steps keep 1..N order)
    entity_ids = ent['entity_id'].astype(int).tolist()
    N = max(entity_ids)
    # pad arrays if needed
    def pad_to(arr, N):
        if arr.shape[0] < N:
            pad = np.zeros((N - arr.shape[0], arr.shape[1]), dtype=arr.dtype)
            arr = np.vstack([arr, pad])
        return arr

    text, txn, graph = pad_to(text, N), pad_to(txn, N), pad_to(graph, N)
    # L2 normalize per block
    def norm(x):
        n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
        return x / n
    text, txn, graph = norm(text), norm(txn), norm(graph)
    Z = np.hstack([W['text']*text, W['txn']*txn, W['graph']*graph]).astype('float32')
    np.save(os.path.join(PROC_DIR, 'Z.npy'), Z)

    # index_to_entity mapping (row index → entity_id)
    index_to_entity = {i: i+1 for i in range(N)}
    with open(os.path.join(PROC_DIR, 'index_to_entity.json'), 'w') as f:
        json.dump(index_to_entity, f)
    print('Saved Z:', Z.shape)

if __name__ == '__main__':
    main()
