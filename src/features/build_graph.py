import os, yaml, pandas as pd, numpy as np, networkx as nx
from sklearn.decomposition import TruncatedSVD

CFG = yaml.safe_load(open('config.yaml'))
PROC_DIR = CFG['paths']['processed_dir']
SUPPLIERS = CFG['paths']['suppliers_csv']
DIRECTORS = CFG['paths']['directors_csv']
GRAPH_DIM = CFG['features']['graph_embed_dim']

os.makedirs(PROC_DIR, exist_ok=True)

def main():
    sup = pd.read_csv(SUPPLIERS)
    dirc = pd.read_csv(DIRECTORS)

    G = nx.Graph()
    # Add supplier edges (buyer -- supplier)
    G.add_edges_from([(int(r.buyer_entity_id), int(r.supplier_entity_id), {'etype':'supply'}) for r in sup.itertuples()])
    # Add director co-ownership edges (entities sharing a director)
    for person, ents in dirc.groupby('person_name')['entity_id']:
        ents = list(set(map(int, ents)))
        for i in range(len(ents)):
            for j in range(i+1, len(ents)):
                a,b = ents[i], ents[j]
                # increase weight if multiple shared directors happen to exist
                if G.has_edge(a,b):
                    G[a][b]['weight'] = G[a][b].get('weight',1.0) + 1.0
                else:
                    G.add_edge(a,b, weight=1.0, etype='common_owner')

    nodes = sorted(G.nodes())
    idx = {n:i for i,n in enumerate(nodes)}
    # Build sparse adjacency matrix (as dense for simplicity on demo scale)
    A = np.zeros((len(nodes), len(nodes)), dtype='float32')
    for a,b,data in G.edges(data=True):
        w = data.get('weight',1.0)
        ia, ib = idx[a], idx[b]
        A[ia, ib] = w
        A[ib, ia] = w

    # Simple embedding via SVD on adjacency
    k = min(GRAPH_DIM, max(2, min(A.shape)-1))
    svd = TruncatedSVD(n_components=k, random_state=42)
    Z = svd.fit_transform(A)
    # Normalize rows
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)

    # Save embedding aligned with entity_id order
    # We'll write mapping matrices
    emb = np.zeros((int(max(nodes)), k), dtype='float32')
    for n in nodes:
        emb[n-1] = Z[idx[n]]  # entity_id assumed 1..N

    np.save(os.path.join(PROC_DIR, 'graph_embed.npy'), emb)
    print('Saved graph_embed:', emb.shape)

if __name__ == '__main__':
    main()
