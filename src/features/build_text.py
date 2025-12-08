import os, yaml, pandas as pd, numpy as np
from sentence_transformers import SentenceTransformer

CFG = yaml.safe_load(open('config.yaml'))
ENTITIES = CFG['paths']['entities_csv']
PROC_DIR = CFG['paths']['processed_dir']

os.makedirs(PROC_DIR, exist_ok=True)

def main():
    df = pd.read_csv(ENTITIES)
    texts = (df['about_text'].fillna('') + ' Industry:' + df['industry_code'].astype(str))
    # Small general-purpose model; downloads on first use
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    emb = model.encode(texts.tolist(), batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
    np.save(os.path.join(PROC_DIR, 'text_embed.npy'), emb)
    df[['entity_id']].to_parquet(os.path.join(PROC_DIR, 'entity_ids.parquet'), index=False)
    print('Saved:', emb.shape, '→', os.path.join(PROC_DIR, 'text_embed.npy'))

if __name__ == '__main__':
    main()
