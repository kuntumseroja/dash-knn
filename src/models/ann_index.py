import os, yaml, numpy as np
from sklearn.neighbors import NearestNeighbors
import joblib

CFG = yaml.safe_load(open('config.yaml'))
PROC_DIR = CFG['paths']['processed_dir']
MODELS_DIR = CFG['paths']['models_dir']
os.makedirs(MODELS_DIR, exist_ok=True)

def main():
    Z = np.load(os.path.join(PROC_DIR, 'Z.npy')).astype('float32')
    index = NearestNeighbors(metric='cosine', algorithm='brute')
    index.fit(Z)
    joblib.dump(index, os.path.join(MODELS_DIR, 'ann_sklearn.joblib'))
    print('Built sklearn NearestNeighbors index over', Z.shape, '→ models/ann_sklearn.joblib')

if __name__ == '__main__':
    main()