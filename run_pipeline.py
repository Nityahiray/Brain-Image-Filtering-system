import sys
import logging
import numpy as np
import tifffile
import os
import time
import json
from brainglobe_atlasapi import BrainGlobeAtlas
from tqdm import tqdm

sys.path.insert(0, 'src')
logging.basicConfig(level=logging.INFO)

from preprocessing import BrainSlicePreprocessor
from feature_extraction import BrainFeatureExtractor
from clustering import BrainClusteringEngine
from filtering import DatasetFilter
from trainer import CNNTrainer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from config import PCA_COMPONENTS

START = time.time()

print('='*60)
print('BRAIN IMAGE FILTERING PIPELINE')
print('='*60)

print('\n[1/6] Loading atlas...')
atlas = BrainGlobeAtlas('allen_mouse_25um')
os.makedirs('data/raw_images', exist_ok=True)
for i in tqdm(range(atlas.reference.shape[0]), desc='Saving slices'):
    tifffile.imwrite(f'data/raw_images/slice_{i:04d}.tif', atlas.reference[i])

print('\n[2/6] Preprocessing...')
stats = BrainSlicePreprocessor().process_directory(
    'data/raw_images', 'data/processed_images')
print(f'Accepted: {stats["accepted"]} Rejected: {stats["rejected"]}')

print('\n[3/6] Extracting features...')
features = BrainFeatureExtractor().extract_directory(
    'data/processed_images','results/embeddings')
filenames = np.load('results/embeddings/filenames.npy', allow_pickle=True)

print('\n[4/6] Clustering...')
engine = BrainClusteringEngine(features)
best_k = engine.find_optimal_k()
labels = engine.run_kmeans(best_k)
np.save('results/embeddings/cluster_labels.npy', labels)

print('\n[5/6] Filtering...')
filt = DatasetFilter(engine.features_pca, labels, filenames, 0.30)
result=filt.run()
filt.apply(result, 'data/processed_images', 'data/filtered_images')
np.save('results/embeddings/kept_files.npy', np.array(result.kept_files))

print('\n[6/6] Training...')
all_files = list(filenames)
kept_files = list(result.kept_files)
kept_labels = np.array([labels[all_files.index(f)] for f in kept_files])

r_base = CNNTrainer('baseline', 'data/processed_images',all_files, labels).train()
r_filt = CNNTrainer('filtered', 'data/filtered_images', kept_files, kept_labels).train()

total = time.time() - START
speedup = (r_base['total_time'] - r_filt['total_time']) / r_base['total_time'] * 100

report = {
    'baseline': r_base,
    'filtered': r_filt,
    'dataset_reduction':result.reduction_pct,
    'speedup_pct': round(speedup, 2),
    'acc_delta': round(r_filt['final_acc'] - r_base['final_acc'],2),
    'pipeline_time': round(total, 2),
}

with open('results/final_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print('\n' + '='*60)
print('FINAL RESULTS')
print('='*60)
print(f' Dataset reduction : {result.reduction_pct:.1f}%')
print(f' Training speedup : {speedup:.1f}%')
print(f' Accuracy delta : {report["acc_delta"]:+.1f}%')
print(f' Total time : {total/60:.1f} minutes')
print('Report saved to results/final_report.json')
print('='*60)