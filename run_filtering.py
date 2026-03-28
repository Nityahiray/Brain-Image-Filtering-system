import sys
import logging
import numpy as np

sys.path.insert(0,'src')
logging.basicConfig(level=logging.INFO)

from clustering import BrainClusteringEngine
from filtering import DatasetFilter

features = np.load('results/embeddings/features.npy')
labels = np.load('results/embeddings/cluster_labels.npy')
filenames = np.load('results/embeddings/filenames.npy',allow_pickle=True)

print(f'Features: {features.shape}')
print(f'Labels: {labels.shape}')

engine = BrainClusteringEngine(features)

filt = DatasetFilter(
    features_pca = engine.features_pca,
    cluster_labels = labels,
    filenames = filenames,
    redundancy_ratio = 0.30
)

result = filt.run()

print(f'Original : {result.original_count}')
print(f'Kept: {result.kept_count}')
print(f'Removed: {result.removed_count}')
print(f'Reduction: {result.reduction_pct}%')

filt.apply(result, 'data/processed_images', 'data/filtered_images')

np.save('results/embeddings/kept_files.npy',np.array(result.kept_files))

print('Filtering complete!')
print('Filtered images saved to data/filtered_images/')