import sys
import logging
import numpy as np

sys.path.insert(0, 'src')
logging.basicConfig(level=logging.INFO)

from clustering import BrainClusteringEngine
from visualization import ClusterVisualizer

features = np.load('results/embeddings/features.npy')
labels = np.load('results/embeddings/cluster_labels.npy')
filenames = np.load('results/embeddings/filenames.npy', allow_pickle=True)

print(f'Features : {features.shape}')
print(f'Labels: {labels.shape}')

engine = BrainClusteringEngine(features)

viz = ClusterVisualizer(engine.features_pca, labels, filenames)

viz.tsne_plot()

viz.cluster_grid(raw_dir = 'data/raw_images', n_per_cluster=5)

report = viz.quality_report()
print('\nCluster quality report:')
for cluster, stats in report.items():
    print(f' {cluster}: {stats}')

print('\n All plots saved to results/plots/')











