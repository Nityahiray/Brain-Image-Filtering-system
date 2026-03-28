import sys 
import logging
import numpy as np

sys.path.insert(0, 'src')
logging.basicConfig(level=logging.INFO)

from clustering import BrainClusteringEngine

features = np.load('results/embeddings/features.npy')
print(f'Features loaded - shape: {features.shape}')

engine = BrainClusteringEngine(features)

best_k = engine.find_optimal_k()
print(f'Optimal K: {best_k}')

labels = engine.run_kmeans(best_k)
print(f'Labels shape: {labels.shape}')
print(f'Unique clusters: {len(set(labels))}')

np.save('results/embeddings/cluster_labels.npy', labels)
print('Cluster labels saved to results/embeddings/cluster_labels.npy')

print('\n Comparing all algorithms...')
comparison = engine.compare_algorithms(best_k)
for algo, metrics in comparison.items():
    print(f' {algo}: {metrics}')