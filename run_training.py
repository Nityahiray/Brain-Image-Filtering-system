import sys
import logging
import numpy as np

sys.path.insert(0, 'src')
logging.basicConfig(level=logging.INFO)

from trainer import CNNTrainer

labels = np.load('results/embeddings/cluster_labels.npy')
filenames = list(np.load('results/embeddings/filenames.npy', allow_pickle=True))
kept_files = list(np.load('results/embeddings/kept_files.npy', allow_pickle=True))
kept_label = np.array([labels[filenames.index(f)] for f in kept_files])

print(f'Baseline samples: {len(filenames)}')
print(f'Filtered samples: {len(kept_files)}')
print(f'Classes : {len(set(labels))}')

print('\nStarting Experiment 1 - Baseline...')
baseline = CNNTrainer('baseline', 'data/processed_images', filenames, labels)
res_base = baseline.train()

print('\nStarting Experiment 2 - Filtered...')
filtered = CNNTrainer('filtered', 'data/filtered_images', kept_files, kept_label)
res_filt = filtered.train()

print('\n' + '='*50)
print('COMPARISON RESULTS')
print('='*50)
print(f'Baseline | time={res_base["total_time"]}s acc={res_base["final_acc"]}% samples={res_base["train_samples"]}')
print(f'Filtered | time={res_filt["total_time"]}s acc={res_filt["final_acc"]}% samples={res_filt["train_samples"]}')

speedup = (res_base['total_time'] - res_filt['total_time']) / res_base['total_time'] * 100
acc_delta = res_filt['final_acc'] - res_base['final_acc']

print(f'Speedup : {speedup:.1f}%')
print(f'Acc delta: {acc_delta:+.1f}%')