import numpy as np
import os, shutil, json
from pathlib import Path 
from dataclasses import dataclass, asdict
from typing import List
import logging
import sys

sys.path.insert(0, 'src')
from config import REDUNDANCY_RATIO

log = logging.getLogger(__name__)

@dataclass
class FilterResult:
    original_count: int
    kept_count: int
    removed_count: int
    reduction_pct: float
    kept_files: List[str]
    removed_files: List[str]
    per_cluster_stats: dict

class DatasetFilter:
    def __init__(self, features_pca: np.ndarray, cluster_labels: np.ndarray,
                 filenames: np.ndarray, redundancy_ratio: float = REDUNDANCY_RATIO):
        self.features = features_pca
        self.labels = cluster_labels
        self.filenames = filenames
        self.ratio = redundancy_ratio
        assert 0 < redundancy_ratio <= 1.0, 'ratio must be in[0, 1]'

    def run(self) -> FilterResult:
        kept = []
        removed =[]
        per_cluster = {}

        for cluster_id in np.unique(self.labels):
            if cluster_id == -1:
                noise_idx = np.where(self.labels == -1)[0]
                kept.extend(self.filenames[noise_idx].tolist())
                per_cluster['noise'] = {
                    'kept': len(noise_idx), 'removed':0
                }

                continue

            mask = self.labels == cluster_id
            indices = np.where(mask)[0]
            feats = self.features[indices]
            centroid = feats.mean(axis = 0)
            distances = np.linalg.norm(feats - centroid, axis = 1)
            sorted_pos = np.argsort(distances)

            n_total = len(indices)
            n_keep = max(1, int(n_total * self.ratio))
            keep_pos = sorted_pos[:n_keep]
            remove_pos = sorted_pos[n_keep:]

            kept.extend(self.filenames[indices[keep_pos]].tolist())
            removed.extend(self.filenames[indices[remove_pos]].tolist())

            per_cluster[str(cluster_id)] = {
                'total': n_total,
                'kept': n_keep,
                'removed': len(remove_pos),
                'pct_kept': round(n_keep/n_total * 100,1),
            }

        n_orig = len(self.filenames)
        return FilterResult(
            original_count = n_orig,
            kept_count = len(kept),
            removed_count = len(removed),
            reduction_pct = round((1 - len(kept) / n_orig) * 100, 2),
            kept_files = kept,
            removed_files = removed,
            per_cluster_stats = per_cluster,
        )
    
    def apply(self, result: FilterResult, src_dir:str, dst_dir: str) -> None:
        os.makedirs(dst_dir, exist_ok=True)
        for fname in result.kept_files:
            src = Path(src_dir) / fname
            dst = Path(dst_dir) / fname
            if src.exists():
                shutil.copy(str(src), str(dst))

        report = asdict(result)
        report.pop('kept_files')
        report.pop('removed_files')
        os.makedirs('results', exist_ok=True)
        with open('results/filter_report.json','w') as f:
            json.dump(report, f, indent = 2)

        log.info(f'Filtered dataset saved. Reduction: {result.reduction_pct:.1f}%')
        print(f'Filter report saved to results/filter_report.json')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print('DatasetFilter loaded successfully!')


        
        