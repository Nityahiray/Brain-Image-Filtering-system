import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
import os
import logging

log = logging.getLogger(__name__)

class ClusterVisualizer:

    def __init__(self, features_pca, labels, filename=None):
        self.feats = features_pca
        self.labels = labels
        self.filenames = filename
        os.makedirs('results/plots', exist_ok=True)

    def tsne_plot(self, perplexity=30):
        print('Running t-SNE...')
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        embed = tsne.fit_transform(self.feats)
        self._scatter(embed, 't-SNE', 'cluster_tsne.png')
        np.save('results/embeddings/tsne_embedding.npy', embed)
        print('t-SNE embedding saved.')
        return embed
    
    def umap_plot(self):
        import umap
        print('Running UMAP...')
        reducer = umap.UMAP(n_components=2, random_state=42)
        embed = reducer.fit_transform(self.feats)
        self._scatter(embed,'UMAP','cluster_umap.png')
        np.save('results/embeddings/umap_embedding.npy',embed)
        print('UMAP embedding saved.')
        return embed
    
    def _scatter(self, embed, title, filename):
        unique = np.unique(self.labels)
        colors = cm.tab20(np.linspace(0, 1, len(unique)))
        fig, ax = plt.subplots(figsize=(12,8))
        for c, col in zip(unique, colors):
            mask = self.labels == c
            label = f'Cluster {c}' if c != -1 else 'Noise'
            ax.scatter(embed[mask, 0], embed[mask, 1],
                       c=[col], label=label, alpha=0.7,
                       s=20 + (10 if c == -1 else 0))
            
        ax.set_title(
            f'{title} - Brain Slice Clusters (n = {len(self.labels)})',
            fontsize=13, fontweight='bold'
        )

        ax.legend(bbox_to_anchor=(1.01, 1), fontsize = 7, ncol = 2)
        plt.tight_layout()
        plt.savefig(f'results/plots/{filename}', dpi=150, bbox_inches='tight')
        plt.close()
        print(f'saved results/plots/{filename}')

    def cluster_grid(self, raw_dir='data/raw_images', n_per_cluster=5):
        import tifffile
        unique = [c for c in np.unique(self.labels) if c != -1]
        fig, axes = plt.subplots(
            len(unique), n_per_cluster,
            figsize=(n_per_cluster * 2, len(unique) * 2)
        ) 

        for row, cid in enumerate(unique):
            idxs = np.where(self.labels -- cid)[0][:n_per_cluster]
            for col,idx in enumerate(idxs):
                fname = self.filenames[idx].replace('.npy', '.tif')
                img = tifffile.imread(f'{raw_dir}/{fname}')
                axes[row,col].imshow(img, cmap='gray')
                axes[row,col].axis('off')
            axes[row, 0].set_ylabel(f'C{cid}', fontsize = 8, fontweight='bold')
        plt.suptitle('Representative Slices per cluster',
                     fontsize = 12, fontweight='bold')
        plt.tight_layout()
        plt.savefig('results/plots/cluster_grid.png', dpi=120,bbox_inches='tight')
        plt.close()
        print('Saved results/plots/cluster_grid.png')

    def quality_report(self) -> dict:
        unique = np.unique(self.labels)
        report = {}
        for c in unique:
            mask = self.labels == c
            label = 'noise' if c == -1 else f'cluster_{c}'
            report[label] = {
                'count': int(mask.sum()),
                'percentage': round(float(mask.sum()) / len(self.labels) * 100, 1),
            }
        return report
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print('ClusterVisualizer loaded successfully!')

                       
