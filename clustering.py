import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, 'src')
from config import DBSCAN_EPS, DBSCAN_MINPTS,K_MAX,K_MIN, PCA_COMPONENTS

log = logging.getLogger(__name__)

class BrainClusteringEngine:

    def __init__(self, features:np.ndarray):
        self.raw_features = features
        self.n_samples = features.shape[0]

        self.scaler = StandardScaler()
        self.features_scaled = self.scaler.fit_transform(features)

        n_components = min(PCA_COMPONENTS, features.shape[0], features.shape[1])
        self.pca = PCA(n_components=n_components, random_state=42)
        self.features_pca = self.pca.fit_transform(self.features_scaled)

        variance_kept = self.pca.explained_variance_ratio_.sum()
        log.info(
            f'PCA: {features.shape[1]}D -> {n_components}D'
            f'({variance_kept * 100:.1f}% variance retained)'
        )

    def find_optimal_k(self, k_range=None) -> int:
        k_range = k_range or range(K_MIN, min(K_MAX, self.n_samples - 1) + 1)
        inertias = []
        silhouettes = []
        dbs = []

        log.info(f'Testing K from {min(k_range)} to {max(k_range)}...')
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(self.features_pca)
            inertias.append(km.inertia_)

            sample_size = min(300, self.n_samples)
            silhouettes.append(
                silhouette_score(self.features_pca, labels, sample_size=sample_size)
            )
            dbs.append(davies_bouldin_score(self.features_pca, labels))

            log.info(
                f' k={k:2d} inertia={km.inertia_:8.0f} '
                f'silhouette = {silhouettes[-1]:.3f} DB={dbs[-1]:.3f}'
            )

        best_k = list(k_range)[int(np.argmax(silhouettes))]
        log.info(f'Optimal K = {best_k}')

        self._plot_k_selection(k_range, inertias, silhouettes, dbs, best_k)

        return best_k

    def run_kmeans(self, k:int) -> np.ndarray:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(self.features_pca)

        self.kmeans_model = km
        self.kmeans_labels = labels

        sample_size = min(300, self.n_samples)
        score = silhouette_score(self.features_pca, labels, sample_size=sample_size)
        log.info(f'K-Means (k={k}): silhouette={score:.3f}')

        return labels

    def run_dbscan(self, eps=DBSCAN_EPS, min_samples=DBSCAN_MINPTS) -> np.ndarray:
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(self.features_pca)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int((labels == -1).sum())
        self.dbscan_labels = labels

        log.info(f'DBSCAN: {n_clusters} clusters, {n_noise} noise points')

        return labels
    def run_agglomerative(self, n_clusters: int) -> np.ndarray:
        agg = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        labels = agg.fit_predict(self.features_pca)

        self.agg_labels = labels
        return labels
    
    def compare_algorithms(self, k:int) -> dict:
        results = {}

        for name, labels in [
            ('KMeans', self.run_kmeans(k)),
            ('DBSCAN', self.run_dbscan()),
            ('Agglomerative', self.run_agglomerative(k)),
        ]:
            valid_mask = labels != -1
            valid_feats = self.features_pca[valid_mask]
            valid_lbls = labels[valid_mask]

            if len(set(valid_lbls)) > 1:
                sample_size = min(300, len(valid_lbls))
                sil = silhouette_score(valid_feats, valid_lbls, sample_size=sample_size)
                db = davies_bouldin_score(valid_feats, valid_lbls)
            else:
                sil = db = float('nan')

            results[name] = {
                'silhouette': round(sil, 3),
                'davies_bouldin': round(db, 3),
                'n_clusters': len(set(valid_lbls)),
                'noise_points': int((labels == -1).sum()),
            }
        return results
    
    def _plot_k_selection(self,k_range,inertias, silhouettes, dbs, best_k):
        os.makedirs('results/plots', exist_ok=True)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16,5))
        ks = list(k_range)

        ax1.plot(ks, inertias, 'bo-')
        ax1.set_title('Elbow method')
        ax1.set_xlabel('K')
        ax1.set_ylabel('Inertia')
        ax1.grid(True, alpha=0.3)

        ax2.plot(ks, silhouettes, 'rs-')
        ax2.axvline(best_k, color='green', linestyle='--', label=f'Best K={best_k}')
        ax2.set_title('Silhouette Score')
        ax2.set_xlabel('K')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        ax3.plot(ks, dbs, 'g^-')
        ax3.set_title('Davies-Bouldin (lower = better)')
        ax3.set_xlabel('K')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('results/plots/k_selection.png', dpi=150, bbox_inches='tight')
        plt.close()
        log.info('K-selection plot saved to results/plots/k_selection.png')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    features = np.load('results/embeddings/features.npy')
    engine = BrainClusteringEngine(features)
    best_k = engine.find_optimal_k()
    labels = engine.run_kmeans(best_k)

    np.save('results/embeddings/cluster_label.npy', labels)

    comparison = engine.compare_algorithms(best_k)
    print('\nAlgorithm comparison:')
    for algo, metrics in comparison.items():
        print(f' {algo}: {metrics}')
        

