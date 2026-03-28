import numpy as np
import json
import os
import logging
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional
import sys
sys.path.insert(0, '.')

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

@dataclass
class LabeledSample:
    filename: str
    original_label: int
    human_label: int
    entropy: float
    round_number: str=""
    brain_region: str=""

@dataclass
class ActiveLearningSession:
    session_id: str
    n_classes: int
    features_path: str
    labels_path: str
    filenames_path: str
    rounds_completed: int = 0
    labeled_samples: list[LabeledSample] = field(default_factory  = list)
    accuracy_history: list[float] = field(default_factory=list)
    correction_rate_history: list[float] = field(default_factory=list)
    total_queries: int = 0

    def save(self, path:str) -> None:
        d = asdict(self)
        with open(path, 'w') as f:
            json.dump(d, f, indent=2)
        log.info(f"Session saved {path}")

    @classmethod
    def Load(cls, path: str) -> "ActiveLearningSession":
        with open(path) as f:
            d = json.load(f)
        d['labeled_samples'] = [LabeledSample(**s) for s in d['labeled_samples']]
        return cls(**d)

class UncertaintySampler:
    def __init__(self,n_classes: int, random_state: int = 42):
        self.n_classes = n_classes
        self.random_state = random_state
        self.model = None

    def fit(self, features: np.ndarray, labels: np.ndarray):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import cross_val_score
        from sklearn.pipeline import Pipeline

        n_samples = len(labels)
        n_unique = len(np.unique(labels))

        if n_samples < n_unique * 2:
            log.warning(f"Only {n_samples} labeled samples for {n_unique} classes."
                        f"Skipping cross-validation.")
            acc = float('nan')
        else:
            cv_folds = min(5, n_samples // n_unique)
            cv_folds = max(2, cv_folds)

        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                C=1.0,
                solver='lbfgs',
            ))
        ])
        self.model.fit(features, labels)

        if n_samples >= n_unique * 2:
            scores = cross_val_score(self.model, features, labels,
                                     cv=cv_folds, scoring='accuracy')
            acc = float(scores.mean())
            log.info(f"Classifier CV accuracy: {acc:.3f} ± {scores.std():.3f}"
                    f"({cv_folds}-fold, n={n_samples})")
        return acc if not np.isnan(acc) else 0.0
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Call fit() before predict_proba().")
        return self.model.predict_proba(features)
    
    def entropy(self, proba: np.ndarray) -> np.ndarray:
        eps = 1e-10
        return -np.sum(proba * np.log(proba + eps), axis=1)
    
    def query(self, features_pool: np.ndarray, indices_pool: np.ndarray,
             n_query: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        proba = self.predict_proba(features_pool)
        entropies = self.entropy(proba)

        sorted_pos = np.argsort(entropies)[::-1]
        top_pos = sorted_pos[:n_query]

        selected_indices = indices_pool[top_pos]
        return selected_indices, entropies[top_pos], proba[top_pos]
    
class ActiveLearner:

    BRAIN_REGIONS = [
        "frontal cortex", "motor cortex", "somatosensory cortex",
        "hippocampus", "entorhinal cortex", "amygdala",
        "thalamus", "hypothalamus", "midbrain",
        "cerebellum", "brainstem", "olfactory bulb",
        "visual cortex", "auditory cortex", "parietal cortex",
        "edge/artifact", "unknown"
    ]

    def __init__(self, features: np.ndarray, labels: np.ndarray,
                 filenames: np.ndarray, n_classes: int):
        self.features = features
        self.labels = labels.copy()
        self.filenames = filenames
        self.n_classes = n_classes
        self.sampler = UncertaintySampler(n_classes)

        log.info(f"ActiveLearner ready."
                 f"Samples={len(features)}, Classes={n_classes}, Dim={features.shape[1]}")
        
    @classmethod
    def from_embeddings(cls, features_path: str, labels_path: str,
                        filenames_path: str) -> "ActiveLearner":
        features = np.load(features_path)
        labels = np.load(labels_path)
        filenames = np.load(filenames_path, allow_pickle=True)
        n_classes = len(np.unique(labels))

        log.info(f"Loaded: features{features.shape},"
                f"labels{labels.shape}, n_classes={n_classes}")
        return cls(features, labels, filenames, n_classes)
        
    def start_session(self, session_id: str = "session_001") -> ActiveLearningSession:
        session = ActiveLearningSession(
            session_id=session_id,
            n_classes=self.n_classes,
            features_path="results/embeddings/features.npy",
            labels_path="results/embeddings/cluster_labels.npy",
            filenames_path="results/embeddings/filenames.npy",
        )

        log.info(f"New session started: {session_id}")
        return session
        
    def run_round(self, session: ActiveLearningSession,
                n_query: int = 10) -> dict:
            
        round_num = session.rounds_completed + 1
        log.info(f"\n{'='*50}")
        log.info(f"ACTIVE LEARNING - Round {round_num}")
        log.info(f"{'='*50}")

        labeled_mask = np.zeros(len(self.features), dtype = bool)
        working_labels = self.labels.copy()

        if session.labeled_samples:
            for sample in session.labeled_samples:
                idx = np.where(self.filenames == sample.filename)[0]
                if len(idx) > 0:
                    labeled_mask[idx[0]] = True
                    working_labels[idx[0]] = sample.human_label
            labeled_idx = np.where(labeled_mask)[0]
            unlabeled_idx = np.where(~labeled_mask)[0]
            feat_labeled = self.features[labeled_idx]
            labels_labeled = working_labels[labeled_idx]

            log.info(f"Labeled pool: {len(labeled_idx)} samples")
            log.info(f"Unlabeled pool: {len(unlabeled_idx)} samples")
        else:
            log.info("Round 1: using K-Means labels as pseudo-labels for seeding")
            labeled_idx = np.arange(len(self.features))
            unlabeled_idx = np.arange(len(self.features))
            feat_labeled = self.features
            labels_labeled = self.labels
            labeled_mask = np.ones(len(self.features), dtype=bool)
        accuracy = self.sampler.fit(feat_labeled, labels_labeled)
        session.accuracy_history.append(accuracy)

        if len(unlabeled_idx) == 0:
            log.info("All samples are labeled. Session complete.")
            return {'queries': [], 'accuracy': accuracy, 'round':round_num,
                        'message': 'All samples labeled. No more queries needed.'}
        feat_pool = self.features[unlabeled_idx]
        actual_n = min(n_query, len(unlabeled_idx))

        selected, entropies, probas = self.sampler.query(
            feat_pool, unlabeled_idx, n_query=actual_n
        )

        queries = []
        for i, (idx, ent, prob) in enumerate(zip(selected, entropies, probas)):
            fname = self.filenames[idx]
            current_lbl = int(working_labels[idx])
            max_prob_lbl = int(np.argmax(prob))

            q = {
                'query_number' : i + 1,
                'filename' : str(fname),
                'slice_index' : int(idx),
                'current_label' : current_lbl,
                'predicted_label': max_prob_lbl,
                'confidence' : round(float(prob[max_prob_lbl]) * 100, 1),
                'entropy' : round(float(ent), 4),
                'top_3_classes' : [
                    {'class': int(c), 'prob': round(float(prob[c]) * 100, 1)}
                    for c in np.argsort(prob)[::-1][:3]
                ],
                'image_path' : f"data/processed_images/{fname}",
                'round' : round_num,
                'needs_label' : current_lbl != max_prob_lbl,
            }
            queries.append(q)
            log.info(f" Query {i+1}: {fname} | "
                    f"current={current_lbl} predicted={max_prob_lbl} | "
                    f"confidence={q['confidence']}% entropy={ent:.3f}")
        session.total_queries += len(queries)
        log.info(f"\nRound {round_num} ready."
                     f"Showing {len(queries)} uncertain slices to researcher.")
        log.info(f"Current accuracy: {accuracy:.3f}")

        return{
            'queries' : queries,
            'accuracy' : accuracy,
            'round' : round_num,
            'n_labeled' : int(labeled_mask.sum()),
            'n_unlabeled' : int((~labeled_mask).sum()),
            'total_queries' : session.total_queries,
        }
        
    def apply_corrections(self, session:ActiveLearningSession,
                        corrections: dict[str, dict]) -> ActiveLearningSession:
        round_num = session.rounds_completed + 1
        n_corrections = 0
        n_confirmed = 0

        for fname, correction in corrections.items():
            idx = np.where(self.filenames == fname)[0]
            if len(idx) == 0:
                log.warning(f"Filename not found: {fname}")
                continue

            original_lbl = int(self.labels[idx[0]])
            human_lbl = int(correction['label'])
            region = correction.get('brain_region', '')

            if original_lbl != human_lbl:
                n_corrections += 1
            else:
                n_confirmed += 1

            sample = LabeledSample(
                filename=fname,
                original_label=original_lbl,
                human_label=human_lbl,
                entropy=correction.get('entropy',0.0),
                round_number=round_num,
                brain_region=region,
            )
            session.labeled_samples.append(sample)
        correction_rate = n_corrections / max(1, n_corrections + n_confirmed)
        session.correction_rate_history.append(correction_rate)
        session.rounds_completed += 1

        log.info(f"\nRound {round_num} corrections applied:")
        log.info(f" Corrections: {n_corrections}")
        log.info(f" Confirmed : {n_confirmed}")
        log.info(f" Correction rate: {correction_rate:.1%}")
        log.info(f" Total labeled samples: {len(session.labeled_samples)}")

        return session
        
    def retrain_clustering(self, session: ActiveLearningSession) -> np.ndarray:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from sklearn.semi_supervised import LabelSpreading

        log.info("\nRetraining clustering with human corrections...")

        semi_labels = np.full(len(self.features), -1, dtype=int)
        for sample in session.labeled_samples:
            idx = np.where(self.filenames == sample.filename)[0]
            if len(idx) > 0:
                semi_labels[idx[0]] = sample.human_label
        n_labeled = (semi_labels != -1).sum()
        log.info(f"Semi-supervised: {n_labeled} labeled, "
                f"{(semi_labels == -1).sum()} unlabeled")
        scaler = StandardScaler()
        feat_std = scaler.fit_transform(self.features)

        if n_labeled >= self.n_classes:
            ls = LabelSpreading(
                kernel='rbf',
                gamma=0.25,
                alpha=0.2,
                max_iter=200,
            )               
            ls.fit(feat_std, semi_labels)
            new_labels=ls.transduction_
            log.info(f"Label spreading complete. "
                    f"Unique labels: {np.unique(new_labels)}")
        else:
            log.warning(f"Not enough labeled samples ({n_labeled}) for "
                        f"semi-supervised. Need at least {self.n_classes}."
                        f"Using original K-Means labels.")
            new_labels = self.labels.copy()
        return new_labels
        
    def convergence_check(self, session: ActiveLearningSession,
                        window: int = 3, threshold: float = 0.01) -> bool:
        if len(session.accuracy_history) < window + 1:
            return False
        recent = session.accuracy_history[-window:]
        improvement = max(recent) - min(recent)
        converged = improvement < threshold

        if converged:
            log.info(F"Convergence detected. Accuracy stable at "
                    f"{recent[-1]:.3f} (improvement = {improvement:.4f} < {threshold})")
        return converged
        
    def generate_report(self, session: ActiveLearningSession) -> dict:
        corrections = [s for s in session.labeled_samples
                      if s.original_label != s.human_label]
        region_counts: dict[str,int] = {}
        for s in session.labeled_samples:
            if s.brain_region:
                region_counts[s.brain_region] = region_counts.get(
                        s.brain_region, 0) + 1
        return{
            'session_id' : session.session_id,
            'rounds_completed': session.rounds_completed,
            'total_queries': session.total_queries,
            'total_labeled': len(session.labeled_samples),
            'total_corrections': len(corrections),
            'correction_rate': round(len(corrections) / max(1, len(session.labeled_samples)), 3),
            'accuracy_history': [round(a, 4) for a in session.accuracy_history],
            'correction_rate_history': [round(r, 4)
                                            for r in session.correction_rate_history],
            'final_accuracy': round(session.accuracy_history[-1], 4)
                                  if session.accuracy_history else None,
            'converged' : self.convergence_check(session),
            'region_annotations' : region_counts,
        }
        
                    



        
        
    
    

     
    