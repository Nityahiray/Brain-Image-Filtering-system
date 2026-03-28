import numpy as np
import pytest
import tempfile
import os
import sys
sys.path.insert(0, '.')
from src.active_learning import (
    ActiveLearner, ActiveLearningSession, UncertaintySampler, LabeledSample
)

@pytest.fixture
def synthetic_data():
    rng = np.random.RandomState(42)
    n_per_class, n_classes, n_dim = 20, 3, 32

    features = np.vstack([
        rng.randn(n_per_class, n_dim) + np.array([i * 5] + [0] * (n_dim - 1))
        for i in range(n_classes)
    ]).astype(np.float32)

    labels = np.repeat(np.arange(n_classes), n_per_class)
    filenames = np.array([f"slice_{i:04d}.npy" for i in range(len(features))])

    return features, labels, filenames, n_classes

@pytest.fixture
def learner(synthetic_data):
    features, labels, filenames, n_classes = synthetic_data
    return ActiveLearner(features, labels, filenames, n_classes)

@pytest.fixture
def session(learner):
    return learner.start_session("test_session")

class TestUncertaintySampler:

    def test_fit_returns_accuracy(self, synthetic_data):
        features, labels, _, n_classes = synthetic_data
        sampler = UncertaintySampler(n_classes)
        acc = sampler.fit(features, labels)
        assert 0.0 <= acc <= 1.0, f"Accuracy {acc} out of [0,1]"

    def test_fit_well_separated_clusters_high_accuracy(self, synthetic_data):
        features, labels, _, n_classes = synthetic_data
        sampler = UncertaintySampler(n_classes)
        acc = sampler.fit(features, labels)
        assert acc >= 0.40, f"Expected > random chance, got {acc:.3f}"

    def test_predict_proba_shape(self, synthetic_data):
        features, labels, _, n_classes = synthetic_data
        sampler = UncertaintySampler(n_classes)
        sampler.fit(features, labels)
        proba = sampler.predict_proba(features)
        assert proba.shape == (len(features), n_classes)

    def test_predict_proba_sums_to_one(self, synthetic_data):
        features, labels, _, n_classes = synthetic_data
        sampler = UncertaintySampler(n_classes)
        sampler.fit(features, labels)
        proba = sampler.predict_proba(features)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_entropy_range(self, synthetic_data):
        features, labels, _, n_classes = synthetic_data
        sampler = UncertaintySampler(n_classes)
        sampler.fit(features, labels)
        proba = sampler.predict_proba(features)
        entropies = sampler.entropy(proba)
        assert (entropies >= 0).all(), "Entropy must be non-negative"
        max_entropy = np.log(n_classes)
        assert (entropies <= max_entropy + 1e-6).all(), \
             f"Entropy exceeds max={max_entropy:.3f}"
        
    def test_query_returns_correct_count(self, synthetic_data):
        features, labels, _, n_classes = synthetic_data
        sampler = UncertaintySampler(n_classes)
        sampler.fit(features, labels)
        indices = np.arange(len(features))
        n_query = 5
        sel, ent, prob = sampler.query(features, indices, n_query=n_query)
        assert len(sel) == n_query
        assert len(ent) == n_query
        assert prob.shape == (n_query, n_classes)

    def test_query_sorted_descending_entropy(self, synthetic_data):
        features, labels, _, n_classes = synthetic_data
        sampler = UncertaintySampler(n_classes)
        sampler.fit(features, labels)
        indices = np.arange(len(features))
        _, ent, _ = sampler.query(features, indices, n_query=10)
        assert all(ent[i] >= ent[i+1] - 1e-8 for i in range(len(ent) -1)), \
              "Queries not sorted by descending entropy"
        
    def test_predict_before_fit_raises(self, synthetic_data):
        _, _, _, n_classes = synthetic_data
        sampler = UncertaintySampler(n_classes)
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            sampler.predict_proba(np.zeros((5, 32)))

class TestActiveLearner:

    def test_start_session(self, learner):
        session = learner.start_session("test_001")
        assert session.session_id == "test_001"
        assert session.n_classes == learner.n_classes
        assert session.rounds_completed == 0
        assert len(session.labeled_samples) == 0

    def test_run_round_queries(self, learner, session):
        result = learner.run_round(session, n_query = 5)
        assert 'queries' in result
        assert 'accuracy' in result
        assert 'round' in result
        assert len(result['queries']) == 5

    def test_query_has_required_fields(self, learner, session):
        result = learner.run_round(session, n_query=3)
        required = ['filename' , 'current_label' , 'predicted_label',
                    'confidence' , 'entropy' , 'top_3_classes']
        for q in result['queries']:
            for field in required:
                assert field in q, f"Missinf field '{field}' in query"
                 
    def test_query_confidence_in_range(self, learner, session):
        result = learner.run_round(session, n_query=10)
        for q in result['queries']:
            assert 0 <= q['confidence'] <= 100, \
            f"Confidence {q['confidence']} out of [0, 100]"

    def test_apply_corrections_updates_session(self, learner, session):
        result = learner.run_round(session, n_query=3)
        queries = result['queries']

        corrections = {
            queries[0]['filename']: {'label': 1, 'brain_region': 'hippocampus',
                                     'entropy': queries[0]['entropy']},
            queries[1]['filename']: {'label': queries[1]['current_label'],
                                     'brain_region': '', 'entropy': 0.0},
        }

        session = learner.apply_corrections(session, corrections)

        assert session.rounds_completed == 1
        assert len(session.labeled_samples) == 2

    def test_corrections_track_original_vs_human(self, learner, session):
        result = learner.run_round(session, n_query=3)
        q = result['queries'][0]
        new_label = (q['current_label'] + 1) % learner.n_classes
        corrections={
            q['filename']: {'label': new_label, 'brain_region': 'cerebellum',
                            'entropy': q['entropy']},
        }
        session = learner.apply_corrections(session, corrections)
        sample = session.labeled_samples[0]

        assert sample.filename == q['filename']
        assert sample.original_label == q['current_label']
        assert sample.human_label == new_label
        assert sample.brain_region == 'cerebellum'
        assert sample.round_number == 1

    def test_second_round_has_fewer_unlabeled(self, learner, session):
        result1 = learner.run_round(session, n_query=5)
        queries1 = result1['queries']
        corrections = {
            q['filename']: {'label': q['current_label'],
                            'brain_region': '', 'entropy': q['entropy']}
            for q in queries1
        }

        session = learner.apply_corrections(session, corrections)

        result2 = learner.run_round(session, n_query=5)
        assert result2['n_labeled'] == len(corrections)
        assert result2['n_unlabeled'] == len(learner.features) - len(corrections)

    def test_convergence_check_false_with_few_rounds(self, learner, session):
        converged = learner.convergence_check(session)
        assert converged is False

    def test_convergence_check_true_when_stable(self, learner, session):
        session.accuracy_history = [0.90, 0.901, 0.900, 0.901]
        converged = learner.convergence_check(session, window=3, threshold=0.01)
        assert converged is True

    def test_convergence_check_false_when_improving(self, learner, session):
        session.accuracy_history = [0.70, 0.80, 0.90, 0.95]
        converged = learner.convergence_check(session, window=3, threshold=0.01)
        assert converged is False

    def test_retrain_clustering_returns_full_labels(self, learner, session):
        result = learner.run_round(session, n_query=10)
        corrections = {
            q['filename']: {'label': q['current_label'],
                            'brain_region': '', 'entropy':0.0}
            for q in result['queries']
        }

        session = learner.apply_corrections(session, corrections)
        new_labels = learner.retrain_clustering(session)

        assert len(new_labels) == len(learner.features)
        assert new_labels.dtype in (np.int32, np.int64, int)

    def test_generate_report_keys(self, learner, session):
        result = learner.run_round(session, n_query=3)
        corrections = {
            q['filename']: {'label': q['current_label'],
                            'brain_region': 'thalamus', 'entropy': q['entropy']}
            for q in result['queries']
        }

        session = learner.apply_corrections(session, corrections)
        report = learner.generate_report(session)

        required_keys = [
            'session_id', 'rounds_completed', 'total_queries',
            'total_labeled', 'total_corrections', 'correction_rate'
            'accuracy_history', 'converged', 'region_annotations'
        ]

        for key in required_keys:
            assert key in report, f"Missing key '{key}' in report"

    def test_region_annotations_collected(self, learner, session):
        result = learner.run_round(session, n_query=3)
        corrections = {
            result['queries'][0]['filename']: {
                'label': 0, 'brain_region': 'hippocampus', 'entropy': 0.0},
            result['queries'][1]['filename']: {
                'label': 1, 'brain_region': 'hippocampus', 'entropy': 0.0},
            result['queries'][2]['filename']: {
                'label': 2, 'brain_region': 'cerebellum',  'entropy': 0.0},
        }
        session = learner.apply_corrections(session, corrections)
        report  = learner.generate_report(session)

        assert report['region_annotations'].get('hippocampus') == 2
        assert report['region_annotations'].get('cerebellum')  == 1


class TestSessionPersistence:

    def test_save_and_load_roundtrip(self, learner, session, tmp_path):
        session.labeled_samples.append(LabeledSample(
            filename='slice_0001.npy', original_label=0, human_label=1,
            entropy=0.85, round_number=1, brain_region='hippocampus'
        ))
        session.accuracy_history       = [0.75, 0.88]
        session.correction_rate_history= [0.30]
        session.rounds_completed       = 1
        session.total_queries          = 10

        path = str(tmp_path / "session.json")
        session.save(path)

        loaded = ActiveLearningSession.load(path)
        assert loaded.session_id               == session.session_id
        assert loaded.rounds_completed         == 1
        assert loaded.total_queries            == 10
        assert loaded.accuracy_history         == [0.75, 0.88]
        assert len(loaded.labeled_samples)     == 1
        assert loaded.labeled_samples[0].filename     == 'slice_0001.npy'
        assert loaded.labeled_samples[0].brain_region == 'hippocampus'

    def test_load_nonexistent_raises(self):
        with pytest.raises((FileNotFoundError, Exception)):
            ActiveLearningSession.load("/nonexistent/path/session.json")

class TestFromEmbeddings:

    def test_from_embeddings_loads_correctly(self, synthetic_data, tmp_path):
        features, labels, filenames, n_classes = synthetic_data

        fp = str(tmp_path / "features.npy")
        lp = str(tmp_path / "labels.npy")
        fnp= str(tmp_path / "filenames.npy")
        np.save(fp, features)
        np.save(lp, labels)
        np.save(fnp, filenames)

        learner = ActiveLearner.from_embeddings(fp, lp, fnp)
        assert learner.n_classes        == n_classes
        assert learner.features.shape   == features.shape
        assert len(learner.filenames)   == len(filenames)

