from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional
import numpy as np
import os, json, uuid, logging
from pathlib import Path
import sys
sys.path.insert(0, '.')
from src.active_learning import ActiveLearner, ActiveLearningSession

log = logging.getLogger(__name__)

al_router = APIRouter()

_sessions: dict[str, ActiveLearningSession] = {}
_learners: dict[str, ActiveLearner] = {}
SESSION_DIR = Path("results/active_learning_sessions")
SESSION_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_PATH = "results/embeddings/features.npy"
LABELS_PATH = "results/embeddings/cluster_labels.npy"
FILENAMES_PATH = "results/embeddings/filenames.npy"

class StartSessionRequest(BaseModel):
    session_name: str = Field(default="",
                              description="Optional name for this session")
    
class StartSessionResponse(BaseModel):
    session_id: str
    n_classes: int
    n_samples: int
    message: str

class QueryResponse(BaseModel):
    session_id: str
    round: int
    n_queries: int
    accuracy: float
    n_labeled: int
    n_unlabeled: int
    queries: list[dict]
    message: str

class CorrectionItem(BaseModel):
    filename: str = Field(...,description="e.g. 'slice_0043.npy")
    label: int = Field(...,description="Corrected cluster ID (0-indexed)")
    brain_region: str = Field(default="", description="Optional anatomical note")
    entropy: float = Field(default=0.0)

class SubmitCorrectionsRequest(BaseModel):
    corrections: list[CorrectionItem]

class RetrainResponse(BaseModel):
    session_id: str
    new_labels_saved: bool
    n_labeled: int
    message: str

class ReportResponse(BaseModel):
    session_id: str
    rounds_completed: int
    total_queries: int
    total_labeled: int
    total_corrections: int
    correction_rate: float
    accuracy_history: list[float]
    correction_rate_history: list[float]
    final_accuracy: Optional[float]
    converged: bool
    region_annotations: dict

def _get_learner(session_id: str) -> tuple[ActiveLearner,ActiveLearningSession]:
    if session_id not in _sessions:
        raise HTTPException(status_code=404,
                            detail=f"Session '{session_id}' not found."
                            f"Call POST / start first.")
    return _learners[session_id], _sessions[session_id]

def _check_embeddings() -> None:
    for p in [FEATURES_PATH, LABELS_PATH, FILENAMES_PATH]:
        if not os.path.exists(p):
            raise HTTPException(
                status_code=422,
                detail=f"Required file not found: {p}. "
                       f"Run the main pipeline first (python run_pipeline.py). "
            )
        
@al_router.post("/start", response_model=StartSessionResponse)
def start_session(req: StartSessionRequest) -> StartSessionResponse:
    _check_embeddings()
    session_id = req.session_name.strip() or str(uuid.uuid4())[:8]
    if session_id in _sessions:
        raise HTTPException(status_code=409,
                            detail=f"Session '{session_id}' already exists.")
    
    learner = ActiveLearner.from_embeddings(
        features_path = FEATURES_PATH,
        labels_path = LABELS_PATH,
        filenames_path = FILENAMES_PATH,
    )
    session = learner.start_session(session_id)

    _learners[session_id] = learner
    _sessions[session_id] = session

    return StartSessionResponse(
        session_id=session_id,
        n_classes=learner.n_classes,
        n_samples=len(learner.features),
        message=(f"Session '{session_id}' started. "
                 f"{len(learner.features)} samples, "
                 f"{learner.n_classes} clusters."
                 f"Call GET /query/{session_id} to begin."),
    )

@al_router.get("/query/{session_id}", response_model=QueryResponse)
def get_queries(session_id: str, n_query: int = 10) -> QueryResponse:
    n_query = min(n_query, 50)
    learner, session = _get_learner(session_id)

    result = learner.run_round(session,n_query=n_query)
    session.save(str(SESSION_DIR / f"{session_id}.json"))

    return QueryResponse(
        session_id=session_id,
        round=result['round'],
        n_queries=len(result['queries']),
        accuracy=round(result.get('accuracy', 0.0), 4),
        n_labeled=result.get('n_labeled', 0),
        n_unlabeled=result.get('n_unlabeled',0),
        queries=result['queries'],
        message=(f"Round {result['round']}: showing {len(result['queries'])}"
                 f"uncertain slices. Review and submit corrections"
                 f"POST / label/{session_id}."),

    )

@al_router.post("/label/{session_id}")
def submit_labels(session_id: str,
                  req: SubmitCorrectionsRequest) -> dict:
    learner, session = _get_learner(session_id)

    if not req.corrections:
        raise HTTPException(status_code=422,
                            detail="corrections list is empty.")
    corrections_dict = {
        item.filename: {
            'label': item.label,
            'brain_region': item.brain_region,
            'entropy': item.entropy,
        }
        for item in req.corrections
    }

    session = learner.apply_corrections(session, corrections_dict)
    _sessions[session_id] = session
    session.save(str(SESSION_DIR / f"{session_id}.json"))

    converged = learner.convergence_check(session)

    return {
        'session_id': session_id,
        'round_completed': session.rounds_completed,
        'total_labeled' : len(session.labeled_samples),
        'corrections_applied': len(req.corrections),
        'converged' : converged,
        'next_step' : (
            f"Call GET /retrain/{session_id} to apply results."
            if converged else
            f"Call GET /query/{session_id} to get next round of uncertain slices."
        ),
        'accuracy_so_far': session.accuracy_history,
    }

@al_router.get("/retrain/{session_id}", response_model=RetrainResponse)
def retrain(session_id:str,
            background_tasks: BackgroundTasks) -> RetrainResponse:
    learner, session = _get_learner(session_id)

    if len(session.labeled_samples) == 0:
        raise HTTPException(status_code=422,
                            detail="No labeled samples yet."
                            "Complete at least one labeling round first.")
    new_labels = learner.retrain_clustering(session)

    out_path = "results/embeddings/cluster_labels_refined.npy"
    np.save(out_path, new_labels)

    audit = [
        {
            'filename' : s.filename,
            'original_label' : s.original_label,
            'human_label' : s.human_label,
            'brain_region': s.brain_region,
            'round': s.round_number,
        }
        for s in session.labeled_samples
    ]

    audit_path = f"results/active_learning_sessions/{session_id}_audit.json"
    with open(audit_path, 'w') as f:
        json.dump(audit, f, indent=2)
    n_corrections = sum(1 for s in session.labeled_samples
                        if s.original_label != s.human_label)
    return RetrainResponse(
        session_id= session_id,
        new_labels_saved=True,
        n_labeled= len(session.labeled_samples),
        message=(f"Refined labels saved to {out_path}."
                 f"{n_corrections} corrections applied across"
                 f"{session.rounds_completed} rounds. "
                 f"Audit trail saved to {audit_path}."),
    )

@al_router.get("/report/{session_id}", response_model=ReportResponse)
def get_report(session_id: str) -> ReportResponse:
    learner, session = _get_learner(session_id)
    report = learner.generate_report(session)
    return ReportResponse(**report)

@al_router.get("/sessions")
def list_sessions() -> list[dict]:
    return [
        {
            'session_id': sid,
            'rounds' : s.rounds_completed,
            'labeled' : len(s.labeled_samples),
            'total_queries': s.total_queries,
            'accuracy': round(s.accuracy_history[-1], 4)
            if s.accuracy_history else None,
        }
        for sid, s in _sessions.items()
    ]

@al_router.delete("/sessions/{session_id}")
def delete_session(session_id: str) -> dict:
    if session_id not in _sessions:
        raise HTTPException(status_code=404,
                            detail=f"session '{session_id}' not found.")
    del _sessions[session_id]
    del _learners[session_id]
    return {'deleted': session_id, 'message': 'Session removed from memory.'
            'Disk files preserved.'}
