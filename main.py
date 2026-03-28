from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import shutil, uuid, os, json, time
from pathlib import Path
from typing import Optional
from pydantic import BaseModel
import numpy as np
import sys

sys.path.insert(0, '.')
from src.preprocessing  import BrainSlicePreprocessor
from src.feature_extraction import BrainFeatureExtractor
from src.clustering import BrainClusteringEngine
from src.filtering import DatasetFilter
from api.active_learning_routes import al_router

app = FastAPI(
    title = 'Brain Image Filtering API',
    description= 'Unsupervised filtering of brain atlas images.',
    version = '1.0.0'
)

app.include_router(
    al_router,
    prefix="/active-learning",
    tags=["Active Learning"]
)

app.add_middleware(CORSMiddleware, allow_origins=['*'],
                   allow_methods=['*'], allow_headers=['*'])
jobs: dict = {}

UPLOAD_DIR = Path('data/uploads')
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

class FilterConfig(BaseModel):
    redundancy_ratio:float = 0.30
    n_clusters: Optional[int] = None

class JobStatus(BaseModel):
    job_id: str
    status: str
    message: str
    result: Optional[dict] = None

@app.get('/')
def home():
    return{
        'message': 'Brain Image Filtering API',
        'status': 'running',
        'docs': 'http://127.0.0.1:8000/docs'
    }

@app.get('/health')
def health():
    import torch, sklearn, cv2
    return {
        'status': 'healthy',
        'torch': torch.__version__,
        'sklearn': sklearn.__version__,
        'opencv': cv2.__version__,
        'gpu_available': torch.cuda.is_available(),
    }

@app.post('/upload-images')
async def upload_images(files: list[UploadFile] = File(...)):
    job_id = str(uuid.uuid4())[:8]
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir()

    saved = []
    for file in files:
        if not file.filename.endswith(('.tif','.tiff','.png','.jpg')):
            raise HTTPException(400, f'Unsupported file type: {file.filename}')
        dest = job_dir / file.filename
        with open(dest, 'wb') as buf:
            shutil.copyfileobj(file.file, buf)
        saved.append(file.filename)
    jobs[job_id] = {'status': 'uploaded', 'files':saved,'dir':str(job_dir)}
    return {'job_id': job_id, 'files_uploaded': len(saved), 'filenames': saved}

def _run_pipeline(job_id: str, config: FilterConfig):
    try:
        jobs[job_id]['status'] = 'running'
        job_dir = Path(jobs[job_id]['dir'])
        start = time.time()

        jobs[job_id]['message'] = 'Preprocessing images...'
        proc_dir = job_dir / 'processed'
        preprocessor = BrainSlicePreprocessor()
        stats = preprocessor.process_directory(str(job_dir), str(proc_dir))

        if stats['accepted'] == 0:
            raise ValueError('No valid images after preprocessing')
        jobs[job_id]['message'] = 'Extracting CNN features...'
        embed_dir = job_dir / 'embeddings'
        extractor = BrainFeatureExtractor()
        features = extractor.extract_directory(str(proc_dir), str(embed_dir))
        filenames = np.load(str(embed_dir / 'filenames.npy'), allow_pickle=True)

        jobs[job_id]['message'] = 'Clustering images...'
        engine = BrainClusteringEngine(features)
        best_k = config.n_clusters or engine.find_optimal_k()
        labels = engine.run_kmeans(best_k)
        
        jobs[job_id]['message'] = 'Filtering redundant images...'
        filt = DatasetFilter(
            engine.features_pca, labels, filenames,
            redundancy_ratio = config.redundancy_ratio
        )
        result = filt.run()

        jobs[job_id].update({
            'status': 'done',
            'message': 'Pipeline complete',
            'elapsed_seconds': round(time.time() - start, 2),
            'result': {
                'original_count': result.original_count,
                'kept_count': result.kept_count,
                'removed_count': result.removed_count,
                'reduction_pct': result.reduction_pct,
                'n_clusters': best_k,
                'kept_files': result.kept_files,
            }
        })
    except Exception as e:
        jobs[job_id].update({'status':'failed','message': str(e)})

@app.post('/run-pipeline/{job_id}')
def run_pipeline(job_id: str, config: FilterConfig,
                 background_tasks: BackgroundTasks):
    if job_id not in jobs:
        raise HTTPException(404, f'Job {job_id} not found')
    if jobs[job_id]['status'] != 'uploaded':
        raise HTTPException(400, 'Job already running or completed')
    background_tasks.add_task(_run_pipeline, job_id, config)
    return {'job_id': job_id, 'status': 'queued',
            'message': f'Pipeline started. Poll /status/{job_id} for updates.'}

@app.get('/status/{job_id}', response_model=JobStatus)
def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, f'Job {job_id} not found')
    j = jobs[job_id]
    return JobStatus(
        job_id = job_id,
        status = j.get('status', 'unknown'),
        message = j.get('message', ''),
        result = j.get('result'),
    )

@app.get('/results/{job_id}')
def get_results(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, 'Job not found')
    if jobs[job_id]['status'] != 'done':
        raise HTTPException(400, f'Job not done.Status: {jobs[job_id]["status"]}')
    return jobs[job_id]['result']

@app.get('/jobs')
def list_jobs():
    return[
        {'job_id': jid, 'status': j['status'], 'files': len(j.get('files',[]))}
        for jid,j in jobs.items()
    ]
