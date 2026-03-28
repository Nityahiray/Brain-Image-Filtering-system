from fastapi.testclient import TestClient
import sys
sys.path.insert(0, 'src')
from api.main import app

client = TestClient(app)

def test_home():
    r = client.get('/')
    assert r.status_code == 200
    assert r.json()['status'] == 'running'

def test_health():
    r = client.get('/health')
    assert r.status_code == 200
    assert 'torch' in r.json()

def test_upload_invalid_format():
    r = client.post('/upload-images',
                    files=[('files', ('bad.txt', b'data', 'text/plain'))])
    assert r.status_code == 400

def test_status_unknown_job():
    r = client.get('/status/nonexistent')
    assert r.status_code == 404

def test_list_jobs():
    r = client.get('/jobs')
    assert r.status_code == 200
    assert isinstance(r.json(), list)