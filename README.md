Brain Image Filtering System - GSoC 2026 
Unsupervised Pipeline for FAIR Brain Atlas Dataset Curation

Solution: brain-filter
End-to-end unsupervised pipeline:

# Clone & install
git clone https://github.com/NilayHair/Brain-Image-Filtering-system.git
cd Brain-Image-Filtering-system
pip install -r requirements.txt

# full src demo
python src/file.py

# Full pipeline demo (Allen Mouse Atlas)
python run_file.py

# Jupyter exploration
jupyter notebook notebook/exploration.ipynb

# FastAPI server
uvicorn main:app --reload
Brain-Image-Filtering-System/
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── clustering.py
│   ├── filtering.py
│   ├── trainer.py
│   ├── visualization.py
│   └── active_learning.py
│
├── pipelines/
│   ├── run_preprocessing.py
│   ├── run_feature_extraction.py
│   ├── run_clustering.py
│   ├── run_filtering.py
│   ├── run_training.py
│   ├── run_visualization.py
│   └── run_active_learning.py
│
├── api/
│   ├── main.py
│   ├── active_learning_routes.py
│
├── data/
│   ├── raw_images/
│   ├── filtered_images/
│   └── sample_atlas/
│
├── notebooks/
│   └── exploration.ipynb
│
├── tests/
│   ├── test_Active_Learning.py
│
├── scripts/
│   └── setup_folder.py
│
├── requirements.txt




requirements.txt
├── README.md
└── .gitignore
