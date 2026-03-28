import os

#paths
RAW_DIR = 'data/raw_images'
PROCESSED_DIR = 'data/processed_images'
FILTERED_DIR = 'data/filtered_images'
RESULTS_DIR = 'results'
EMBEDDINGS_DIR = 'results/embeddings'
MODELS_DIR = 'results/models'

#processing
IMG_SIZE = 224
CLAHE_CLIP = 2.0
CLAHE_GRID = (8,8)

#feature extraction
FEAT_DIM = 215
BATCH_SIZE = 32

#clustering
K_MIN, K_MAX = 3,20
PCA_COMPONENTS = 50
DBSCAN_EPS = 3.5
DBSCAN_MINPTS = 5

#filtering
REDUNDANCY_RATIO = 0.30

#training
EPOCHS = 15
LEARNING_RATE = 1e-3
