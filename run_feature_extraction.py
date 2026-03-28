import sys
import logging 
sys.path.insert(0, 'src')

logging.basicConfig(level=logging.INFO)

from feature_extraction import BrainFeatureExtractor

extractor = BrainFeatureExtractor()

feature = extractor.extract_directory(
    input_dir = 'data/processed_images',
    output_dir = 'results/embeddings'
)

print(f'Feature shape: {feature.shape}')
print(f'Saved to : results/embeddings/features.npy')

