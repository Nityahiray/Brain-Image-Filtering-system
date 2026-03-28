import sys
sys.path.insert(0, 'src')

from preprocessing import BrainSlicePreprocessor

Preprocessor = BrainSlicePreprocessor()

stats = Preprocessor.process_directory(
    input_dir = 'data/raw_images',
    output_dir = 'data/processed_images'
)

print(f'Total : {stats["total"]}')
print(f'Accepted : {stats["accepted"]}')
print(f'Rejected : {stats["rejected"]}')
print(f'Rejection rate:{stats["rejection_rate"]*100:.1f}%')