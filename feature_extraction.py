import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import logging, os
import sys

sys.path.insert(0, 'src')
from config import FEAT_DIM, BATCH_SIZE

log = logging.getLogger(__name__)

class BrainFeatureExtractor:
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        log.info(f'Using device: {self.device}')

        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.model = torch.nn.Sequential(*list(backbone.children())[:-1])
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
            transforms.Normalize(self.IMAGENET_MEAN, self.IMAGENET_STD), 
        ])

    def _load_image(self, path:str) -> torch.Tensor:
        if path.endswith('.npy'):
            arr = np.load(path)
            img = (arr * 255).astype(np.uint8)
            pil = Image.fromarray(img, mode='L')
        else:
            pil = Image.open(path).convert('L')
        return self.transform(pil)
    
    @torch.no_grad()
    def extract_batch(self, paths: list) -> np.ndarray:
        tensors = [self._load_image(p) for p in paths]
        batch = torch.stack(tensors).to(self.device)
        feats = self.model(batch)
        return feats.squeeze(-1).squeeze(-1).cpu().numpy()
    
    def extract_directory(self, input_dir: str, output_dir: str,
                          batch_size: int = BATCH_SIZE) -> np.ndarray:
        os.makedirs(output_dir, exist_ok=True)
        files = sorted(Path(input_dir).glob('*.npy'))
        paths = [str(f) for f in files]
        names = [f.name for f in files]

        all_features = []

        for i in tqdm(range(0, len(paths), batch_size), desc='Extracting features'):
            batch_paths = paths[i: i + batch_size]
            batch_feats = self.extract_batch(batch_paths)
            all_features.append(batch_feats)

        features = np.vstack(all_features)

        np.save(f'{output_dir}/features.npy', features)
        np.save(f'{output_dir}/filenames.npy', np.array(names))

        log.info(f'Features extracted: {features.shape}')
        return features
    
    def analyze_embeddings(self, features: np.ndarray) -> dict:
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA

        scaler = StandardScaler()
        fs = scaler.fit_transform(features)
        pca = PCA(n_components=2)
        pca.fit_transform(fs)

        return{
            'n_samples': features.shape[0],
            'feat_dim': features.shape[1],
            'mean_norm': float(np.linalg.norm(features, axis=1).mean()),
            'pca_variance_2d': float(pca.explained_variance_ratio_.sum()),
        }
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    extractor = BrainFeatureExtractor()
    print('BrainFeatureExtractor loaded successfully!')
    print(f' Device: {extractor.device}')
    print(f' FEAT_DIM: {FEAT_DIM}')
    
