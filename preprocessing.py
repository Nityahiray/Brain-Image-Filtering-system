import cv2
import numpy as np
import tifffile
import os
import logging
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional
import sys

sys.path.insert(0, 'src')
from config import IMG_SIZE, CLAHE_CLIP, CLAHE_GRID

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

@dataclass
class PreprocessConfig:
    img_size: tuple = IMG_SIZE
    clahe_clip: float = CLAHE_CLIP
    clahe_grid: tuple = CLAHE_GRID
    denoise_h: float = 10.0
    min_content: float = 0.05

class BrainSlicePreprocessor:
    def __init__(self, config: PreprocessConfig = None):
        self.cfg = config or PreprocessConfig()
        self.clahe = cv2.createCLAHE(
            clipLimit = self.cfg.clahe_clip,
            tileGridSize = self.cfg.clahe_grid
        )

    def _load(self, path: str) -> np.ndarray:
        img = tifffile.imread(path)
        if img.dtype == np.uint16:
            p2, p98 = np.percentile(img, [2, 98])
            img = np.clip(img, p2, p98)
            img = ((img - p2) / (p98 - p2 + 1e-8) * 255).astype(np.uint8)
        return img

    def _quality_check(self, img: np.ndarray) -> bool:
        nonzero_ratio = (img > 5).mean()
        return nonzero_ratio >= self.cfg.min_content

    def _denoise(self, img: np.ndarray) -> np.ndarray:
        return cv2.fastNlMeansDenoising(
            img,
            h = self.cfg.denoise_h,
            templateWindowSize = 7,
            searchWindowSize = 21
        )

    def process_single(self, path:str) -> Optional[np.ndarray]:
        img = self._load(path)

        if not self._quality_check(img):
            log.debug(f'Rejected (low content): {path} ')
            return None

        img = self._denoise(img)

        img = self.clahe.apply(img)

        img = cv2.resize(
            img,
            (self.cfg.img_size, self.cfg.img_size),
            interpolation = cv2.INTER_LINEAR
        )

        img = img.astype(np.float32) / 255.0

        return img

    def process_directory(self, input_dir: str, output_dir: str) -> dict:
        os.makedirs(output_dir, exist_ok = True)
        files = sorted(Path(input_dir). glob('*.tif'))

        if len(files) == 0:
            log.warning(f'No .tif files found in {input_dir}')
            return {'total':0, 'accepted':0, 'rejected':0, 'rejection_rate':0}

        accepted = rejected = 0

        for fpath in tqdm(files, desc='Preprocessing'):
            result = self.process_single(str(fpath))
            if result is not None:
                out = Path(output_dir) / fpath.with_suffix('.npy').name
                np.save(str(out), result)
                accepted += 1
            else:
                rejected += 1

        stats = {
            'total': len(files),
            'accepted': accepted,
            'rejected': rejected,
            'rejection_rate': rejected / len(files)
        }
        log.info(f'Preprocessing done. Accepted: {accepted}, Rejected: {rejected}')
        return stats
    
if __name__ == '__main__':
    print('BrainSlicePreprocessor loaded successfully!')