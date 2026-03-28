import torch 
import torch.nn as nn
import torchvision.models as models
import numpy as np
import time, json, os
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import logging

sys.path.insert(0, 'src')
from config import EPOCHS, LEARNING_RATE, BATCH_SIZE

log = logging.getLogger(__name__)

class BrainDataset(Dataset):

    def __init__(self, files, data_dir, labels):
        self.files = files
        self.data_dir = data_dir
        self.labels = labels

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img = np.load(f'{self.data_dir}/{self.files[idx]}')
        t = torch.tensor(img).unsqueeze(0).repeat(3, 1, 1).float()
        return t, int(self.labels[idx])
    
class CNNTrainer:

    def __init__(self, name: str, data_dir: str,
                 file_list: list, labels:np.ndarray):
        self.name = name
        self.data_dir = data_dir
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.n_classes = len(np.unique(labels))

        log.info(f'Training: {name} | device: {self.device} |'
                 f'classes: {self.n_classes} | samples: {len(file_list)}')
        
        tr_f, va_f, tr_l, va_l = train_test_split(
            file_list, labels,
            test_size = 0.20,
            random_state = 42,
            stratify = labels
        )
        self.train_loader = DataLoader(
            BrainDataset(tr_f, data_dir, tr_l),
            batch_size = BATCH_SIZE,
            shuffle = True,
            num_workers = 0
        )
        self.val_loader = DataLoader(
            BrainDataset(va_f, data_dir, va_l),
            batch_size = BATCH_SIZE,
            shuffle = False,
            num_workers = 0
        )
        backbone = models.resnet18(weights = None)
        backbone.fc = nn.Linear(512, self.n_classes)
        self.model = backbone.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=LEARNING_RATE
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=5, gamma=0.5
        )

        self.criterion = nn.CrossEntropyLoss()

    def train(self, epochs: int = EPOCHS) -> dict:
        history = {'train_loss': [], 'val_acc': [], 'epoch_times': []}
        total_start = time.time()

        for epoch in range(epochs):
            ep_start = time.time()

            self.model.train()
            running_loss = 0

            for imgs, labels in self.train_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(imgs), labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            self.scheduler.step()

            acc = self._validate()
            ep_time = time.time()
            
            history['train_loss'].append(running_loss / len(self.train_loader))
            history['val_acc'].append(acc)
            history['epoch_times'].append(ep_time)

            print(f'[{self.name}] Epoch {epoch+1:2d}/{epochs}'
                  f'loss={history["train_loss"][-1]:.4f}'
                  f'val_acc={acc:.1f}%'
                  f'time={ep_time:.1f}s')
        
        total_time = time.time() - total_start
        final_acc = history['val_acc'][-1]

        results = {
            'name': self.name,
            'train_samples': len(self.train_loader.dataset),
            'val_samples': len(self.val_loader.dataset),
            'total_time': round(total_time, 2),
            'final_acc': round(final_acc, 2),
            'best_acc': round(max(history['val_acc']), 2),
            'history': history,
        }

        os.makedirs('results/metrics', exist_ok=True)
        os.makedirs('results/models', exist_ok=True)

        with open(f'results/metrics/{self.name}.json', 'w') as f:
            json.dump(results, f, indent=2)
        self._plot_history(history)

        torch.save(
            self.model.state_dict(),
            f'results/models/{self.name}.pth'
        )

        log.info(f'{self.name} training complete.'
                 f'Final acc: {final_acc:.1f}% |'
                 f'Time: {total_time:.1f}s')
        return results
    
    def _validate(self) -> float:
        self.model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in self.val_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                preds = self.model(imgs).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += len(labels)
        return correct / total * 100
    
    def _plot_history(self, history):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(history['train_loss'],'b-o')
        ax1.set_title(f'{self.name} - Loss')
        ax1.set_xlabel('Epoch')
        ax1.grid(True, alpha=0.3)

        ax2.plot(history['val_acc'], 'r-s')
        ax2.set_title(f'{self.name} - Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy %')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'results/plots/{self.name}_history.png', dpi=120)
        plt.close()
        log.info(f'Training history plot saved for {self.name}')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print('CNNTrainer loaded Successfully!')


        