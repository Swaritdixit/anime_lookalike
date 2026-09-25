import os
import cv2
import torch
from torch.utils.data import Dataset

class AnimeDataset(Dataset):
    def __init__(self, root_dir):
        self.data = []
        self.labels = []
        self.class_names = sorted(os.listdir(root_dir))

        for idx, name in enumerate(self.class_names):
            folder = os.path.join(root_dir, name)
            for img in os.listdir(folder):
                self.data.append(os.path.join(folder, img))
                self.labels.append(idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = cv2.imread(self.data[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))
        img = img / 255.0
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        label = self.labels[index]
        return img, label
