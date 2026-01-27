import cv2
import torch
from model import AnimeCNN
from dataset import AnimeDataset

dataset = AnimeDataset("anime_faces")
model = AnimeCNN(len(dataset.class_names))
model.load_state_dict(torch.load("anime_model.pth"))
model.eval()

img = cv2.imread("test.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (128, 128))
img = img / 255.0
img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

with torch.no_grad():
    output = model(img)
    pred = output.argmax(1).item()

print("You look like:", dataset.class_names[pred])
