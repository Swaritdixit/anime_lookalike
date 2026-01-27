import torch
from torch.utils.data import DataLoader
from dataset import AnimeDataset
from model import AnimeCNN

dataset = AnimeDataset("anime_faces")
loader = DataLoader(dataset, batch_size=8, shuffle=True)

model = AnimeCNN(num_classes=len(dataset.class_names))
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(14):
    total_loss = 0
    for images, labels in loader:
        preds = model(images)
        loss = loss_fn(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss:.2f}")

torch.save(model.state_dict(), "anime_model.pth")
print("Model saved")
