import cv2
import torch
import numpy as np
from model import AnimeCNN
from dataset import AnimeDataset

# -------------------------------
# Load dataset and model
# -------------------------------
dataset = AnimeDataset("anime_faces")

model = AnimeCNN(len(dataset.class_names))
model.load_state_dict(
    torch.load("anime_model.pth", map_location="cpu")
)
model.eval()

# -------------------------------
# Load face detector
# -------------------------------
face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    raise IOError("Cannot load haarcascade_frontalface_default.xml")

# -------------------------------
# Open USB camera
# -------------------------------
# 0 = laptop cam
# 1 = USB cam (most common)
cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)


# Optional: reduce lag
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    raise IOError("Cannot open USB camera")

print("USB Camera opened successfully")

# -------------------------------
# Live camera loop
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(60, 60)
    )

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        # Preprocess face
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (128, 128))
        face = face.astype(np.float32) / 255.0

        face = torch.tensor(face)
        face = face.permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            output = model(face)
            probs = torch.softmax(output, dim=1)
            confidence, prediction = probs.max(1)

        label = dataset.class_names[prediction.item()]
        conf_percent = confidence.item() * 100

        # Draw box + label
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"{label} ({conf_percent:.1f}%)",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.imshow("Anime Lookalike - USB Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# -------------------------------
# Cleanup
# -------------------------------
cap.release()
cv2.destroyAllWindows()
