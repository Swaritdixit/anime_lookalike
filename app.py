import streamlit as st
import cv2
import torch
import numpy as np
from PIL import Image
from model import AnimeCNN
from dataset import AnimeDataset

st.set_page_config(
    page_title="Anime Lookalike",
    page_icon="🎌"
)

st.title("🎌 Anime Lookalike")
st.write("Take a photo and find the anime character you look like!")

@st.cache_resource
def load_model():
    dataset = AnimeDataset("anime_faces")
    model = AnimeCNN(len(dataset.class_names))

    model.load_state_dict(
        torch.load("anime_model.pth", map_location="cpu")
    )

    model.eval()

    return model, dataset.class_names

model, class_names = load_model()

@st.cache_resource
def load_face_detector():
    detector = cv2.CascadeClassifier(
        "haarcascade_frontalface_default.xml"
    )

    if detector.empty():
        raise IOError("Cannot load face detector")

    return detector

face_cascade = load_face_detector()

st.subheader("📷 Take a photo")

camera_image = st.camera_input("Take a picture of your face")

if camera_image is not None:
    image = Image.open(camera_image).convert("RGB")
    image = np.array(image)

    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(60, 60)
    )

    if len(faces) == 0:
        st.warning("No face detected. Please try again.")

    else:
        x, y, w, h = faces[0]

        face = frame[y:y+h, x:x+w]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (128, 128))
        face = face.astype(np.float32) / 255.0

        face = torch.tensor(face, dtype=torch.float32)
        face = face.permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            output = model(face)
            probabilities = torch.softmax(output, dim=1)
            confidence, prediction = probabilities.max(1)

        predicted_class = class_names[prediction.item()]
        confidence_percent = confidence.item() * 100

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"{predicted_class} ({confidence_percent:.1f}%)",
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        result_image = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        st.image(
            result_image,
            caption="Result",
            use_container_width=True
        )

        st.success(f"You look like: {predicted_class}")
        st.write(f"Confidence: {confidence_percent:.1f}%")
