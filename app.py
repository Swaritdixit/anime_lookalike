import streamlit as st
import cv2
import torch
import numpy as np
from PIL import Image
from model import AnimeCNN
from dataset import AnimeDataset

st.set_page_config(page_title="Anime Lookalike", page_icon="🎌")

st.title("🎌 Anime Lookalike")
st.write("Take a photo and find the anime character you look like!")

@st.cache_resource
def load_model():
    dataset = AnimeDataset("anime_faces")

    model = AnimeCNN(len(dataset.class_names))
    model.load_state_dict(
        torch.load("anime_model.pth", map_location=torch.device("cpu"))
    )
    model.eval()

    return model, dataset.class_names

@st.cache_resource
def load_face_detector():
    cascade_path = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)

    if detector.empty():
        raise RuntimeError("Could not load the face detector.")

    return detector

model, class_names = load_model()
face_cascade = load_face_detector()

photo = st.camera_input("Take a photo")

if photo is not None:
    image = Image.open(photo).convert("RGB")
    image_array = np.array(image)

    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    if len(faces) == 0:
        st.warning("No face detected. Please try another photo.")
    else:
        x, y, w, h = faces[0]

        face = image_array[y:y+h, x:x+w]
        face = cv2.resize(face, (128, 128))

        face = face.astype(np.float32) / 255.0
        face = np.transpose(face, (2, 0, 1))

        tensor = torch.tensor(face).unsqueeze(0)

        with torch.no_grad():
            output = model(tensor)
            probabilities = torch.softmax(output, dim=1)

        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item() * 100

        result_image = image_array.copy()

        cv2.rectangle(
            result_image,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

        st.image(result_image, caption="Detected face")

        st.subheader(f"🎌 {class_names[prediction]}")
        st.write(f"Confidence: {confidence:.2f}%")
