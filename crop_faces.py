import cv2
import os

BASE_DIR = "anime_faces"

face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

for character in os.listdir(BASE_DIR):
    char_path = os.path.join(BASE_DIR, character)

    if not os.path.isdir(char_path):
        continue

    print(f"Processing {character}...")

    for img_name in os.listdir(char_path):
        img_path = os.path.join(char_path, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = img[y:y+h, x:x+w]
            face = cv2.resize(face, (128, 128))
            cv2.imwrite(img_path, face)
            break  # keep only one face per image

print("Face cropping completed ✅")
