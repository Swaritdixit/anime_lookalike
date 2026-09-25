# 🎌 Anime Lookalike

Take a photo and find the anime character you look like!

🔗 **Live Demo:** https://animelookalike-hlhkzrnmtxdpr4i4qpvqcw.streamlit.app/

Anime Lookalike is a deep learning web application that uses computer vision and a custom Convolutional Neural Network (CNN) to detect a face from a captured photo and predict the anime character that the person most closely resembles based on the trained model.

## ✨ Features

- 📷 Capture a photo directly from the browser
- 🙂 Detect faces using OpenCV Haar Cascade
- ✂️ Automatically crop the detected face
- 🖼️ Resize and preprocess the face image
- 🧠 Classify the face using a custom CNN
- 🎌 Predict the closest anime character
- 📊 Display the predicted character and confidence
- 🌐 Interactive web interface using Streamlit
- 🚀 Deployed using Streamlit Community Cloud

## 🛠️ Tech Stack

- **Python** — Core programming language
- **PyTorch** — Deep learning framework
- **Torchvision** — Image transformations
- **OpenCV** — Face detection and image processing
- **NumPy** — Numerical operations
- **Pillow** — Image handling
- **Streamlit** — Web application and deployment

## 🧠 How It Works

The application follows a computer vision and deep learning pipeline. The user captures a photo through the browser using Streamlit's camera input. OpenCV detects the face in the image using a Haar Cascade classifier. The detected face is cropped and resized to 128 × 128 pixels. The image is then preprocessed and converted into a PyTorch tensor before being passed to the trained CNN model. The model produces probabilities for the available anime-character classes, and the class with the highest probability is displayed as the prediction along with its confidence.

```text
User
  ↓
Take a Photo
  ↓
Face Detection
  ↓
Face Crop
  ↓
Resize to 128 × 128
  ↓
Image Preprocessing
  ↓
CNN Model
  ↓
Anime Character Prediction
  ↓
Character + Confidence
```

## 🤖 Machine Learning Model

The application uses a custom Convolutional Neural Network (CNN) trained on anime character face images.

The trained model is stored in:

```text
anime_model.pth
```

Input face images are resized to:

```text
128 × 128 pixels
```

before being passed to the model.

The CNN predicts the probability of each anime-character class. The class with the highest predicted probability is selected as the final result.

## 🔍 Face Detection

The application uses OpenCV's Haar Cascade Classifier to detect faces.

The project includes:

```text
haarcascade_frontalface_default.xml
```

The detected face is extracted from the captured image before being passed to the CNN model.

## 📊 Dataset

The training dataset is organized into separate folders for different anime characters. Each folder contains images belonging to the corresponding character.

```text
anime_faces/
├── character_1/
├── character_2/
├── character_3/
├── ...
└── character_n/
```

The folder structure is used to identify the different classes available to the model.

## 🔄 Prediction Pipeline

```text
Camera Input
     ↓
Image Conversion
     ↓
Face Detection
     ↓
Face Extraction
     ↓
Resize
     ↓
Normalization
     ↓
PyTorch Tensor
     ↓
CNN Inference
     ↓
Class Probabilities
     ↓
Highest Probability Class
     ↓
Anime Character
```

## 📂 Project Structure

```text
anime_lookalike/
│
├── app.py
├── model.py
├── dataset.py
├── predict.py
├── train.py
├── crop_faces.py
├── camera.py
│
├── anime_model.pth
├── haarcascade_frontalface_default.xml
├── requirements.txt
├── runtime.txt
├── README.md
│
└── anime_faces/
    ├── anya/
    ├── ash/
    ├── asta/
    └── ...
```

### Main Files

**`app.py`**

The main Streamlit application. It provides the user interface, captures the image, performs face detection and preprocessing, loads the trained model, performs prediction, and displays the result.

**`model.py`**

Contains the CNN architecture used by the project.

**`dataset.py`**

Handles the anime face dataset and class names used during model training.

**`train.py`**

Used to train the CNN model using the anime face dataset.

**`predict.py`**

Contains prediction-related functionality for running the trained model.

**`crop_faces.py`**

Used for detecting and cropping faces from images for dataset preparation.

**`camera.py`**

Contains local camera functionality for testing camera input outside the Streamlit web application.

**`anime_model.pth`**

Contains the trained PyTorch model.

**`haarcascade_frontalface_default.xml`**

OpenCV Haar Cascade model used for face detection.

## 🚀 Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/Swaritdixit/anime_lookalike.git
```

### 2. Enter the Project Directory

```bash
cd anime_lookalike
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

## 📦 Requirements

The project uses the following Python packages:

```text
streamlit
torch
torchvision
opencv-python-headless==4.10.0.84
numpy
Pillow
```

The OpenCV version is pinned to provide a consistent environment during deployment.

## 🌐 Deployment

The application is deployed using **Streamlit Community Cloud**.

🔗 **Live Application:** https://animelookalike-hlhkzrnmtxdpr4i4qpvqcw.streamlit.app/

The main Streamlit entry point is:

```text
app.py
```

The deployment flow is:

```text
GitHub Repository
       ↓
Streamlit Community Cloud
       ↓
Install Dependencies
       ↓
Load Trained Model
       ↓
Run app.py
       ↓
Live Web Application
```

## 🎯 Training Workflow

The model training process follows a dataset preparation and CNN training pipeline.

Anime character images are organized into their respective classes. Faces can be cropped using the face detection utility, the images are preprocessed, and the CNN is trained on the resulting dataset. The trained model is then saved as `anime_model.pth` and used by the Streamlit application for inference.

```text
Anime Face Dataset
       ↓
Dataset Preparation
       ↓
Face Cropping
       ↓
Image Preprocessing
       ↓
CNN Training
       ↓
anime_model.pth
       ↓
Streamlit Application
```

## ⚠️ Limitations

- Prediction quality depends on the quality and diversity of the training dataset.
- Face detection can be affected by lighting conditions, image quality, face angle, and occlusion.
- The application is limited to the anime-character classes included in the training dataset.
- The prediction is intended as an entertainment and machine-learning demonstration rather than an objective measurement of appearance.
- The model's confidence represents its classification probability and does not necessarily indicate how visually similar two faces are.

## 🔮 Future Improvements

- Improve face detection under different lighting conditions
- Support multiple faces in a single image
- Add more anime characters
- Increase the size and diversity of the training dataset
- Improve model accuracy
- Experiment with pretrained computer vision models
- Display the top 3 anime-character predictions
- Add prediction history
- Support image uploads in addition to camera input
- Improve the user interface and animations
- Add better confidence visualization
- Improve preprocessing and data augmentation

## 🔐 Privacy

Images captured through the application are used for the prediction process. Users should avoid uploading sensitive or private images that they do not want processed by an online application.

## 👨‍💻 Author

**Swarit Dixit**

B.Tech Electronics & Communication Engineering  
IIT Bhilai

- **GitHub:** https://github.com/Swaritdixit
- **LinkedIn:** https://www.linkedin.com/in/swarit-dixit-b907b8309/
- **Live Demo:** https://animelookalike-hlhkzrnmtxdpr4i4qpvqcw.streamlit.app/

## ⭐ Project

If you find the project interesting, consider giving the repository a ⭐ on GitHub!
