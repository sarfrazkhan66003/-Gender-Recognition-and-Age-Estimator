# 👤 Gender Recognition System using Deep Learning

> “Artificial Intelligence is not about replacing humans,  
> it is about understanding humans better.”

This project is a **Deep Learning–based Gender Recognition System** that detects  
**gender (Man / Woman)** from:
- 🖼️ Static images
- 🎥 Live webcam feed

It combines **Computer Vision** and **Convolutional Neural Networks (CNNs)**  
to deliver accurate and real-time predictions.

---

## 📌 Project Overview

The system first detects **human faces** from an image or video stream,  
then applies a **pre-trained CNN model** to classify gender.

✔ No training required  
✔ Beginner-friendly  
✔ Real-world AI use case  

---

## 🧠 Algorithm Used (Step-by-Step)

### 🔹 Step 1: Input Acquisition
- User provides:
  - An image file OR
  - Live webcam feed

---

### 🔹 Step 2: Face Detection
- Faces are detected using **Computer Vision techniques**
- Only face regions are passed to the deep learning model

> “Good input leads to good predictions.”

---

### 🔹 Step 3: Image Preprocessing
Each detected face is:
- Cropped ✂️
- Resized to **96 × 96**
- Normalized (pixel values scaled between 0–1)

---

### 🔹 Step 4: Gender Classification (CNN)
- A **pre-trained Convolutional Neural Network** predicts gender
- Model outputs probabilities for:
  - `Man`
  - `Woman`

---

### 🔹 Step 5: Visualization
- Bounding box drawn around face
- Gender label + confidence displayed
- Output saved automatically

---

# 🏷️ GitHub Topics
- deep-learning
- computer-vision
- gender-recognition
- face-detection
- opencv
- keras
- tensorflow
- cnn
- machine-learning
- python-project
- ai-project
- mediapipe
- image-processing
- webcam-detection

---

## 🗂️ Project Structure

    Gender-Recognition-and-Age-Estimator/
    │
    ├── detect_gender.py # Gender detection from image
    ├── detect_gender_webcam.py # Real-time webcam detection
    ├── fork-of-gender-classification-cnn-image-dataset.ipynb
    ├── requirements.txt
    ├── README.md

---

- Accuracy: Woman: 91.73%
- Bounding box drawn around face
- Gender label with confidence score displayed
- Output image saved automatically

---

## 📁 Project Structure


---
 - Accuracy - Man: 94.32%
   - ✔ Bounding box on face  
   - ✔ Confidence score displayed  

---

### 🎥 Webcam Output
- Real-time face detection
- Live gender prediction
- Press **Q** to exit

---

## 📊 Output Example

| Input Type | Output |
|-----------|--------|
| Image | Gender + Confidence |
| Webcam | Live Bounding Box + Label |

---

## 📚 Technologies Used

| Technology | Purpose |
|----------|--------|
| Python | Programming language |
| TensorFlow | Deep learning framework |
| Keras | Model loading & prediction |
| OpenCV | Image & video processing |
| MediaPipe | Face detection |
| NumPy | Numerical operations |

---

## ⚙️ Installation & Setup

✔ Bounding box on face  
✔ Confidence score displayed  

---

### 🎥 Webcam Output
- Real-time face detection
- Live gender prediction
- Press **Q** to exit

---

## 📊 Output Example

| Input Type | Output |
|-----------|--------|
| Image | Gender + Confidence |
| Webcam | Live Bounding Box + Label |

---

## 📚 Technologies Used

| Technology | Purpose |
|----------|--------|
| Python | Programming language |
| TensorFlow | Deep learning framework |
| Keras | Model loading & prediction |
| OpenCV | Image & video processing |
| MediaPipe | Face detection |
| NumPy | Numerical operations |

---

# ⚠️ Common Issues & Fix
    - ❌ cvlib / TensorFlow error
    
    - ✔ Use Python ≤ 3.10
    - ✔ Avoid Python 3.11+
    - ✔ Prefer OpenCV / MediaPipe

    
#🎓 Learning Outcomes

- After completing this project, you will understand:
- ✅ Face detection techniques
- ✅ CNN-based classification
- ✅ Image preprocessing pipelines
- ✅ Real-time computer vision
- ✅ How to use pre-trained models
- ✅ AI project structuring


# 📚 What You Will Learn from This Project

-
-  “Learning by building is the best form of learning.”
  
- Practical Deep Learning
- Computer Vision fundamentals
- Model inference (not training)
- AI project deployment basics
- Debugging ML dependencies

# 🔮 Future Enhancements

- ✨ Emotion detection
- ✨ Mobile camera support

# 👨‍💻 Developer
## Sarfraz Khan
- 🎓 Data Science & AI Enthusiast
- 💡 Passionate about Deep Learning & Computer Vision

- “Turning ideas into intelligent systems.”
