BLOOD GROUP DETECTION USING FINGERPRINTS
This repository contains the source code, training notebook, and demo system for my **Final Year Project (FYP)**:  
a **deep learning–based system** that predicts **blood groups from fingerprint images** using EfficientNet CNNs.  
It includes **model training**, a trained model, and a **Flask-based demo UI**.

Project Overview
- **Goal**: Predict human blood groups using fingerprint images in a non-invasive manner.  
- **Dataset**: SOCOFing fingerprint dataset (synthetically labeled).  
- **Model**: EfficientNetB0 trained on 6,000+ fingerprint images across 8 blood groups.  
- **Accuracy**: Achieved **90.33% test accuracy**.  
- **Demo System**: Flask backend + HTML/CSS/JS frontend with a cyberpunk dark theme.  




## 🛠 Tech Stack
- **Programming**: Python 3.10+, TensorFlow/Keras, scikit-learn, OpenCV  
- **Frontend**: HTML, CSS, JavaScript  
- **Backend**: Flask (API + templates)  
- **Visualization**: Matplotlib, Seaborn  
- **Deployment**: Local demo (Docker/Render compatible)  

 Training

* Open `Model Training and Testing Code.ipynb` in Jupyter.
* Requires the SOCOFing dataset (not included due to size).
* Trains EfficientNetB0 with preprocessing, normalization, and dropout.
* Evaluated with accuracy, precision/recall, confusion matrix.

---

 Features

* End-to-end ML pipeline: dataset → training → evaluation → deployment.
* Flask demo app with cyberpunk dark-themed UI.
* Easy to run locally (requirements + model file).
* Extendable to MLOps tools (Docker, MLflow, GitHub Actions).

---

 Results

* **Test Accuracy**: 90.33%
* **Model**: EfficientNetB0 (transfer learning)
* **Classes**: 8 blood groups → `['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']`



-

* **Dataset**: [SOCOFing – Sokoto Coventry Fingerprint Dataset](https://www.kaggle.com/datasets/ruizgara/socofing)
