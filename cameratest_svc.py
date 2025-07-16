#!/usr/bin/env python3
import os
import cv2
import numpy as np
from concrete.ml.sklearn import LinearSVC
import joblib  # for saving/loading sklearn models

# Constants
RESIZED_IMAGE_SIZE = (24, 24)
SVD_K = 10
MODEL_PATH = "crime_svc.joblib"        # where you’ll save your trained model
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# 1) TRAIN & SAVE (run once, then comment out or skip)
def apply_svd(image_array, k):
    U, S, VT = np.linalg.svd(image_array, full_matrices=False)
    k = min(k, len(S))
    return np.clip((U[:, :k] * S[:k]) @ VT[:k, :], 0, 255)

def load_and_prepare(folder, label):
    data, labels = [], []
    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    for fname in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, fname))
        if img is None: continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 5)
        if not len(faces): continue
        x,y,w,h = max(faces, key=lambda b: b[2]*b[3])
        crop = cv2.resize(gray[y:y+h, x:x+w], RESIZED_IMAGE_SIZE)
        comp = apply_svd(crop, SVD_K)
        data.append(comp.flatten()); labels.append(label)
    return np.array(data, dtype=np.float32), np.array(labels)

def train_and_save():
    X1, y1 = load_and_prepare("dataset/Criminal", label=1)
    X0, y0 = load_and_prepare("dataset/General",  label=0)
    X, y = np.vstack((X1, X0)), np.concatenate((y1, y0))
    from sklearn.model_selection import train_test_split
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    svc = LinearSVC(n_bits=6)
    svc.fit(X_train, y_train)
    joblib.dump(svc, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

# Uncomment to train:
#train_and_save()
#exit()

# 2) LIVE INFERENCE
model: LinearSVC = joblib.load(MODEL_PATH)
cascade = cv2.CascadeClassifier(CASCADE_PATH)
cap = cv2.VideoCapture(0)   # 0 = default webcam

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

print("Press ESC to exit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.1, 5)
    for (x,y,w,h) in faces:
        # take the largest face only (optional)
        face = gray[y:y+h, x:x+w]
        resized = cv2.resize(face, RESIZED_IMAGE_SIZE)
        compressed = apply_svd(resized, SVD_K).flatten().astype(np.float32).reshape(1, -1)
        pred = model.predict(compressed)[0]
        label = "Criminal" if pred==1 else "General"
        color = (0,0,255) if pred==1 else (0,255,0)
        cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
        cv2.putText(frame, label, (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Real‑Time Face Classification", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
