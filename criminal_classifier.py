#!/usr/bin/env python
"""
Criminal vs General Inference (128×128, k=32)
  • Trains on full dataset with SVD k=32 on 128×128 crops
  • LinearSVC(C=0.01, n_bits=6)
  • Decision threshold = 0.0
  • Tighter Haar cascade + Non‑Max Suppression
  • Outputs group_analysis.png + group_results.json
"""
import os
import sys
import time
import json
import random
import logging

import cv2
import numpy as np
from sklearn.utils.extmath import randomized_svd
from concrete.ml.sklearn import LinearSVC

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
RESIZED       = (128, 128)   # Increased resolution
SVD_K         = 32           # Top‑32 singular values
C_PARAM       = 0.01         # SVC regularization
N_BITS        = 6            # FHE quantization bits
THRESHOLD     = 0.0          # SVM decision boundary
RANDOM_STATE  = 42

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(BASE_DIR, "dataset")
CRIMINAL_DIR  = os.path.join(DATA_DIR, "Criminal")
GENERAL_DIR   = os.path.join(DATA_DIR, "General")
RESULTS_DIR   = os.path.join(BASE_DIR, "results")
GROUP_IMG     = os.path.join(BASE_DIR, "sample_images", "diddy.jpg")

CASCADE_FRONTAL = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
CASCADE_PROFILE = cv2.data.haarcascades + "haarcascade_profileface.xml"

# ─── LOGGING ────────────────────────────────────────────────────────────────────
os.makedirs(RESULTS_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(RESULTS_DIR, "inference.log"))
    ]
)
logger = logging.getLogger()

# ─── HELPERS ────────────────────────────────────────────────────────────────────
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)

def apply_svd(img, k=SVD_K):
    U, S, VT = randomized_svd(img, n_components=k, n_iter=5, random_state=RANDOM_STATE)
    return U @ np.diag(S) @ VT

def non_max_suppression(boxes, overlap_thresh=0.3):
    if len(boxes) == 0: return []
    x1 = boxes[:,0]; y1 = boxes[:,1]
    x2 = boxes[:,0] + boxes[:,2]; y2 = boxes[:,1] + boxes[:,3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = np.argsort(y2)
    keep = []
    while order.size:
        i = order[-1]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[:-1]])
        yy1 = np.maximum(y1[i], y1[order[:-1]])
        xx2 = np.minimum(x2[i], x2[order[:-1]])
        yy2 = np.minimum(y2[i], y2[order[:-1]])
        w = np.maximum(0, xx2-xx1+1); h = np.maximum(0, yy2-yy1+1)
        inter = w*h
        ovr = inter / (areas[order[:-1]] + areas[i] - inter)
        order = np.delete(order, np.concatenate(([order.size-1], np.where(ovr>overlap_thresh)[0])))
    return boxes[keep].astype(int)

def load_images(folder, label):
    data = []
    fr = cv2.CascadeClassifier(CASCADE_FRONTAL)
    pr = cv2.CascadeClassifier(CASCADE_PROFILE)
    for fn in sorted(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder, fn))
        if img is None: continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dets = []
        dets += list(fr.detectMultiScale(gray, 1.05, 5, minSize=(50,50)))
        dets += list(pr.detectMultiScale(gray, 1.05, 5, minSize=(50,50)))
        dets = np.array(dets) if dets else np.empty((0,4),dtype=int)
        dets = non_max_suppression(dets)
        for (x,y,w,h) in dets:
            face = cv2.resize(gray[y:y+h, x:x+w], RESIZED)
            data.append((face.flatten(), label))
    return data

def train_and_compile(X, y):
    model = LinearSVC(
        n_bits=N_BITS,
        C=C_PARAM,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )
    logger.info(json.dumps({"event":"train_start","C":C_PARAM}))
    model.fit(X, y)
    logger.info(json.dumps({"event":"compile_start"}))
    model.compile(X)
    logger.info(json.dumps({"event":"train_compile_done"}))
    return model

def predict_fhe(model, x):
    xq = model.quantize_input(x.reshape(1,-1).astype(np.float32))
    t0 = time.time()
    logit = model.fhe_circuit.encrypt_run_decrypt(xq)[0]
    return float(logit), float(time.time()-t0)

def analyze_group(model):
    if not os.path.exists(GROUP_IMG):
        logger.info("No sample_group.jpg; skipping group analysis.")
        return

    img  = cv2.imread(GROUP_IMG)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fr, pr = (cv2.CascadeClassifier(CASCADE_FRONTAL),
              cv2.CascadeClassifier(CASCADE_PROFILE))

    dets = []
    dets += list(fr.detectMultiScale(gray,1.05,5,minSize=(50,50)))
    dets += list(pr.detectMultiScale(gray,1.05,5,minSize=(50,50)))
    dets = np.array(dets) if dets else np.empty((0,4),dtype=int)
    boxes = non_max_suppression(dets)

    results = []
    for i,(x,y,w,h) in enumerate(boxes):
        face = cv2.resize(gray[y:y+h, x:x+w], RESIZED)
        svd_face = apply_svd(face).flatten()
        logit, t_fhe = predict_fhe(model, svd_face)
        label = "Criminal" if logit>THRESHOLD else "General"
        conf  = 1/(1+np.exp(-logit))
        results.append({
            "face_id":    i+1,
            "bbox":       [int(x),int(y),int(w),int(h)],
            "label":      label,
            "confidence": conf,
            "fhe_time_s": t_fhe
        })
        color = (0,0,255) if label=="Criminal" else (0,255,0)
        cv2.rectangle(img, (x,y),(x+w,y+h), color, 2)
        cv2.putText(img, f"{label} ({conf:.2f})", (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    out_img  = os.path.join(BASE_DIR, "group_analysis.png")
    out_json = os.path.join(BASE_DIR, "group_results.json")
    cv2.imwrite(out_img, img)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Group analysis saved: {out_img}, {out_json}")

# ─── MAIN ────────────────────────────────────────────────────────────────────────
def main():
    seed_everything(RANDOM_STATE)

    # Load full dataset
    crim = load_images(CRIMINAL_DIR, 1)
    gen  = load_images(GENERAL_DIR,  0)
    logger.info(json.dumps({
        "event":"data_stats",
        "n_criminal": len(crim),
        "n_general":  len(gen)
    }))
    data = crim + gen
    X,y  = zip(*data)
    X, y = np.array(X), np.array(y)

    # Build SVD features & train
    X_svd = np.array([apply_svd(x.reshape(RESIZED), SVD_K).flatten() for x in X])
    model = train_and_compile(X_svd, y)

    # Sanity check first few samples
    for sample, true in zip(X_svd[:5], y[:5]):
        logit, t = predict_fhe(model, sample)
        logger.info(f"Sample true={true} logit={logit:.2f} pred={int(logit>THRESHOLD)} t={t:.3f}s")

    # Run group analysis
    analyze_group(model)
    logger.info("✅ Inference complete. See results in ‘results/’ directory.")

if __name__ == "__main__":
    main()
