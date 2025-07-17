# debug_live_classifier.py
#!/usr/bin/env python
import os
import cv2
import numpy as np
from concrete.ml.deployment import FHEModelClient, FHEModelServer

# ——— CONFIG —————————————————————————————————————
BASE_DIR     = os.path.dirname(__file__)
MODEL_DIR    = os.path.join(BASE_DIR, "models", "fhe_criminal_detector")
DATA_DIR     = os.path.join(BASE_DIR, "dataset")
CRIMINAL_DIR = os.path.join(DATA_DIR, "Criminal")
GENERAL_DIR  = os.path.join(DATA_DIR, "General")
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
RESIZED      = (24, 24)
# ——————————————————————————————————————————————

# load k
with open(os.path.join(MODEL_DIR, "svd_k.txt")) as f:
    svd_k = int(f.read().strip())

# spin up FHE client/server
client    = FHEModelClient(path_dir=MODEL_DIR, key_dir=MODEL_DIR)
eval_keys = client.get_serialized_evaluation_keys()
server    = FHEModelServer(path_dir=MODEL_DIR)
server.load()

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

def apply_svd(img, k):
    U, S, VT = np.linalg.svd(img, full_matrices=False)
    k = min(k, len(S))
    return (U[:, :k] @ np.diag(S[:k]) @ VT[:k, :])

def fhe_predict(face_flat):
    # 1) encrypt
    enc       = client.quantize_encrypt_serialize(face_flat.reshape(1, -1))
    # 2) homomorphic eval
    enc_res   = server.run(enc, eval_keys)
    # 3) decrypt
    res_array = client.deserialize_decrypt_dequantize(enc_res)
    # pull scalar logit
    logit     = float(np.asarray(res_array).flatten()[0])
    # confidence
    confidence= 1/(1+np.exp(-logit))
    # debug print
    print(f"[DEBUG] logit={logit:.3f}, conf={confidence:.3f}")
    return "Criminal" if logit > 0 else "General"

# 1️⃣ Quick test on dataset samples
def quick_dataset_test():
    for folder, label in [(CRIMINAL_DIR, "Criminal"), (GENERAL_DIR, "General")]:
        print(f"\nTesting {label} samples:")
        for fname in os.listdir(folder)[:3]:
            img = cv2.imread(os.path.join(folder, fname))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)
            if not len(faces):
                print("  no face:", fname); continue
            x,y,w,h = faces[0]
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, RESIZED)
            comp = apply_svd(face, svd_k).flatten().astype(np.float32)
            pred = fhe_predict(comp)
            print(f"  {fname}: predicted={pred}")

if __name__ == "__main__":
    quick_dataset_test()

    # 2️⃣ Then go live
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")
    print("\nPress ‘q’ to quit.")
    while True:
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, RESIZED)
            comp = apply_svd(face, svd_k).flatten().astype(np.float32)
            label = fhe_predict(comp)
            color = (0, 0, 255) if label=="Criminal" else (0,255,0)
            cv2.rectangle(frame, (x,y),(x+w,y+h), color, 2)
            cv2.putText(frame, label, (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow("DEBUG FHE+SVD", frame)
        if cv2.waitKey(1)&0xFF==ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
