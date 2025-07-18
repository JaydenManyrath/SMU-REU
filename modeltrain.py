#!/usr/bin/env python
"""
Criminal vs General: Train→Save→Webcam Inference (fixed conf formatting)
"""
import os, sys, time, json, random, logging
import cv2
import numpy as np
from sklearn.utils.extmath import randomized_svd
from concrete.ml.sklearn import LinearSVC
from concrete.ml.common.serialization.loaders import load

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
RESIZED      = (128, 128)
SVD_K        = 32
C_PARAM      = 0.01
N_BITS       = 6
THRESHOLD    = 0.0
RANDOM_STATE = 42

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "dataset")
CR_DIR      = os.path.join(DATA_DIR, "Criminal")
GN_DIR      = os.path.join(DATA_DIR, "General")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODEL_DIR   = os.path.join(RESULTS_DIR, "fhe_model")
MODEL_PATH  = os.path.join(MODEL_DIR, "model.json")
CASCADE     = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# ─── LOGGING ────────────────────────────────────────────────────────────────────
os.makedirs(RESULTS_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(RESULTS_DIR, "run.log"))
    ]
)
logger = logging.getLogger()

# ─── HELPERS ────────────────────────────────────────────────────────────────────
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)

def apply_svd(img, k=SVD_K):
    U, S, VT = randomized_svd(
        img, n_components=k, n_iter=5, random_state=RANDOM_STATE
    )
    return U @ np.diag(S) @ VT

def non_max_suppression(boxes, thresh=0.3):
    if len(boxes)==0: return []
    x1,y1 = boxes[:,0], boxes[:,1]
    x2 = boxes[:,0]+boxes[:,2]; y2 = boxes[:,1]+boxes[:,3]
    areas = (x2-x1+1)*(y2-y1+1)
    order = np.argsort(y2)
    keep=[]
    while order.size:
        i = order[-1]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[:-1]])
        yy1 = np.maximum(y1[i], y1[order[:-1]])
        xx2 = np.minimum(x2[i], x2[order[:-1]])
        yy2 = np.minimum(y2[i], y2[order[:-1]])
        w = np.maximum(0, xx2-xx1+1); h = np.maximum(0, yy2-yy1+1)
        inter = w*h
        ovr = inter / (areas[order[:-1]] + areas[i] - inter)
        order = np.delete(
            order,
            np.concatenate(
                ([order.size-1], np.where(ovr>thresh)[0])
            )
        )
    return boxes[keep].astype(int)

def load_images(folder, label):
    det = cv2.CascadeClassifier(CASCADE)
    data=[]
    for fn in sorted(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder, fn))
        if img is None: continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dets = det.detectMultiScale(gray,1.05,5, minSize=(50,50))
        dets = np.array(dets) if len(dets) else np.empty((0,4),dtype=int)
        dets = non_max_suppression(dets)
        for x,y,w,h in dets:
            face = cv2.resize(gray[y:y+h, x:x+w], RESIZED)
            data.append((face.flatten(), label))
    return data

# ─── TRAIN & SAVE ───────────────────────────────────────────────────────────────
def train_and_save():
    crim = load_images(CR_DIR, 1)
    gen  = load_images(GN_DIR, 0)
    logger.info(f"Loaded {len(crim)} criminal, {len(gen)} general images")
    data = crim + gen
    X,y = zip(*data); X,y = np.array(X), np.array(y)

    X_svd = np.array([
        apply_svd(x.reshape(RESIZED), SVD_K).flatten()
        for x in X
    ])

    model = LinearSVC(
        n_bits=N_BITS, C=C_PARAM,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )
    logger.info("⏳ Training model...")
    model.fit(X_svd, y)
    logger.info("⏳ Compiling for FHE...")
    model.compile(X_svd)
    logger.info("✅ Done.")

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "w") as f:
        model.dump(f)
    logger.info(f"Model dumped to {MODEL_PATH}")
    return model

# ─── REAL‑TIME WEBCAM ────────────────────────────────────────────────────────────
def realtime_inference(model):
    cap = cv2.VideoCapture(0)
    det = cv2.CascadeClassifier(CASCADE)
    if not cap.isOpened():
        logger.error("Cannot open webcam"); return

    logger.info("▶ Webcam inference started. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dets = det.detectMultiScale(gray,1.05,5, minSize=(50,50))
        dets = np.array(dets) if len(dets) else np.empty((0,4),dtype=int)
        boxes = non_max_suppression(dets)

        for x,y,w,h in boxes:
            crop = cv2.resize(gray[y:y+h, x:x+w], RESIZED)
            svd_f = apply_svd(crop, SVD_K).flatten()
            xq = model.quantize_input(svd_f.reshape(1,-1).astype(np.float32))
            t0 = time.time()
            logit = model.fhe_circuit.encrypt_run_decrypt(xq)[0]
            t_fhe = time.time() - t0
            label = "Criminal" if logit>THRESHOLD else "General"
            conf  = float(1/(1+np.exp(-logit)))   # <-- cast to float
            color = (0,0,255) if label=="Criminal" else (0,255,0)
            cv2.rectangle(frame, (x,y),(x+w,y+h), color, 2)
            cv2.putText(
                frame,
                f"{label} ({conf:.2f})",
                (x,y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        cv2.imshow("Webcam: Criminal vs General", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.info("▶ Webcam inference ended.")

# ─── MAIN ────────────────────────────────────────────────────────────────────────
def main():
    seed_everything(RANDOM_STATE)

    if os.path.exists(MODEL_PATH):
        logger.info(f"⏳ Loading model from {MODEL_PATH}")
        with open(MODEL_PATH, "r") as f:
            model = load(f)
        logger.info("⏳ Recompiling FHE circuit...")
        dim = RESIZED[0]*RESIZED[1]
        model.compile(np.zeros((1, dim), dtype=np.float32))
        logger.info("✅ Model loaded & compiled.")
    else:
        model = train_and_save()

    realtime_inference(model)

if __name__ == "__main__":
    main()
