import os
import cv2
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve, auc
)
from concrete.ml.sklearn import LogisticRegression

IMAGE_SIZE = (32, 32)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CRIMINAL_DIR = os.path.join(BASE_DIR, "dataset", "Criminal")
GENERAL_DIR = os.path.join(BASE_DIR, "dataset", "General")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DRONE_INPUT_DIR = os.path.join(RESULTS_DIR, "drone")
SAMPLES_DIR = os.path.join(RESULTS_DIR, "samples")
SUMMARY_DIR = os.path.join(RESULTS_DIR, "summary")
TIMING_DIR = os.path.join(RESULTS_DIR, "timing")

SAMPLE_DRONE_IMG = os.path.join(BASE_DIR, "sample_images", "sample_drone.jpg")

CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

model = None  # Global model variable

def clean_results_folder(folder=RESULTS_DIR):
    if not os.path.exists(folder):
        return
    for entry in os.listdir(folder):
        full_path = os.path.join(folder, entry)
        if os.path.isdir(full_path):
            for root, dirs, files in os.walk(full_path, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(full_path)

def load_images_from_folder(folder, label, size=IMAGE_SIZE):
    data = []
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        img = cv2.imread(path)
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            if len(faces) == 0:
                continue
            x, y, w, h = sorted(faces, key=lambda b: b[2]*b[3], reverse=True)[0]
            face = gray[y:y+h, x:x+w]
            resized = cv2.resize(face, size)
            data.append((resized.flatten(), label))
    return data

def train_and_compile_model(progress_callback=None):
    global model

    criminal_data = load_images_from_folder(CRIMINAL_DIR, label=1)
    general_data = load_images_from_folder(GENERAL_DIR, label=0)
    combined = criminal_data + general_data
    if len(combined) == 0:
        raise RuntimeError("No images loaded! Check dataset folders.")

    X, y = zip(*combined)
    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    model = LogisticRegression(n_bits=6)
    model.fit(X, y)

    if progress_callback:
        progress_callback("Compiling model (this may take several minutes)...")

    model.compile(X)

    if progress_callback:
        progress_callback("Training and compilation finished.")

def predict_image_plaintext(image_array):
    global model
    if model is None:
        raise RuntimeError("Model not trained yet. Call train_and_compile_model() first.")
    x = image_array.flatten().astype(np.float32).reshape(1, -1)
    return model.predict(x)[0]

def predict_image_fhe(image_array):
    global model
    if model is None:
        raise RuntimeError("Model not trained yet. Call train_and_compile_model() first.")
    x = image_array.flatten().astype(np.float32).reshape(1, -1)
    x_q = model.quantize_input(x)
    logit = model.fhe_circuit.encrypt_run_decrypt(x_q)[0]
    return 1 if logit > 0 else 0

def preprocess_drone_image(image_path, output_dir=DRONE_INPUT_DIR, output_file="transferred_image.npy"):
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Drone image not found or unreadable at {image_path}")
        return None
    gray = cv2.cvtColor(cv2.resize(img, IMAGE_SIZE), cv2.COLOR_BGR2GRAY)
    array = gray.flatten().astype(np.float32)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)
    np.save(output_path, array)
    print(f"📡 Drone image processed and saved as {output_path}")
    return output_path

def plot_roc_curve(y_true, y_scores, output_path):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(output_path)
    plt.close()
    print(f"📊 Saved ROC curve to {output_path}")

def print_performance(plain_acc, fhe_acc, fhe_time, total, correct):
    print("\n================ Performance Summary ================\n")
    print(f"Plaintext Accuracy: {plain_acc:.2f}%")
    print(f"FHE Accuracy: {correct}/{total} = {fhe_acc:.2f}%")
    print(f"Average Encrypted Inference Time: {fhe_time:.3f} sec\n")
    print("=====================================================\n")

def main():
    print("🧹 Cleaning previous results...")
    clean_results_folder()

    os.makedirs(SAMPLES_DIR, exist_ok=True)
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    os.makedirs(TIMING_DIR, exist_ok=True)
    os.makedirs(DRONE_INPUT_DIR, exist_ok=True)

    print("📥 Loading images and training model...")
    train_and_compile_model()

    criminal_data = load_images_from_folder(CRIMINAL_DIR, label=1)
    general_data = load_images_from_folder(GENERAL_DIR, label=0)
    combined = criminal_data + general_data
    X, y = zip(*combined)
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_pred = [predict_image_plaintext(X_test[i].reshape(IMAGE_SIZE)) for i in range(len(X_test))]
    print("\n📊 Plaintext Evaluation:\n")
    print(classification_report(y_test, y_pred, target_names=["General", "Criminal"]))
    plain_acc = accuracy_score(y_test, y_pred) * 100

    fhe_times = []
    correct, fhe_preds = 0, []
    total = len(X_test)
    for i in range(total):
        x_sample = X_test[i].reshape(IMAGE_SIZE)
        start = time.time()
        pred = predict_image_fhe(x_sample)
        end = time.time()
        fhe_times.append(end - start)
        fhe_preds.append(pred)
        if pred == y_test[i]:
            correct += 1
    fhe_acc = correct / total * 100
    avg_fhe_time = sum(fhe_times) / len(fhe_times)

    summary_path = os.path.join(SUMMARY_DIR, "performance_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Plaintext Accuracy: {plain_acc:.2f}%\n")
        f.write(f"FHE Accuracy: {correct}/{total} = {fhe_acc:.2f}%\n")
        f.write(f"Average FHE Inference Time (sec): {avg_fhe_time:.3f}\n")

    predictions_csv = os.path.join(SUMMARY_DIR, "predictions.csv")
    with open(predictions_csv, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Index", "True Label", "FHE Prediction"])
        for i in range(total):
            writer.writerow([i, y_test[i], fhe_preds[i]])

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["General", "Criminal"])
    disp.plot()
    plt.title("Confusion Matrix (Plaintext)")
    cm_path = os.path.join(RESULTS_DIR, "confusion_plaintext.png")
    plt.savefig(cm_path)
    plt.close()

    plt.figure(figsize=(6,4))
    plt.bar(["Plaintext", "FHE"], [plain_acc, fhe_acc], color=["green", "blue"])
    plt.ylabel("Accuracy (%)")
    plt.title("Plaintext vs FHE Accuracy")
    plt.ylim(0, 100)
    acc_comp_path = os.path.join(RESULTS_DIR, "accuracy_comparison.png")
    plt.savefig(acc_comp_path)
    plt.close()

    try:
        y_scores = model.decision_function(X_test)
        roc_path = os.path.join(RESULTS_DIR, "roc_curve.png")
        plot_roc_curve(y_test, y_scores, roc_path)
    except Exception as e:
        print(f"⚠️ Could not generate ROC curve: {e}")

    for i in range(min(5, total)):
        label = "Criminal" if y_test[i] else "General"
        pred_label = "Criminal" if y_pred[i] else "General"
        img = X_test[i].reshape(IMAGE_SIZE)
        sample_path = os.path.join(SAMPLES_DIR, f"sample_{i}_true_{label}_pred_{pred_label}.png")
        cv2.imwrite(sample_path, img)

    print("\n🔍 Misclassifications (Plaintext vs FHE):")
    mismatch_count = 0
    for i in range(total):
        if y_pred[i] != fhe_preds[i]:
            print(f"Index {i}: Plaintext={y_pred[i]}, FHE={fhe_preds[i]}, True={y_test[i]}")
            mismatch_count += 1
    if mismatch_count == 0:
        print("No mismatches found.")

    drone_npy_path = preprocess_drone_image(SAMPLE_DRONE_IMG)
    if drone_npy_path and os.path.exists(drone_npy_path):
        print("\n📥 Received image from drone...")
        drone_img = np.load(drone_npy_path).reshape(IMAGE_SIZE)
        drone_pred = predict_image_fhe(drone_img)
        drone_pred_label = "Criminal" if drone_pred else "General"
        print(f"🧠 Encrypted Drone Prediction: {drone_pred_label}")
    else:
        print("⚠️ No valid drone image found or preprocessing failed.")

    timing_path = os.path.join(TIMING_DIR, "fhe_inference_times.csv")
    with open(timing_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Sample Index", "FHE Inference Time (sec)"])
        for i, t in enumerate(fhe_times):
            writer.writerow([i, f"{t:.5f}"])

    print_performance(plain_acc, fhe_acc, avg_fhe_time, total, correct)

if __name__ == "__main__":
    main()
