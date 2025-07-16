#!/usr/bin/env python
"""
This script trains a facial recognition model to classify images as 'Criminal' or 'General'.
It uses Singular Value Decomposition (SVD) for image compression and Concrete-ML for
performing predictions using Fully Homomorphic Encryption (FHE).

The core logic has been corrected to prevent data leakage by splitting the dataset into
training and testing sets *before* model training, ensuring a valid evaluation of
model performance on unseen data.
"""
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve, auc
)
from concrete.ml.sklearn import LinearSVC

# --- Constants ---
IMAGE_SIZE = (32, 32)
RESIZED_IMAGE_SIZE = (24, 24)

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

# --- Global Variables & Initializations ---
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
model = None  # Global model variable

# --- Helper Functions ---

def clean_results_folder(folder=RESULTS_DIR):
    """Recursively cleans the results folder."""
    if not os.path.exists(folder):
        os.makedirs(folder)
        return
    for entry in os.listdir(folder):
        full_path = os.path.join(folder, entry)
        if os.path.isdir(full_path):
            # Deeper cleaning for subdirectories
            for root, dirs, files in os.walk(full_path, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(full_path)
        else:
            os.remove(full_path)

def apply_svd(image_array, k):
    """
    Apply SVD to a grayscale image and reconstruct it using the top-k singular values.
    Returns a rank-k approximation of the original image.
    """
    U, S, VT = np.linalg.svd(image_array, full_matrices=False)
    k = min(k, len(S))
    S_reduced = np.diag(S[:k])
    U_reduced = U[:, :k]
    VT_reduced = VT[:k, :]
    compressed = np.dot(U_reduced, np.dot(S_reduced, VT_reduced))
    return np.clip(compressed, 0, 255)

def load_images_from_folder(folder, label, size=RESIZED_IMAGE_SIZE, svd_k=50):
    """
    Loads images, detects the largest face, resizes it, and compresses it using SVD.
    The result is a flattened, SVD-compressed grayscale image used for training or inference.
    """
    data = []
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        img = cv2.imread(path)
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            if len(faces) == 0:
                continue
            # Get the largest face detected
            x, y, w, h = sorted(faces, key=lambda b: b[2]*b[3], reverse=True)[0]
            face = gray[y:y+h, x:x+w]
            resized = cv2.resize(face, size)
            compressed = apply_svd(resized, svd_k)
            data.append((compressed.flatten(), label))
    return data

def train_and_compile_model(X, y, progress_callback=None):
    """Trains a LinearSVC Regression model and compiles it for FHE."""
    global model
    model = LinearSVC(n_bits=6)
    model.fit(X, y)
    if progress_callback:
        progress_callback("Compiling model (this may take several minutes)...")
    model.compile(X)
    if progress_callback:
        progress_callback("Training and compilation finished.")

def predict_image_plaintext(image_array):
    """Predicts using the simulated (plaintext) FHE model."""
    global model
    if model is None:
        raise RuntimeError("Model not trained yet. Call train_and_compile_model() first.")
    x = image_array.flatten().astype(np.float32).reshape(1, -1)
    return model.predict(x)[0]

def predict_image_fhe(image_array):
    """Predicts using the actual FHE circuit."""
    global model
    if model is None:
        raise RuntimeError("Model not trained yet. Call train_and_compile_model() first.")
    x = image_array.flatten().astype(np.float32).reshape(1, -1)
    x_q = model.quantize_input(x)
    logit = model.fhe_circuit.encrypt_run_decrypt(x_q)[0]
    return 1 if logit > 0 else 0

def preprocess_drone_image(image_path, output_dir=DRONE_INPUT_DIR, output_file="transferred_image.npy"):
    """Simulates receiving and processing a drone image for FHE inference."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Drone image not found or unreadable at {image_path}")
        return None
    gray = cv2.cvtColor(cv2.resize(img, RESIZED_IMAGE_SIZE), cv2.COLOR_BGR2GRAY)
    array = gray.flatten().astype(np.float32)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)
    np.save(output_path, array)
    print(f"📡 Drone image processed and saved as {output_path}")
    return output_path

def plot_roc_curve(y_true, y_scores, output_path):
    """Plots and saves the ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(output_path)
    plt.close()
    print(f"📊 Saved ROC curve to {output_path}")

# --- Main Execution Logic ---

def main():
    print("🧹 Cleaning previous results...")
    clean_results_folder()

    os.makedirs(SAMPLES_DIR, exist_ok=True)
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    os.makedirs(TIMING_DIR, exist_ok=True)
    os.makedirs(DRONE_INPUT_DIR, exist_ok=True)

    k_values = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
    summary_rows = []

    print("📥 Loading and preparing dataset...")
    criminal_data = load_images_from_folder(CRIMINAL_DIR, label=1)
    general_data = load_images_from_folder(GENERAL_DIR, label=0)
    combined = criminal_data + general_data
    if len(combined) == 0:
        raise RuntimeError("No images loaded! Check dataset folders and image paths.")

    X_all, y_all = zip(*combined)
    X_all = np.array(X_all, dtype=np.float32)
    y_all = np.array(y_all)

    print(" splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )

    for svd_k in k_values:
        print(f"\n🚀 Running experiment for SVD k = {svd_k}")

        criminal_data_k = load_images_from_folder(CRIMINAL_DIR, label=1, svd_k=svd_k)
        general_data_k = load_images_from_folder(GENERAL_DIR, label=0, svd_k=svd_k)
        combined_k = criminal_data_k + general_data_k
        X_k, y_k = zip(*combined_k)
        X_k, y_k = np.array(X_k), np.array(y_k)

        X_train_k, X_test_k, y_train_k, y_test_k = train_test_split(
            X_k, y_k, test_size=0.2, random_state=42, stratify=y_k
        )

        print("💪 Training and compiling model on the training set...")
        train_and_compile_model(X_train_k, y_train_k)

        print("\n📊 Evaluating model on the test set (using FHE)...")
        fhe_times = []
        fhe_predictions = []
        total = len(X_test_k)

        for i in range(total):
            x_sample = X_test_k[i]
            start = time.time()
            pred = predict_image_fhe(x_sample)
            end = time.time()
            fhe_times.append(end - start)
            fhe_predictions.append(pred)
            print(f"  Processed {i + 1}/{total}", end='\r')

        print("\n")
        fhe_acc = accuracy_score(y_test_k, fhe_predictions) * 100
        correct = np.sum(np.array(fhe_predictions) == y_test_k)
        avg_fhe_time = sum(fhe_times) / len(fhe_times) if fhe_times else 0

        print("\n📊 Evaluation Report (FHE on Test Set):\n")
        print(classification_report(y_test_k, fhe_predictions, target_names=["General", "Criminal"]))

        print("\n================ Performance Summary ================\n")
        print(f"Model Accuracy (on unseen data): {correct}/{total} = {fhe_acc:.2f}%")
        print(f"Average Encrypted Inference Time: {avg_fhe_time:.3f} sec\n")
        print("=====================================================\n")

        print("\n🔍 Evaluating model on the test set (plaintext)...")
        start_plain = time.time()
        plain_predictions = [predict_image_plaintext(x) for x in X_test_k]
        end_plain = time.time()
        plain_acc = accuracy_score(y_test_k, plain_predictions) * 100
        plain_latency = (end_plain - start_plain) / len(X_test_k)

        summary_rows.append([
            svd_k,
            f"{plain_acc:.2f}",
            f"{fhe_acc:.2f}",
            f"{avg_fhe_time:.4f}"
        ])

    # --- Summary Table ---
    print("\n📊 Model Evaluation Summary:")
    print("-" * 83)
    print(f"{'k':>5} | {'Plain Accuracy (%)':>20} | {'FHE Accuracy (%)':>20} | {'Avg FHE Time (s)':>20}")
    print("-" * 83)
    for row in summary_rows:
        print(f"{row[0]:>5} | {row[1]:>20} | {row[2]:>20} | {row[3]:>20}")
    print("-" * 83)

    # --- Plotting Section ---
    k_vals = [row[0] for row in summary_rows]
    accuracies = [float(row[2]) for row in summary_rows]
    inference_times = [float(row[3]) for row in summary_rows]

    fig, ax1 = plt.subplots()

    color1 = 'tab:blue'
    ax1.set_xlabel('SVD k-value')
    ax1.set_ylabel('Accuracy (%)', color=color1)
    ax1.plot(k_vals, accuracies, marker='o', color=color1, label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Avg FHE Inference Time (s)', color=color2)
    ax2.plot(k_vals, inference_times, marker='s', color=color2, label='Inference Time')
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title('FHE Inference Time vs Accuracy for Varying SVD k-values')
    fig.tight_layout()
    plt.savefig(os.path.join(SUMMARY_DIR, "svd_k_comparison.png"))
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(k_vals, accuracies, marker='o', linestyle='-', color='purple', linewidth=2)
    plt.title('Effect of SVD k-value on Classification Accuracy')
    plt.xlabel('SVD k-value (Number of Singular Values Kept)')
    plt.ylabel('Classification Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(k_vals)
    plt.tight_layout()
    svd_exp_plot_path = os.path.join(SUMMARY_DIR, "svd_compression_accuracy.png")
    plt.savefig(svd_exp_plot_path)
    plt.show()
    print(f"📊 Saved SVD Compression Experimentation plot to {svd_exp_plot_path}")


if __name__ == "__main__":
    main()
