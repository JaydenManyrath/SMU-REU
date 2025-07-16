#!/usr/bin/env python
"""
This script provides a comprehensive framework for training and evaluating an FHE-based
facial recognition model using a Linear Support Vector Classifier (LinearSVC),
designed for drone applications.

Key Features:
- SVD analysis to balance accuracy, inference time, and data payload size.
- Generates a visual comparison of SVD compression levels.
- Dual face detection (frontal + profile) with Non-Maximum Suppression.
- Simulation of real-world drone conditions (motion blur, variable altitude).
- Detailed, per-face analysis with confidence scores and bounding boxes.
- Professional, structured JSON logging for audit and data integration.
"""
import os
import cv2
import time
import json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from concrete.ml.sklearn import LinearSVC, LogisticRegression

# --- Constants ---
RESIZED_IMAGE_SIZE = (24, 24)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CRIMINAL_DIR = os.path.join(BASE_DIR, "dataset", "Criminal")
GENERAL_DIR = os.path.join(BASE_DIR, "dataset", "General")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
SAMPLES_DIR = os.path.join(RESULTS_DIR, "samples")
SUMMARY_DIR = os.path.join(RESULTS_DIR, "summary")

SAMPLE_GROUP_IMG = os.path.join(BASE_DIR, "sample_images", "sample_group.jpg")

# --- Cascade Classifiers ---
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
PROFILE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_profileface.xml'

# --- Global Variables & Initializations ---
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
profile_cascade = cv2.CascadeClassifier(PROFILE_CASCADE_PATH)
model = None  # Global model variable

# --- Helper & Core Functions ---

def clean_results_folder(folder=RESULTS_DIR):
    """Recursively cleans and recreates the results folder."""
    if os.path.exists(folder):
        for root, dirs, files in os.walk(folder, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(folder)
    os.makedirs(folder, exist_ok=True)
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    os.makedirs(SUMMARY_DIR, exist_ok=True)


def apply_svd(image_array, k):
    """Apply SVD and reconstruct with top-k singular values."""
    U, S, VT = np.linalg.svd(image_array, full_matrices=False)
    k = min(k, len(S))
    S_reduced = np.diag(S[:k])
    U_reduced = U[:, :k]
    VT_reduced = VT[:k, :]
    return np.dot(U_reduced, np.dot(S_reduced, VT_reduced))

def generate_svd_visualization(original_image, k_values, optimal_k, output_path):
    """Generates and saves a visual comparison of SVD compression levels."""
    print("🖼️  Generating SVD compression visualization...")
    num_k = len(k_values)
    # Adjust layout to be more compact
    fig, axs = plt.subplots(1, num_k + 1, figsize=(18, 4))
    axs = axs.flatten()
    
    # Display original image
    axs[0].imshow(original_image, cmap='gray')
    axs[0].set_title("Original")
    axs[0].axis('off')
    
    # Display SVD compressed images
    for i, k in enumerate(k_values):
        ax = axs[i+1]
        compressed_img = apply_svd(original_image, k)
        ax.imshow(compressed_img, cmap='gray')
        ax.set_title(f"k = {k}")
        ax.axis('off')
        
        # Highlight the optimal k
        if k == optimal_k:
            for spine in ax.spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(4)
            ax.set_title(f"k = {k} (Optimal)", color='red', fontweight='bold')

    # Turn off any unused subplots
    for i in range(num_k + 1, len(axs)):
        axs[i].axis('off')
        
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✅ SVD visualization saved to {output_path}")

def non_max_suppression(boxes, overlapThresh):
    """Elegant non-maximum suppression to remove overlapping bounding boxes."""
    if len(boxes) == 0: return []
    pick = []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    while len(idxs) > 0:
        last = len(idxs) - 1; i = idxs[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]]); yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]]); yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w, h = np.maximum(0, xx2 - xx1 + 1), np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
    return boxes[pick].astype("int")

def load_images_from_folder(folder, label, size=RESIZED_IMAGE_SIZE):
    """Loads images, detects all faces, and returns them without SVD."""
    data = []
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        img = cv2.imread(path)
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            frontal_faces = face_cascade.detectMultiScale(gray, 1.1, 5)
            for (x, y, w, h) in frontal_faces:
                face = gray[y:y+h, x:x+w]
                resized = cv2.resize(face, size)
                data.append((resized.flatten(), label))
    return data

def train_and_compile_model(X, y, model_class=LinearSVC, n_bits=6):
    """Trains a specified model and compiles it for FHE."""
    global model
    model = model_class(n_bits=n_bits)
    print(f"💪 Training {model.__class__.__name__}...")
    model.fit(X, y)
    print("⚡ Compiling model for FHE (this may take several minutes)...")
    model.compile(X)
    print("✅ Training and compilation finished.")

def predict_image_fhe_with_confidence(image_array):
    """Predicts using FHE and returns label and confidence score (logit)."""
    global model
    x = image_array.flatten().astype(np.float32).reshape(1, -1)
    x_q = model.quantize_input(x)
    logit = model.fhe_circuit.encrypt_run_decrypt(x_q)[0]
    prediction = 1 if logit > 0 else 0
    return prediction, logit

def get_calibrated_confidence(logit):
    """Converts a logit to a probability-like score using the sigmoid function."""
    return 1 / (1 + np.exp(-logit))

def analyze_group_image_fhe(image_path, svd_k, size=RESIZED_IMAGE_SIZE):
    """Comprehensive analysis of a group image using dual detectors and FHE."""
    img = cv2.imread(image_path)
    if img is None: return [], None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    frontal_faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    profile_faces = profile_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
    all_boxes = [box for box in np.concatenate((frontal_faces, profile_faces))]
    all_boxes_coords = [[x, y, x + w, y + h] for x, y, w, h in all_boxes]
    clean_boxes_coords = non_max_suppression(np.array(all_boxes_coords), 0.3)
    results = []
    for i, (x1, y1, x2, y2) in enumerate(clean_boxes_coords):
        w, h = x2 - x1, y2 - y1
        face = gray[y1:y2, x1:x2]
        resized = cv2.resize(face, size)
        compressed = apply_svd(resized, svd_k)
        prediction, logit = predict_image_fhe_with_confidence(compressed)
        results.append({
            "face_id": i + 1, "bounding_box": [int(x1), int(y1), int(w), int(h)],
            "prediction": "Criminal" if prediction == 1 else "General",
            "fhe_logit": float(logit), "confidence": get_calibrated_confidence(logit)
        })
    return results, img

def draw_bounding_boxes(image, results):
    """Draws annotated bounding boxes on an image."""
    output_image = image.copy()
    for res in results:
        x, y, w, h = res['bounding_box']
        label = res['prediction']; confidence = res['confidence']
        color = (0, 0, 255) if label == "Criminal" else (0, 255, 0)
        cv2.rectangle(output_image, (x, y), (x+w, y+h), color, 2)
        cv2.putText(output_image, f"{label} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return output_image

def perturb_image(image, blur_level=0, resolution_scale=1.0):
    """Simulates motion blur and variable altitude for robustness testing."""
    perturbed = image.copy()
    if blur_level > 0: perturbed = cv2.GaussianBlur(perturbed, (blur_level*2+1, blur_level*2+1), 0)
    if resolution_scale < 1.0:
        h, w = perturbed.shape[:2]
        downscaled = cv2.resize(perturbed, (int(w*resolution_scale), int(h*resolution_scale)), cv2.INTER_AREA)
        perturbed = cv2.resize(downscaled, (w, h), cv2.INTER_LINEAR)
    return perturbed

def main():
    print("🧹 Cleaning previous results...")
    clean_results_folder()
    
    # --- Experiment Configuration ---
    k_values = [4, 8, 12, 16, 20, 24]
    model_to_use = LinearSVC  # CORRECTED: Using LinearSVC as requested
    optimal_k_for_demo = 12
    summary_rows = []

    print("📥 Loading and preparing dataset (without SVD)...")
    criminal_data = load_images_from_folder(CRIMINAL_DIR, label=1)
    general_data = load_images_from_folder(GENERAL_DIR, label=0)
    if not criminal_data or not general_data: raise RuntimeError("Dataset empty. Check folders.")
        
    X_all, y_all = zip(*(criminal_data + general_data))
    X_all = np.array(X_all)
    y_all = np.array(y_all)
    
    X_train_orig, X_test_orig, y_train, y_test = train_test_split(X_all, y_all, test_size=0.25, random_state=42, stratify=y_all)
    
    for svd_k in k_values:
        print(f"\n🚀 Running experiment for SVD k = {svd_k}")
        X_train_k = np.array([apply_svd(x.reshape(RESIZED_IMAGE_SIZE), svd_k).flatten() for x in X_train_orig])
        X_test_k = np.array([apply_svd(x.reshape(RESIZED_IMAGE_SIZE), svd_k).flatten() for x in X_test_orig])
        
        train_and_compile_model(X_train_k, y_train, model_class=model_to_use)
        
        fhe_times = []
        preds = []
        for x_sample in X_test_k:
            start_time = time.time()
            pred, _ = predict_image_fhe_with_confidence(x_sample)
            end_time = time.time()
            fhe_times.append(end_time - start_time)
            preds.append(pred)
            
        acc = accuracy_score(y_test, preds) * 100
        avg_fhe_time = np.mean(fhe_times)
        print(f"Accuracy for k={svd_k}: {acc:.2f}% | Avg FHE Inference Time: {avg_fhe_time:.4f}s")
        summary_rows.append([svd_k, acc, avg_fhe_time])

    # --- SVD Visualization ---
    sample_face = X_test_orig[0].reshape(RESIZED_IMAGE_SIZE)
    generate_svd_visualization(sample_face, k_values, optimal_k_for_demo, os.path.join(RESULTS_DIR, "svd_compression_visualization.png"))

    # --- Detailed Group Analysis & Robustness Demonstration ---
    # ... (This section remains unchanged and will now use the trained LinearSVC model) ...
    print("\n" + "="*60)
    print("🕵️‍♂️  Running Detailed Analysis & Robustness Demonstration")
    print("="*60)
    if os.path.exists(SAMPLE_GROUP_IMG):
        # 1. Standard Analysis
        print(f"\n--- Analyzing Standard Image: {os.path.basename(SAMPLE_GROUP_IMG)} ---")
        analysis_results, original_img = analyze_group_image_fhe(SAMPLE_GROUP_IMG, svd_k=optimal_k_for_demo)
        if analysis_results:
            annotated_img = draw_bounding_boxes(original_img, analysis_results)
            out_path = os.path.join(RESULTS_DIR, "group_analysis_STANDARD.jpg")
            cv2.imwrite(out_path, annotated_img)
            print(f"🖼️  Standard analysis visual saved to: {out_path}")
            
            # Log results to JSON
            log_path = os.path.join(RESULTS_DIR, "analysis_log.json")
            log_entry = {
                "analysis_type": "standard", "source_image": os.path.basename(SAMPLE_GROUP_IMG),
                "timestamp_utc": datetime.utcnow().isoformat(),
                "model_params": {"svd_k": optimal_k_for_demo, "model": model_to_use.__name__},
                "detections": analysis_results
            }
            with open(log_path, 'w') as f: json.dump(log_entry, f, indent=4)
            print(f"📋  Detailed log saved to: {log_path}")

        # 2. Robustness Test on Challenged Image
        print(f"\n--- Analyzing Challenged Image (Blur + Low-Res) ---")
        challenged_img = perturb_image(original_img, blur_level=3, resolution_scale=0.6)
        ch_path = os.path.join(RESULTS_DIR, "group_analysis_CHALLENGED_INPUT.jpg")
        cv2.imwrite(ch_path, challenged_img)
        
        results_ch, _ = analyze_group_image_fhe(ch_path, svd_k=optimal_k_for_demo)
        if results_ch:
            annotated_ch = draw_bounding_boxes(challenged_img, results_ch)
            out_path_ch = os.path.join(RESULTS_DIR, "group_analysis_CHALLENGED_OUTPUT.jpg")
            cv2.imwrite(out_path_ch, annotated_ch)
            print(f"🖼️  Challenged analysis visual saved to: {out_path_ch}")
            
            # Append to the same log file
            log_entry_ch = {
                "analysis_type": "challenged", "source_image": os.path.basename(ch_path),
                "timestamp_utc": datetime.utcnow().isoformat(),
                "model_params": {"svd_k": optimal_k_for_demo, "model": model_to_use.__name__},
                "detections": results_ch
            }
            with open(log_path, 'a') as f: f.write('\n' + json.dumps(log_entry_ch, indent=4))
            print(f"📋  Challenged results appended to log: {log_path}")
    else:
        print(f"Sample group image not found at {SAMPLE_GROUP_IMG}, skipping demonstration.")

    # --- Plotting Section ---
    print("\n📊 Generating summary plots...")
    k_vals_plot = [row[0] for row in summary_rows]
    accuracies_plot = [row[1] for row in summary_rows]
    inference_times_plot = [row[2] for row in summary_rows]

    # Plot 1: Accuracy vs. Inference Time Trade-off
    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax1.set_xlabel('SVD k-value')
    ax1.set_ylabel('Accuracy (%)', color='tab:blue')
    ax1.plot(k_vals_plot, accuracies_plot, color='tab:blue', marker='o', label='Accuracy')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, linestyle='--')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Avg FHE Inference Time (s)', color='tab:red')
    ax2.plot(k_vals_plot, inference_times_plot, color='tab:red', marker='x', linestyle='--', label='Inference Time')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.title('Accuracy vs. Inference Time Trade-off (LinearSVC)')
    fig.tight_layout()
    plt.savefig(os.path.join(SUMMARY_DIR, "accuracy_vs_inference_time_tradeoff.png"))
    
    # Plot 2: Accuracy vs. Payload Size Trade-off
    payload_sizes = [k * (RESIZED_IMAGE_SIZE[0] + RESIZED_IMAGE_SIZE[1] + 1) * 4 for k in k_vals_plot] # 4 bytes per float32
    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax1.set_xlabel('SVD k-value')
    ax1.set_ylabel('Accuracy (%)', color='tab:blue')
    ax1.plot(k_vals_plot, accuracies_plot, color='tab:blue', marker='o', label='Accuracy')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, linestyle='--')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Estimated Data Payload Size (bytes)', color='tab:green')
    ax2.plot(k_vals_plot, payload_sizes, color='tab:green', marker='s', linestyle=':', label='Payload Size')
    ax2.tick_params(axis='y', labelcolor='tab:green')
    
    plt.title('Accuracy vs. Data Payload Size Trade-off (LinearSVC)')
    fig.tight_layout()
    plt.savefig(os.path.join(SUMMARY_DIR, "accuracy_vs_payload_tradeoff.png"))
    
    print(f"✅ Summary plots saved to '{SUMMARY_DIR}' directory.")
    
if __name__ == "__main__":
    main()
