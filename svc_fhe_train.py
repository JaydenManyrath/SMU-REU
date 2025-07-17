#!/usr/bin/env python
"""
Enhanced FHE-based facial recognition model with improved robustness,
data augmentation, feature engineering, and comprehensive validation.
"""
import os
import shutil
import json
import cv2
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_curve, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from concrete.ml.sklearn import LinearSVC
from concrete.ml.deployment import FHEModelDev
import logging
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ——————————————————————————————————————————————
BASE_DIR      = os.path.dirname(__file__)
CRIMINAL_DIR  = os.path.join(BASE_DIR, "dataset", "Criminal")
GENERAL_DIR   = os.path.join(BASE_DIR, "dataset", "General")
MODEL_OUT_DIR = os.path.join(BASE_DIR, "models", "fhe_criminal_detector")

# Enhanced hyperparameters
SVD_K         = 16    # Increased for better feature retention
N_BITS        = 8     # Increased for better precision
RESIZED       = (32, 32)  # Slightly larger for more detail
MIN_SAMPLES   = 50    # Minimum samples per class
CASCADE_PATH  = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
ALT_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_profileface.xml"
# ——————————————————————————————————————————————

class EnhancedFaceDetector:
    """Enhanced face detection with multiple cascades and preprocessing."""
    
    def __init__(self):
        self.frontal_cascade = cv2.CascadeClassifier(CASCADE_PATH)
        self.profile_cascade = cv2.CascadeClassifier(ALT_CASCADE_PATH)
        
    def detect_faces(self, img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using multiple cascade classifiers."""
        faces = []
        
        # Try frontal face detection
        frontal_faces = self.frontal_cascade.detectMultiScale(
            img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        faces.extend(frontal_faces)
        
        # Try profile face detection if no frontal faces found
        if len(faces) == 0:
            profile_faces = self.profile_cascade.detectMultiScale(
                img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            faces.extend(profile_faces)
            
        return faces

def apply_enhanced_svd(img: np.ndarray, k: int) -> np.ndarray:
    """Apply SVD compression with enhanced error handling."""
    try:
        # Ensure image is properly normalized
        img = img.astype(np.float32) / 255.0
        
        U, S, VT = np.linalg.svd(img, full_matrices=False)
        k = min(k, len(S))
        
        # Apply compression
        compressed = (U[:, :k] @ np.diag(S[:k]) @ VT[:k, :])
        
        # Ensure output is in valid range
        compressed = np.clip(compressed, 0, 1)
        
        return compressed
    except Exception as e:
        logger.error(f"SVD compression failed: {e}")
        return img.astype(np.float32) / 255.0

def augment_image(img: np.ndarray) -> List[np.ndarray]:
    """Generate augmented versions of the image."""
    augmented = [img]  # Original image
    
    # Slight rotation
    center = (img.shape[1] // 2, img.shape[0] // 2)
    for angle in [-5, 5]:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        augmented.append(rotated)
    
    # Horizontal flip
    flipped = cv2.flip(img, 1)
    augmented.append(flipped)
    
    # Brightness adjustment
    bright = cv2.convertScaleAbs(img, alpha=1.1, beta=10)
    dark = cv2.convertScaleAbs(img, alpha=0.9, beta=-10)
    augmented.extend([bright, dark])
    
    return augmented

def extract_additional_features(img: np.ndarray) -> np.ndarray:
    """Extract additional features beyond SVD compression."""
    # Histogram features
    hist = cv2.calcHist([img], [0], None, [16], [0, 256])
    hist_features = hist.flatten() / hist.sum()  # Normalize
    
    # Texture features using Gabor filters
    texture_features = []
    for theta in [0, 45, 90, 135]:
        kernel = cv2.getGaborKernel((5, 5), 1, np.radians(theta), 2*np.pi/3, 0.5, 0, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(img, cv2.CV_8UC3, kernel)
        texture_features.append(filtered.mean())
        texture_features.append(filtered.std())
    
    # Edge density
    edges = cv2.Canny(img, 50, 150)
    edge_density = np.sum(edges > 0) / (img.shape[0] * img.shape[1])
    
    additional_features = np.concatenate([
        hist_features,
        texture_features,
        [edge_density]
    ])
    
    return additional_features

def load_and_process_enhanced(folder: str, label: int, face_detector: EnhancedFaceDetector, 
                            augment: bool = True) -> List[Tuple[np.ndarray, int]]:
    """Enhanced data loading with augmentation and additional features."""
    data = []
    processed_count = 0
    
    logger.info(f"Processing {folder} (label={label})...")
    
    for fname in os.listdir(folder):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue
            
        img_path = os.path.join(folder, fname)
        img = cv2.imread(img_path)
        
        if img is None:
            logger.warning(f"Could not load image: {fname}")
            continue
            
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Enhance image quality
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Detect faces
        faces = face_detector.detect_faces(gray)
        
        if len(faces) == 0:
            logger.warning(f"No face detected in: {fname}")
            continue
            
        # Process the largest face
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face
        
        # Extract face with some padding
        padding = int(0.1 * min(w, h))
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(gray.shape[1] - x, w + 2 * padding)
        h = min(gray.shape[0] - y, h + 2 * padding)
        
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, RESIZED)
        
        # Apply data augmentation
        if augment and len(data) < MIN_SAMPLES:
            faces_to_process = augment_image(face)
        else:
            faces_to_process = [face]
        
        for processed_face in faces_to_process:
            # SVD compression
            svd_compressed = apply_enhanced_svd(processed_face, SVD_K)
            svd_features = svd_compressed.flatten()
            
            # Additional features
            additional_features = extract_additional_features(processed_face)
            
            # Combine all features
            combined_features = np.concatenate([svd_features, additional_features])
            
            data.append((combined_features, label))
            processed_count += 1
            
            # Limit augmentation
            if len(data) >= MIN_SAMPLES * 2:
                break
                
        if len(data) >= MIN_SAMPLES * 2:
            break
    
    logger.info(f"Processed {processed_count} samples from {folder}")
    return data

def validate_dataset(pos_data: List, neg_data: List) -> bool:
    """Validate dataset balance and quality."""
    pos_count = len(pos_data)
    neg_count = len(neg_data)
    
    logger.info(f"Dataset stats: Criminal={pos_count}, General={neg_count}")
    
    if pos_count < MIN_SAMPLES or neg_count < MIN_SAMPLES:
        logger.error(f"Insufficient samples. Need at least {MIN_SAMPLES} per class.")
        return False
        
    # Check class balance
    ratio = min(pos_count, neg_count) / max(pos_count, neg_count)
    if ratio < 0.5:
        logger.warning(f"Dataset imbalanced (ratio: {ratio:.2f}). Consider balancing.")
    
    return True

def cross_validate_model(X: np.ndarray, y: np.ndarray, n_folds: int = 3) -> float:
    """Perform cross-validation to estimate model performance."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        logger.info(f"Cross-validation fold {fold + 1}/{n_folds}")
        
        X_train_cv, X_val_cv = X[train_idx], X[val_idx]
        y_train_cv, y_val_cv = y[train_idx], y[val_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_train_cv_scaled = scaler.fit_transform(X_train_cv)
        X_val_cv_scaled = scaler.transform(X_val_cv)
        
        # Train model
        model_cv = LinearSVC(n_bits=N_BITS, random_state=42)
        model_cv.fit(X_train_cv_scaled, y_train_cv)
        
        # Evaluate
        score = model_cv.score(X_val_cv_scaled, y_val_cv)
        scores.append(score)
        logger.info(f"Fold {fold + 1} accuracy: {score:.3f}")
    
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    logger.info(f"Cross-validation: {mean_score:.3f} ± {std_score:.3f}")
    
    return mean_score

def main():
    """Main training pipeline with enhanced robustness."""
    try:
        # Initialize
        logger.info("Starting enhanced FHE criminal detection model training...")
        
        # Cleanup old artifacts
        if os.path.exists(MODEL_OUT_DIR):
            shutil.rmtree(MODEL_OUT_DIR)
        
        # Initialize face detector
        face_detector = EnhancedFaceDetector()
        
        # Load and process data
        logger.info("Loading and processing dataset...")
        pos_data = load_and_process_enhanced(CRIMINAL_DIR, 1, face_detector, augment=True)
        neg_data = load_and_process_enhanced(GENERAL_DIR, 0, face_detector, augment=True)
        
        # Validate dataset
        if not validate_dataset(pos_data, neg_data):
            raise RuntimeError("Dataset validation failed")
        
        # Combine and prepare data
        all_data = pos_data + neg_data
        X, y = zip(*all_data)
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int8)
        
        logger.info(f"Dataset prepared: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Cross-validation
        logger.info("Performing cross-validation...")
        cv_score = cross_validate_model(X_train_scaled, y_train)
        
        if cv_score < 0.7:
            logger.warning(f"Low cross-validation score: {cv_score:.3f}")
        
        # Train final model
        logger.info("Training final FHE model...")
        model = LinearSVC(n_bits=N_BITS, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Compile for FHE
        logger.info("Compiling for FHE (this may take several minutes)...")
        model.compile(X_train_scaled)
        
        # Evaluate model
        train_acc = model.score(X_train_scaled, y_train)
        test_acc = model.score(X_test_scaled, y_test)
        
        logger.info(f"Training accuracy: {train_acc:.3f}")
        logger.info(f"Test accuracy: {test_acc:.3f}")
        
        # Detailed evaluation
        y_pred = model.predict(X_test_scaled)
        logger.info("Classification Report:")
        logger.info(f"\n{classification_report(y_test, y_pred, target_names=['General', 'Criminal'])}")
        
        # Save model and artifacts
        os.makedirs(MODEL_OUT_DIR, exist_ok=True)
        
        # Save FHE model
        dev = FHEModelDev(path_dir=MODEL_OUT_DIR, model=model)
        dev.save(via_mlir=True)
        
        # Save configuration
        config = {
            'svd_k': SVD_K,
            'n_bits': N_BITS,
            'resized': RESIZED,
            'feature_dim': X.shape[1],
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
            'cv_score': float(cv_score)
        }
        
        with open(os.path.join(MODEL_OUT_DIR, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        
        # Save scaler parameters
        scaler_params = {
            'mean': scaler.mean_.tolist(),
            'std': scaler.scale_.tolist()
        }
        
        with open(os.path.join(MODEL_OUT_DIR, "scaler.json"), "w") as f:
            json.dump(scaler_params, f, indent=2)
        
        # Calibrate threshold using encrypted inference
        logger.info("Calibrating decision threshold on encrypted logits...")
        encrypted_logits = []
        true_labels = []
        
        for i, (x_sample, true_label) in enumerate(zip(X_test_scaled, y_test)):
            if i % 10 == 0:
                logger.info(f"Processing sample {i+1}/{len(X_test_scaled)}")
            
            # Encrypt and run inference
            encrypted_input = model.quantize_input(x_sample.reshape(1, -1))
            encrypted_logit = model.fhe_circuit.encrypt_run_decrypt(encrypted_input)[0]
            
            encrypted_logits.append(float(encrypted_logit))
            true_labels.append(int(true_label))
        
        # Find optimal threshold
        fpr, tpr, thresholds = roc_curve(true_labels, encrypted_logits)
        
        # Youden's J statistic for optimal threshold
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = float(thresholds[optimal_idx])
        
        # Save threshold
        with open(os.path.join(MODEL_OUT_DIR, "threshold.txt"), "w") as f:
            f.write(str(optimal_threshold))
        
        # Final summary
        logger.info(f"✅ Model training completed successfully!")
        logger.info(f"   Training accuracy: {train_acc:.3f}")
        logger.info(f"   Test accuracy: {test_acc:.3f}")
        logger.info(f"   Cross-validation score: {cv_score:.3f}")
        logger.info(f"   Optimal threshold: {optimal_threshold:.3f}")
        logger.info(f"   Model saved to: {MODEL_OUT_DIR}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()