#!/usr/bin/env python
"""
Enhanced live webcam classification with improved face detection,
feature extraction, and FHE inference using the enhanced model.
"""
import os
import json
import cv2
import numpy as np
from concrete.ml.deployment import FHEModelClient, FHEModelServer
import logging
from typing import List, Tuple, Optional
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ——————————————————————————————————————————————
BASE_DIR     = os.path.dirname(__file__)
MODEL_DIR    = os.path.join(BASE_DIR, "models", "fhe_criminal_detector")
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
ALT_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_profileface.xml"
# ——————————————————————————————————————————————

class EnhancedFaceDetector:
    """Enhanced face detection matching training pipeline."""
    
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

class EnhancedFHEInference:
    """Enhanced FHE inference system matching the training pipeline."""
    
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.face_detector = EnhancedFaceDetector()
        self.load_model_artifacts()
        self.setup_fhe_components()
        
    def load_model_artifacts(self):
        """Load all model artifacts and configuration."""
        # Load configuration
        config_path = os.path.join(self.model_dir, "config.json")
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.svd_k = self.config['svd_k']
        self.resized = tuple(self.config['resized'])
        self.feature_dim = self.config['feature_dim']
        
        logger.info(f"Model config: SVD_K={self.svd_k}, Size={self.resized}, Features={self.feature_dim}")
        
        # Load scaler parameters
        scaler_path = os.path.join(self.model_dir, "scaler.json")
        with open(scaler_path, 'r') as f:
            scaler_data = json.load(f)
        
        self.scaler_mean = np.array(scaler_data["mean"], dtype=np.float32)
        self.scaler_std = np.array(scaler_data["std"], dtype=np.float32)
        
        # Load threshold
        threshold_path = os.path.join(self.model_dir, "threshold.txt")
        with open(threshold_path, 'r') as f:
            self.threshold = float(f.read().strip())
        
        logger.info(f"Threshold: {self.threshold:.3f}")
        
    def setup_fhe_components(self):
        """Initialize FHE client and server."""
        logger.info("Setting up FHE components...")
        
        try:
            self.client = FHEModelClient(path_dir=self.model_dir, key_dir=self.model_dir)
            eval_keys = self.client.get_serialized_evaluation_keys()
            
            self.server = FHEModelServer(path_dir=self.model_dir)
            self.server.load()
            
            # Store eval keys for inference
            self.eval_keys = eval_keys
            
            logger.info("FHE components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize FHE components: {e}")
            raise
    
    def apply_enhanced_svd(self, img: np.ndarray, k: int) -> np.ndarray:
        """Apply SVD compression matching training pipeline."""
        try:
            # Normalize image
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
    
    def extract_additional_features(self, img: np.ndarray) -> np.ndarray:
        """Extract additional features matching training pipeline."""
        # Convert back to uint8 for OpenCV operations
        img_uint8 = (img * 255).astype(np.uint8)
        
        # Histogram features
        hist = cv2.calcHist([img_uint8], [0], None, [16], [0, 256])
        hist_features = hist.flatten() / (hist.sum() + 1e-8)  # Normalize with epsilon
        
        # Texture features using Gabor filters
        texture_features = []
        for theta in [0, 45, 90, 135]:
            kernel = cv2.getGaborKernel((5, 5), 1, np.radians(theta), 2*np.pi/3, 0.5, 0, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(img_uint8, cv2.CV_8UC3, kernel)
            texture_features.append(filtered.mean())
            texture_features.append(filtered.std())
        
        # Edge density
        edges = cv2.Canny(img_uint8, 50, 150)
        edge_density = np.sum(edges > 0) / (img_uint8.shape[0] * img_uint8.shape[1])
        
        additional_features = np.concatenate([
            hist_features,
            texture_features,
            [edge_density]
        ])
        
        return additional_features.astype(np.float32)
    
    def extract_features(self, face: np.ndarray) -> np.ndarray:
        """Extract complete feature vector matching training pipeline."""
        # Enhance image quality
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_face = clahe.apply(face)
        
        # Resize to match training
        resized_face = cv2.resize(enhanced_face, self.resized)
        
        # SVD compression
        svd_compressed = self.apply_enhanced_svd(resized_face, self.svd_k)
        svd_features = svd_compressed.flatten()
        
        # Additional features
        additional_features = self.extract_additional_features(resized_face)
        
        # Combine all features
        combined_features = np.concatenate([svd_features, additional_features])
        
        return combined_features.astype(np.float32)
    
    def fhe_predict(self, features: np.ndarray) -> Tuple[str, float, float]:
        """Perform FHE inference and return prediction."""
        try:
            # Feature scaling
            scaled_features = (features - self.scaler_mean) / self.scaler_std
            
            # Encrypt and run inference
            encrypted_input = self.client.quantize_encrypt_serialize(scaled_features.reshape(1, -1))
            encrypted_result = self.server.run(encrypted_input, self.eval_keys)
            
            # Decrypt and get logit
            result_array = self.client.deserialize_decrypt_dequantize(encrypted_result)
            logit = float(np.asarray(result_array).flatten()[0])
            
            # Make prediction
            prediction = "Criminal" if logit > self.threshold else "General"
            confidence = abs(logit - self.threshold)
            
            return prediction, logit, confidence
            
        except Exception as e:
            logger.error(f"FHE prediction failed: {e}")
            return "Error", 0.0, 0.0
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame and return annotated result."""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_detector.detect_faces(gray)
        
        for face_coords in faces:
            x, y, w, h = face_coords
            
            # Extract face with padding
            padding = int(0.1 * min(w, h))
            x_pad = max(0, x - padding)
            y_pad = max(0, y - padding)
            w_pad = min(gray.shape[1] - x_pad, w + 2 * padding)
            h_pad = min(gray.shape[0] - y_pad, h + 2 * padding)
            
            face = gray[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]
            
            # Extract features
            features = self.extract_features(face)
            
            # Predict
            prediction, logit, confidence = self.fhe_predict(features)
            
            # Draw bounding box and label
            if prediction == "Criminal":
                color = (0, 0, 255)  # Red
                box_thickness = 3
            elif prediction == "General":
                color = (0, 255, 0)  # Green
                box_thickness = 2
            else:  # Error
                color = (0, 255, 255)  # Yellow
                box_thickness = 2
            
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, box_thickness)
            
            # Prepare label with confidence
            label = f"{prediction} ({confidence:.2f})"
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x, y-30), (x + label_size[0], y), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Debug info
            debug_text = f"Logit: {logit:.3f} | T: {self.threshold:.3f}"
            cv2.putText(frame, debug_text, (x, y+h+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame

def main():
    """Main inference loop."""
    logger.info("Starting enhanced FHE inference system...")
    
    try:
        # Initialize inference system
        inference_system = EnhancedFHEInference(MODEL_DIR)
        
        # Open webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Cannot open webcam")
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        logger.info("Webcam opened successfully")
        logger.info("Press 'q' to quit, 's' to save screenshot")
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame")
                break
            
            # Process frame
            processed_frame = inference_system.process_frame(frame)
            
            # Add FPS counter
            frame_count += 1
            if frame_count % 30 == 0:  # Update every 30 frames
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time
                cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow("Enhanced FHE Criminal Detection", processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save screenshot
                timestamp = int(time.time())
                filename = f"fhe_detection_{timestamp}.jpg"
                cv2.imwrite(filename, processed_frame)
                logger.info(f"Screenshot saved: {filename}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Application closed successfully")
        
    except Exception as e:
        logger.error(f"Application failed: {e}")
        raise

if __name__ == "__main__":
    main()