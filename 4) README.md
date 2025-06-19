# FACECLASS_FHE: Privacy-Preserving Face Classification with FHE

## Overview
**FACECLASS_FHE** uses Fully Homomorphic Encryption (FHE) to enable privacy-preserving classification of facial images. Built on Zama’s Concrete-ML framework and compatible with Superpod compute environments, this system performs secure, encrypted inference—allowing face classification without ever revealing the original image.

The model is designed to distinguish between two categories (e.g., "Criminal" vs. "General") and features a fully interactive UI to visualize predictions, performance, and encryption overhead.

---

## 📁 Directory Structure
├── dataset/
│ ├── Criminal/ # Images labeled as 'Criminal'
│ └── General/ # Images labeled as 'General'
├── results/ # Output directory for predictions, charts, and logs
│ ├── drone/ # Input/output for drone-based test images
│ ├── samples/ # Sample prediction outputs
│ ├── summary/ # Accuracy reports, CSVs
│ └── timing/ # Inference timing logs
├── sample_images/
│ └── sample_drone.jpg # Example input image for encrypted drone inference
├── faceclass_fhe.py # Main training and evaluation script
├── ui_app.py # Streamlit user interface app
└── README.md # This file

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Zama’s [Concrete-ML](https://github.com/zama-ai/concrete-ml)
- OpenCV
- NumPy
- scikit-learn
- matplotlib
- Streamlit

### Installation (Superpod or Local)
```bash
# Create and activate virtual environment
python3 -m venv fhe-env
source fhe-env/bin/activate        # Windows: fhe-env\Scripts\activate

# Install required packages
pip install --upgrade pip
pip install -r requirements.txt

Using the Streamlit Interface

Launch the App
streamlit run ui_app.py

UI Features
📤 Upload a test image for classification

🧠 Choose between standard or encrypted inference

⏱️ View timing comparison between unencrypted and encrypted modes

🔍 Visualize model predictions with confidence and accuracy

📊 Analyze encrypted overhead and verify prediction match

🗂️ Browse dataset samples directly in-app

📖 Learn about FHE through expandable technical explanations

No command-line coding is required after launch—the app guides users through every step.

🧠 Model Pipeline Summary
The classification flow:

A face image (from user or drone) is uploaded.

The image is preprocessed (face detection, resized to 32×32 grayscale).

The image is vectorized and quantized.

It is encrypted using Concrete-ML’s FHE compiler.

Encrypted data is passed through the compiled model.

The encrypted result is decrypted to reveal the prediction (e.g., "Criminal").

📊 Outputs
After running predictions, outputs are saved to the results/ directory:

summary/: Accuracy reports and prediction CSVs

samples/: Labeled prediction outputs

timing/: Encrypted vs. unencrypted inference timings

roc_curve.png: ROC performance chart

confusion_plaintext.png: Confusion matrix (plaintext)

accuracy_comparison.png: Accuracy of FHE vs. plaintext models

🛠️ Customizing for Your Dataset
1. Replace Dataset Categories
You can rename your folders to fit a new binary classification task:

Copy
Edit
dataset/
├── WithHat/
└── WithoutHat/
Update these paths in main.py:

python
Copy
Edit
CRIMINAL_DIR = os.path.join("dataset", "WithHat")
GENERAL_DIR = os.path.join("dataset", "WithoutHat")
2. Adjust Image Size
By default: IMAGE_SIZE = (32, 32)
You can increase to (64, 64) for improved resolution—note this increases FHE latency.

3. Add More Categories (Advanced)
To convert from binary to multi-class:

Use sklearn.linear_model.LogisticRegression(multi_class='multinomial')

Adjust label encoding and prediction logic

Update thresholding in decryption step

✈️ Drone Use Case
To simulate encrypted classification from drone imagery:

Replace sample_images/sample_drone.jpg with your image.

Launch the UI.

Upload the image for encrypted classification.

The system will preprocess and classify it without ever revealing the image content.

📦 Packaging the Project
To share or move the project:
# Optional cleanup
rm -rf __pycache__/ fhe-env/ *.pyc

# Create a zip file
zip -r faceclass_fhe_project.zip .

