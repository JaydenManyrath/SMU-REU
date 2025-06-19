import threading
import time

# CRITICAL: Apply fix BEFORE any other imports (exactly like the working minimal version)
class FixedLocal(threading.local):
    def __init__(self):
        super().__init__()
        self.stack = []

threading.local = FixedLocal

# Now import everything else
import streamlit as st
import numpy as np
import cv2
import os
import glob
from pathlib import Path
from concrete.ml.sklearn import LogisticRegression

# === Page Configuration ===
st.set_page_config(
    page_title="Encrypted Image Classifier",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/zama-ai/concrete-ml',
        'Report a bug': None,
        'About': "# Encrypted Image Classifier\nBuilt with Concrete ML for homomorphic encryption"
    }
)

# === Updated CSS to completely hide progress bars ===
st.markdown("""
<style>
/* Main container styling */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* HIDE ALL PROGRESS BARS - Add this section */
.stProgress {
    display: none !important;
}
.stProgress > div {
    display: none !important;
}
.stProgress > div > div {
    display: none !important;
}
.stProgress > div > div > div {
    display: none !important;
}
.stProgress > div > div > div > div {
    display: none !important;
}

/* Header styling */
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 10px;
    margin-bottom: 2rem;
    color: white;
    text-align: center;
}

/* Rest of your existing CSS... */
/* Card styling */
.info-card {
    background: #f8f9fa;
    padding: 1.5rem;
    border-radius: 8px;
    border-left: 4px solid #667eea;
    margin: 1rem 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.success-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem;
    border-radius: 8px;
    color: white;
    margin: 1rem 0;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}
.warning-card {
    background: #fff3cd;
    padding: 1.5rem;
    border-radius: 8px;
    border-left: 4px solid #ffc107;
    margin: 1rem 0;
}
.timing-card {
    background: #e8f5e8;
    padding: 1.5rem;
    border-radius: 8px;
    border-left: 4px solid #28a745;
    margin: 1rem 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
/* Status indicators */
.status-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.875rem;
    font-weight: 600;
    margin: 0.25rem;
}
.status-success {
    background-color: #d4edda;
    color: #155724;
}
.status-error {
    background-color: #f8d7da;
    color: #721c24;
}

/* Button styling */
.stButton > button {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.75rem 1.5rem;
    font-weight: 600;
    transition: all 0.3s ease;
}
.stButton > button:hover {
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    transform: translateY(-2px);
}
/* Sidebar styling */
.css-1d391kg {
    background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
}
/* Metric cards */
.metric-card {
    background: white;
    padding: 1rem;
    border-radius: 8px;
    border: 1px solid #e9ecef;
    margin: 0.5rem 0;
    text-align: center;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}
/* Image upload area */
.uploadedFile {
    border: 2px dashed #667eea;
    border-radius: 8px;
    padding: 2rem;
    text-align: center;
    background: #f8f9ff;
}
/* Timing comparison styling */
.timing-comparison {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
    border: 1px solid #dee2e6;
}
.timing-fast {
    background: linear-gradient(90deg, #28a745, #20c997);
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-weight: bold;
}
.timing-slow {
    background: linear-gradient(90deg, #dc3545, #fd7e14);
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# === Configuration ===
# Default dataset paths - can be modified
DEFAULT_DATASET_PATHS = {
    "General": "dataset/General",    # Path to Class A images
    "Criminal": "dataset/Criminal"   # Path to Class B images
}

# Supported image extensions
SUPPORTED_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']

# === Dataset loading functions ===
def load_images_from_path(path, target_size=(32, 32)):
    """Load and preprocess images from a directory"""
    images = []
    if not os.path.exists(path):
        st.warning(f"Path does not exist: {path}")
        return images
    
    for ext in SUPPORTED_EXTENSIONS:
        pattern = os.path.join(path, '**', ext)
        image_files = glob.glob(pattern, recursive=True)
        
        for img_path in image_files:
            try:
                # Read image
                img = cv2.imread(img_path)
                if img is not None:
                    # Resize and convert to grayscale
                    img_resized = cv2.resize(img, target_size)
                    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
                    # Flatten for model input
                    img_flat = img_gray.flatten().astype(np.float32)
                    images.append(img_flat)
            except Exception as e:
                st.warning(f"Could not load image {img_path}: {str(e)}")
                continue
    
    return images

def load_dataset(dataset_paths):
    """Load dataset from specified paths - NO PROGRESS BARS"""
    X = []
    y = []
    class_counts = {}
    
    # Create a container for loading status (text only)
    loading_container = st.container()
    
    with loading_container:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("**📊 Loading Dataset**")
        status_text = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)
        
        for class_idx, (class_name, path) in enumerate(dataset_paths.items()):
            status_text.write(f"🔍 Loading {class_name} from: `{path}`")
            
            # Load images for this class
            class_images = load_images_from_path(path)
            
            # Add to dataset
            X.extend(class_images)
            y.extend([class_idx] * len(class_images))
            class_counts[class_name] = len(class_images)
        
        # Show completion
        status_text.write("✅ Dataset loading complete!")
        time.sleep(0.3)  # Brief pause to show completion
    
    # Clear ALL loading elements
    loading_container.empty()
    
    if len(X) == 0:
        st.error("❌ No images found in the specified paths!")
        return None, None, None
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    return X, y, class_counts
    
def handle_model_training(dataset_paths, use_dataset):
    """Handle model training with clean progress management"""
    if train_button or 'fhe_model' not in st.session_state:
        with st.spinner("🔄 Training and compiling encrypted model..."):
            model, class_names, class_counts = load_or_train_model(
                dataset_paths, use_dataset
            )
            
            if model is not None:
                st.session_state.fhe_model = model
                st.session_state.class_names = class_names
                st.session_state.class_counts = class_counts
                
                # Show success message only after everything is complete
                display_model_success(class_counts, class_names)
            else:
                st.error("❌ Failed to train/load model. Please check your dataset paths and try again.")
                st.stop()

def display_model_success(class_counts, class_names):
    """Display success message and dataset info after model is loaded"""
    st.markdown('<div class="success-card">', unsafe_allow_html=True)
    st.markdown("### ✅ Model Trained and Compiled Successfully!")
    
    # Dataset info
    if class_counts:
        total_samples = sum(class_counts.values())
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", total_samples)
        with col2:
            st.metric("Feature Dimension", "1024 (32x32)")
        with col3:
            st.metric("Classes", len(class_names))
        
        st.markdown("**Class Distribution:**")
        for class_name, count in class_counts.items():
            percentage = (count / total_samples) * 100 if total_samples > 0 else 0
            st.write(f"• **{class_name}**: {count} samples ({percentage:.1f}%)")
    
    st.markdown('</div>', unsafe_allow_html=True)
# === Prediction functions with timing ===
def encrypted_face_predict(img_array, model, class_names):
    """Predict using encrypted computation with timing"""
    try:
        start_time = time.time()
        img = cv2.resize(img_array, (32, 32))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x = gray.flatten().reshape(1, -1).astype(np.float32)
        x_q = model.quantize_input(x)
        prediction = model.fhe_circuit.encrypt_run_decrypt(x_q)[0]
        end_time = time.time()
        inference_time = end_time - start_time
        
        # Convert prediction to class name
        class_idx = int(prediction > 0.5) if len(class_names) == 2 else int(prediction)
        class_idx = max(0, min(class_idx, len(class_names) - 1))  # Ensure valid index
        
        return class_names[class_idx], inference_time
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return "Error", 0

def unencrypted_face_predict(img_array, model, class_names):
    """Predict using standard (unencrypted) computation with timing"""
    try:
        start_time = time.time()
        img = cv2.resize(img_array, (32, 32))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x = gray.flatten().reshape(1, -1).astype(np.float32)
        prediction = model.predict(x)[0]
        end_time = time.time()
        inference_time = end_time - start_time
        
        # Convert prediction to class name
        class_idx = int(prediction > 0.5) if len(class_names) == 2 else int(prediction)
        class_idx = max(0, min(class_idx, len(class_names) - 1))  # Ensure valid index
        
        return class_names[class_idx], inference_time
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return "Error", 0

# === Load or train model ===
@st.cache_resource
def load_or_train_model(dataset_paths, use_dataset=True):
    """Load/train and compile the FHE model"""
    try:
        # Create container for ALL model creation elements
        model_container = st.container()
        
        with model_container:
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.markdown("**🤖 Creating LogisticRegression Model**")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Clear the initial info
        model_container.empty()
        
        model = LogisticRegression(n_bits=3)  # Keep low for stability
        
        if use_dataset:
            # Load real dataset
            X, y, class_counts = load_dataset(dataset_paths)
            if X is None:
                st.error("Failed to load dataset. Falling back to dummy data.")
                use_dataset = False
        
        if not use_dataset:
            # Create container for dummy data warning
            dummy_container = st.container()
            with dummy_container:
                st.markdown('<div class="warning-card">', unsafe_allow_html=True)
                st.markdown("**⚠️ Using dummy training data**")
                st.markdown('</div>', unsafe_allow_html=True)
            
            time.sleep(1)  # Show warning briefly
            dummy_container.empty()  # Clear warning
            
            np.random.seed(42)  # For reproducibility
            X = np.random.rand(10, 32*32).astype(np.float32)
            y = np.random.randint(0, 2, size=10)
            class_counts = {"General": 5, "Criminal": 5}
        
        # Training progress with proper cleanup
        training_container = st.container()
        with training_container:
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.markdown("**🏋️ Training Progress**")
            training_progress = st.progress(0)
            status_text = st.empty()
            st.markdown('</div>', unsafe_allow_html=True)
            
            status_text.text("🏋️ Training model...")
            training_progress.progress(25)
            model.fit(X, y)
            
            status_text.text("🔧 Compiling for FHE...")
            training_progress.progress(75)
            
            # Use small compilation set for faster compilation
            compile_set_size = min(5, len(X))
            model.compile(X[:compile_set_size])
            
            status_text.text("✅ Compilation complete!")
            training_progress.progress(100)
            
            # Small delay to show completion
            time.sleep(0.5)
        
        # Clear ALL training elements
        training_container.empty()
        
        class_names = list(class_counts.keys()) if use_dataset else ["General", "Criminal"]
        return model, class_names, class_counts
        
    except Exception as e:
        st.error(f"❌ Model training error: {str(e)}")
        return None, None, None

# === Main UI ===
# Header
st.markdown("""
<div class="main-header">
    <h1>🔒 Encrypted Image Classifier</h1>
    <p style="font-size: 1.2rem; margin: 0;">Privacy-Preserving Machine Learning with Homomorphic Encryption</p>
</div>
""", unsafe_allow_html=True)

# === Sidebar Configuration ===
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    # Dataset path configuration (fixed paths)
    with st.expander("📁 Dataset Paths", expanded=True):
        st.markdown("**Default Dataset Paths:**")
        dataset_paths = DEFAULT_DATASET_PATHS
    
    # Training options
    with st.expander("🎯 Training Options", expanded=True):
        use_dataset = st.checkbox("Train with dataset (vs dummy data)", value=True)
    
    # Path status
    st.markdown("### 📊 Path Status")
    for class_name, path in dataset_paths.items():
        exists = os.path.exists(path)
        status_class = "status-success" if exists else "status-error"
        icon = "✅" if exists else "❌"
        st.markdown(f'<span class="status-badge {status_class}">{icon} {class_name}</span>', unsafe_allow_html=True)
        st.caption(f"`{path}`")

# === Model Training Section ===
st.markdown("## 🤖 Model Training & Loading")

# Check if paths exist when using dataset
if use_dataset:
    missing_paths = [path for path in dataset_paths.values() if not os.path.exists(path)]
    if missing_paths:
        st.markdown('<div class="warning-card">', unsafe_allow_html=True)
        st.markdown("**⚠️ Some dataset paths don't exist:**")
        for path in missing_paths:
            st.markdown(f"• `{path}`")
        st.markdown("The app will fall back to dummy data if dataset loading fails.")
        st.markdown('</div>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])
with col1:
    train_button = st.button("🚀 Train & Load FHE Model", type="primary", use_container_width=True)
with col2:
    if 'fhe_model' in st.session_state:
        st.success("✅ Model Ready")
    else:
        st.warning("⏳ Model Not Loaded")

if train_button or 'fhe_model' not in st.session_state:
    with st.spinner("🔄 Training and compiling encrypted model..."):
        model, class_names, class_counts = load_or_train_model(
            dataset_paths, use_dataset
        )
        
        if model is not None:
            st.session_state.fhe_model = model
            st.session_state.class_names = class_names
            st.session_state.class_counts = class_counts
        else:
            st.error("❌ Failed to train/load model. Please check your dataset paths and try again.")
            st.stop()

# === Image Classification Section ===
if 'fhe_model' in st.session_state:
    st.markdown("## 🖼️ Image Classification")
    
    # Model info in expandable card
    with st.expander("📋 Model Information", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Quantization", f"{st.session_state.fhe_model.n_bits} bits")
        with col2:
            total_samples = sum(st.session_state.class_counts.values())
            st.metric("Training Samples", total_samples)
        with col3:
            st.metric("Classes", len(st.session_state.class_names))
        
        st.markdown("**Class Distribution:**")
        for class_name, count in st.session_state.class_counts.items():
            percentage = (count / total_samples) * 100
            st.write(f"• **{class_name}**: {count} samples ({percentage:.1f}%)")
    
    # File upload section
    st.markdown("### 📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        help="Upload any image to test the encrypted classification"
    )
    
    if uploaded_file is not None:
        try:
            # Read and decode image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if img is None:
                st.error("❌ Could not decode the uploaded image. Please try a different file.")
            else:
                # Display images in columns
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### 🖼️ Original Image")
                    st.image(img, caption=f"Uploaded: {uploaded_file.name}", channels="BGR", use_container_width=True)
                
                with col2:
                    st.markdown("#### 🔄 Preprocessed Image")
                    preprocessed = cv2.resize(img, (32, 32))
                    preprocessed_gray = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2GRAY)
                    st.image(preprocessed_gray, caption="32x32 grayscale for model", use_container_width=True)
                
                # Prediction section with comparison options
                st.markdown("### 🔐 Inference Comparison")
                col1, col2 = st.columns(2)
                with col1:
                    run_encrypted = st.button("🔒 Run Encrypted Inference", type="primary", use_container_width=True)
                with col2:
                    run_comparison = st.button("⚡ Compare Both Methods", type="secondary", use_container_width=True)
                
                if run_encrypted:
                    prediction_container = st.container()
                    with prediction_container:
                        with st.spinner("🔒 Running encrypted inference..."):
                            prediction, inference_time = encrypted_face_predict(
                                img,
                                st.session_state.fhe_model,
                                st.session_state.class_names
                            )
                        
                        if prediction != "Error":
                            st.markdown('<div class="success-card">', unsafe_allow_html=True)
                            st.markdown(f"### 🎯 Encrypted Prediction: **{prediction}**")
                            st.markdown(f"**⏱️ Inference Time: {inference_time:.3f} seconds**")
                            st.markdown("*Computed entirely on encrypted data!*")
                            st.markdown('</div>', unsafe_allow_html=True)
                
                if run_comparison:
                    comparison_container = st.container()
                    with comparison_container:
                        st.markdown("#### 🏁 Performance Comparison")
                        
                        # Initialize results
                        results = {}
                        
                        # Run unencrypted inference
                        with st.spinner("⚡ Running standard inference..."):
                            unencrypted_pred, unencrypted_time = unencrypted_face_predict(
                                img,
                                st.session_state.fhe_model,
                                st.session_state.class_names
                            )
                            results['unencrypted'] = {
                                'prediction': unencrypted_pred,
                                'time': unencrypted_time
                            }
                        
                        # Run encrypted inference
                        with st.spinner("🔒 Running encrypted inference..."):
                            encrypted_pred, encrypted_time = encrypted_face_predict(
                                img,
                                st.session_state.fhe_model,
                                st.session_state.class_names
                            )
                            results['encrypted'] = {
                                'prediction': encrypted_pred,
                                'time': encrypted_time
                            }
                        
                        # Display comparison results
                        if results['unencrypted']['prediction'] != "Error" and results['encrypted']['prediction'] != "Error":
                            st.markdown('<div class="timing-card">', unsafe_allow_html=True)
                            st.markdown("### 📊 Inference Comparison Results")
                            
                            # Results table
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.markdown("**Method**")
                                st.write("🔓 Standard")
                                st.write("🔒 Encrypted")
                            with col2:
                                st.markdown("**Prediction**")
                                st.write(f"**{results['unencrypted']['prediction']}**")
                                st.write(f"**{results['encrypted']['prediction']}**")
                            with col3:
                                st.markdown("**Time (seconds)**")
                                st.write(f"**{results['unencrypted']['time']:.3f}**")
                                st.write(f"**{results['encrypted']['time']:.3f}**")
                            
                            # Performance analysis
                            st.markdown("---")
                            
                            # Calculate overhead
                            if results['unencrypted']['time'] > 0:
                                overhead = (results['encrypted']['time'] / results['unencrypted']['time'])
                                st.markdown(f"**🔍 Performance Analysis:**")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Encryption Overhead", f"{overhead:.1f}x",
                                            f"+{(overhead-1)*100:.1f}%")
                                with col3:
                                    accuracy_match = "✅ Match" if results['unencrypted']['prediction'] == results['encrypted']['prediction'] else "❌ Differ"
                                    st.metric("Prediction Accuracy", accuracy_match)
                                
                                # Interpretation
                                if overhead < 5:
                                    performance_msg = "🚀 **Excellent**: Low encryption overhead!"
                                elif overhead < 20:
                                    performance_msg = "✅ **Good**: Reasonable encryption overhead for privacy benefits."
                                elif overhead < 100:
                                    performance_msg = "⚠️ **Moderate**: Higher overhead but strong privacy guarantees."
                                else:
                                    performance_msg = "🐌 **High**: Significant overhead - consider optimization for production."
                                
                                st.markdown(performance_msg)
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Technical details in expandable section
                            with st.expander("🔧 Technical Details", expanded=False):
                                st.markdown("**Homomorphic Encryption Process:**")
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown("""
                                    **Input Processing:**
                                    - Original size: {}
                                    - Resized to: 32×32 grayscale
                                    - Features: {} pixel values
                                    - Quantization: {} bits
                                    """.format(img.shape, 32*32, st.session_state.fhe_model.n_bits))
                                
                                with col2:
                                    st.markdown("""
                                    **Encryption Flow:**
                                    - Classes: {}
                                    - Computation: Encrypted data only
                                    - Privacy: Input never decrypted
                                    - Output: Secure prediction result
                                    """.format(', '.join(st.session_state.class_names)))
                                
                                st.markdown("**Timing Breakdown:**")
                                st.markdown(f"- **Standard Inference**: {results['unencrypted']['time']:.4f}s")
                                st.markdown(f"- **Encrypted Inference**: {results['encrypted']['time']:.4f}s")
                                st.markdown(f"- **Encryption Overhead**: {results['encrypted']['time'] - results['unencrypted']['time']:.4f}s")
                                
                                st.info("🔒 The encrypted prediction was computed entirely on encrypted data without ever decrypting the input!")
        
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")
            st.write("Please try uploading a different image format.")

# === Dataset Exploration ===
if 'class_counts' in st.session_state:
    with st.expander("🔍 Dataset Exploration", expanded=False):
        st.markdown("### Sample Images from Dataset")
        
        for class_name, path in dataset_paths.items():
            if os.path.exists(path):
                st.markdown(f"#### {class_name} Class Samples")
                
                # Get sample images
                sample_files = []
                for ext in SUPPORTED_EXTENSIONS:
                    pattern = os.path.join(path, '**', ext)
                    files = glob.glob(pattern, recursive=True)[:3]
                    sample_files.extend(files)
                
                if sample_files:
                    cols = st.columns(min(len(sample_files), 3))
                    for i, img_path in enumerate(sample_files[:3]):
                        try:
                            img = cv2.imread(img_path)
                            if img is not None:
                                with cols[i]:
                                    st.image(img, caption=os.path.basename(img_path),
                                           channels="BGR", use_container_width=True)
                        except Exception as e:
                            st.write(f"Could not display {img_path}")
                else:
                    st.warning(f"No sample images found in {path}")

# === Instructions Footer ===
st.markdown("---")
st.markdown("## 📚 Setup Instructions")

with st.expander("📖 How to Use This App", expanded=False):
    st.markdown("""
    ### 🗂️ Dataset Structure
    Create folders with your training images:
    ```
    data/
    ├── class_a/
    │   ├── image1.jpg
    │   ├── image2.png
    │   └── ...
    └── class_b/
        ├── image1.jpg
        ├── image2.png
        └── ...
    ```
    
    ### ⚙️ Configuration Steps
    1. **Custom Paths**: Use the sidebar to specify custom dataset paths
    2. **Training Options**: Configure training parameters
    3. **Model Training**: Click "Train & Load FHE Model"
    4. **Classification**: Upload test images for encrypted predictions
    5. **Performance**: Use "Compare Both Methods" to see timing differences
    
    ### 🔒 About Homomorphic Encryption
    - **Privacy-Preserving**: Computations on encrypted data
    - **Secure**: Input data never decrypted during processing
    - **Innovative**: Uses Concrete ML for FHE implementation
    - **Fallback**: Automatically uses dummy data if dataset fails
    - **Performance**: Compare encrypted vs standard inference times
    
    ### ⏱️ Performance Analysis Features
    - **Timing Comparison**: See exact inference times for both methods
    - **Overhead Analysis**: Understand the cost of encryption
    - **Accuracy Verification**: Confirm predictions match between methods
    - **Real-time Metrics**: Get immediate performance feedback
    """)