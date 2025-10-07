import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import base64
from datetime import datetime
import json

# Import TensorFlow components more carefully
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    # Set memory growth to avoid memory issues
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
except ImportError as e:
    st.error(f"TensorFlow import error: {e}")
    st.stop()

# GradCAM Utilities
def get_img_array(img, size=(224,224)):
    arr = img.resize(size).convert("RGB")
    arr = np.array(arr).astype(np.float32)
    arr /= 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(img, heatmap, alpha=0.5):
    img = np.array(img.resize((224,224))).astype(np.uint8)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    combined = cv2.addWeighted(img, 1-alpha, heatmap_color, alpha, 0)
    return combined

# Helper functions for enhancements
def get_confidence_color(confidence):
    """Return color based on confidence level"""
    if confidence >= 80:
        return "🟢", "#28a745"  # Green
    elif confidence >= 50:
        return "🟡", "#ffc107"  # Yellow
    else:
        return "🔴", "#dc3545"  # Red

def create_download_report(pred_class, confidence, all_predictions, class_names):
    """Create a downloadable analysis report"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = {
        "timestamp": timestamp,
        "predicted_class": pred_class,
        "confidence": f"{confidence:.1f}%",
        "all_predictions": {
            class_names[i]: f"{float(all_predictions[0][i]) * 100:.1f}%"
            for i in range(len(class_names))
        },
        "analysis_summary": f"Brain MRI scan analysis completed with {confidence:.1f}% confidence for {pred_class.title()}"
    }
    
    return json.dumps(report, indent=2)

def image_to_base64(img):
    """Convert PIL image to base64 string for download"""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

# Load model (cache speeds up)
@st.cache_resource
def load_densenet_model():
    try:
        with st.spinner("Loading AI model... Please wait"):
            # Try to load local model first
            try:
                model = load_model("densenet121_brain_tumor_best.h5")
            except FileNotFoundError:
                # For cloud deployment, you can add model download logic here
                st.error("❌ Model file not found. Please ensure densenet121_brain_tumor_best.h5 is available.")
                st.info("📋 For deployment instructions, see MODEL_SETUP.md in the repository.")
                st.stop()
                return None
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()
        return None

# Initialize model
try:
    model = load_densenet_model()
    last_conv_layer_name = "conv5_block16_concat"
    class_names = ["glioma", "meningioma", "notumor", "pituitary"]
except Exception as e:
    st.error(f"Failed to initialize model: {e}")
    st.stop()

st.title("🧠 Brain Tumor MRI Classification & Analysis")
st.markdown("""
This AI-powered tool analyzes MRI brain scans to classify different types of brain tumors.
Upload an MRI image to get instant classification results with confidence scores and visual explanations.
""")

# Tumor information
tumor_info = {
    "glioma": {
        "description": "A type of brain tumor that occurs in the brain and spinal cord. Gliomas begin in the gluey supportive cells (glial cells) that surround nerve cells.",
        "color": "🔴",
        "severity": "Variable (depends on grade)"
    },
    "meningioma": {
        "description": "A tumor that arises from the meninges — the membranes that surround the brain and spinal cord. Most meningiomas are benign.",
        "color": "🟡",
        "severity": "Usually benign"
    },
    "notumor": {
        "description": "No tumor detected. The brain tissue appears normal in the MRI scan.",
        "color": "🟢",
        "severity": "Normal/Healthy"
    },
    "pituitary": {
        "description": "A tumor in the pituitary gland, which is located at the base of the brain. These tumors can affect hormone production.",
        "color": "🟠",
        "severity": "Usually benign"
    }
}

uploaded_file = st.file_uploader("📁 Upload an MRI image (JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # First row: Image and initial results
    col1, col2 = st.columns([1, 1])
    
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption="📷 Uploaded MRI Scan", use_container_width=True)
    
    with col2:
        with st.spinner("Analyzing MRI scan..."):
            img_array = get_img_array(image)
            preds = model.predict(img_array)
            pred_idx = np.argmax(preds)
            pred_class = class_names[pred_idx]
            confidence = float(preds[0][pred_idx]) * 100
            
            # Get confidence color and emoji
            confidence_emoji, confidence_color = get_confidence_color(confidence)
            
            # Display results with color-coded confidence
            st.markdown("### 🎯 Classification Results")
            info = tumor_info[pred_class]
            st.markdown(f"""
            **{info['color']} Predicted Type:** **{pred_class.title()}**  
            **{confidence_emoji} Confidence:** <span style="color: {confidence_color}; font-weight: bold;">{confidence:.1f}%</span>  
            **⚕️ Severity:** {info['severity']}
            """, unsafe_allow_html=True)
    
    # Second row: Detailed results and Grad-CAM side by side
    st.markdown("---")
    col3, col4 = st.columns([1, 1])
    
    with col3:
        # Show confidence for all classes with color-coded progress bars
        st.markdown("### 📈 All Confidence Scores")
        for i, class_name in enumerate(class_names):
            conf = float(preds[0][i]) * 100
            conf_emoji, conf_color = get_confidence_color(conf)
            
            # Custom progress bar with color
            progress_html = f"""
            <div style="margin-bottom: 10px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                    <span style="font-weight: bold;">{conf_emoji} {class_name.title()}</span>
                    <span style="color: {conf_color}; font-weight: bold;">{conf:.1f}%</span>
                </div>
                <div style="background-color: #e0e0e0; border-radius: 10px; height: 20px; overflow: hidden;">
                    <div style="background-color: {conf_color}; height: 100%; width: {conf}%; border-radius: 10px; transition: width 0.3s ease;"></div>
                </div>
            </div>
            """
            st.markdown(progress_html, unsafe_allow_html=True)
        
        # Tumor information
        st.markdown("### 📚 About This Tumor Type")
        st.info(tumor_info[pred_class]["description"])
    
    with col4:
        # GradCAM Analysis
        st.markdown(" 🔍 Grad-CAM")
        st.markdown("Heatmap showing AI focus areas. Red/yellow = high importance.")
        
        with st.spinner("Generating AI explanation..."):
            heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
            cam_img = overlay_heatmap(image, heatmap)
            st.image(cam_img, caption="Grad-CAM Heatmap", use_container_width=True)
            
            # Convert grad-cam to PIL for download
            cam_pil = Image.fromarray(cam_img)
    
    # Download Section
    st.markdown("---")
    st.markdown("### 💾 Download Analysis Results")
    
    col5, col6 = st.columns(2)
    
    with col5:
        # Download analysis report
        report_data = create_download_report(pred_class, confidence, preds, class_names)
        st.download_button(
            label="📄 Download Analysis Report (JSON)",
            data=report_data,
            file_name=f"brain_tumor_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    with col6:
        # Download Grad-CAM image
        cam_buffer = io.BytesIO()
        cam_pil.save(cam_buffer, format='PNG')
        cam_buffer.seek(0)
        
        st.download_button(
            label="🖼️ Download Grad-CAM Image",
            data=cam_buffer.getvalue(),
            file_name=f"gradcam_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            mime="image/png"
        )
    
    # Disclaimer
    st.warning("⚠️ **Medical Disclaimer:** This tool is for educational purposes only and should not be used for actual medical diagnosis. Always consult with qualified healthcare professionals for medical advice.")
