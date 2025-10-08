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

# Simple and working GradCAM implementation
def get_img_array(img, size=(224,224)):
    arr = img.resize(size).convert("RGB")
    arr = np.array(arr).astype(np.float32)
    arr /= 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Bulletproof Grad-CAM that works in production"""
    try:
        # This is the EXACT working approach from marine classification
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            # Convert tensor to int to avoid indexing issues
            pred_index = int(pred_index)
            class_channel = predictions[0][pred_index]
        
        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = tf.image.resize(heatmap[..., tf.newaxis], [224, 224])
        return tf.squeeze(heatmap).numpy()
        
    except Exception as e:
        st.error(f"Grad-CAM failed: {e}")
        return np.zeros((224, 224))

def overlay_heatmap(image, heatmap, alpha=0.4):
    """Production-ready overlay - same as marine classification"""
    try:
        # Convert to arrays
        img = np.array(image.resize((224, 224)))
        heatmap = np.array(heatmap)
        
        # Normalize heatmap
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        heatmap = np.uint8(255 * heatmap)
        
        # Use JET colormap (most reliable across environments)
        colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)
        
        # Ensure RGB
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=-1)
        
        # Overlay
        result = img * (1 - alpha) + colored_heatmap * alpha
        return result.astype(np.uint8)
        
    except Exception as e:
        st.error(f"Overlay failed: {e}")
        return np.array(image.resize((224, 224)))

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
    """Create downloadable analysis report"""
    report = {
        "timestamp": datetime.now().isoformat(),
        "predicted_class": pred_class,
        "confidence": float(confidence),
        "all_predictions": {
            class_names[i]: float(all_predictions[0][i] * 100) 
            for i in range(len(class_names))
        },
        "model_info": {
            "architecture": "DenseNet121",
            "input_size": "224x224",
            "classes": len(class_names)
        }
    }
    return json.dumps(report, indent=2)

def image_to_base64(img):
    """Convert PIL image to base64 string"""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

# Load model (cache speeds up)
@st.cache_resource
def load_densenet_model():
    try:
        with st.spinner("Loading AI model... Please wait"):
            model = load_model("densenet121_brain_tumor_best.h5")
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.info("📋 Make sure densenet121_brain_tumor_best.h5 is available in the repository.")
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
st.markdown("### Upload an MRI scan to get instant AI-powered analysis with explainable results")

# Tumor information
tumor_info = {
    "glioma": {
        "description": "A type of tumor that occurs in the brain and spinal cord. Gliomas begin in the gluey supportive cells (glial cells) that surround nerve cells.",
        "severity": "High",
        "color": "🔴"
    },
    "meningioma": {
        "description": "A tumor that arises from the meninges — the membranes that surround the brain and spinal cord. Most meningiomas are benign.",
        "severity": "Low to Medium", 
        "color": "🟡"
    },
    "notumor": {
        "description": "No tumor detected. The brain tissue appears normal in the MRI scan.",
        "severity": "None",
        "color": "🟢"
    },
    "pituitary": {
        "description": "A tumor in the pituitary gland, which is located at the base of the brain. These tumors can affect hormone production.",
        "severity": "Medium",
        "color": "🟠"
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
            
            # Display results
            st.markdown("### 🎯 Classification Results")
            info = tumor_info[pred_class]
            st.markdown(f"""
            **{info['color']} Predicted Type:** **{pred_class.title()}**  
            **📊 Confidence:** {confidence:.1f}%  
            **⚕️ Severity:** {info['severity']}
            """)
    
    # Second row: Detailed results and Grad-CAM side by side
    st.markdown("---")
    col3, col4 = st.columns([1, 1])
    
    with col3:
        # Show confidence for all classes
        st.markdown("### 📈 All Confidence Scores")
        for i, class_name in enumerate(class_names):
            conf = float(preds[0][i]) * 100
            emoji, color = get_confidence_color(conf)
            st.markdown(f"**{emoji} {class_name.title()}:** {conf:.1f}%")
            st.progress(conf/100.0)
        
        # Tumor information
        st.markdown("### 📚 About This Tumor Type")
        st.info(tumor_info[pred_class]["description"])
    
    with col4:
        # GradCAM Analysis
        st.markdown("### 🔥 Grad-CAM")
        st.markdown("Heatmap showing AI focus areas. Red/yellow = high importance.")
        
        with st.spinner("Generating AI explanation..."):
            try:
                heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
                cam_img = overlay_heatmap(image, heatmap)
                st.image(cam_img, caption="Grad-CAM Heatmap", use_container_width=True)
                
                # Convert grad-cam to PIL for download
                cam_pil = Image.fromarray(cam_img)
            except Exception as e:
                st.error(f"Error generating Grad-CAM: {str(e)}")
                st.info("Grad-CAM visualization is temporarily unavailable, but the classification results above are still accurate.")
                # Create a placeholder image for download
                cam_pil = image.resize((224, 224))
    
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
        if 'cam_pil' in locals():
            img_buffer = io.BytesIO()
            cam_pil.save(img_buffer, format='PNG')
            st.download_button(
                label="🖼️ Download Grad-CAM Image",
                data=img_buffer.getvalue(),
                file_name=f"gradcam_{pred_class}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png"
            )
    
    # Disclaimer
    st.warning("⚠️ **Medical Disclaimer:** This tool is for educational purposes only and should not be used for actual medical diagnosis. Always consult with qualified healthcare professionals for medical advice.")