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
    """Generate Grad-CAM heatmap for model predictions"""
    try:
        # Get the gradient of the output neuron (top predicted or chosen)
        # with respect to the output feature map of the last conv layer
        grad_model = tf.keras.Model(
            model.inputs, 
            [model.get_layer(last_conv_layer_name).output, model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]
        
        # Compute gradients of the class output value with respect to feature map
        grads = tape.gradient(class_channel, conv_outputs)
        
        if grads is None:
            return np.zeros((224, 224), dtype=np.float32)
        
        # Pool the gradients over all the axes leaving out the channel dimension
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Multiply each channel in the feature map array by 
        # "how important this channel is" with regard to the top predicted class
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # For visualization purpose, normalize the heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        # Resize to original image size
        heatmap = tf.image.resize(tf.expand_dims(heatmap, -1), [224, 224])
        heatmap = tf.squeeze(heatmap)
        
        return heatmap.numpy()
        
    except Exception as e:
        print(f"Error in make_gradcam_heatmap: {e}")
        return np.zeros((224, 224), dtype=np.float32)

def overlay_heatmap(img, heatmap, alpha=0.4):
    """Create a colored heatmap overlay on the original image"""
    try:
        # Convert PIL image to numpy array
        img_array = np.array(img.resize((224, 224)))
        
        # Ensure heatmap is the right size and type
        if heatmap.shape != (224, 224):
            heatmap = cv2.resize(heatmap, (224, 224))
        
        # Normalize heatmap to 0-255 range
        heatmap = np.clip(heatmap, 0, 1)
        heatmap = (heatmap * 255).astype(np.uint8)
        
        # Apply colormap to create red-yellow visualization
        # COLORMAP_JET: blue (low) -> green -> yellow -> red (high)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Convert from BGR to RGB (OpenCV uses BGR)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Ensure image is RGB
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_rgb = img_array
        else:
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        
        # Create overlay: blend original image with colored heatmap
        overlay = cv2.addWeighted(img_rgb, 1-alpha, heatmap_colored, alpha, 0)
        
        return overlay.astype(np.uint8)
        
    except Exception as e:
        print(f"Error in overlay_heatmap: {e}")
        # Return original image if overlay fails
        return np.array(img.resize((224, 224)))

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
