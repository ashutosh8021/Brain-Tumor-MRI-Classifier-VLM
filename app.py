"""
Brain Tumor MRI Classification & Analysis v2.0
DenseNet121-based classifier with VLM explanations and failure detection
"""

# CRITICAL: Load .env BEFORE any other imports
from dotenv import load_dotenv
load_dotenv(override=True)

import streamlit as st

# MUST be first Streamlit command
st.set_page_config(page_title="Brain Tumor Classifier v2.0", page_icon="🧠", layout="wide")

import numpy as np
import cv2
from PIL import Image
import io
import base64
from datetime import datetime
import json
import os

# Import TensorFlow components
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
except ImportError as e:
    st.error(f"TensorFlow import error: {e}")
    st.stop()

# Import PDF generation
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    from reportlab.lib import colors
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Groq API setup
try:
    from groq import Groq
    # Use dict.get() to avoid Streamlit command before page config
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
    if GROQ_API_KEY is None and hasattr(st, 'secrets'):
        try:
            GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", None)
        except:
            GROQ_API_KEY = None
    
    GROQ_AVAILABLE = GROQ_API_KEY is not None
    VLM_ENABLED = GROQ_AVAILABLE and bool(GROQ_API_KEY)
except ImportError:
    GROQ_AVAILABLE = False
    VLM_ENABLED = False
    GROQ_API_KEY = None

# ============================================================================
# CORE FUNCTIONS (UNCHANGED)
# ============================================================================

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
            # Use tf.gather instead of direct indexing to avoid conversion issues
            class_channel = tf.gather(predictions[0], pred_index)
        
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
    """OpenCV-based overlay with proper error handling for cloud deployment"""
    try:
        # Convert to arrays
        img = np.array(image.resize((224, 224)))
        heatmap = np.array(heatmap)
        
        # Fix the division by zero issue that was causing NaN values
        heatmap_min = heatmap.min()
        heatmap_max = heatmap.max()
        
        # Normalize heatmap properly (avoiding division by zero)
        if heatmap_max > heatmap_min and not np.isnan(heatmap_max) and not np.isnan(heatmap_min):
            heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)
        else:
            # If all values are the same or NaN, create a uniform low-intensity heatmap
            heatmap = np.full_like(heatmap, 0.1)
        
        # Convert to uint8 safely (this was line 70 in the error)
        heatmap = np.clip(heatmap, 0, 1)  # Ensure values are in [0,1]
        heatmap = np.uint8(255 * heatmap)
        
        # Use OpenCV JET colormap (red=high, blue=low) - more advanced and reliable
        colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)
        
        # Ensure image is RGB
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.shape[2] == 4:  # RGBA
            img = img[:, :, :3]
        
        # Overlay with proper data types to avoid casting issues
        img = img.astype(np.float32)
        colored_heatmap = colored_heatmap.astype(np.float32)
        
        # Blend the images
        result = img * (1 - alpha) + colored_heatmap * alpha
        return np.clip(result, 0, 255).astype(np.uint8)
        
    except Exception as e:
        st.error(f"Overlay failed: {e}")
        return np.array(image.resize((224, 224)))

# ============================================================================
# V2.0 NEW FUNCTIONS
# ============================================================================

def vlm_explanation_groq(cam_pil, pred_class, confidence, preds, class_names):
    """Generate VLM-style explanation using Groq's text models (vision models deprecated)"""
    if not VLM_ENABLED:
        return (None, "error")
    
    try:
        # Get API key fresh each time (Streamlit caching issue workaround)
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            try:
                api_key = st.secrets.get("GROQ_API_KEY")
            except:
                pass
        
        if not api_key:
            st.warning("GROQ_API_KEY not found in environment or secrets")
            return (None, "error")
        
        # Since Groq removed vision models, use text-only approach
        # Analyze heatmap and describe it in text for the LLM
        
        # Convert heatmap to analyzable data
        cam_array = np.array(cam_pil)
        heatmap_data = cam_array[:,:,0]  # Red channel (heatmap intensity)
        
        # Find regions of high activation
        high_activation_mask = heatmap_data > 200
        activation_pct = float(np.mean(high_activation_mask) * 100)
        
        # Determine region (simplified brain quadrants)
        height, width = heatmap_data.shape
        top_activation = np.mean(heatmap_data[:height//2, :])
        bottom_activation = np.mean(heatmap_data[height//2:, :])
        left_activation = np.mean(heatmap_data[:, :width//2])
        right_activation = np.mean(heatmap_data[:, width//2:])
        
        regions = []
        if top_activation > 150: regions.append("superior regions")
        if bottom_activation > 150: regions.append("inferior regions")
        if left_activation > 150: regions.append("left hemisphere")
        if right_activation > 150: regions.append("right hemisphere")
        region_str = " and ".join(regions) if regions else "central areas"
        
        # Get runner-up prediction
        runner_up_idx = np.argsort(preds[0])[-2]
        runner_up_class = class_names[runner_up_idx]
        runner_up_conf = float(preds[0][runner_up_idx]) * 100
        
        # Build detailed prompt
        prompt = f"""You are an expert neuroradiologist analyzing brain MRI classification results with AI model explainability data.

**Classification Results:**
- Predicted Class: {pred_class.title()}
- Confidence: {confidence:.1f}%
- Runner-up: {runner_up_class.title()} ({runner_up_conf:.1f}%)

**Grad-CAM Analysis (AI attention heatmap):**
- High activation area: {activation_pct:.1f}% of image
- Primary activation regions: {region_str}
- Activation pattern: {"Focal" if activation_pct < 15 else "Moderate" if activation_pct < 35 else "Diffuse"}

Provide a clinical explanation (3-4 sentences) covering:
1. What the classification detected and the significance of the Grad-CAM activation pattern
2. Typical imaging features associated with {pred_class}
3. {"Confidence is moderate - recommend additional imaging for confirmation" if confidence < 80 else "High confidence - findings are reliable"}
4. Clinical recommendation (consult specialist, follow-up, etc.)

Write professionally for medical practitioners."""

        # Call Groq with text-only model
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # Fast, accurate text model
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300
        )
        
        explanation = response.choices[0].message.content.strip()
        return (explanation, "vlm_groq_text")
        
    except Exception as e:
        st.warning(f"Groq explanation failed: {str(e)}")
        return (None, "error")

def template_explanation(pred_class, confidence, preds, class_names, heatmap):
    """Generate template-based explanation with uncertainty and activation analysis"""
    # Compute uncertainty metrics
    probs = preds[0]
    num_classes = len(class_names)
    
    # Shannon entropy normalized
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    max_entropy = np.log(num_classes)
    uncertainty_ratio = float(entropy / max_entropy)
    
    # High activation percentage
    heatmap_normalized = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-10)
    high_activation_pct = float(np.mean(heatmap_normalized > 0.7) * 100)
    
    # Classify activation focus
    if high_activation_pct < 10:
        activation_focus = "focal"
    elif high_activation_pct < 30:
        activation_focus = "moderate"
    else:
        activation_focus = "diffuse"
    
    # Class-specific explanations
    explanations = {
        "glioma": f"The model detected a **glioma** pattern with {confidence:.1f}% confidence. The Grad-CAM heatmap shows {activation_focus} activation ({high_activation_pct:.1f}% high-intensity regions), typically indicating infiltrative tumor characteristics in brain parenchyma. Gliomas often present with irregular borders and variable enhancement patterns. Clinical correlation with contrast-enhanced sequences and patient symptoms is recommended for grading and treatment planning.",
        
        "meningioma": f"The analysis identified a **meningioma** signature at {confidence:.1f}% confidence. The {activation_focus} activation pattern ({high_activation_pct:.1f}% high-intensity areas) is consistent with extra-axial tumor location, often showing well-defined margins along the dural surface. Meningiomas typically display homogeneous enhancement and may show calcification or hyperostosis. Surgical consultation recommended for symptomatic cases or large lesions.",
        
        "notumor": f"The scan appears **normal** with {confidence:.1f}% confidence for absence of tumor. The {activation_focus} activation pattern ({high_activation_pct:.1f}% highlighted regions) shows no focal pathological features. The model detected routine anatomical structures without suspicious enhancement or mass effect. If clinical symptoms persist, consider follow-up imaging or evaluation for non-neoplastic conditions.",
        
        "pituitary": f"A **pituitary tumor** pattern was identified with {confidence:.1f}% confidence. The Grad-CAM shows {activation_focus} activation ({high_activation_pct:.1f}% high-intensity focus) in the sellar region, consistent with pituitary adenoma or other sellar/parasellar masses. These lesions may affect hormonal function or cause visual field defects via optic chiasm compression. Endocrinology referral and visual field testing recommended."
    }
    
    explanation = explanations.get(pred_class, "Analysis completed.")
    
    # Add secondary consideration if runner-up is significant
    sorted_indices = np.argsort(probs)[::-1]
    runner_up_idx = sorted_indices[1]
    runner_up_prob = float(probs[runner_up_idx]) * 100
    
    if runner_up_prob > 15:
        runner_up_class = class_names[runner_up_idx]
        explanation += f"\n\n**Secondary Consideration:** {runner_up_class.title()} shows {runner_up_prob:.1f}% probability. Differential diagnosis should include this possibility, especially if clinical presentation is atypical."
    
    return (explanation, uncertainty_ratio, high_activation_pct)

def detect_failure(confidence, preds, class_names, uncertainty_ratio, 
                  conf_thresh=50.0, entropy_thresh=0.65, margin_thresh=20.0):
    """Detect potential failure cases based on multiple criteria"""
    probs = preds[0]
    sorted_probs = np.sort(probs)[::-1]
    
    # Check 1: Low confidence
    if confidence < conf_thresh:
        return (True, "low_confidence", 
                f"⚠️ **Low Confidence Alert**: Prediction confidence ({confidence:.1f}%) is below {conf_thresh}% threshold. Results may be unreliable.")
    
    # Check 2: High entropy (uncertain distribution)
    if uncertainty_ratio > entropy_thresh:
        return (True, "high_entropy",
                f"⚠️ **High Uncertainty Alert**: Prediction entropy ({uncertainty_ratio:.2f}) indicates high uncertainty across classes. Consider additional imaging.")
    
    # Check 3: Borderline (small margin between top 2)
    top2_margin = float((sorted_probs[0] - sorted_probs[1]) * 100)
    if top2_margin < margin_thresh:
        return (True, "borderline",
                f"⚠️ **Borderline Case Alert**: Top two predictions are very close (margin: {top2_margin:.1f}%). Differential diagnosis recommended.")
    
    return (False, None, None)

def log_failure_case(filename, pred_class, confidence, failure_type, 
                    uncertainty_ratio, preds, class_names):
    """Log failure case to session state"""
    if 'failure_log' not in st.session_state:
        st.session_state.failure_log = []
    
    case = {
        "timestamp": datetime.now().isoformat(),
        "filename": filename,
        "predicted_class": pred_class,
        "confidence": float(confidence),
        "failure_type": failure_type,
        "uncertainty_score": float(uncertainty_ratio),
        "all_predictions": {
            class_names[i]: float(preds[0][i] * 100) 
            for i in range(len(class_names))
        }
    }
    
    st.session_state.failure_log.append(case)

def create_report_v2(pred_class, confidence, preds, class_names, 
                     high_activation_pct, explanation, explanation_source,
                     uncertainty_ratio, is_failure, failure_type):
    """Create v2.0 JSON report with full analytics"""
    sorted_probs = np.sort(preds[0])[::-1]
    top2_margin = float((sorted_probs[0] - sorted_probs[1]) * 100)
    
    # Source label mapping
    source_labels = {
        "vlm_groq_text": "Groq AI (Llama-3.3-70B)",
        "template": "Template-based Analysis"
    }
    
    # Uncertainty label
    if uncertainty_ratio < 0.3:
        uncertainty_label = "Low"
    elif uncertainty_ratio < 0.65:
        uncertainty_label = "Moderate"
    else:
        uncertainty_label = "High"
    
    report = {
        "report_version": "2.0",
        "timestamp": datetime.now().isoformat(),
        "scan_info": {
            "predicted_class": pred_class,
            "confidence_pct": float(confidence),
            "all_class_probabilities_pct": {
                class_names[i]: float(preds[0][i] * 100) 
                for i in range(len(class_names))
            },
            "top2_margin_pct": top2_margin
        },
        "explainability": {
            "gradcam_high_activation_pct": float(high_activation_pct),
            "explanation_source": explanation_source,
            "explanation_source_label": source_labels.get(explanation_source, "Unknown"),
            "ai_explanation": explanation
        },
        "reliability": {
            "uncertainty_score": float(uncertainty_ratio),
            "uncertainty_label": uncertainty_label,
            "is_failure_case": is_failure,
            "failure_type": failure_type if is_failure else None,
            "reliability_passed": not is_failure
        },
        "model_info": {
            "architecture": "DenseNet121",
            "last_conv_layer": "conv5_block16_concat",
            "input_size": "224x224",
            "num_classes": 4,
            "overall_accuracy": "91.2%"
        }
    }
    
    return json.dumps(report, indent=2)

def create_pdf_report(pred_class, confidence, preds, class_names, 
                      high_activation_pct, explanation, explanation_source,
                      uncertainty_ratio, is_failure, failure_type,
                      cam_pil, original_image, filename="scan"):
    """Generate PDF report with images and analytics"""
    if not PDF_AVAILABLE:
        return None
    
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=12,
            spaceBefore=12
        )
        
        # Title
        story.append(Paragraph("🧠 Brain Tumor MRI Analysis Report v2.0", title_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Report metadata
        metadata = [
            ["Report Generated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ["Scan File:", filename],
            ["Analysis Version:", "2.0 (VLM Enhanced)"]
        ]
        t = Table(metadata, colWidths=[2*inch, 4*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ecf0f1')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.3*inch))
        
        # Classification Results
        story.append(Paragraph("🎯 Classification Results", heading_style))
        
        # Tumor info
        tumor_info_map = {
            "glioma": ("High", "Surgery + Radiation + Chemotherapy"),
            "meningioma": ("Low-Medium", "Surgery or Observation"),
            "notumor": ("None", "No treatment needed"),
            "pituitary": ("Medium", "Surgery or Medication")
        }
        severity, treatment = tumor_info_map.get(pred_class, ("Unknown", "Consult physician"))
        
        results_data = [
            ["Metric", "Value"],
            ["Predicted Type", pred_class.upper()],
            ["Confidence", f"{confidence:.1f}%"],
            ["Severity Level", severity],
            ["Recommended Treatment", treatment],
        ]
        
        t = Table(results_data, colWidths=[2.5*inch, 3.5*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('BACKGROUND', (0, 1), (0, -1), colors.HexColor('#ecf0f1')),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.2*inch))
        
        # All class probabilities
        story.append(Paragraph("📊 All Class Probabilities", heading_style))
        prob_data = [["Class", "Probability"]]
        for i, class_name in enumerate(class_names):
            prob = float(preds[0][i] * 100)
            prob_data.append([class_name.upper(), f"{prob:.1f}%"])
        
        t = Table(prob_data, colWidths=[3*inch, 3*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2ecc71')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('BACKGROUND', (0, 1), (0, -1), colors.HexColor('#ecf0f1')),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.3*inch))
        
        # Add page break before images
        story.append(PageBreak())
        
        # Images section
        story.append(Paragraph("📷 MRI Scans & Analysis", heading_style))
        story.append(Spacer(1, 0.1*inch))
        
        # Original and Grad-CAM side by side
        img_data = []
        
        # Save original image to buffer
        orig_buffer = io.BytesIO()
        original_image.save(orig_buffer, format='PNG')
        orig_buffer.seek(0)
        orig_img = RLImage(orig_buffer, width=2.5*inch, height=2.5*inch)
        
        # Save grad-cam to buffer
        cam_buffer = io.BytesIO()
        cam_pil.save(cam_buffer, format='PNG')
        cam_buffer.seek(0)
        cam_img = RLImage(cam_buffer, width=2.5*inch, height=2.5*inch)
        
        img_table_data = [
            [Paragraph("<b>Original MRI Scan</b>", styles['Normal']), 
             Paragraph("<b>Grad-CAM Heatmap</b>", styles['Normal'])],
            [orig_img, cam_img]
        ]
        
        img_table = Table(img_table_data, colWidths=[3*inch, 3*inch])
        img_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        story.append(img_table)
        story.append(Spacer(1, 0.3*inch))
        
        # AI Explanation
        story.append(Paragraph("🤖 AI Explanation", heading_style))
        source_label = "Groq Llama-3.2-11B Vision" if explanation_source == "vlm_llama32_groq" else "Template-based Analysis"
        story.append(Paragraph(f"<i>Source: {source_label}</i>", styles['Normal']))
        story.append(Spacer(1, 0.1*inch))
        
        explanation_style = ParagraphStyle(
            'Explanation',
            parent=styles['BodyText'],
            fontSize=10,
            alignment=TA_JUSTIFY,
            spaceAfter=12
        )
        story.append(Paragraph(explanation.replace('\n', '<br/>'), explanation_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Reliability Metrics
        story.append(Paragraph("📈 Reliability & Quality Metrics", heading_style))
        
        uncertainty_label = "Low" if uncertainty_ratio < 0.3 else ("Moderate" if uncertainty_ratio < 0.65 else "High")
        sorted_probs = np.sort(preds[0])[::-1]
        top2_margin = float((sorted_probs[0] - sorted_probs[1]) * 100)
        
        reliability_data = [
            ["Metric", "Value", "Assessment"],
            ["Uncertainty Score", f"{uncertainty_ratio:.3f}", uncertainty_label],
            ["High Activation Area", f"{high_activation_pct:.1f}%", "Grad-CAM focus"],
            ["Top-2 Margin", f"{top2_margin:.1f}%", "Prediction separation"],
            ["Reliability Status", "FAIL" if is_failure else "PASS", failure_type or "All checks passed"],
        ]
        
        t = Table(reliability_data, colWidths=[2*inch, 2*inch, 2*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e74c3c')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('BACKGROUND', (0, 1), (0, -1), colors.HexColor('#ecf0f1')),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.3*inch))
        
        # Model Information
        story.append(Paragraph("⚙️ Model Information", heading_style))
        model_data = [
            ["Architecture", "DenseNet121"],
            ["Convolutional Layer", "conv5_block16_concat"],
            ["Input Size", "224×224 RGB"],
            ["Number of Classes", "4"],
            ["Overall Accuracy", "91.2%"],
        ]
        
        t = Table(model_data, colWidths=[2.5*inch, 3.5*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ecf0f1')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.4*inch))
        
        # Disclaimer
        story.append(Paragraph("⚠️ Medical Disclaimer", heading_style))
        disclaimer_text = """This analysis is generated by an AI system for educational and research purposes only. 
        It should NOT be used for actual medical diagnosis or treatment decisions. Always consult qualified 
        healthcare professionals for medical advice, diagnosis, and treatment. The AI model may produce errors 
        or misclassifications."""
        
        disclaimer_style = ParagraphStyle(
            'Disclaimer',
            parent=styles['BodyText'],
            fontSize=9,
            textColor=colors.HexColor('#7f8c8d'),
            alignment=TA_JUSTIFY,
            leftIndent=10,
            rightIndent=10
        )
        story.append(Paragraph(disclaimer_text, disclaimer_style))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
        
    except Exception as e:
        st.error(f"PDF generation failed: {str(e)}")
        return None

# ============================================================================
# MODEL LOADING
# ============================================================================

class PatchedInputLayer(tf.keras.layers.InputLayer):
    def __init__(self, *args, **kwargs):
        if "batch_shape" in kwargs and "batch_input_shape" not in kwargs:
            kwargs["batch_input_shape"] = kwargs.pop("batch_shape")
        super().__init__(*args, **kwargs)


@st.cache_resource
def load_densenet_model():
    try:
        with st.spinner("Loading AI model..."):
            model = load_model(
                "densenet121_brain_tumor_best.h5",
                custom_objects={"InputLayer": PatchedInputLayer},
                compile=False,
            )
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()
        return None

# Initialize
model = load_densenet_model()
last_conv_layer_name = "conv5_block16_concat"
class_names = ["glioma", "meningioma", "notumor", "pituitary"]

# Tumor metadata
tumor_info = {
    "glioma": {"severity": "High", "color": "🔴", "treatment": "Surgery + Radiation + Chemo"},
    "meningioma": {"severity": "Low-Medium", "color": "🟡", "treatment": "Surgery or Observation"},
    "notumor": {"severity": "None", "color": "🟢", "treatment": "No treatment needed"},
    "pituitary": {"severity": "Medium", "color": "🟠", "treatment": "Surgery or Medication"}
}

# ============================================================================
# UI LAYOUT
# ============================================================================

# Sidebar
with st.sidebar:
    st.title("🧠 Brain Tumor Classifier")
    st.markdown("### v2.0 - AI Enhanced")
    
    st.markdown("---")
    st.markdown("#### 🤖 AI Status")
    if VLM_ENABLED:
        st.success("✅ AI Explanations Active")
    else:
        st.warning("⚠️ Template Mode")
    
    st.markdown("---")
    
    # Smart defaults for failure detection (hidden from users)
    conf_threshold = 50.0
    entropy_threshold = 0.65
    margin_threshold = 20.0
    
    st.markdown("#### 📊 Model Info")
    st.markdown("""
    **Architecture:** DenseNet121  
    **Classes:** 4  
    **Accuracy:** 91.2%  
    **Explainability:** Grad-CAM  
    **AI Engine:** Groq Llama-3.3-70B
    """)
    
    st.markdown("---")
    st.markdown("#### ℹ️ About")
    st.markdown("""
    Deep learning system with explainable 
    AI for brain tumor classification.
    
    **Automated Quality Checks:**
    - Low confidence detection
    - High uncertainty alerts  
    - Borderline case flagging
    """)
    
    st.markdown("---")
    st.markdown("**⚠️ For Research Use Only**")
    st.caption("*Always consult medical professionals*")

# Main content - Tabs
tab1, tab2 = st.tabs(["📊 Analysis", "⚠️ Failure Dashboard"])

# ============================================================================
# TAB 1: ANALYSIS
# ============================================================================

with tab1:
    st.title("🧠 Brain Tumor MRI Classification & Analysis v2.0")
    st.markdown("Upload an MRI scan for AI-powered analysis with VLM explanations and reliability scoring")
    
    uploaded_file = st.file_uploader("📁 Upload MRI Image (JPEG, PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        filename = uploaded_file.name
        
        # Load and analyze image
        image = Image.open(uploaded_file)
        img_array = get_img_array(image)
        preds = model.predict(img_array, verbose=0)
        pred_idx = np.argmax(preds)
        pred_class = class_names[pred_idx]
        confidence = float(preds[0][pred_idx]) * 100
        
        # Generate Grad-CAM
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
        cam_img = overlay_heatmap(image, heatmap)
        cam_pil = Image.fromarray(cam_img)
        
        # Row 1: Image + Results Table
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption="📷 Uploaded MRI Scan", use_container_width=True)
        
        with col2:
            st.markdown("### 🎯 Classification Results")
            info = tumor_info[pred_class]
            
            # Results table
            st.markdown(f"""
            | **Field** | **Value** |
            |-----------|-----------|
            | **{info['color']} Type** | **{pred_class.upper()}** |
            | **📊 Confidence** | **{confidence:.1f}%** |
            | **⚕️ Severity** | {info['severity']} |
            | **💊 Treatment** | {info['treatment']} |
            """)
        
        st.markdown("---")
        
        # Row 2: Probabilities + Grad-CAM
        col3, col4 = st.columns([1, 1])
        
        with col3:
            st.markdown("### 📈 All Class Probabilities")
            for i, class_name in enumerate(class_names):
                prob = float(preds[0][i]) * 100
                st.markdown(f"**{class_name.upper()}**")
                st.progress(prob / 100.0)
                st.markdown(f"{prob:.1f}%")
                st.markdown("")
            
            st.markdown("### 📚 About This Type")
            about_text = {
                "glioma": "Tumor originating from glial cells in brain/spinal cord. Ranges from slow-growing to highly aggressive.",
                "meningioma": "Usually benign tumor from meninges (brain covering). Often curable with surgical removal.",
                "notumor": "No tumor detected. Brain tissue appears normal. No immediate intervention needed.",
                "pituitary": "Tumor in pituitary gland affecting hormone production. May cause endocrine or visual symptoms."
            }
            st.info(about_text[pred_class])
        
        with col4:
            st.markdown("### 🔥 Grad-CAM Heatmap")
            st.markdown("*Red/yellow regions = high AI attention*")
            st.image(cam_pil, caption="Grad-CAM Overlay", use_container_width=True)
        
        st.markdown("---")
        
        # AI Explanation Section
        st.markdown("### 🤖 AI Explanation")
        
        explanation = None
        explanation_source = "template"
        high_activation_pct = 0
        uncertainty_ratio = 0
        
        with st.spinner("Generating explanation..."):
            # Try VLM first
            if VLM_ENABLED:
                vlm_result, vlm_source = vlm_explanation_groq(cam_pil, pred_class, confidence, preds, class_names)
                if vlm_result:
                    explanation = vlm_result
                    explanation_source = vlm_source
                    # Still need template for metrics
                    _, uncertainty_ratio, high_activation_pct = template_explanation(
                        pred_class, confidence, preds, class_names, heatmap
                    )
            
            # Fallback to template
            if explanation is None:
                explanation, uncertainty_ratio, high_activation_pct = template_explanation(
                    pred_class, confidence, preds, class_names, heatmap
                )
                explanation_source = "template"
        
        # Display explanation
        source_badge = "🤖 AI-Generated (Groq)" if explanation_source == "vlm_groq_text" else "📋 Template-Based"
        st.info(f"**{source_badge}**\n\n{explanation}")
        
        # Uncertainty metrics
        col5, col6 = st.columns(2)
        with col5:
            st.markdown("**🎲 Uncertainty Score**")
            st.progress(uncertainty_ratio)
            st.caption(f"{uncertainty_ratio:.2f} (0=certain, 1=maximum uncertainty)")
        
        with col6:
            st.markdown("**🔥 High Activation Area**")
            st.progress(high_activation_pct / 100.0)
            st.caption(f"{high_activation_pct:.1f}% of heatmap shows high intensity")
        
        st.markdown("---")
        
        # Failure detection
        is_failure, failure_type, failure_msg = detect_failure(
            confidence, preds, class_names, uncertainty_ratio,
            conf_threshold, entropy_threshold, margin_threshold
        )
        
        if is_failure:
            st.error(failure_msg)
            st.markdown("**Recommendation:** Consider additional imaging modalities, clinical correlation, or expert radiologist review.")
            
            # Log failure
            log_failure_case(filename, pred_class, confidence, failure_type, 
                           uncertainty_ratio, preds, class_names)
        else:
            st.success("✅ **Reliability Check Passed** — All quality metrics within acceptable ranges.")
        
        st.markdown("---")
        
        # Download section
        st.markdown("### 💾 Download Results")
        
        col7, col8, col9 = st.columns(3)
        
        with col7:
            # JSON Report
            report_json = create_report_v2(
                pred_class, confidence, preds, class_names,
                high_activation_pct, explanation, explanation_source,
                uncertainty_ratio, is_failure, failure_type
            )
            
            st.download_button(
                label="📄 Download JSON Report",
                data=report_json,
                file_name=f"brain_tumor_report_{pred_class}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col8:
            # PDF Report
            if PDF_AVAILABLE:
                pdf_data = create_pdf_report(
                    pred_class, confidence, preds, class_names,
                    high_activation_pct, explanation, explanation_source,
                    uncertainty_ratio, is_failure, failure_type,
                    cam_pil, image, filename
                )
                
                if pdf_data:
                    st.download_button(
                        label="📑 Download PDF Report",
                        data=pdf_data,
                        file_name=f"brain_tumor_report_{pred_class}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )
                else:
                    st.error("PDF generation failed")
            else:
                st.warning("Install reportlab for PDF")
        
        with col9:
            # Grad-CAM Image
            img_buffer = io.BytesIO()
            cam_pil.save(img_buffer, format='PNG')
            
            st.download_button(
                label="🖼️ Download Grad-CAM",
                data=img_buffer.getvalue(),
                file_name=f"gradcam_{pred_class}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png"
            )
        
        st.markdown("---")
        st.warning("⚠️ **Medical Disclaimer:** This tool is for educational purposes only. Not for actual diagnosis. Always consult qualified healthcare professionals.")
    
    else:
        # Empty state
        st.info("👆 Upload an MRI scan above to begin analysis")
        st.markdown("""
        ### Features
        - **4-Class Classification**: Glioma, Meningioma, No Tumor, Pituitary
        - **VLM Explanations**: AI-generated clinical interpretations (when enabled)
        - **Grad-CAM**: Visual explanation of model decisions
        - **Failure Detection**: Automated quality and reliability checks
        - **Downloadable Reports**: JSON format with full analytics
        """)

# ============================================================================
# TAB 2: FAILURE DASHBOARD
# ============================================================================

with tab2:
    st.title("⚠️ Failure Case Dashboard")
    st.markdown("Monitor and analyze cases that triggered reliability alerts")
    
    if 'failure_log' not in st.session_state:
        st.session_state.failure_log = []
    
    failure_log = st.session_state.failure_log
    
    if len(failure_log) > 0:
        # Metrics row
        total_failures = len(failure_log)
        low_conf_count = sum(1 for f in failure_log if f['failure_type'] == 'low_confidence')
        high_entropy_count = sum(1 for f in failure_log if f['failure_type'] == 'high_entropy')
        borderline_count = sum(1 for f in failure_log if f['failure_type'] == 'borderline')
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📊 Total Failures", total_failures)
        col2.metric("⬇️ Low Confidence", low_conf_count)
        col3.metric("🎲 High Entropy", high_entropy_count)
        col4.metric("⚖️ Borderline", borderline_count)
        
        st.markdown("---")
        
        # Class breakdown
        st.markdown("### 📈 Failure Breakdown by Class")
        class_counts = {}
        for f in failure_log:
            cls = f['predicted_class']
            class_counts[cls] = class_counts.get(cls, 0) + 1
        
        cols = st.columns(4)
        for i, cls in enumerate(class_names):
            count = class_counts.get(cls, 0)
            cols[i].metric(cls.upper(), count)
        
        st.markdown("---")
        
        # Case log
        st.markdown("### 📋 Failure Case Log")
        st.caption(f"Showing {len(failure_log)} cases (newest first)")
        
        for idx, case in enumerate(reversed(failure_log)):
            with st.expander(f"Case #{len(failure_log) - idx}: {case['filename']} — {case['predicted_class'].upper()} ({case['confidence']:.1f}%)"):
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.markdown(f"""
                    **Timestamp:** {case['timestamp']}  
                    **Predicted:** {case['predicted_class'].upper()}  
                    **Confidence:** {case['confidence']:.1f}%  
                    **Failure Type:** {case['failure_type']}  
                    **Uncertainty:** {case['uncertainty_score']:.3f}
                    """)
                
                with col_b:
                    st.markdown("**All Predictions:**")
                    for cls, prob in case['all_predictions'].items():
                        st.write(f"- {cls.upper()}: {prob:.1f}%")
        
        st.markdown("---")
        
        # Download and clear
        col_dl, col_clear = st.columns([1, 1])
        
        with col_dl:
            failure_json = json.dumps(failure_log, indent=2)
            st.download_button(
                label="📥 Download Failure Log (JSON)",
                data=failure_json,
                file_name=f"failure_cases_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col_clear:
            if st.button("🗑️ Clear Failure Log"):
                st.session_state.failure_log = []
                st.rerun()
    
    else:
        # Empty state
        st.info("No failure cases logged yet")
        st.markdown("### Failure Detection Types")
        st.markdown("""
        | **Type** | **Trigger Condition** | **Interpretation** |
        |----------|----------------------|-------------------|
        | **Low Confidence** | Prediction confidence < threshold | Model is uncertain about classification |
        | **High Entropy** | Shannon entropy > threshold | Probability distribution is flat/uncertain |
        | **Borderline** | Top-2 margin < threshold | Two classes have similar probabilities |
        """)
        st.markdown("Failure cases will appear here after analyzing scans that trigger these conditions.")