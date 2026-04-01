
# 🧠 Brain Tumor MRI Classification & Analysis v2.0

An AI-powered web application that classifies brain tumors from MRI scans using deep learning. Built with DenseNet121 architecture and featuring explainable AI through Grad-CAM visualization, VLM-generated clinical explanations, and automated failure detection.

## 🎯 Features

### Phase 1 (Core Classification)
- **4-Class Classification**: Glioma, Meningioma, Pituitary Tumor, No Tumor
- **High Accuracy**: DenseNet121 model trained on medical imaging dataset (91.2%)
- **Explainable AI**: Grad-CAM heatmaps show AI decision-making process
- **User-Friendly Interface**: Clean Streamlit web application
- **Confidence Scoring**: Color-coded confidence levels for predictions
- **Download Reports**: Export analysis results and visualizations
- **Medical Information**: Detailed descriptions of each tumor type

### Phase 2 (Advanced Analytics) ✨ NEW
- **🤖 VLM Integration**: AI-generated clinical explanations using Groq's Llama-3.2-11B Vision model
- **⚠️ Failure Detection**: Automated quality checks with three detection modes:
  - Low Confidence: Predictions below confidence threshold
  - High Entropy: Uncertain probability distributions
  - Borderline Cases: Small margins between top predictions
- **📊 Failure Dashboard**: Monitor and analyze cases triggering reliability alerts
- **📈 Uncertainty Metrics**: Shannon entropy analysis and activation percentage tracking
- **📋 Enhanced Reports v2.0**: Comprehensive JSON + PDF reports with explainability and reliability data
- **📑 PDF Generation**: Professional printable reports with images, metrics, and clinical explanations
- **🔍 Template Fallback**: Advanced rule-based explanations when VLM unavailable

## 🚀 Live Demo

**🔗 [Try the Live Application](https://brain-tumor-classification-80.streamlit.app/)**

Experience the brain tumor classification system in action! Upload your MRI scans and get instant AI-powered analysis with explainable Grad-CAM visualizations.

## 🎥 Video Demo

**📹 [Watch Full Demo Video](https://drive.google.com/file/d/15w1Yy5LujRQ3Bt9QcRs8dDEfMjoQvL_S/view?usp=sharing)**

See the complete application workflow including:
- MRI image upload process
- Real-time classification results
- Grad-CAM heatmap visualization
- Confidence scoring system
- Medical information display

## 📸 Screenshots

### Main Interface
![Brain Tumor Classification Interface](https://raw.githubusercontent.com/ashutosh8021/brain-tumor-classification/main/screenshots/main_interface.png)
*Clean two-column layout with instant classification results and confidence scoring*

### Grad-CAM Analysis
![Grad-CAM Heatmap Visualization](https://raw.githubusercontent.com/ashutosh8021/brain-tumor-classification/main/screenshots/gradcam_demo.png)
*AI explanation through heatmap visualization showing decision-making process*

### Real-time Results
![Classification Results](https://raw.githubusercontent.com/ashutosh8021/brain-tumor-classification/main/screenshots/classification_results.png)
*Color-coded confidence levels and detailed medical information*

### Download Features
![Download Features](https://raw.githubusercontent.com/ashutosh8021/brain-tumor-classification/main/screenshots/download_features.png)
*Export analysis results and visualizations for record keeping*

## 🛠️ Technology Stack

- **Deep Learning**: TensorFlow/Keras, DenseNet121
- **VLM**: Groq API with Llama-3.2-11B Vision (Phase 2)
- **Web Framework**: Streamlit
- **Image Processing**: OpenCV, PIL
- **Visualization**: Grad-CAM, Matplotlib
- **PDF Generation**: ReportLab
- **Data Science**: NumPy

## 📦 Installation

### Prerequisites
- Python 3.8+
- pip package manager
- (Optional) Groq API key for VLM explanations

### Local Setup
```bash
# Clone the repository
git clone https://github.com/ashutosh8021/brain-tumor-classification.git
cd brain-tumor-classification

# Create virtual environment
python -m venv env
# On Windows:
env\Scripts\activate
# On macOS/Linux:
source env/bin/activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Set up environment variables for VLM
# Copy .env.example to .env and add your Groq API key
cp .env.example .env
# Edit .env and set: GROQ_API_KEY=gsk_your_actual_key_here

# Run the application
streamlit run app.py
```

### Streamlit Cloud Setup

1. **Fork this repository** to your GitHub account

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select this repository
   - Click "Deploy"

3. **Configure Secrets** (for VLM features):
   - In Streamlit Cloud dashboard, go to App Settings → Secrets
   - Add the following:
   ```toml
   GROQ_API_KEY = "gsk_your_actual_key_here"
   ```
   - Get your API key from: [console.groq.com/keys](https://console.groq.com/keys)

**Note:** The app works fully without the API key - it will use template-based explanations as fallback.

## 🌍 Environment Variables

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `GROQ_API_KEY` | No | Groq API key for VLM explanations | None (uses template fallback) |

**Setting Environment Variables:**
- **Local Development**: Create `.env` file (see `.env.example`)
- **Streamlit Cloud**: Add to App Settings → Secrets → `secrets.toml`
- **Heroku/AWS**: Use platform-specific environment variable settings

## 📁 Project Structure

```
brain-tumor-classification/
├── app.py                          # Main Streamlit application v2.0
├── densenet121_brain_tumor_best.h5 # Trained DenseNet121 model
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variables template
├── .streamlit/
│   └── secrets.toml               # Streamlit secrets template
├── README.md                      # Project documentation
├── .gitignore                     # Git ignore file
└── screenshots/                   # Application screenshots
```

## 🧠 Model Information & Dataset

### Dataset Source
- **Dataset**: [Brain Tumor MRI Dataset - Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **Classes**: 4 tumor types (Glioma, Meningioma, No Tumor, Pituitary)
- **Images**: High-quality MRI scans for medical classification
- **Format**: JPEG images, 224x224 resolution after preprocessing

### Model Architecture
- **Architecture**: DenseNet121 (pre-trained on ImageNet)
- **Input Size**: 224x224 RGB images
- **Classes**: 4 (Glioma, Meningioma, No Tumor, Pituitary)
- **Framework**: TensorFlow/Keras
- **Last Convolutional Layer**: conv5_block16_concat (for Grad-CAM)
- **Parameters**: ~8M trainable parameters

### Performance Metrics

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Glioma | 0.94 | 0.89 | 0.92 |
| Meningioma | 0.87 | 0.79 | 0.83 |
| Pituitary | 0.88 | 1.00 | 0.94 |
| No Tumor | 0.95 | 0.97 | 0.96 |

**Overall Accuracy**: 91.2% on validation dataset

## 🚀 Phase 2 Features (v2.0)

### 🤖 VLM-Powered Explanations

The app now integrates **Vision Language Models (VLM)** via Groq API for AI-generated clinical explanations:

- **Model**: Llama-3.2-11B Vision Preview
- **Input**: Grad-CAM overlay image + classification results
- **Output**: 3-4 sentence clinical explanation covering:
  - What was detected and heatmap activation locations
  - Supporting imaging features visible in the scan
  - Uncertainty notes for low-confidence cases
  - Clinical recommendations (e.g., specialist referral, follow-up imaging)
- **Fallback**: Template-based explanations always available when VLM unavailable
- **Configuration**: Environment variable only (no UI input field)

**VLM Setup:**
```bash
# Option 1: Local development (.env file)
GROQ_API_KEY=gsk_your_actual_key_here

# Option 2: Streamlit Cloud (App Settings → Secrets)
GROQ_API_KEY = "gsk_your_actual_key_here"
```

Get your free API key: [console.groq.com/keys](https://console.groq.com/keys)

### ⚠️ Failure Detection System

Automated quality control with three detection modes:

| **Failure Type** | **Trigger Condition** | **Interpretation** | **Default Threshold** |
|------------------|----------------------|-------------------|---------------------|
| **Low Confidence** | Prediction confidence < threshold | Model uncertain about classification | 50% |
| **High Entropy** | Shannon entropy > threshold | Probability distribution is flat/uncertain | 0.65 (normalized) |
| **Borderline** | Top-2 margin < threshold | Two classes have similar probabilities | 20% |

**How it works:**
1. Every prediction is evaluated against all three criteria in order
2. If any criterion triggers, case is logged as a failure
3. User receives specific alert with recommendation
4. Case appears in Failure Dashboard for review

**Customizable Thresholds:**
- Adjust via sidebar sliders in real-time
- Thresholds persist within session
- No configuration file needed

### 📊 Failure Dashboard

Track and analyze problematic cases:

- **Metrics Overview**: Total failures, breakdown by type
- **Class Distribution**: See which tumor types trigger most failures
- **Case Log**: Expandable entries with full prediction data (newest first)
- **Export**: Download failure log as JSON for external analysis
- **Clear Log**: Reset dashboard with one-click (uses `st.rerun()`)
- **Empty State**: Helpful table explaining failure types when no cases logged

**Dashboard Fields per Case:**
```json
{
  "timestamp": "2026-04-01T08:30:00.000Z",
  "filename": "scan_123.jpg",
  "predicted_class": "glioma",
  "confidence": 48.3,
  "failure_type": "low_confidence",
  "uncertainty_score": 0.72,
  "all_predictions": {
    "glioma": 48.3,
    "meningioma": 31.2,
    "pituitary": 12.1,
    "notumor": 8.4
  }
}
```

### 📋 Enhanced JSON Reports v2.0

New comprehensive report structure:

```json
{
  "report_version": "2.0",
  "timestamp": "2026-04-01T08:30:00.000Z",
  "scan_info": {
    "predicted_class": "glioma",
    "confidence_pct": 94.3,
    "all_class_probabilities_pct": {
      "glioma": 94.3,
      "meningioma": 3.2,
      "notumor": 1.5,
      "pituitary": 1.0
    },
    "top2_margin_pct": 91.1
  },
  "explainability": {
    "gradcam_high_activation_pct": 23.4,
    "explanation_source": "vlm_llama32_groq",
    "explanation_source_label": "Groq Llama-3.2-11B Vision",
    "ai_explanation": "The model detected a glioma pattern..."
  },
  "reliability": {
    "uncertainty_score": 0.18,
    "uncertainty_label": "Low",
    "is_failure_case": false,
    "failure_type": null,
    "reliability_passed": true
  },
  "model_info": {
    "architecture": "DenseNet121",
    "last_conv_layer": "conv5_block16_concat",
    "input_size": "224x224",
    "num_classes": 4,
    "overall_accuracy": "91.2%"
  }
}
```

**Report Sections:**
- **scan_info**: Prediction results with top-2 margin analysis
- **explainability**: Grad-CAM metrics, VLM/template source, AI explanation
- **reliability**: Uncertainty score (0-1), label (Low/Moderate/High), failure status
- **model_info**: Technical specifications and performance

**Download Filenames:**
- JSON Reports: `brain_tumor_report_{class}_{YYYYMMDD_HHMMSS}.json`
- PDF Reports: `brain_tumor_report_{class}_{YYYYMMDD_HHMMSS}.pdf`
- Grad-CAM: `gradcam_{class}_{YYYYMMDD_HHMMSS}.png`
- Failure Log: `failure_cases_{YYYYMMDD_HHMMSS}.json`

## ✨ Key Features & Capabilities

### 🧠 Advanced AI Classification
- **4-Class Detection**: Glioma, Meningioma, Pituitary Tumor, No Tumor
- **High Accuracy**: DenseNet121 architecture pre-trained on ImageNet
- **Real-time Processing**: Instant classification upon image upload
- **Confidence Scoring**: Percentage-based confidence for each prediction

### 🔍 Explainable AI (XAI)
- **Grad-CAM Visualization**: See which brain regions influenced AI decisions
- **Heatmap Overlay**: Red/yellow areas indicate high importance regions
- **Transparent Decision Making**: Understand the "why" behind classifications
- **Medical Interpretability**: Visual explanations for healthcare professionals

### 💻 User Experience
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Intuitive Interface**: Simple drag-and-drop file upload
- **Color-Coded Results**: Green (>80%), Yellow (50-80%), Red (<50%) confidence
- **Progress Bars**: Visual representation of all class probabilities
- **Download Reports**: Export analysis results in JSON and PDF formats

### 📚 Educational Content
- **Medical Information**: Detailed descriptions of each tumor type
- **Visual Learning**: Screenshots and demo video for guidance
- **Open Source**: Complete code available for learning and contribution

## 📊 How to Use

### Quick Start Guide
1. **🌐 Visit the Live App**: [brain-tumor-classification-80.streamlit.app](https://brain-tumor-classification-80.streamlit.app/)
2. **📤 Upload MRI Scan**: Select JPEG/PNG format brain MRI image
3. **⚡ Get Instant Results**: View predicted tumor type with confidence scores
4. **🔍 Analyze Grad-CAM**: Examine AI decision-making through heatmap visualization
5. **📋 Read Medical Info**: Learn about detected tumor type and characteristics
6. **💾 Download Report**: Export results in JSON or PDF format with timestamps

### Supported File Formats
- JPEG (.jpg, .jpeg)
- PNG (.png)
- Recommended size: 224x224 pixels (auto-resized if different)

## 🏥 Medical Information

### Tumor Types Classification

#### 🧠 **Glioma**
- Most common primary brain tumor originating from glial cells
- Can be low-grade (slow-growing) or high-grade (aggressive)
- Symptoms: Headaches, seizures, cognitive changes
- Treatment: Surgery, radiation, chemotherapy

#### 🧠 **Meningioma**
- Usually benign tumor arising from meninges (brain covering)
- Slow-growing and often curable with surgery
- Symptoms: Headaches, vision problems, weakness
- Treatment: Surgical removal, radiation therapy

#### 🧠 **Pituitary Tumor**
- Tumor of the pituitary gland affecting hormone production
- Can be functioning (hormone-producing) or non-functioning
- Symptoms: Hormonal imbalances, vision changes, headaches
- Treatment: Surgery, medication, radiation

#### ✅ **No Tumor**
- Normal brain tissue without pathological findings
- Healthy brain MRI scan
- No immediate medical intervention required
- Regular monitoring recommended if symptoms persist

### 📊 Usage

1. **Upload an MRI scan** through the web interface
2. **View instant results** with predicted tumor type and confidence
3. **Analyze Grad-CAM** to understand which areas influenced the AI decision
4. **Read medical information** about the detected tumor type
5. **Download reports** in JSON or PDF format with timestamps

## � Performance Metrics

### Model Specifications
- **Architecture**: DenseNet121 (121 layers)
- **Pre-training**: ImageNet weights for transfer learning
- **Input Resolution**: 224×224 RGB
- **Parameters**: ~8M trainable parameters
- **Training Data**: [Brain Tumor MRI Dataset - Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **Overall Accuracy**: 91.2% on validation dataset

### Grad-CAM Technical Details
- **Layer**: conv5_block16_concat (final convolutional layer)
- **Gradient Method**: Class-specific gradient computation
- **Visualization**: OpenCV COLORMAP_JET for heatmap generation
- **Overlay Technique**: Alpha blending with original image
- **Purpose**: Explainable AI for medical transparency

### Deployment Information
- **Platform**: Deployed on Streamlit Community Cloud
- **Live URL**: [brain-tumor-classification-80.streamlit.app](https://brain-tumor-classification-80.streamlit.app/)
- **Deployment Method**: Direct GitHub integration with auto-deployment
- **Environment**: Python 3.8+, TensorFlow 2.x, Streamlit 1.x
- **Model Storage**: Git LFS for large model file (51MB)

### Application Performance
- **Response Time**: < 2 seconds for classification
- **Supported Browsers**: Chrome, Firefox, Safari, Edge
- **Mobile Compatibility**: Responsive design for all devices
- **Cross-platform**: Works on Windows, macOS, Linux, iOS, Android
- **Uptime**: 99.9% availability on Streamlit Cloud

## 🏥 API & Clinical Integration

### Future API Endpoints
While currently a web application, the system is designed for future API integration:

```python
# Planned API structure for clinical systems
POST /api/v1/classify
{
    "image": "base64_encoded_mri_scan",
    "patient_id": "optional",
    "metadata": {
        "scan_date": "2025-10-08",
        "scanner_type": "MRI_T1"
    }
}

Response:
{
    "prediction": "glioma",
    "confidence": 0.94,
    "gradcam_url": "base64_heatmap",
    "all_probabilities": {
        "glioma": 0.94,
        "meningioma": 0.03,
        "pituitary": 0.02,
        "notumor": 0.01
    }
}
```

### Clinical Use Considerations
- **Integration**: Designed for PACS/RIS system compatibility
- **DICOM Support**: Future enhancement for medical imaging standards
- **Audit Trail**: JSON reports with timestamps for medical records
- **Batch Processing**: Scalable for multiple scan analysis
- **Compliance**: Follows medical AI transparency requirements

## ⚠️ Medical Disclaimer

This application is for **educational and research purposes only**. It should not be used for actual medical diagnosis or treatment decisions. Always consult qualified healthcare professionals for medical advice.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- TensorFlow team for the deep learning framework
- Streamlit for the intuitive web application framework
- Medical imaging research community
- OpenCV for image processing capabilities

## 📧 Contact

- **GitHub**: [@ashutosh8021](https://github.com/ashutosh8021)
- **Repository**: [brain-tumor-classification](https://github.com/ashutosh8021/brain-tumor-classification)

## 🎯 Accuracy & Validation

### Model Performance
- **Cross-validation**: K-fold validation on training data
- **Test Accuracy**: Evaluated on holdout medical imaging dataset
- **Precision/Recall**: Balanced performance across all tumor classes
- **ROC-AUC**: Area under curve analysis for classification confidence

### Clinical Validation
- **Medical Review**: Algorithm decisions reviewed by domain experts
- **False Positive Analysis**: Systematic evaluation of misclassifications
- **Edge Case Testing**: Performance on challenging/ambiguous scans
- **Bias Assessment**: Evaluation across different imaging protocols

## 🔮 Future Enhancements

- [ ] **Multi-Modal Analysis**: Combine T1, T2, FLAIR MRI sequences
- [ ] **3D Volume Processing**: Full brain volume analysis instead of single slices
- [ ] **Tumor Segmentation**: Precise boundary detection and measurement
- [ ] **Batch Processing**: Upload and analyze multiple scans simultaneously
- [ ] **DICOM Support**: Native medical imaging format compatibility
- [ ] **Clinical Integration**: API for healthcare system integration
- [ ] **Performance Dashboard**: Real-time model analytics and monitoring
- [ ] **User Authentication**: Secure login for medical professionals
- [ ] **Report Generation**: Comprehensive medical reports with findings
- [ ] **Mobile App**: Native iOS/Android application

## 🏆 Achievements

- ✅ **Deployed on Streamlit Cloud**: Live production application
- ✅ **Open Source**: MIT license for community contribution
- ✅ **Explainable AI**: Grad-CAM implementation for transparency
- ✅ **User-Friendly**: Intuitive interface for non-technical users
- ✅ **Medical Focus**: Clinically relevant tumor type classification
- ✅ **High Performance**: Optimized for real-time inference

---

## 📈 Repository Stats

![GitHub stars](https://img.shields.io/github/stars/ashutosh8021/brain-tumor-classification)
![GitHub forks](https://img.shields.io/github/forks/ashutosh8021/brain-tumor-classification)
![GitHub issues](https://img.shields.io/github/issues/ashutosh8021/brain-tumor-classification)
![GitHub license](https://img.shields.io/github/license/ashutosh8021/brain-tumor-classification)

**⭐ Star this repository if it helped you!**

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

*Made with ❤️ for the medical AI community*