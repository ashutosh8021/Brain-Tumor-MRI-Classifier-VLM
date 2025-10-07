# 🧠 Brain Tumor MRI Classification & Analysis

An AI-powered web application that classifies brain tumors from MRI scans using deep learning. Built with DenseNet121 architecture and featuring explainable AI through Grad-CAM visualization.

## 🎯 Features

- **4-Class Classification**: Glioma, Meningioma, Pituitary Tumor, No Tumor
- **High Accuracy**: DenseNet121 model trained on medical imaging dataset
- **Explainable AI**: Grad-CAM heatmaps show AI decision-making process
- **User-Friendly Interface**: Clean Streamlit web application
- **Confidence Scoring**: Color-coded confidence levels for predictions
- **Download Reports**: Export analysis results and visualizations
- **Medical Information**: Detailed descriptions of each tumor type

## 🚀 Live Demo

[**Try the app here**](https://brain-tumor-classification-ai.streamlit.app) 

## 📸 Screenshots

### Main Interface
![Main Interface](https://user-images.githubusercontent.com/yourusername/brain-tumor-app-demo.png)
*Clean two-column layout with instant classification results*

### Grad-CAM Analysis
![Grad-CAM](https://user-images.githubusercontent.com/yourusername/gradcam-demo.png)
*AI explanation through heatmap visualization*

## 🛠️ Technology Stack

- **Deep Learning**: TensorFlow/Keras, DenseNet121
- **Web Framework**: Streamlit
- **Image Processing**: OpenCV, PIL
- **Visualization**: Grad-CAM, Matplotlib
- **Data Science**: NumPy

## 📦 Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup
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

# Run the application
streamlit run app.py
```

## 📁 Project Structure

```
brain-tumor-classification/
├── app.py                          # Main Streamlit application
├── densenet121_brain_tumor_best.h5 # Trained DenseNet121 model
├── requirements.txt                # Python dependencies
├── README.md                      # Project documentation
├── .gitignore                     # Git ignore file
└── screenshots/                   # Application screenshots
```

## 🧠 Model Information

- **Architecture**: DenseNet121 (pre-trained on ImageNet)
- **Input Size**: 224x224 RGB images
- **Classes**: 4 (Glioma, Meningioma, No Tumor, Pituitary)
- **Framework**: TensorFlow/Keras
- **Last Convolutional Layer**: conv5_block16_concat (for Grad-CAM)

## 🎨 Key Features Explained

### 1. Brain Tumor Classification
- Upload MRI scan in JPEG/PNG format
- Instant AI-powered classification
- Confidence scores for all tumor types
- Color-coded results (Green > 80%, Yellow 50-80%, Red < 50%)

### 2. Explainable AI (Grad-CAM)
- Visual explanation of AI decisions
- Heatmap overlay showing important regions
- Red/yellow areas indicate high importance for classification

### 3. Enhanced User Experience
- Two-column layout for better organization
- Progress bars for confidence visualization
- Download functionality for reports and images
- Detailed medical information for each tumor type

### 4. Tumor Type Information

**Glioma**: Most common primary brain tumor, arises from glial cells
**Meningioma**: Usually benign tumor of the meninges
**Pituitary**: Tumor of the pituitary gland, affects hormone production
**No Tumor**: Normal brain tissue without pathological findings

## 📊 Usage

1. **Upload an MRI scan** through the web interface
2. **View instant results** with predicted tumor type and confidence
3. **Analyze Grad-CAM** to understand which areas influenced the AI decision
4. **Read medical information** about the detected tumor type
5. **Download reports** in JSON format with timestamps

## 🔧 Technical Implementation

### Grad-CAM Visualization
```python
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # Generate class activation map
    # Overlay heatmap on original image
    # Return visualization
```

### Color-Coded Confidence
- Implemented dynamic styling based on confidence thresholds
- Visual progress bars for all classification probabilities
- Intuitive green/yellow/red color scheme

### Download Functionality
- JSON reports with analysis metadata
- Base64 encoded images for offline viewing
- Timestamped results for record keeping

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

## 🔮 Future Enhancements

- [ ] Batch processing for multiple scans
- [ ] Model performance analytics dashboard
- [ ] User authentication system
- [ ] Mobile-optimized interface
- [ ] Additional AI model comparisons
- [ ] DICOM file support

---

⭐ **Star this repository if it helped you!**

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)