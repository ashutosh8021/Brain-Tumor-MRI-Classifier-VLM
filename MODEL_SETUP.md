# Model Setup Instructions

Due to GitHub's file size limitations, the trained model file `densenet121_brain_tumor_best.h5` is not included in this repository.

## For Local Development:

1. Ensure you have the model file `densenet121_brain_tumor_best.h5` in the root directory
2. If you don't have the model file, you can:
   - Train your own DenseNet121 model on brain tumor data
   - Download from cloud storage (add your link here)
   - Contact the repository owner

## For Streamlit Cloud Deployment:

The model file will be handled through Streamlit's secrets management or external hosting.

## Model Specifications:

- **Architecture**: DenseNet121
- **Input Size**: 224x224x3
- **Classes**: 4 (glioma, meningioma, notumor, pituitary)
- **File Size**: ~32MB
- **Framework**: TensorFlow/Keras

## Cloud Storage Options:

For deployment, consider hosting the model on:
- Google Drive (with direct download link)
- AWS S3
- Google Cloud Storage
- Hugging Face Model Hub