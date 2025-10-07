# Model Setup Instructions

The trained model file `densenet121_brain_tumor_best.h5` is included in this repository using **Git LFS (Large File Storage)**.

## ✅ Model Available

The model is automatically available when you clone this repository. Git LFS will handle downloading the large file.

## For Local Development:

1. Clone the repository (Git LFS will automatically download the model):
   ```bash
   git clone https://github.com/ashutosh8021/brain-tumor-classification.git
   ```

2. If you already cloned before LFS setup, pull the LFS files:
   ```bash
   git lfs pull
   ```

## For Streamlit Cloud Deployment:

✅ **Ready for deployment!** Streamlit Cloud supports Git LFS, so the model will be automatically available.

## Model Specifications:

- **Architecture**: DenseNet121
- **Input Size**: 224x224x3
- **Classes**: 4 (glioma, meningioma, notumor, pituitary)
- **File Size**: ~51MB
- **Framework**: TensorFlow/Keras
- **Storage**: Git LFS enabled

## Git LFS Setup (for contributors):

If you need to add new model files:
```bash
git lfs track "*.h5"
git add .gitattributes
git add your_model_file.h5
git commit -m "Add model file"
git push
```