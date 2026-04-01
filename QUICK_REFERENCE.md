# Brain Tumor Classifier v2.0 - Quick Reference Card

## 📁 File Locations

- **API Key (Local)**: `D:\braintumorclassification\.env`
- **API Key (Cloud)**: Streamlit App Settings → Secrets → `secrets.toml`
- **App**: `D:\braintumorclassification\app.py`

## 🔑 Setup Your API Key

### Option 1: Local Development (.env file)
1. Open: `D:\braintumorclassification\.env`
2. Find: `GROQ_API_KEY=gsk_your_key_here`
3. Replace with your actual key
4. Save file

### Option 2: Streamlit Cloud (secrets.toml)
1. Go to Streamlit Cloud dashboard
2. Click App Settings → Secrets
3. Add: `GROQ_API_KEY = "gsk_your_actual_key"`
4. Save

**Get API Key**: https://console.groq.com/keys (FREE)

## 🚀 Run the App

```bash
# Install dependencies
pip install reportlab

# Run locally
streamlit run app.py
```

## 📥 Download Options

Your app now generates **3 types** of downloads:

1. **📄 JSON Report**
   - Technical data format
   - For programmatic use
   - Filename: `brain_tumor_report_{class}_{timestamp}.json`

2. **📑 PDF Report** ⭐ NEW
   - Professional format
   - Includes images (MRI + Grad-CAM)
   - Printable and shareable
   - Filename: `brain_tumor_report_{class}_{timestamp}.pdf`

3. **🖼️ Grad-CAM Image**
   - Heatmap visualization
   - PNG format
   - Filename: `gradcam_{class}_{timestamp}.png`

## 📋 PDF Report Contents

**Page 1:**
- Report header with timestamp
- Classification results table
- All class probabilities

**Page 2:**
- Original MRI scan
- Grad-CAM heatmap (side by side)
- AI explanation
- Reliability metrics
- Model information
- Medical disclaimer

## 🎯 Features Checklist

- [x] 4-class tumor classification
- [x] DenseNet121 model (91.2% accuracy)
- [x] Grad-CAM visualization
- [x] VLM-powered explanations (optional)
- [x] Failure detection (3 modes)
- [x] Failure dashboard
- [x] JSON reports
- [x] PDF reports with images
- [x] Uncertainty metrics
- [x] No file writes (session state only)
- [x] Streamlit Cloud ready

## ⚠️ Troubleshooting

**VLM not working?**
- Check .env file has correct API key
- Restart streamlit after editing .env
- Verify API key starts with `gsk_`

**PDF not generating?**
- Run: `pip install reportlab`
- Check error message in app

**App not loading?**
- Check model file exists: `densenet121_brain_tumor_best.h5`
- Run: `pip install -r requirements.txt`

## 📞 Support

- GitHub: https://github.com/ashutosh8021/brain-tumor-classification
- Issues: Create issue on GitHub
- Groq Docs: https://console.groq.com/docs

---

**Version**: 2.0  
**Last Updated**: April 1, 2026  
**Status**: Production Ready ✅
