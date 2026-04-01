# Brain Tumor Classifier - v2.0 Changelog

## 🚀 Release Date
April 1, 2026

## 📋 Overview
Major update introducing Vision Language Model (VLM) integration, automated failure detection, and comprehensive reliability analytics. The app now provides AI-generated clinical explanations and proactive quality control.

---

## ✨ New Features

### 1. 🤖 VLM Integration (Groq API)
- **Model**: Llama-3.2-11B Vision Preview
- **Purpose**: Generate natural language clinical explanations from Grad-CAM overlays
- **Configuration**: Environment variable only (`GROQ_API_KEY`)
- **Output**: 3-4 sentence professional explanations covering:
  - Detected tumor type and heatmap activation locations
  - Supporting imaging features
  - Uncertainty notes (when confidence < 80%)
  - Clinical recommendations
- **Fallback**: Automatic switch to template-based explanations when VLM unavailable
- **API Parameters**:
  - Temperature: 0.3 (focused, deterministic)
  - Max tokens: 300
  - Model: llama-3.2-11b-vision-preview

### 2. ⚠️ Failure Detection System
Three-mode automated quality control:

| Mode | Trigger | Default Threshold | Purpose |
|------|---------|-------------------|---------|
| **Low Confidence** | Prediction confidence below threshold | 50% | Detect uncertain predictions |
| **High Entropy** | Shannon entropy above threshold | 0.65 | Detect flat probability distributions |
| **Borderline** | Top-2 margin below threshold | 20% | Detect competing predictions |

**Features**:
- Real-time evaluation of every prediction
- Customizable thresholds via sidebar sliders
- Specific alert messages with recommendations
- Automatic logging to failure dashboard

### 3. 📊 Failure Dashboard (New Tab)
Comprehensive failure case tracking and analysis:

**Metrics Section**:
- Total failure count
- Breakdown by failure type (Low Confidence, High Entropy, Borderline)
- Class distribution (which tumor types fail most)

**Case Log**:
- Expandable entries with full prediction data
- Sorted newest-first
- Includes: timestamp, filename, predicted class, confidence, failure type, uncertainty score, all probabilities

**Actions**:
- Download failure log as JSON
- Clear log with one-click reset
- Empty state with explanatory table

### 4. 📈 Enhanced Analytics
- **Uncertainty Score**: Shannon entropy normalized to [0, 1]
- **High Activation Percentage**: Percent of heatmap with intensity > 70%
- **Activation Focus Classification**: Focal (<10%) / Moderate (10-30%) / Diffuse (>30%)
- **Top-2 Margin**: Confidence gap between top two predictions
- **Secondary Considerations**: Automatic detection of competing diagnoses (>15% probability)

### 5. 📋 JSON Reports v2.0
Complete restructure with four main sections:

```json
{
  "report_version": "2.0",
  "scan_info": {
    "predicted_class": "...",
    "confidence_pct": 0.0,
    "all_class_probabilities_pct": {},
    "top2_margin_pct": 0.0
  },
  "explainability": {
    "gradcam_high_activation_pct": 0.0,
    "explanation_source": "vlm_llama32_groq | template",
    "explanation_source_label": "...",
    "ai_explanation": "..."
  },
  "reliability": {
    "uncertainty_score": 0.0,
    "uncertainty_label": "Low | Moderate | High",
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

### 6. 🎨 UI/UX Improvements
- **Two-Tab Layout**: Analysis + Failure Dashboard
- **Sidebar Enhancements**:
  - VLM status badge (Active / Set API key)
  - Failure threshold sliders (3 customizable thresholds)
  - Model information card
- **Results Table**: Compact table format for main predictions
- **Progress Bars**: Visual uncertainty and activation metrics
- **Reliability Badge**: Pass/Fail indicator with alert details

---

## 🔧 Technical Changes

### New Functions

#### `vlm_explanation_groq(cam_pil, pred_class, confidence, preds, class_names)`
- Encodes Grad-CAM as base64 PNG
- Calls Groq API with vision model
- Returns `(explanation_text, "vlm_llama32_groq")` on success
- Returns `(None, "error")` on failure
- Temperature: 0.3, Max tokens: 300

#### `template_explanation(pred_class, confidence, preds, class_names, heatmap)`
- Computes Shannon entropy (normalized)
- Calculates high activation percentage
- Classifies activation focus (focal/moderate/diffuse)
- Generates class-specific explanation paragraphs
- Detects secondary considerations (>15% probability)
- Returns `(explanation, uncertainty_ratio, high_activation_pct)`

#### `detect_failure(confidence, preds, class_names, uncertainty_ratio, conf_thresh, entropy_thresh, margin_thresh)`
- Checks three failure conditions in order
- Returns `(is_failure, failure_type, failure_message)`
- Failure types: `"low_confidence"`, `"high_entropy"`, `"borderline"`

#### `log_failure_case(filename, pred_class, confidence, failure_type, uncertainty_ratio, preds, class_names)`
- Appends case to `st.session_state.failure_log`
- Includes: timestamp, filename, predictions, failure details
- No file writes (session state only)

#### `create_report_v2(...)`
- Generates v2.0 JSON report structure
- Includes all four sections: scan_info, explainability, reliability, model_info
- Computes top-2 margin
- Maps uncertainty score to label (Low/Moderate/High)
- Returns formatted JSON string

### Modified Functions
- **None** - `make_gradcam_heatmap()` and `overlay_heatmap()` unchanged as required

### Session State Variables
- `st.session_state.failure_log`: List of failure case dictionaries

### Configuration
- **Environment Variables**:
  - `GROQ_API_KEY`: Optional, enables VLM features
- **Streamlit Secrets**:
  - `GROQ_API_KEY`: Alternative to environment variable for cloud deployment

---

## 📦 Dependencies

### Added
- `groq`: Groq API client for VLM integration

### Unchanged
- `streamlit`
- `tensorflow`
- `keras`
- `pillow`
- `opencv-python-headless`
- `matplotlib`
- `numpy`

---

## 📂 New Files

1. **`.env.example`**
   - Template for local environment variables
   - Instructions for obtaining Groq API key

2. **`.streamlit/secrets.toml`**
   - Template for Streamlit Cloud secrets
   - TOML format for Groq API key

3. **`CHANGELOG_v2.0.md`** (this file)
   - Complete documentation of v2.0 changes

---

## 📝 Updated Files

### 1. `app.py`
- **Lines**: 679 (up from ~280)
- **Changes**:
  - Complete rewrite of UI layout (two tabs)
  - Added 5 new functions
  - VLM integration with Groq
  - Failure detection and logging
  - Enhanced explanations with entropy analysis
  - New sidebar with thresholds and status
  - Maintained backward compatibility (unchanged Grad-CAM functions)

### 2. `requirements.txt`
- Added `groq` package

### 3. `README.md`
- **Lines**: 539 (up from ~355)
- **New Sections**:
  - Phase 2 Features overview
  - VLM setup instructions
  - Environment variables table
  - Failure detection documentation
  - Enhanced reports v2.0 structure
  - Streamlit Cloud deployment guide
- **Updated Sections**:
  - Features list (split Phase 1 / Phase 2)
  - Technology stack (added Groq)
  - Installation (added environment setup)
  - Project structure (new files)

### 4. `.gitignore`
- Already contained `.env` - no changes needed
- Verified `.streamlit/` is excluded

---

## 🔒 Security & Privacy

- **No API Key in UI**: Groq API key only via environment variables
- **No File Writes**: All data stored in session state
- **No External Database**: Self-contained application
- **Graceful Degradation**: Full functionality without API key
- **TOML Template**: Secrets template prevents accidental commits

---

## 🚢 Deployment

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# (Optional) Set up VLM
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# Run app
streamlit run app.py
```

### Streamlit Cloud
1. Push changes to GitHub
2. Deploy on [share.streamlit.io](https://share.streamlit.io)
3. In App Settings → Secrets, add:
   ```toml
   GROQ_API_KEY = "gsk_your_actual_key_here"
   ```

**Note**: App works fully without API key (uses template fallback)

---

## 🔄 Backward Compatibility

- ✅ **Grad-CAM Functions**: `make_gradcam_heatmap()` and `overlay_heatmap()` unchanged
- ✅ **Model Loading**: Same DenseNet121 model file
- ✅ **Class Names**: Same 4 classes (glioma, meningioma, notumor, pituitary)
- ✅ **Core Workflow**: Upload → Analyze → Download still works
- ✅ **No Breaking Changes**: Existing deployments compatible

---

## 🎯 Performance

- **VLM Response Time**: ~2-4 seconds (network dependent)
- **Template Fallback**: Instant (<100ms)
- **Failure Detection**: <10ms per prediction
- **Session State**: In-memory, no performance impact
- **Overall App**: Still <2 seconds for complete analysis

---

## 📊 Testing Checklist

- [x] App starts without API key (template mode)
- [x] App starts with API key (VLM mode)
- [x] Failure detection triggers correctly
- [x] Dashboard displays logged cases
- [x] Reports download with correct filenames
- [x] Grad-CAM unchanged (visual comparison)
- [x] All 4 tumor classes work
- [x] Threshold sliders update detection
- [x] Clear log button resets dashboard
- [x] VLM explanations are clinical and relevant

---

## 🐛 Known Issues

None at release.

---

## 🔮 Future Enhancements

- [ ] Multiple VLM model options (GPT-4V, Claude Vision)
- [ ] Failure case export to CSV
- [ ] Historical analytics dashboard
- [ ] Batch processing mode
- [ ] API endpoint for programmatic access
- [ ] DICOM file support

---

## 👥 Contributors

- **Ashutosh** (@ashutosh8021) - v2.0 implementation

---

## 📄 License

MIT License - Same as v1.0

---

## 🙏 Acknowledgments

- **Groq**: For providing fast VLM inference API
- **Llama Team**: For open-source vision models
- **Streamlit**: For excellent framework updates
- **Community**: For feature requests and feedback

---

**Version**: 2.0  
**Release**: April 1, 2026  
**Status**: Production Ready ✅
