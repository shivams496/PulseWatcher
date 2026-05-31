# 🫀 PulseWatcher — ECG Anomaly Detection System

> Real-time ECG anomaly detection using an LSTM Autoencoder trained on the MIT-BIH Arrhythmia Dataset. Live demo on HuggingFace Spaces.

[![Live Demo](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-blue)](https://huggingface.co/spaces/[your-username]/PulseWatcher)
[![Python](https://img.shields.io/badge/Python-3.9+-green)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)](https://streamlit.io)

---

## Results

| Metric | Before (95th pct) | After (82nd pct) | Change |
|--------|-------------------|------------------|--------|
| Precision | 95.21% | 89.82% | -5.4% |
| **Recall** | **44.64%** | **71.35%** | **+26.7%** |
| **F1 Score** | **60.79%** | **79.53%** | **+18.7%** |
| ROC-AUC | 0.8678 | 0.8678 | same |

**Threshold sensitivity analysis:** At the 95th percentile, Precision = 95.2%, Recall = 44.6%. Sweeping to the 82nd percentile yields Precision = 88%, Recall = 65%, F1 = 0.746 — a 24% improvement in anomaly detection at the cost of 7% more false positives. The optimal threshold is a clinical decision, not a model decision.

### Model Comparison (same test set, same threshold logic)

| Model | Precision | Recall | F1 | AUC |
|-------|-----------|--------|----|-----|
| **LSTM Autoencoder (ours)** | **89.82%** | **71.35%** | **79.53%** | **0.8678** |
| 1D CNN Autoencoder | 73.77% | 28.35% | 40.96% | 0.6737 |

LSTM wins across every metric. ECG beats are temporal sequences with known structure — LSTM's sequential modelling is a natural fit. CNN treats the signal as a spatial pattern and loses the temporal ordering that makes arrhythmias detectable.

---

## Architecture

```
MIT-BIH Dataset
     │
     ▼
Beat Segmentation + Normalisation
     │  (187 timesteps, z-score normalised)
     ▼
LSTM Encoder  →  Latent Space (64 units)  →  LSTM Decoder
     │
     ▼
Reconstruction Error  →  Threshold (82nd percentile of train errors)
     │
     ▼
Anomaly Score (0–100)  +  Binary Classification
```

**Model details:**
- Architecture: LSTM Autoencoder (encoder-decoder)
- Hidden units: 64
- Parameters: ~500K
- Input shape: (N, 187, 1)
- Training: Normal beats only (unsupervised)
- Inference: <5ms per beat on CPU

**Why unsupervised?** In clinical settings, labeled anomaly data is rare and expensive. Training on normal beats only allows the model to generalise to anomaly types it has never seen, as long as they deviate from normal morphology.

---

## Dashboard Features

| Feature | Description |
|---------|-------------|
| **Live ECG Simulation** | Beat-by-beat streaming with `st.empty()` loop. Anomaly score updates in real time. |
| **Explainability Heatmap** | ECG waveform colour-coded by reconstruction error: cyan (low) → amber (mid) → crimson (high) |
| **CSV Upload** | Upload your own ECG signal. Auto-normalises, resamples to 187pts, runs inference. |
| **PDF Report** | One-click dark-themed clinical PDF: beat class, error, anomaly score, top-5 error timesteps |
| **Dynamic Metrics** | All metrics loaded live from `models/metrics.json` — no hardcoded strings |

---

## Project Structure

```
ecg-anomaly-detection/
├── dashboard/
│   └── app.py                  # Streamlit dashboard (all features)
├── src/
│   ├── model.py                # LSTM Autoencoder architecture
│   ├── train.py                # Training loop
│   ├── evaluate.py             # Threshold sweep + metrics
│   ├── cnn_autoencoder.py      # CNN baseline model
│   └── benchmark.py            # Model comparison script
├── models/
│   ├── model.pt                # Trained LSTM weights
│   ├── threshold.npy           # Optimal threshold (82nd percentile)
│   ├── train_errors.npy        # Training reconstruction errors
│   ├── metrics.json            # Live metrics for dashboard
│   └── benchmark.json          # LSTM vs CNN comparison results
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/[your-username]/ecg-anomaly-detection
cd ecg-anomaly-detection
pip install -r requirements.txt
```

**Run the threshold sweep (optional — already done, results saved):**
```bash
python -m src.evaluate
```

**Start the dashboard:**
```bash
streamlit run dashboard/app.py
```

**Run the model benchmark:**
```bash
python -m src.benchmark
```

---

## Dataset

[MIT-BIH Arrhythmia Dataset](https://www.physionet.org/content/mitdb/1.0.0/) via PhysioNet.

| Split | Normal | Anomaly |
|-------|--------|---------|
| Train | 59,816 | — |
| Test | 14,955 | 33,308 |

---

## Known Limitations

1. **Single-beat classification only** — rhythm-level disorders (e.g. AFib) require inter-beat interval (RR) analysis, not just beat morphology.
2. **MIT-BIH distribution** — trained on Holter monitor recordings. Signals from different device types or electrode placements may need threshold recalibration.
3. **Unusual-but-benign beats** — patients with non-standard baseline ECGs may generate higher reconstruction errors without true pathology.
4. **Not FDA/CE approved** — for research and educational use only.

---

## Future Work

- [ ] RR interval analysis for rhythm-level anomaly detection (AFib, heart block)
- [ ] Multi-lead ECG support (currently single-lead)
- [ ] Patient-specific threshold calibration
- [ ] HL7/FHIR integration for hospital data pipelines
- [ ] Transformer baseline comparison

---

## Tech Stack

Python · PyTorch · Streamlit · NumPy · SciPy · Scikit-learn · ReportLab · Pandas

---

