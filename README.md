
---
title: PulseWatcher
emoji: 🫀
colorFrom: red
colorTo: blue
sdk: streamlit
sdk_version: "1.32.0"
app_file: app.py
pinned: false
---

# 🫀 PulseWatcher — ECG Anomaly Detection System
<p align="center">
  <img src="https://img.shields.io/badge/Model-LSTM%20Autoencoder-00d4ff?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Dataset-MIT--BIH%20Arrhythmia-00e87a?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Framework-PyTorch-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Dashboard-Streamlit-ff4b4b?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/Deployed-Hugging%20Face-ffb020?style=for-the-badge&logo=huggingface&logoColor=white"/>
  <img src="https://img.shields.io/badge/Type-Unsupervised-b060ff?style=for-the-badge"/>
</p>

<p align="center">
  Real-time ECG anomaly detection using an LSTM Autoencoder trained <strong>exclusively on normal heartbeats</strong>.<br/>
  Anomalies are detected through reconstruction error — no anomaly labels required during training.
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/shivams496/Pulsewatcher">
    <img src="https://img.shields.io/badge/🌐%20Live%20Demo-Hugging%20Face%20Spaces-yellow?style=for-the-badge"/>
  </a>
</p>

---

## How It Works

```
MIT-BIH Dataset (PhysioNet)
        │
        ▼
Beat Segmentation + Normalisation
  (187 timesteps, min-max normalised per beat)
        │
        ▼
LSTM Encoder → Latent Space (64 units) → LSTM Decoder
        │
        ▼
Reconstruction Error (MSE per timestep)
        │
        ▼
Threshold (82nd percentile of normal train errors)
        │
   ┌────┴────┐
   ▼         ▼
NORMAL    ANOMALY
```

**Key insight:** The model never sees anomalous beats during training. It learns only what *normal* looks like. When an anomalous beat is fed in, reconstruction fails — the error spike is the detection signal.

---

## Results

### Threshold Sensitivity Analysis

| Metric | Before (95th pct) | After (82nd pct) | Change |
|--------|:-----------------:|:----------------:|:------:|
| Precision | 95.21% | 89.82% | -5.4% |
| Recall | 44.64% | **71.35%** | **+26.7%** |
| F1 Score | 60.79% | **79.53%** | **+18.7%** |
| ROC-AUC | 0.8678 | 0.8678 | same |

Sweeping to the 82nd percentile yields Precision = 89%, Recall = 71%, F1 = 0.795 — a **24% improvement in anomaly detection** at the cost of 7% more false positives. The optimal threshold is a **clinical decision, not a model decision**.

### Model Comparison (same test set, same threshold logic)

| Model | Precision | Recall | F1 | AUC |
|-------|:---------:|:------:|:--:|:---:|
| **LSTM Autoencoder (ours)** | **89.82%** | **71.35%** | **79.53%** | **0.8678** |
| 1D CNN Autoencoder | 73.77% | 28.35% | 40.96% | 0.6737 |

LSTM wins across every metric. ECG beats are temporal sequences with known structure (P→QRS→T) — LSTM's sequential memory is a natural fit. CNN treats the signal as a spatial pattern and loses the temporal ordering that makes arrhythmias detectable.

---

## Dashboard Features

| Feature | Description |
|---------|-------------|
| **Live ECG Simulation** | Beat-by-beat streaming with `st.empty()` loop. Anomaly score bar updates in real time. |
| **Explainability Heatmap** | ECG waveform colour-coded by reconstruction error: cyan (low) → amber (mid) → crimson (high). |
| **CSV / ECG Upload** | Upload any ECG signal. Auto-normalises, resamples to 187 pts, runs inference instantly. |
| **PDF Report** | One-click dark-themed clinical PDF: beat class, error, anomaly score, top-5 error timesteps. |
| **Dynamic Metrics** | All metrics loaded live from `models/metrics.json` — no hardcoded strings. |

---

## Model Details

- **Architecture:** LSTM Autoencoder (encoder-decoder)
- **Hidden units:** 64
- **Parameters:** ~110K
- **Input shape:** `(N, 187, 1)`
- **Training:** Normal beats only (unsupervised)
- **Inference:** <5ms per beat on CPU

**Why unsupervised?** In clinical settings, labelled anomaly data is rare and expensive. Training on normal beats only allows the model to generalise to anomaly types it has never seen, as long as they deviate from normal morphology.

---

## Project Structure

```
ecg-anomaly-detection/
├── dashboard/
│   └── app.py              # Streamlit dashboard (all features)
├── src/
│   ├── model.py            # LSTM Autoencoder architecture
│   ├── train.py            # Training loop
│   ├── evaluate.py         # Threshold sweep + metrics
│   ├── cnn_autoencoder.py  # CNN baseline model
│   └── benchmark.py        # Model comparison script
├── models/
│   ├── lstm_autoencoder.pt # Trained LSTM weights
│   ├── threshold.npy       # Optimal threshold (82nd percentile)
│   ├── train_errors.npy    # Training reconstruction errors
│   ├── metrics.json        # Live metrics for dashboard
│   └── benchmark.json      # LSTM vs CNN comparison results
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/shivams496/ecg-anomaly-detection
cd ecg-anomaly-detection
pip install -r requirements.txt
```

**Start the dashboard:**
```bash
streamlit run dashboard/app.py
```

**Run threshold sweep (optional — results already saved):**
```bash
python -m src.evaluate
```

**Run model benchmark:**
```bash
python -m src.benchmark
```

---

## Dataset

[MIT-BIH Arrhythmia Dataset](https://physionet.org/content/mitdb/1.0.0/) via PhysioNet.

| Split | Normal | Anomaly |
|-------|-------:|--------:|
| Train | 59,816 | — |
| Test  | 14,955 | 33,308 |

48 recordings · 30 min each · 360 Hz · 47 patients · 15+ arrhythmia classes

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
- [ ] Transformer autoencoder ablation study

---

## Interview Q&A

**"Why LSTM over Transformer?"**
MIT-BIH beats are 187 timesteps — short structured sequences with known temporal order (P→QRS→T). Transformers excel at long sequences with long-range dependencies. LSTM captures within-beat temporal structure naturally. The benchmark confirms it: LSTM F1 79% vs CNN F1 41%.

**"Why is recall not higher?"**
The threshold is a clinical tuning parameter. At the 95th percentile, precision is 95% but recall is 44%. At the 82nd percentile, precision drops to 89% but recall rises to 71% (+27%). The right operating point depends on the clinical context — a general ward might prefer high precision; a cardiac ICU might prefer high recall.

**"How would this run in a hospital?"**
<5ms inference on CPU. Dockerised. Threshold adjustable per cohort without retraining. The Hugging Face deployment proves it runs without a GPU.

---

## Tech Stack

`Python` · `PyTorch` · `Streamlit` · `NumPy` · `SciPy` · `scikit-learn` · `ReportLab` · `Pandas` · `WFDB`

---

*B.Tech Final Year Project — ECG Anomaly Detection*
