# ❤️ PulseWatcher — Real-Time ECG Anomaly Detection

A deep learning system that detects cardiac arrhythmias in real-time
using an LSTM Autoencoder trained on the MIT-BIH Arrhythmia Database.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Dataset](https://img.shields.io/badge/Dataset-MIT--BIH-green)

---

## 🧠 What This Project Does

Normal ECG classification requires labeled data for every arrhythmia type.
PulseWatcher takes a different approach — **unsupervised anomaly detection**:

1. Train an LSTM Autoencoder **only on normal heartbeats**
2. The model learns what a normal beat looks like
3. When it sees an abnormal beat, reconstruction error spikes
4. High reconstruction error = anomaly alert

This mirrors how a cardiologist thinks: *"this beat doesn't look normal"*
— without needing to know exactly which arrhythmia it is.

---

## 📊 Results

| Metric          | Value    |
|-----------------|----------|
| Training Loss   | 0.001584 |
| Precision       | 95.21%   |
| Recall          | 44.64%   |
| F1 Score        | 60.79%   |
| Normal Beats    | 74,771   |
| Anomalous Beats | 33,308   |

> **95% Precision** means almost zero false alarms —
> critical in a clinical setting where unnecessary alerts cause alert fatigue.

---

## 🏗️ Architecture

```
ECG Signal (187 timesteps)
        ↓
  LSTM Encoder
  (187 → 64 → 32 hidden units)
        ↓
  LSTM Decoder
  (32 → 64 → 187 reconstruction)
        ↓
  Reconstruction Error
        ↓
  Threshold (95th percentile)
        ↓
  NORMAL ✅  or  ANOMALY 🚨
```

---

## 📁 Project Structure

```
pulsewatcher/
├── data/
│   ├── train.npy             # 59,816 normal beats for training
│   ├── test.npy              # 14,955 normal beats for evaluation
│   └── anomaly.npy           # 33,308 anomalous beats
├── models/
│   ├── lstm_autoencoder.pt   # Trained model weights
│   └── threshold.npy         # Decision threshold
├── src/
│   ├── preprocess.py         # Data download and segmentation
│   ├── model.py              # LSTM Autoencoder architecture
│   ├── train.py              # Training loop
│   └── evaluate.py           # Metrics and threshold calculation
├── dashboard/
│   └── app.py                # Streamlit dashboard
└── README.md
```

---

## 🗃️ Dataset

**MIT-BIH Arrhythmia Database** — PhysioNet
- 48 patients, ~30 minutes of ECG each
- Sampled at 360 Hz
- Manually annotated by cardiologists
- Gold standard benchmark in cardiac AI research

---

## 🚀 Run Locally

```bash
# Clone the repo
git clone https://github.com/yourname/pulsewatcher.git
cd pulsewatcher

# Create virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Download data and preprocess
python src/preprocess.py

# Train the model
python src/train.py

# Evaluate
python src/evaluate.py

# Launch dashboard
streamlit run dashboard/app.py
```

---

## 📦 Requirements

```
torch
numpy
pandas
matplotlib
seaborn
scikit-learn
streamlit
plotly
wfdb
```

---

## 💡 Key Design Decisions

**Why LSTM?**
ECG is time-series data. LSTMs have memory — they understand that what
happened 50 timesteps ago affects what's happening now. A regular
feedforward network would treat each timestep independently.

**Why Autoencoder?**
We don't have balanced labeled data for every arrhythmia type. An
autoencoder learns the distribution of normal data in an unsupervised
way. Anything outside that distribution gets flagged.

**Why 95th percentile threshold?**
We allow the model to flag the top 5% of normal beats as anomalous —
this is the standard clinical tradeoff between sensitivity and specificity.

---

## 🔍 Explainability

The dashboard shows **per-timestep reconstruction error** — which exact
part of the 187-point heartbeat the model struggled to reconstruct.
This maps to real cardiac features:

- **Points 0–40**    → P-wave (atrial depolarization)
- **Points 60–100**  → QRS complex (ventricular depolarization)
- **Points 110–160** → T-wave (ventricular repolarization)

---

## 👨‍💻 Author

**Aryan Singh**
B.Tech Computer Science Engineering, Final Year
Built as a Final Year Project — May 2026