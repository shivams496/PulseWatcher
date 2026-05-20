# dashboard/app.py

import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import LSTMAutoencoder

# --- Page config ---
st.set_page_config(
    page_title="PulseWatcher",
    page_icon="❤️",
    layout="wide"
)

# --- Load model and threshold ---
@st.cache_resource
def load_model():
    model = LSTMAutoencoder()
    model.load_state_dict(torch.load("models/lstm_autoencoder.pt",
                          map_location="cpu"))
    model.eval()
    return model

@st.cache_resource
def load_threshold():
    return float(np.load("models/threshold.npy"))

model = load_model()
threshold = load_threshold()

# --- Load sample beats ---
@st.cache_data
def load_beats():
    normal = np.load("data/test.npy")
    anomaly = np.load("data/anomaly.npy")
    return normal, anomaly

normal_beats, anomaly_beats = load_beats()

# --- Helper ---
def get_error(beat_np):
    tensor = torch.tensor(beat_np[np.newaxis, :, np.newaxis].astype(np.float32))
    with torch.no_grad():
        output = model(tensor)
    error = torch.mean((output - tensor) ** 2).item()
    return error, output.numpy()[0, :, 0]

# --- UI ---
st.title("❤️ PulseWatcher")
st.markdown("**Real-time ECG Anomaly Detection using LSTM Autoencoder**")
st.markdown("---")

# Sidebar
st.sidebar.header("Controls")
beat_type = st.sidebar.radio("Select beat type:", ["Normal", "Anomalous"])
beat_index = st.sidebar.slider("Beat index", 0, 99, 0)

# Select beat
if beat_type == "Normal":
    beat = normal_beats[beat_index]
else:
    beat = anomaly_beats[beat_index]

# Get reconstruction and error
error, reconstructed = get_error(beat)
is_anomaly = error > threshold

# --- Metrics row ---
col1, col2, col3 = st.columns(3)
col1.metric("Reconstruction Error", f"{error:.6f}")
col2.metric("Threshold", f"{threshold:.6f}")
col3.metric("Status", "🚨 ANOMALY" if is_anomaly else "✅ NORMAL")

st.markdown("---")

# --- ECG Chart ---
x = list(range(187))

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=x, y=beat,
    name="Original ECG",
    line=dict(color="royalblue", width=2)
))

fig.add_trace(go.Scatter(
    x=x, y=reconstructed,
    name="Reconstructed",
    line=dict(color="tomato", width=2, dash="dash")
))

fig.update_layout(
    title=f"ECG Beat — {'🚨 ANOMALY DETECTED' if is_anomaly else '✅ Normal Beat'}",
    xaxis_title="Timestep",
    yaxis_title="Amplitude (normalized)",
    height=400,
    plot_bgcolor="black",
    paper_bgcolor="#0e1117",
    font=dict(color="white"),
    legend=dict(bgcolor="#0e1117")
)

st.plotly_chart(fig, use_container_width=True)

# --- Explanation ---
st.markdown("### How it works")
st.markdown(f"""
- The LSTM Autoencoder was trained **only on normal heartbeats**
- It learns to reconstruct normal beats with very low error
- When it sees an **abnormal beat**, it struggles to reconstruct it
- High reconstruction error = anomaly alert
- **Current threshold: `{threshold:.6f}`** (95th percentile of normal errors)
""")
# --- Explainability Section ---
if is_anomaly:
    st.markdown("---")
    st.markdown("### 🔍 Why was this flagged? — Feature Importance")

    # Calculate per-timestep reconstruction error
    tensor = torch.tensor(beat[np.newaxis, :, np.newaxis].astype(np.float32))
    with torch.no_grad():
        output = model(tensor)

    per_timestep_error = ((output[0, :, 0].numpy() - beat) ** 2)

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=list(range(187)),
        y=per_timestep_error,
        marker_color=per_timestep_error,
        marker_colorscale="Reds",
        name="Reconstruction Error per Timestep"
    ))
    fig2.update_layout(
        title="Which part of the heartbeat caused the anomaly alert?",
        xaxis_title="Timestep",
        yaxis_title="Reconstruction Error",
        height=300,
        plot_bgcolor="black",
        paper_bgcolor="#0e1117",
        font=dict(color="white")
    )
    st.plotly_chart(fig2, use_container_width=True)  
    st.caption("Red bars show which timesteps the model struggled to reconstruct — these caused the anomaly alert.")
    