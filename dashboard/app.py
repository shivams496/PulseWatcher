# dashboard/app.py

import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys
import os
sys.path.append(os.path.abspath("."))
from src.model import LSTMAutoencoder

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PulseWatcher — ECG Anomaly Detection",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── GLOBAL CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

    /* Base */
    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Main background */
    .stApp {
        background-color: #0a0e1a;
        color: #e2e8f0;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0d1224;
        border-right: 1px solid #1e2d4a;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #0d1224 0%, #111827 100%);
        border: 1px solid #1e2d4a;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        transition: border-color 0.3s ease;
    }
    .metric-card:hover {
        border-color: #3b82f6;
    }
    .metric-label {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #64748b;
        margin-bottom: 8px;
        font-family: 'IBM Plex Mono', monospace;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        font-family: 'IBM Plex Mono', monospace;
        color: #e2e8f0;
    }
    .metric-value.normal { color: #22c55e; }
    .metric-value.anomaly { color: #ef4444; }
    .metric-value.neutral { color: #3b82f6; }

    /* Status banner */
    .status-normal {
        background: linear-gradient(135deg, #052e16, #14532d);
        border: 1px solid #166534;
        border-left: 4px solid #22c55e;
        border-radius: 10px;
        padding: 16px 24px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .status-anomaly {
        background: linear-gradient(135deg, #1c0505, #450a0a);
        border: 1px solid #991b1b;
        border-left: 4px solid #ef4444;
        border-radius: 10px;
        padding: 16px 24px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .status-text-normal {
        font-size: 18px;
        font-weight: 700;
        color: #22c55e;
        font-family: 'IBM Plex Mono', monospace;
        letter-spacing: 1px;
    }
    .status-text-anomaly {
        font-size: 18px;
        font-weight: 700;
        color: #ef4444;
        font-family: 'IBM Plex Mono', monospace;
        letter-spacing: 1px;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }
    .status-sub {
        font-size: 13px;
        color: #94a3b8;
        margin-top: 4px;
    }

    /* Section headers */
    .section-header {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: #3b82f6;
        font-family: 'IBM Plex Mono', monospace;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 1px solid #1e2d4a;
    }

    /* Info card */
    .info-card {
        background: #0d1224;
        border: 1px solid #1e2d4a;
        border-radius: 10px;
        padding: 16px 20px;
        font-size: 13px;
        color: #94a3b8;
        line-height: 1.7;
    }
    .info-card strong { color: #e2e8f0; }

    /* Wave region labels */
    .wave-label {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
        font-family: 'IBM Plex Mono', monospace;
        margin: 3px;
    }
    .p-wave { background: #1e3a5f; color: #60a5fa; }
    .qrs { background: #1c2d1e; color: #4ade80; }
    .t-wave { background: #2d1f1f; color: #f87171; }

    /* Divider */
    hr { border-color: #1e2d4a !important; }

    /* Streamlit selectbox, slider labels */
    .stRadio label, .stSlider label {
        color: #94a3b8 !important;
        font-size: 12px !important;
    }

    /* Title area */
    .hero-title {
        font-size: 36px;
        font-weight: 700;
        color: #e2e8f0;
        letter-spacing: -1px;
        line-height: 1.1;
    }
    .hero-subtitle {
        font-size: 14px;
        color: #64748b;
        font-family: 'IBM Plex Mono', monospace;
        margin-top: 6px;
        letter-spacing: 1px;
    }
    .hero-badge {
        display: inline-block;
        background: #1e2d4a;
        color: #60a5fa;
        border: 1px solid #2563eb;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 11px;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        letter-spacing: 1px;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)


# ─── LOAD MODEL ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = LSTMAutoencoder()
    model.load_state_dict(torch.load(
        "models/lstm_autoencoder.pt", map_location="cpu"
    ))
    model.eval()
    return model

@st.cache_resource
def load_threshold():
    return float(np.load("models/threshold.npy"))

@st.cache_data
def load_beats():
    normal = np.load("data/test.npy")
    anomaly = np.load("data/anomaly.npy")
    return normal, anomaly

model = load_model()
threshold = load_threshold()
normal_beats, anomaly_beats = load_beats()


# ─── HELPER ─────────────────────────────────────────────────────────────────
@st.cache_data
def get_reconstruction(beat_index, beat_type_str):
    if beat_type_str == "Normal Beat":
        beat = normal_beats[beat_index]
    else:
        beat = anomaly_beats[beat_index]
    tensor = torch.tensor(
        beat[np.newaxis, :, np.newaxis].astype(np.float32)
    )
    with torch.no_grad():
        output = model(tensor)
    recon = output.numpy()[0, :, 0]
    error = float(np.mean((recon - beat) ** 2))
    return beat, recon, error


# ─── SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 8px 0 24px 0;'>
        <div style='font-size:22px; font-weight:700; color:#e2e8f0;'>❤️ PulseWatcher</div>
        <div style='font-size:11px; color:#64748b; font-family: IBM Plex Mono; letter-spacing:1px; margin-top:4px;'>
            ECG ANOMALY DETECTION
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">BEAT SELECTOR</div>', unsafe_allow_html=True)

    beat_type = st.radio(
        "Beat type",
        ["Normal Beat", "Anomalous Beat"],
        label_visibility="collapsed"
    )

    max_index = 99
    beat_index = st.slider(
        "Beat index", 0, max_index, 0,
        help="Scroll through individual heartbeats from the dataset"
    )

    st.markdown("---")
    st.markdown('<div class="section-header">MODEL INFO</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-card">
        <strong>Architecture</strong><br/>LSTM Autoencoder<br/><br/>
        <strong>Dataset</strong><br/>MIT-BIH Arrhythmia DB<br/><br/>
        <strong>Training</strong><br/>59,816 normal beats<br/><br/>
        <strong>Threshold</strong><br/><span style='font-family: IBM Plex Mono; color:#3b82f6;'>{threshold:.6f}</span><br/>
        <span style='font-size:11px;'>(95th percentile)</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">PERFORMANCE</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-card">
        <strong>Precision</strong> &nbsp;95.21%<br/>
        <strong>Recall</strong> &nbsp;&nbsp;&nbsp;&nbsp;44.64%<br/>
        <strong>F1 Score</strong> &nbsp;&nbsp;60.79%<br/>
        <strong>Train Loss</strong> &nbsp;0.001584
    </div>
    """, unsafe_allow_html=True)


# ─── MAIN CONTENT ───────────────────────────────────────────────────────────

# Hero header
col_title, col_badges = st.columns([3, 2])
with col_title:
    st.markdown("""
    <div class="hero-title">❤️ PulseWatcher</div>
    <div class="hero-subtitle">REAL-TIME ECG ANOMALY DETECTION SYSTEM</div>
    """, unsafe_allow_html=True)
with col_badges:
    st.markdown("""
    <div style='padding-top: 12px; text-align: right;'>
        <span class="hero-badge">LSTM AUTOENCODER</span>
        <span class="hero-badge">MIT-BIH</span>
        <span class="hero-badge">UNSUPERVISED</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ─── SELECT BEAT & RUN MODEL ────────────────────────────────────────────────
beat_label = "Normal" if beat_type == "Normal Beat" else "Anomalous"
beat, reconstructed, error = get_reconstruction(beat_index, beat_type)
is_anomaly = error > threshold
error_pct = (error / threshold) * 100

# ─── STATUS BANNER ──────────────────────────────────────────────────────────
if is_anomaly:
    st.markdown(f"""
    <div class="status-anomaly">
        <div style='font-size:32px;'>🚨</div>
        <div>
            <div class="status-text-anomaly">ANOMALY DETECTED</div>
            <div class="status-sub">
                Reconstruction error is {error_pct:.1f}% of threshold —
                this beat pattern is outside the normal distribution
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="status-normal">
        <div style='font-size:32px;'>✅</div>
        <div>
            <div class="status-text-normal">NORMAL BEAT</div>
            <div class="status-sub">
                Reconstruction error is {error_pct:.1f}% of threshold —
                this beat matches the learned normal pattern
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)

# ─── METRICS ROW ────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">DIAGNOSTICS</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Reconstruction Error</div>
        <div class="metric-value {'anomaly' if is_anomaly else 'normal'}">{error:.6f}</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Decision Threshold</div>
        <div class="metric-value neutral">{threshold:.6f}</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    delta = error - threshold
    sign = "+" if delta > 0 else ""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Delta from Threshold</div>
        <div class="metric-value {'anomaly' if delta > 0 else 'normal'}">{sign}{delta:.6f}</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Beat Classification</div>
        <div class="metric-value {'anomaly' if is_anomaly else 'normal'}">
            {"ANOMALY" if is_anomaly else "NORMAL"}
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)

# ─── ECG CHART ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">ECG SIGNAL — ORIGINAL vs RECONSTRUCTED</div>',
            unsafe_allow_html=True)

x = list(range(187))
fig = go.Figure()

# Shaded area between original and reconstructed (shows error visually)
fig.add_trace(go.Scatter(
    x=x + x[::-1],
    y=list(beat) + list(reconstructed[::-1]),
    fill='toself',
    fillcolor='rgba(239, 68, 68, 0.08)' if is_anomaly else 'rgba(34, 197, 94, 0.08)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo='skip',
    showlegend=False,
    name='Error region'
))

fig.add_trace(go.Scatter(
    x=x, y=beat,
    name="Original ECG",
    line=dict(color="#60a5fa", width=2),
    hovertemplate="t=%{x}<br>amplitude=%{y:.4f}<extra>Original</extra>"
))

fig.add_trace(go.Scatter(
    x=x, y=reconstructed,
    name="Reconstructed",
    line=dict(color="#f87171" if is_anomaly else "#4ade80",
              width=2, dash="dash"),
    hovertemplate="t=%{x}<br>amplitude=%{y:.4f}<extra>Reconstructed</extra>"
))

# Wave region annotations
fig.add_vrect(x0=0, x1=40, fillcolor="#1e3a5f",
              opacity=0.15, line_width=0, annotation_text="P-wave",
              annotation_position="top left",
              annotation_font=dict(color="#60a5fa", size=10))
fig.add_vrect(x0=60, x1=100, fillcolor="#1c2d1e",
              opacity=0.2, line_width=0, annotation_text="QRS",
              annotation_position="top left",
              annotation_font=dict(color="#4ade80", size=10))
fig.add_vrect(x0=110, x1=160, fillcolor="#2d1f1f",
              opacity=0.15, line_width=0, annotation_text="T-wave",
              annotation_position="top left",
              annotation_font=dict(color="#f87171", size=10))

fig.update_layout(
    height=380,
    plot_bgcolor="#080c18",
    paper_bgcolor="#0a0e1a",
    font=dict(color="#94a3b8", family="IBM Plex Mono"),
    xaxis=dict(
        title="Timestep (samples)",
        gridcolor="#1e2d4a",
        zerolinecolor="#1e2d4a",
        tickfont=dict(size=10)
    ),
    yaxis=dict(
        title="Amplitude (normalized)",
        gridcolor="#1e2d4a",
        zerolinecolor="#1e2d4a",
        tickfont=dict(size=10)
    ),
    legend=dict(
        bgcolor="#0d1224",
        bordercolor="#1e2d4a",
        borderwidth=1,
        font=dict(size=11)
    ),
    margin=dict(l=60, r=30, t=30, b=60),
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

# Wave legend
st.markdown("""
<div style='text-align:center; margin-top:-10px; margin-bottom:20px;'>
    <span class="wave-label p-wave">P-WAVE &nbsp;0–40</span>
    <span class="wave-label qrs">QRS COMPLEX &nbsp;60–100</span>
    <span class="wave-label t-wave">T-WAVE &nbsp;110–160</span>
</div>
""", unsafe_allow_html=True)

# ─── EXPLAINABILITY ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">EXPLAINABILITY — WHERE IS THE ANOMALY?</div>',
            unsafe_allow_html=True)

pointwise_error = (beat - reconstructed) ** 2
mean_err = pointwise_error.mean()
colors = ["#ef4444" if e > mean_err else "#334155" for e in pointwise_error]

fig2 = go.Figure()

fig2.add_trace(go.Bar(
    x=x,
    y=pointwise_error,
    marker_color=colors,
    name="Per-timestep error",
    hovertemplate="t=%{x}<br>error=%{y:.6f}<extra></extra>"
))

fig2.add_hline(
    y=mean_err,
    line_dash="dash",
    line_color="#f59e0b",
    line_width=1.5,
    annotation_text=f"Mean error: {mean_err:.6f}",
    annotation_position="top right",
    annotation_font=dict(color="#f59e0b", size=10)
)

# Wave region shading on error chart too
fig2.add_vrect(x0=0, x1=40, fillcolor="#1e3a5f", opacity=0.1, line_width=0)
fig2.add_vrect(x0=60, x1=100, fillcolor="#1c2d1e", opacity=0.15, line_width=0)
fig2.add_vrect(x0=110, x1=160, fillcolor="#2d1f1f", opacity=0.1, line_width=0)

fig2.update_layout(
    height=280,
    plot_bgcolor="#080c18",
    paper_bgcolor="#0a0e1a",
    font=dict(color="#94a3b8", family="IBM Plex Mono"),
    xaxis=dict(
        title="Timestep",
        gridcolor="#1e2d4a",
        zerolinecolor="#1e2d4a",
        tickfont=dict(size=10)
    ),
    yaxis=dict(
        title="Squared Error",
        gridcolor="#1e2d4a",
        zerolinecolor="#1e2d4a",
        tickfont=dict(size=10)
    ),
    showlegend=False,
    margin=dict(l=60, r=30, t=20, b=60),
)

st.plotly_chart(fig2, use_container_width=True)

# ─── SMART INTERPRETATION ───────────────────────────────────────────────────
high_error_points = [i for i, e in enumerate(pointwise_error) if e > mean_err]

p_wave_hits = [p for p in high_error_points if 0 <= p <= 40]
qrs_hits = [p for p in high_error_points if 60 <= p <= 100]
twave_hits = [p for p in high_error_points if 110 <= p <= 160]

interpretation = []
if qrs_hits:
    interpretation.append("**QRS Complex** anomaly detected → possible ventricular arrhythmia")
if p_wave_hits:
    interpretation.append("**P-Wave** anomaly detected → possible atrial abnormality")
if twave_hits:
    interpretation.append("**T-Wave** anomaly detected → possible ischemia or repolarization issue")

col_interp, col_tip = st.columns([3, 2])

with col_interp:
    st.markdown('<div class="section-header">CLINICAL INTERPRETATION</div>',
                unsafe_allow_html=True)
    if interpretation and is_anomaly:
        for item in interpretation:
            st.markdown(f"""
            <div class="info-card" style='margin-bottom:8px;'>
                ⚠️ &nbsp;{item}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-card">
            ✅ &nbsp;All wave segments reconstructed within normal range.
            No clinical abnormality indicated.
        </div>
        """, unsafe_allow_html=True)

with col_tip:
    st.markdown('<div class="section-header">HOW IT WORKS</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-card">
        The LSTM Autoencoder was trained <strong>only on normal beats</strong>.
        It compresses each beat into a latent representation and reconstructs it.<br/><br/>
        Beats that differ from the learned normal pattern produce
        <strong>high reconstruction error</strong>.<br/><br/>
        Red bars = timesteps the model struggled with.<br/>
        Threshold = <span style='font-family: IBM Plex Mono; color:#3b82f6;'>{threshold:.6f}</span>
        (95th percentile of normal errors).
    </div>
    """, unsafe_allow_html=True)
    # ─── MODEL PERFORMANCE ──────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">MODEL PERFORMANCE METRICS</div>',
            unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Precision</div>
        <div class="metric-value normal">95.21%</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Recall</div>
        <div class="metric-value neutral">44.64%</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">F1 Score</div>
        <div class="metric-value neutral">60.79%</div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">ROC-AUC</div>
        <div class="metric-value normal">0.8678</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)

# ROC Curve + Confusion Matrix
col_roc, col_cm = st.columns(2)

with col_roc:
    st.markdown('<div class="section-header">ROC CURVE</div>',
                unsafe_allow_html=True)
    if os.path.exists("outputs/roc_curve.png"):
        st.image("outputs/roc_curve.png", use_container_width=True)
    else:
        st.warning("Run: python -m src.evaluate")

with col_cm:
    st.markdown('<div class="section-header">CONFUSION MATRIX</div>',
                unsafe_allow_html=True)
    if os.path.exists("outputs/confusion_matrix.png"):
        st.image("outputs/confusion_matrix.png", use_container_width=True)
    else:
        st.warning("Run: python -m src.evaluate")

st.markdown("<br/>", unsafe_allow_html=True)
st.markdown('<div class="section-header">UNDERSTANDING THE METRICS</div>',
            unsafe_allow_html=True)

col_explain1, col_explain2 = st.columns(2)

with col_explain1:
    st.markdown("""
    <div class="info-card">
        <strong>Why is Recall 44%?</strong><br/><br/>
        This is a deliberate tradeoff. In cardiac screening,
        a <strong>false positive</strong> — flagging a healthy patient
        as sick — causes unnecessary panic and expensive tests.<br/><br/>
        We optimized for <strong>high precision first</strong>.
        When PulseWatcher raises an alert, it is right
        <strong>95% of the time</strong>.<br/><br/>
        Recall can be improved by lowering the detection threshold —
        a tunable parameter depending on clinical context.
    </div>
    """, unsafe_allow_html=True)

with col_explain2:
    st.markdown("""
    <div class="info-card">
        <strong>What does ROC-AUC 0.8678 mean?</strong><br/><br/>
        AUC measures the model's ability to <strong>distinguish
        normal from anomalous beats</strong> across all thresholds.<br/><br/>
        <span style='color:#64748b;'>Random classifier &nbsp;= 0.50</span><br/>
        <span style='color:#3b82f6;'>PulseWatcher &nbsp;&nbsp;&nbsp;&nbsp;= 0.8678</span><br/>
        <span style='color:#22c55e;'>Perfect classifier = 1.00</span><br/><br/>
        Strong discriminative power — without ever seeing
        a single anomalous beat during training.
    </div>
    """, unsafe_allow_html=True)

# ─── FOOTER ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; padding: 20px 0 10px 0;'>
    <div style='font-size:11px; color:#334155; font-family: IBM Plex Mono; letter-spacing:2px;'>
        PULSEWATCHER &nbsp;·&nbsp; MIT-BIH ARRHYTHMIA DATABASE
        &nbsp;·&nbsp; LSTM AUTOENCODER &nbsp;·&nbsp; UNSUPERVISED LEARNING
    </div>
</div>
""", unsafe_allow_html=True)