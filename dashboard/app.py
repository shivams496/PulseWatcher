# dashboard/app.py  —  PulseWatcher  |  Redesigned UI v3
# Drop-in replacement: open app.py → Ctrl+A → paste → Ctrl+S

import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import sys, os
sys.path.append(os.path.abspath("."))
from src.model import LSTMAutoencoder

# ═══════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="PulseWatcher — ECG Anomaly Detection",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════
#  GLOBAL CSS  — Aesthetic: Clinical Noir
#  Fonts: Syne (display) + JetBrains Mono (data)
#  Palette: near-black base, electric cyan accent, danger crimson
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500;700&display=swap');

/* ── Reset & Base ─────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #060810 !important;
    color: #c8d6e8;
}

#MainMenu, footer, header { visibility: hidden; }

.stApp {
    background: #060810;
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -20%, rgba(0,200,255,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 40% 30% at 90% 80%, rgba(239,68,68,0.04) 0%, transparent 50%);
    min-height: 100vh;
}

/* ── Sidebar ───────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #080b14 !important;
    border-right: 1px solid #0d1f35;
}
[data-testid="stSidebar"] * { color: #8899aa !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 13px !important; }

/* ── Scrollbar ─────────────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #060810; }
::-webkit-scrollbar-thumb { background: #0d2040; border-radius: 2px; }

/* ══════════════════════════════════════════════════════════
   COMPONENTS
══════════════════════════════════════════════════════════ */

/* ── Hero header ───────────────────────────────────────────── */
.pw-hero {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    padding: 28px 0 20px 0;
    border-bottom: 1px solid #0d1f35;
    margin-bottom: 28px;
    flex-wrap: wrap;
    gap: 12px;
}
.pw-logo {
    display: flex;
    align-items: center;
    gap: 14px;
}
.pw-logo-icon {
    width: 44px; height: 44px;
    background: linear-gradient(135deg, #00c8ff22, #00c8ff08);
    border: 1px solid #00c8ff44;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 22px;
}
.pw-logo-text {
    font-size: 26px;
    font-weight: 800;
    color: #e8f4ff;
    letter-spacing: -0.5px;
    line-height: 1;
}
.pw-logo-sub {
    font-size: 10px;
    font-family: 'JetBrains Mono', monospace;
    color: #3a5a7a;
    letter-spacing: 3px;
    margin-top: 4px;
    text-transform: uppercase;
}
.pw-badges {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    align-items: center;
}
.pw-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    font-weight: 500;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 5px 12px;
    border-radius: 4px;
    border: 1px solid;
}
.pw-badge-cyan  { color: #00c8ff; border-color: #00c8ff33; background: #00c8ff0a; }
.pw-badge-slate { color: #64748b; border-color: #1e2d4a;   background: #0d1224; }

/* ── Status Banner ─────────────────────────────────────────── */
.pw-status {
    position: relative;
    border-radius: 12px;
    padding: 20px 28px;
    display: flex;
    align-items: center;
    gap: 18px;
    overflow: hidden;
    margin-bottom: 24px;
}
.pw-status::before {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: 12px;
    pointer-events: none;
}
.pw-status-normal {
    background: linear-gradient(135deg, #00170d 0%, #002a18 100%);
    border: 1px solid #00ff8833;
    border-left: 3px solid #00ff88;
}
.pw-status-anomaly {
    background: linear-gradient(135deg, #1a0404 0%, #2d0808 100%);
    border: 1px solid #ff224433;
    border-left: 3px solid #ff2244;
}
.pw-status-icon { font-size: 36px; line-height: 1; flex-shrink: 0; }
.pw-status-title-normal {
    font-size: 20px; font-weight: 800;
    color: #00ff88; letter-spacing: 1px;
    font-family: 'JetBrains Mono', monospace;
}
.pw-status-title-anomaly {
    font-size: 20px; font-weight: 800;
    color: #ff2244; letter-spacing: 1px;
    font-family: 'JetBrains Mono', monospace;
    animation: pulse-red 1.8s ease-in-out infinite;
}
@keyframes pulse-red {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.65; }
}
.pw-status-sub {
    font-size: 13px; color: #5a7a9a;
    margin-top: 5px; line-height: 1.5;
    font-family: 'JetBrains Mono', monospace;
}
.pw-status-bg-text {
    position: absolute; right: 24px; top: 50%;
    transform: translateY(-50%);
    font-size: 72px; font-weight: 800;
    opacity: 0.04; letter-spacing: -2px;
    font-family: 'Syne', sans-serif;
    pointer-events: none; user-select: none;
    color: #fff;
}

/* ── Section label ─────────────────────────────────────────── */
.pw-section {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; font-weight: 700;
    letter-spacing: 3px; text-transform: uppercase;
    color: #00c8ff;
    padding-bottom: 10px;
    border-bottom: 1px solid #0d1f35;
    margin-bottom: 16px;
}

/* ── Metric Card ───────────────────────────────────────────── */
.pw-metric {
    background: #080b14;
    border: 1px solid #0d1f35;
    border-radius: 10px;
    padding: 18px 16px;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: border-color 0.25s, transform 0.2s;
}
.pw-metric:hover {
    border-color: #00c8ff33;
    transform: translateY(-1px);
}
.pw-metric::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    border-radius: 10px 10px 0 0;
}
.pw-metric-cyan::after   { background: #00c8ff; }
.pw-metric-green::after  { background: #00ff88; }
.pw-metric-red::after    { background: #ff2244; }
.pw-metric-amber::after  { background: #f59e0b; }

.pw-metric-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; font-weight: 500;
    letter-spacing: 2px; text-transform: uppercase;
    color: #3a5a7a; margin-bottom: 10px;
}
.pw-metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 24px; font-weight: 700;
    line-height: 1; letter-spacing: -0.5px;
}
.pw-metric-value.cyan  { color: #00c8ff; }
.pw-metric-value.green { color: #00ff88; }
.pw-metric-value.red   { color: #ff2244; }
.pw-metric-value.amber { color: #f59e0b; }
.pw-metric-value.white { color: #e8f4ff; }

/* ── Info / Interpretation card ────────────────────────────── */
.pw-card {
    background: #080b14;
    border: 1px solid #0d1f35;
    border-radius: 10px;
    padding: 18px 20px;
    font-size: 13px;
    line-height: 1.75;
    color: #6a8aaa;
    height: 100%;
}
.pw-card strong { color: #c8d6e8; font-weight: 600; }
.pw-card .accent { color: #00c8ff; font-family: 'JetBrains Mono', monospace; }

/* ── Alert chip ────────────────────────────────────────────── */
.pw-chip {
    display: inline-flex; align-items: center; gap: 7px;
    padding: 8px 16px; border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px; font-weight: 500;
    margin-bottom: 8px; width: 100%;
}
.pw-chip-warn {
    background: #1a0d00; border: 1px solid #f59e0b44;
    color: #f59e0b;
}
.pw-chip-ok {
    background: #00170d; border: 1px solid #00ff8844;
    color: #00ff88;
}

/* ── Wave legend pills ─────────────────────────────────────── */
.pw-wave-legend {
    display: flex; justify-content: center; gap: 10px;
    flex-wrap: wrap; margin: -4px 0 20px 0;
}
.pw-pill {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; font-weight: 600; letter-spacing: 1px;
    padding: 4px 12px; border-radius: 20px; border: 1px solid;
}
.pw-pill-p   { color: #60a5fa; border-color: #60a5fa44; background: #0a1628; }
.pw-pill-qrs { color: #4ade80; border-color: #4ade8044; background: #061408; }
.pw-pill-t   { color: #f87171; border-color: #f8717144; background: #160606; }

/* ── Performance grid ──────────────────────────────────────── */
.pw-perf-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-bottom: 16px;
}
.pw-perf-item {
    background: #080b14;
    border: 1px solid #0d1f35;
    border-radius: 8px;
    padding: 14px 16px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.pw-perf-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; color: #3a5a7a; letter-spacing: 1px;
}
.pw-perf-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 15px; font-weight: 700;
}

/* ── Sidebar custom ────────────────────────────────────────── */
.sb-section {
    font-family: 'JetBrains Mono', monospace;
    font-size: 8px; font-weight: 700;
    letter-spacing: 3px; text-transform: uppercase;
    color: #00c8ff !important;
    padding-bottom: 8px;
    border-bottom: 1px solid #0d1f35;
    margin-bottom: 12px; margin-top: 20px;
}
.sb-card {
    background: #06090f;
    border: 1px solid #0d1f35;
    border-radius: 8px;
    padding: 14px 16px;
    font-size: 12px;
    line-height: 1.8;
    color: #4a6a8a !important;
}
.sb-card strong { color: #8899aa !important; }
.sb-mono {
    font-family: 'JetBrains Mono', monospace !important;
    color: #00c8ff !important;
    font-size: 12px !important;
}

/* ── Footer ────────────────────────────────────────────────── */
.pw-footer {
    text-align: center;
    padding: 32px 0 16px 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; letter-spacing: 3px;
    color: #1a2a3a;
    text-transform: uppercase;
    border-top: 1px solid #0d1f35;
    margin-top: 40px;
}

hr { border-color: #0d1f35 !important; margin: 28px 0 !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    model = LSTMAutoencoder()
    model.load_state_dict(torch.load("models/lstm_autoencoder.pt", map_location="cpu"))
    model.eval()
    return model

@st.cache_resource
def load_threshold():
    return float(np.load("models/threshold.npy"))

@st.cache_data
def load_beats():
    normal  = np.load("data/test.npy")
    anomaly = np.load("data/anomaly.npy")
    return normal, anomaly

@st.cache_data
def compute_roc_data():
    """Compute ROC curve data once and cache it."""
    model     = load_model()
    threshold = load_threshold()
    normal_beats, anomaly_beats = load_beats()

    # Sample for speed (500 each)
    rng = np.random.default_rng(42)
    n_idx = rng.choice(len(normal_beats),  min(500, len(normal_beats)),  replace=False)
    a_idx = rng.choice(len(anomaly_beats), min(500, len(anomaly_beats)), replace=False)

    def errors(data):
        t = torch.tensor(data[:, :, np.newaxis].astype(np.float32))
        with torch.no_grad():
            out = model(t)
        return torch.mean((out - t) ** 2, dim=(1, 2)).numpy()

    n_err = errors(normal_beats[n_idx])
    a_err = errors(anomaly_beats[a_idx])

    y_true   = np.concatenate([np.zeros(len(n_err)), np.ones(len(a_err))])
    y_scores = np.concatenate([n_err, a_err])
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)

    y_pred = (y_scores > threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    return fpr.tolist(), tpr.tolist(), float(auc), cm.tolist()

@st.cache_data
def get_reconstruction(beat_index, beat_type_str):
    model = load_model()
    normal_beats, anomaly_beats = load_beats()
    beat = normal_beats[beat_index] if beat_type_str == "Normal Beat" else anomaly_beats[beat_index]
    tensor = torch.tensor(beat[np.newaxis, :, np.newaxis].astype(np.float32))
    with torch.no_grad():
        output = model(tensor)
    recon = output.numpy()[0, :, 0]
    error = float(np.mean((recon - beat) ** 2))
    return beat.tolist(), recon.tolist(), error

model     = load_model()
threshold = load_threshold()
normal_beats, anomaly_beats = load_beats()


# ═══════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding: 16px 0 8px 0;'>
        <div style='font-size:18px; font-weight:800; color:#c8d6e8; letter-spacing:-0.3px;'>🫀 PulseWatcher</div>
        <div style='font-family: JetBrains Mono; font-size:9px; color:#1e3a5a; letter-spacing:3px; margin-top:5px;'>
            ECG ANOMALY DETECTION
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-section">BEAT SELECTOR</div>', unsafe_allow_html=True)

    beat_type = st.radio(
        "Select beat type",
        ["Normal Beat", "Anomalous Beat"],
        label_visibility="collapsed"
    )

    max_idx   = min(99, len(anomaly_beats) - 1) if beat_type == "Anomalous Beat" else min(99, len(normal_beats) - 1)
    beat_index = st.slider("Beat index", 0, max_idx, 0,
                           help="Browse individual heartbeats from the dataset")

    st.markdown('<div class="sb-section">MODEL INFO</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="sb-card">
        <strong>Architecture</strong><br/>LSTM Autoencoder<br/><br/>
        <strong>Dataset</strong><br/>MIT-BIH Arrhythmia DB<br/><br/>
        <strong>Training beats</strong><br/>59,816 normal<br/><br/>
        <strong>Threshold</strong><br/>
        <span class="sb-mono">{threshold:.6f}</span><br/>
        <span style='font-size:10px; color:#1e3a5a;'>95th percentile of normal errors</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-section">PERFORMANCE</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-card">
        <strong>Precision</strong>&nbsp;&nbsp;95.21%<br/>
        <strong>Recall</strong>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;44.64%<br/>
        <strong>F1 Score</strong>&nbsp;&nbsp;&nbsp;60.79%<br/>
        <strong>ROC-AUC</strong>&nbsp;&nbsp;&nbsp;0.8678<br/>
        <strong>Train Loss</strong>&nbsp;0.001584
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  MAIN — RUN INFERENCE
# ═══════════════════════════════════════════════════════════════
beat, reconstructed, error = get_reconstruction(beat_index, beat_type)
is_anomaly  = error > threshold
error_pct   = (error / threshold) * 100
delta       = error - threshold
sign        = "+" if delta > 0 else ""
x           = list(range(187))
pointwise   = [(b - r) ** 2 for b, r in zip(beat, reconstructed)]
mean_err    = float(np.mean(pointwise))


# ═══════════════════════════════════════════════════════════════
#  HERO HEADER
# ═══════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="pw-hero">
    <div class="pw-logo">
        <div class="pw-logo-icon">🫀</div>
        <div>
            <div class="pw-logo-text">PulseWatcher</div>
            <div class="pw-logo-sub">Real-Time ECG Anomaly Detection System</div>
        </div>
    </div>
    <div class="pw-badges">
        <span class="pw-badge pw-badge-cyan">LSTM Autoencoder</span>
        <span class="pw-badge pw-badge-cyan">MIT-BIH</span>
        <span class="pw-badge pw-badge-slate">Unsupervised</span>
        <span class="pw-badge pw-badge-slate">PyTorch</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  STATUS BANNER
# ═══════════════════════════════════════════════════════════════
if is_anomaly:
    st.markdown(f"""
    <div class="pw-status pw-status-anomaly">
        <div class="pw-status-icon">🚨</div>
        <div>
            <div class="pw-status-title-anomaly">ANOMALY DETECTED</div>
            <div class="pw-status-sub">
                Reconstruction error {error:.6f} is {error_pct:.1f}% of threshold —
                beat pattern outside the learned normal distribution
            </div>
        </div>
        <div class="pw-status-bg-text">ANOMALY</div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="pw-status pw-status-normal">
        <div class="pw-status-icon">✅</div>
        <div>
            <div class="pw-status-title-normal">NORMAL BEAT</div>
            <div class="pw-status-sub">
                Reconstruction error {error:.6f} is {error_pct:.1f}% of threshold —
                beat matches the learned normal pattern
            </div>
        </div>
        <div class="pw-status-bg-text">NORMAL</div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  DIAGNOSTICS ROW  (4 cards)
# ═══════════════════════════════════════════════════════════════
st.markdown('<div class="pw-section">DIAGNOSTICS</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
cards = [
    (c1, "Reconstruction Error", f"{error:.6f}",
     "red" if is_anomaly else "green", "red" if is_anomaly else "green"),
    (c2, "Decision Threshold",   f"{threshold:.6f}", "cyan",  "cyan"),
    (c3, "Delta from Threshold", f"{sign}{delta:.6f}",
     "red" if delta > 0 else "green", "red" if delta > 0 else "green"),
    (c4, "Classification",
     "ANOMALY" if is_anomaly else "NORMAL",
     "red" if is_anomaly else "green", "red" if is_anomaly else "green"),
]
for col, label, value, accent, val_class in cards:
    with col:
        st.markdown(f"""
        <div class="pw-metric pw-metric-{accent}">
            <div class="pw-metric-label">{label}</div>
            <div class="pw-metric-value {val_class}">{value}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  ECG CHART  (Original vs Reconstructed)
# ═══════════════════════════════════════════════════════════════
st.markdown('<div class="pw-section">ECG SIGNAL — ORIGINAL vs RECONSTRUCTED</div>',
            unsafe_allow_html=True)

accent_color = "#ff2244" if is_anomaly else "#00ff88"

fig_ecg = go.Figure()

# Shaded error fill
fig_ecg.add_trace(go.Scatter(
    x=x + x[::-1],
    y=beat + reconstructed[::-1],
    fill='toself',
    fillcolor=f'rgba({"255,34,68" if is_anomaly else "0,255,136"}, 0.06)',
    line=dict(color='rgba(0,0,0,0)'),
    hoverinfo='skip', showlegend=False
))

# Original ECG
fig_ecg.add_trace(go.Scatter(
    x=x, y=beat,
    name="Original ECG",
    line=dict(color="#00c8ff", width=2),
    hovertemplate="t=%{x} | amp=%{y:.4f}<extra>Original</extra>"
))

# Reconstructed
fig_ecg.add_trace(go.Scatter(
    x=x, y=reconstructed,
    name="Reconstructed",
    line=dict(color=accent_color, width=1.8, dash="dot"),
    hovertemplate="t=%{x} | amp=%{y:.4f}<extra>Reconstructed</extra>"
))

# Wave region shading
for x0, x1, fc, label, fc2 in [
    (0,   40,  "#0a1628", "P-wave", "#60a5fa"),
    (60,  100, "#061408", "QRS",    "#4ade80"),
    (110, 160, "#160606", "T-wave", "#f87171"),
]:
    fig_ecg.add_vrect(x0=x0, x1=x1, fillcolor=fc, opacity=0.5, line_width=0,
                      annotation_text=label, annotation_position="top left",
                      annotation_font=dict(color=fc2, size=9, family="JetBrains Mono"))

fig_ecg.update_layout(
    height=360,
    plot_bgcolor="#04060e",
    paper_bgcolor="#060810",
    font=dict(color="#3a5a7a", family="JetBrains Mono", size=10),
    xaxis=dict(title="Timestep", gridcolor="#0a1422", zerolinecolor="#0a1422",
               title_font=dict(size=10), tickfont=dict(size=9)),
    yaxis=dict(title="Amplitude (norm.)", gridcolor="#0a1422", zerolinecolor="#0a1422",
               title_font=dict(size=10), tickfont=dict(size=9)),
    legend=dict(bgcolor="#080b14", bordercolor="#0d1f35", borderwidth=1,
                font=dict(size=10), orientation="h", yanchor="bottom", y=1.02,
                xanchor="right", x=1),
    margin=dict(l=55, r=20, t=40, b=55),
    hovermode="x unified"
)
st.plotly_chart(fig_ecg, use_container_width=True)

st.markdown("""
<div class="pw-wave-legend">
    <span class="pw-pill pw-pill-p">P-WAVE · 0–40</span>
    <span class="pw-pill pw-pill-qrs">QRS COMPLEX · 60–100</span>
    <span class="pw-pill pw-pill-t">T-WAVE · 110–160</span>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  EXPLAINABILITY + INTERPRETATION  (side by side)
# ═══════════════════════════════════════════════════════════════
st.markdown('<div class="pw-section">EXPLAINABILITY — WHERE IS THE ANOMALY?</div>',
            unsafe_allow_html=True)

col_bar, col_interp = st.columns([3, 2])

with col_bar:
    bar_colors = ["#ff2244" if e > mean_err else "#0d2040" for e in pointwise]

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=x, y=pointwise,
        marker_color=bar_colors,
        hovertemplate="t=%{x}<br>error=%{y:.6f}<extra></extra>",
        name="Per-timestep error"
    ))
    fig_bar.add_hline(
        y=mean_err, line_dash="dash", line_color="#f59e0b", line_width=1.2,
        annotation_text=f"mean {mean_err:.5f}",
        annotation_position="top right",
        annotation_font=dict(color="#f59e0b", size=9, family="JetBrains Mono")
    )
    for x0, x1, fc in [(0,40,"#0a1628"),(60,100,"#061408"),(110,160,"#160606")]:
        fig_bar.add_vrect(x0=x0, x1=x1, fillcolor=fc, opacity=0.5, line_width=0)

    fig_bar.update_layout(
        height=240,
        plot_bgcolor="#04060e", paper_bgcolor="#060810",
        font=dict(color="#3a5a7a", family="JetBrains Mono", size=9),
        xaxis=dict(title="Timestep", gridcolor="#0a1422", zerolinecolor="#0a1422",
                   tickfont=dict(size=8)),
        yaxis=dict(title="Squared Error", gridcolor="#0a1422", zerolinecolor="#0a1422",
                   tickfont=dict(size=8)),
        showlegend=False,
        margin=dict(l=50, r=20, t=20, b=45),
        bargap=0.05
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with col_interp:
    high_pts   = [i for i, e in enumerate(pointwise) if e > mean_err]
    p_hits     = [p for p in high_pts if 0   <= p <= 40]
    qrs_hits   = [p for p in high_pts if 60  <= p <= 100]
    t_hits     = [p for p in high_pts if 110 <= p <= 160]

    st.markdown('<div class="pw-section" style="margin-top:4px;">CLINICAL INTERPRETATION</div>',
                unsafe_allow_html=True)

    if is_anomaly:
        findings = []
        if qrs_hits: findings.append(("⚡", "QRS Complex anomaly → possible ventricular arrhythmia"))
        if p_hits:   findings.append(("〰", "P-Wave anomaly → possible atrial abnormality"))
        if t_hits:   findings.append(("🌊", "T-Wave anomaly → possible ischemia / repolarization issue"))

        if findings:
            for icon, text in findings:
                st.markdown(f'<div class="pw-chip pw-chip-warn">{icon}&nbsp; {text}</div>',
                            unsafe_allow_html=True)
        else:
            st.markdown('<div class="pw-chip pw-chip-warn">⚠️&nbsp; Anomaly detected — diffuse error across beat</div>',
                        unsafe_allow_html=True)
    else:
        st.markdown('<div class="pw-chip pw-chip-ok">✓&nbsp; All wave segments within normal range</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class="pw-card" style="margin-top:8px;">
            The model reconstructed this beat with <strong>low error</strong>
            across all cardiac phases. No clinical abnormality indicated.
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="pw-card" style="margin-top:10px; font-size:12px;">
        <strong>Red bars</strong> = timesteps the model struggled to reconstruct.<br/><br/>
        Threshold = <span class="accent">{threshold:.6f}</span>
        &nbsp;(95th percentile of normal reconstruction errors).<br/><br/>
        Trained on <strong>normal beats only</strong> — anomaly detection
        is purely from reconstruction failure.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  MODEL PERFORMANCE  (ROC + Confusion Matrix side by side)
# ═══════════════════════════════════════════════════════════════
st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown('<div class="pw-section">MODEL PERFORMANCE</div>', unsafe_allow_html=True)

# Performance metric row
pm1, pm2, pm3, pm4 = st.columns(4)
perf = [
    (pm1, "Precision",   "95.21%", "green",  "green"),
    (pm2, "Recall",      "44.64%", "amber",  "amber"),
    (pm3, "F1 Score",    "60.79%", "cyan",   "cyan"),
    (pm4, "ROC-AUC",     "0.8678", "green",  "green"),
]
for col, label, value, accent, val_class in perf:
    with col:
        st.markdown(f"""
        <div class="pw-metric pw-metric-{accent}" style="margin-bottom:16px;">
            <div class="pw-metric-label">{label}</div>
            <div class="pw-metric-value {val_class}">{value}</div>
        </div>
        """, unsafe_allow_html=True)

# ROC + CM charts
col_roc, col_cm = st.columns(2)

with col_roc:
    st.markdown('<div class="pw-section">ROC CURVE</div>', unsafe_allow_html=True)
    try:
        fpr, tpr, auc, cm = compute_roc_data()

        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            line=dict(color="#0d2040", dash="dash", width=1),
            showlegend=False, hoverinfo="skip"
        ))
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr,
            fill='tozeroy',
            fillcolor="rgba(0,200,255,0.07)",
            line=dict(color="#00c8ff", width=2),
            name=f"AUC = {auc:.4f}",
            hovertemplate="FPR=%{x:.3f}<br>TPR=%{y:.3f}<extra></extra>"
        ))
        fig_roc.update_layout(
            height=280,
            plot_bgcolor="#04060e", paper_bgcolor="#060810",
            font=dict(color="#3a5a7a", family="JetBrains Mono", size=9),
            xaxis=dict(title="False Positive Rate", gridcolor="#0a1422",
                       zerolinecolor="#0a1422", range=[0,1]),
            yaxis=dict(title="True Positive Rate",  gridcolor="#0a1422",
                       zerolinecolor="#0a1422", range=[0,1]),
            legend=dict(bgcolor="#080b14", bordercolor="#0d1f35",
                        borderwidth=1, font=dict(size=10)),
            margin=dict(l=50, r=20, t=20, b=50)
        )
        st.plotly_chart(fig_roc, use_container_width=True)
    except Exception as e:
        st.markdown(f'<div class="pw-card">ROC curve unavailable: {e}</div>',
                    unsafe_allow_html=True)

with col_cm:
    st.markdown('<div class="pw-section">CONFUSION MATRIX</div>', unsafe_allow_html=True)
    try:
        fpr, tpr, auc, cm = compute_roc_data()
        cm_arr = np.array(cm)
        labels = ["Normal", "Anomaly"]
        z_text = [[str(cm_arr[i][j]) for j in range(2)] for i in range(2)]

        fig_cm = go.Figure(go.Heatmap(
            z=cm_arr,
            x=labels, y=labels,
            colorscale=[[0,"#04060e"],[0.5,"#0a1e3a"],[1,"#00c8ff"]],
            showscale=False,
            text=z_text,
            texttemplate="%{text}",
            textfont=dict(size=20, family="JetBrains Mono", color="#e8f4ff"),
            hovertemplate="Actual=%{y}<br>Predicted=%{x}<br>Count=%{z}<extra></extra>"
        ))
        fig_cm.update_layout(
            height=280,
            plot_bgcolor="#04060e", paper_bgcolor="#060810",
            font=dict(color="#3a5a7a", family="JetBrains Mono", size=10),
            xaxis=dict(title="Predicted", side="bottom"),
            yaxis=dict(title="Actual", autorange="reversed"),
            margin=dict(l=70, r=20, t=20, b=60)
        )
        st.plotly_chart(fig_cm, use_container_width=True)
    except Exception as e:
        st.markdown(f'<div class="pw-card">Confusion matrix unavailable: {e}</div>',
                    unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  METRIC EXPLANATION  (2 cards)
# ═══════════════════════════════════════════════════════════════
ex1, ex2 = st.columns(2)
with ex1:
    st.markdown("""
    <div class="pw-card">
        <strong>Why is Recall 44%?</strong><br/><br/>
        A deliberate clinical tradeoff. In cardiac screening, a
        <strong>false positive</strong> — flagging a healthy patient
        as sick — triggers unnecessary panic and expensive tests.<br/><br/>
        PulseWatcher optimises <strong>precision first</strong>.
        When it raises an alert, it is correct <span class="accent">95% of the time</span>.
        Recall can be improved by lowering the detection threshold — a tunable parameter
        depending on clinical context.
    </div>
    """, unsafe_allow_html=True)

with ex2:
    st.markdown("""
    <div class="pw-card">
        <strong>What does ROC-AUC 0.8678 mean?</strong><br/><br/>
        AUC measures the model's ability to <strong>distinguish
        normal from anomalous beats</strong> across all thresholds.<br/><br/>
        Random classifier &nbsp;→ <span style='color:#3a5a7a;'>0.50</span><br/>
        PulseWatcher &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;→ <span class="accent">0.8678</span><br/>
        Perfect classifier &nbsp;→ <span style='color:#00ff88;'>1.00</span><br/><br/>
        Strong discriminative power — without ever seeing
        a single anomalous beat during training.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  FOOTER
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="pw-footer">
    PulseWatcher &nbsp;·&nbsp; MIT-BIH Arrhythmia Database
    &nbsp;·&nbsp; LSTM Autoencoder &nbsp;·&nbsp; Unsupervised Anomaly Detection
    &nbsp;·&nbsp; B.Tech Final Year Project · May 2026
</div>
""", unsafe_allow_html=True)