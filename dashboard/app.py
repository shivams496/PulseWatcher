# dashboard/app.py  --  PulseWatcher  |  Redesigned UI v3
# Drop-in replacement: open app.py -> Ctrl+A -> paste -> Ctrl+S

import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import sys, os
import time
import io
from datetime import datetime
sys.path.append(os.path.abspath("."))
from src.model import LSTMAutoencoder
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

# &#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;
#  PAGE CONFIG
# &#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;

# ═══════════════════════════════════════════════════════════════
#  PDF REPORT GENERATOR  (Task 6 - Phase 2)
# ═══════════════════════════════════════════════════════════════
def ordinal(n):
    """Return ordinal string: 82 -> '82nd', 90 -> '90th', etc."""
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    return f"{n}" + {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def generate_pdf_report(beat, reconstructed, error, threshold, is_anomaly,
                        a_score, pointwise, metrics, beat_label, upload_source):
    """Generate a clinical-style PDF summary report. Returns bytes."""
    buf = io.BytesIO()

    # ── Colour palette matching the dashboard ──
    C_BG       = colors.HexColor("#060810")
    C_CYAN     = colors.HexColor("#00c8ff")
    C_GREEN    = colors.HexColor("#00ff88")
    C_RED      = colors.HexColor("#ff2244")
    C_AMBER    = colors.HexColor("#f59e0b")
    C_SLATE    = colors.HexColor("#3a5a7a")
    C_CARD     = colors.HexColor("#0a1422")
    C_WHITE    = colors.HexColor("#c8d6e8")
    C_ACCENT   = C_RED if is_anomaly else C_GREEN

    # Dark page background via canvas callback
    def dark_background(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(C_BG)
        canvas.rect(0, 0, A4[0], A4[1], fill=1, stroke=0)
        canvas.restoreState()

    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=18*mm, bottomMargin=18*mm
    )

    styles = getSampleStyleSheet()

    def S(name, **kw):
        base = styles["Normal"]
        return ParagraphStyle(name, parent=base, **kw)

    title_style   = S("T",  fontSize=24, textColor=C_CYAN,  fontName="Helvetica-Bold",
                       spaceAfter=0, alignment=TA_LEFT)
    sub_style     = S("Su", fontSize=8,  textColor=C_SLATE, fontName="Helvetica",
                       spaceAfter=8, spaceBefore=4)
    section_style = S("Se", fontSize=8,  textColor=C_CYAN,  fontName="Helvetica-Bold",
                       spaceBefore=10, spaceAfter=4, leading=12)
    body_style    = S("B",  fontSize=9,  textColor=C_WHITE, fontName="Helvetica",
                       leading=14, spaceAfter=4)
    verdict_style = S("V",  fontSize=20, textColor=C_ACCENT, fontName="Helvetica-Bold",
                       alignment=TA_CENTER, spaceAfter=4, spaceBefore=4)
    small_style   = S("Sm", fontSize=8,  textColor=C_SLATE, fontName="Helvetica",
                       leading=12)
    right_style   = S("R",  fontSize=8,  textColor=C_SLATE, fontName="Helvetica",
                       alignment=TA_RIGHT)

    story = []
    ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")

    # ── Header: title on its own line, subtitle below ──
    story.append(Paragraph("PulseWatcher", title_style))
    story.append(Paragraph(
        "ECG Anomaly Detection System  \u2022  LSTM Autoencoder  \u2022  MIT-BIH Dataset",
        sub_style
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=C_CYAN, spaceAfter=10))

    # ── Report meta ──
    meta_data = [
        ["Report generated", ts],
        ["Signal source",    "Uploaded CSV" if upload_source else f"Preset — {beat_label}"],
        ["Model",            "LSTM Autoencoder  (64 hidden units, PyTorch)"],
        ["Dataset",          "MIT-BIH Arrhythmia Database"],
        ["Threshold",        f"{threshold:.6f}  ({ordinal(metrics.get('threshold_percentile', 82))} percentile)"],
    ]
    meta_table = Table(meta_data, colWidths=[45*mm, 120*mm])
    meta_table.setStyle(TableStyle([
        ("FONTNAME",    (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE",    (0,0), (-1,-1), 8),
        ("TEXTCOLOR",   (0,0), (0,-1),  C_SLATE),
        ("TEXTCOLOR",   (1,0), (1,-1),  C_WHITE),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [C_BG, C_CARD]),
        ("TOPPADDING",  (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # ── Verdict ──
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_SLATE, spaceAfter=8))
    verdict_text = "ANOMALY DETECTED" if is_anomaly else "NORMAL BEAT"
    story.append(Paragraph(verdict_text, verdict_style))

    # ── Key metrics table ──
    story.append(Paragraph("BEAT ANALYSIS", section_style))
    error_pct = (error / threshold) * 100
    delta     = error - threshold
    sign      = "+" if delta > 0 else ""
    kpi_data  = [
        ["RECONSTRUCTION ERROR", "DECISION THRESHOLD", "DELTA", "ANOMALY SCORE"],
        [f"{error:.6f}", f"{threshold:.6f}", f"{sign}{delta:.6f}", f"{a_score} / 100"],
    ]
    kpi_table = Table(kpi_data, colWidths=[42*mm]*4)
    kpi_table.setStyle(TableStyle([
        ("FONTNAME",     (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTNAME",     (0,1), (-1,1),  "Helvetica-Bold"),
        ("FONTSIZE",     (0,0), (-1,0),  7),
        ("FONTSIZE",     (0,1), (-1,1),  13),
        ("TEXTCOLOR",    (0,0), (-1,0),  C_SLATE),
        ("TEXTCOLOR",    (0,1), (-1,1),  C_ACCENT),
        ("ALIGN",        (0,0), (-1,-1), "CENTER"),
        ("BACKGROUND",   (0,0), (-1,-1), C_CARD),
        ("BOX",          (0,0), (-1,-1), 0.5, C_SLATE),
        ("INNERGRID",    (0,0), (-1,-1), 0.3, C_SLATE),
        ("TOPPADDING",   (0,0), (-1,-1), 6),
        ("BOTTOMPADDING",(0,0), (-1,-1), 6),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 8))

    # ── Waveform ASCII sparkline (compact visual) ──
    story.append(Paragraph("SIGNAL RECONSTRUCTION SUMMARY", section_style))
    top3   = sorted(range(len(pointwise)), key=lambda i: pointwise[i], reverse=True)[:5]
    mean_e = float(sum(pointwise) / len(pointwise))
    high_pts = [i for i, e in enumerate(pointwise) if e > mean_e]
    p_hits   = [p for p in high_pts if 0   <= p <= 40]
    qrs_hits = [p for p in high_pts if 60  <= p <= 100]
    t_hits   = [p for p in high_pts if 110 <= p <= 160]

    recon_data = [
        ["Mean reconstruction error",  f"{mean_e:.6f}"],
        ["Error vs threshold",          f"{error_pct:.1f}%  of threshold"],
        ["Top-5 highest-error timesteps", ", ".join([f"t={t}" for t in top3])],
        ["High-error timesteps (total)", f"{len(high_pts)} of 187"],
    ]
    recon_table = Table(recon_data, colWidths=[70*mm, 97*mm])
    recon_table.setStyle(TableStyle([
        ("FONTNAME",     (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE",     (0,0), (-1,-1), 9),
        ("TEXTCOLOR",    (0,0), (0,-1),  C_SLATE),
        ("TEXTCOLOR",    (1,0), (1,-1),  C_WHITE),
        ("ROWBACKGROUNDS",(0,0),(-1,-1), [C_BG, C_CARD]),
        ("TOPPADDING",   (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0), (-1,-1), 5),
        ("LEFTPADDING",  (0,0), (-1,-1), 6),
    ]))
    story.append(recon_table)
    story.append(Spacer(1, 8))

    # ── Clinical interpretation ──
    story.append(Paragraph("CLINICAL INTERPRETATION", section_style))
    if is_anomaly:
        findings = []
        if qrs_hits: findings.append("QRS Complex anomaly detected -- possible ventricular arrhythmia")
        if p_hits:   findings.append("P-Wave anomaly detected -- possible atrial abnormality")
        if t_hits:   findings.append("T-Wave anomaly detected -- possible ischemia / repolarization issue")
        if not findings:
            findings.append("Anomaly detected -- diffuse reconstruction error across beat")
        for f in findings:
            story.append(Paragraph(f"[!]  {f}", S("F", fontSize=9, textColor=C_RED,
                                                    fontName="Helvetica-Bold", spaceAfter=3)))
    else:
        story.append(Paragraph(
            "[OK]  All wave segments within normal reconstruction range. "
            "No clinical abnormality indicated.",
            S("OK", fontSize=9, textColor=C_GREEN, fontName="Helvetica-Bold", spaceAfter=3)
        ))

    story.append(Spacer(1, 6))

    # ── Model performance ──
    story.append(Paragraph("MODEL PERFORMANCE  (current threshold)", section_style))
    perf_data = [
        ["Precision", "Recall", "F1 Score", "ROC-AUC"],
        [
            f"{metrics['precision']*100:.2f}%",
            f"{metrics['recall']*100:.2f}%",
            f"{metrics['f1']*100:.2f}%",
            f"{metrics['auc']:.4f}",
        ],
    ]
    perf_table = Table(perf_data, colWidths=[42*mm]*4)
    perf_table.setStyle(TableStyle([
        ("FONTNAME",     (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTNAME",     (0,1), (-1,1),  "Helvetica-Bold"),
        ("FONTSIZE",     (0,0), (-1,0),  7),
        ("FONTSIZE",     (0,1), (-1,1),  12),
        ("TEXTCOLOR",    (0,0), (-1,0),  C_SLATE),
        ("TEXTCOLOR",    (0,1), (-1,1),  C_CYAN),
        ("ALIGN",        (0,0), (-1,-1), "CENTER"),
        ("BACKGROUND",   (0,0), (-1,-1), C_CARD),
        ("BOX",          (0,0), (-1,-1), 0.5, C_SLATE),
        ("INNERGRID",    (0,0), (-1,-1), 0.3, C_SLATE),
        ("TOPPADDING",   (0,0), (-1,-1), 6),
        ("BOTTOMPADDING",(0,0), (-1,-1), 6),
    ]))
    story.append(perf_table)
    story.append(Spacer(1, 8))

    # ── Disclaimer ──
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_SLATE, spaceAfter=6))
    story.append(Paragraph(
        "DISCLAIMER: This report is generated by an automated ML system for research "
        "and educational purposes only. It does not constitute medical advice. "
        "All findings must be reviewed by a qualified clinician before any clinical decision.",
        S("D", fontSize=7, textColor=C_SLATE, fontName="Helvetica", leading=11)
    ))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "PulseWatcher  |  B.Tech Final Year Project  |  May 2026  |  MIT-BIH Arrhythmia Database",
        right_style
    ))

    doc.build(story, onFirstPage=dark_background, onLaterPages=dark_background)
    buf.seek(0)
    return buf.read()


st.set_page_config(
    page_title="PulseWatcher -- ECG Anomaly Detection",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# &#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;
#  GLOBAL CSS  -- Aesthetic: Clinical Noir
#  Fonts: Syne (display) + JetBrains Mono (data)
#  Palette: near-black base, electric cyan accent, danger crimson
# &#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500;700&display=swap');

/* &#9472;&#9472; Reset & Base &#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472; */
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

/* &#9472;&#9472; Sidebar &#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472; */
[data-testid="stSidebar"] {
    background: #080b14 !important;
    border-right: 1px solid #0d1f35;
}
[data-testid="stSidebar"] * { color: #8899aa !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 13px !important; }

/* &#9472;&#9472; Scrollbar &#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472; */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #060810; }
::-webkit-scrollbar-thumb { background: #0d2040; border-radius: 2px; }

/* &#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;
   COMPONENTS
&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552; */

/* &#9472;&#9472; Hero header &#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472; */
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

/* &#9472;&#9472; Status Banner &#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472; */
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

/* &#9472;&#9472; Section label &#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472; */
.pw-section {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; font-weight: 700;
    letter-spacing: 3px; text-transform: uppercase;
    color: #00c8ff;
    padding-bottom: 10px;
    border-bottom: 1px solid #0d1f35;
    margin-bottom: 16px;
}

/* &#9472;&#9472; Metric Card &#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472; */
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

/* &#9472;&#9472; Info / Interpretation card &#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472; */
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

/* &#9472;&#9472; Alert chip &#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472; */
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

/* &#9472;&#9472; Wave legend pills &#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472; */
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

/* &#9472;&#9472; Performance grid &#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472; */
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

/* &#9472;&#9472; Sidebar custom &#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472; */
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

/* &#9472;&#9472; Footer &#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472; */
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


# &#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;
#  DATA LOADING
# &#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;
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
def load_metrics():
    """Load saved evaluation metrics. Falls back to defaults if file missing."""
    import json, os
    defaults = {"threshold_percentile": 82, "precision": 0.8982,
                "recall": 0.7135, "f1": 0.7953, "auc": 0.8678}
    path = "models/metrics.json"
    if not os.path.exists(path):
        return defaults
    with open(path) as fp:
        return json.load(fp)

@st.cache_data
def load_train_errors():
    """Load training errors for anomaly score calculation."""
    import os
    if os.path.exists("models/train_errors.npy"):
        return np.load("models/train_errors.npy")
    return None

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

model        = load_model()
threshold    = load_threshold()
metrics      = load_metrics()
train_errors = load_train_errors()
normal_beats, anomaly_beats = load_beats()


# &#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;
#  SIDEBAR
# &#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;
with st.sidebar:
    st.markdown("""
    <div style='padding: 16px 0 8px 0;'>
        <div style='font-size:18px; font-weight:800; color:#c8d6e8; letter-spacing:-0.3px;'>🫀 PulseWatcher</div>
        <div style='font-family: JetBrains Mono; font-size:9px; color:#1e3a5a; letter-spacing:3px; margin-top:5px;'>
            ECG ANOMALY DETECTION
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-section">UPLOAD YOUR ECG</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload a CSV file (one column, 187 samples)",
        type=["csv"],
        help="Upload your own ECG beat as a CSV. One numeric column, ideally 187 timesteps (will be resampled if not)."
    )

    st.markdown('<div class="sb-section">-- OR -- PRESET BEATS</div>', unsafe_allow_html=True)

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
        <span style='font-size:10px; color:#1e3a5a;'>{metrics.get("threshold_percentile", 95)}th percentile of normal errors</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-section">PERFORMANCE</div>', unsafe_allow_html=True)
    recall_color = "#00ff88" if metrics["recall"] >= 0.60 else "#f59e0b" if metrics["recall"] >= 0.45 else "#ff2244"
    st.markdown(f"""
    <div class="sb-card">
        <strong>Precision</strong>&nbsp;&nbsp;{metrics["precision"]*100:.2f}%<br/>
        <strong>Recall</strong>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span style="color:{recall_color}">{metrics["recall"]*100:.2f}%</span><br/>
        <strong>F1 Score</strong>&nbsp;&nbsp;&nbsp;{metrics["f1"]*100:.2f}%<br/>
        <strong>ROC-AUC</strong>&nbsp;&nbsp;&nbsp;{metrics["auc"]:.4f}<br/>
        <strong>Train Loss</strong>&nbsp;0.001584
    </div>
    """, unsafe_allow_html=True)


# &#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;
#  MAIN -- RUN INFERENCE
# &#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;

def run_inference_on_signal(signal_np):
    """Run model on a raw 1D numpy array (187,). Returns (beat, recon, error)."""
    model_ = load_model()
    tensor = torch.tensor(signal_np[np.newaxis, :, np.newaxis].astype(np.float32))
    with torch.no_grad():
        output = model_(tensor)
    recon = output.numpy()[0, :, 0]
    err   = float(np.mean((recon - signal_np) ** 2))
    return signal_np.tolist(), recon.tolist(), err

def process_uploaded_csv(file):
    """
    Parse uploaded CSV -> normalised 187-point numpy array.
    Handles: with/without header, millivolt scale, arbitrary length (resamples).
    Returns (signal_np, warning_msg or None).
    """
    import pandas as pd
    from scipy.signal import resample

    try:
        df = pd.read_csv(file, header=None)
    except Exception:
        file.seek(0)
        df = pd.read_csv(file)

    # Pick first numeric column
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        # Try dropping header row -- seek back and re-read from the file object
        file.seek(0)
        df = pd.read_csv(file, skiprows=1, header=None)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return None, "No numeric column found in CSV."

    signal = df[numeric_cols[0]].dropna().values.astype(np.float64)

    if len(signal) < 10:
        return None, f"Signal too short ({len(signal)} samples). Need at least 10."

    warn = None

    # Millivolt -> normalised: if range looks like raw mV (>1.0 range), normalise
    sig_range = signal.max() - signal.min()
    if sig_range > 1.0:
        signal = (signal - signal.min()) / (sig_range + 1e-8)
        warn = f"Signal was in raw scale (range {sig_range:.2f}) -- auto-normalised to 0&#8211;1."

    # Resample to 187 if needed
    if len(signal) != 187:
        original_len = len(signal)
        signal = resample(signal, 187)
        msg = f"Signal had {original_len} samples -- resampled to 187."
        warn = (warn + " " + msg) if warn else msg

    signal = signal.astype(np.float32)
    return signal, warn

# &#9472;&#9472; Choose source: upload or preset &#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;&#9472;
upload_source = False
upload_warning = None

if uploaded_file is not None:
    signal_np, upload_warning = process_uploaded_csv(uploaded_file)
    if signal_np is not None:
        upload_source = True
        beat, reconstructed, error = run_inference_on_signal(signal_np)
    else:
        st.error(f"&#9888; Could not parse CSV: {upload_warning}")
        upload_warning = None
        beat, reconstructed, error = get_reconstruction(beat_index, beat_type)
else:
    beat, reconstructed, error = get_reconstruction(beat_index, beat_type)

is_anomaly  = error > threshold
error_pct   = (error / threshold) * 100
delta       = error - threshold
sign        = "+" if delta > 0 else ""
x           = list(range(187))
pointwise   = [(b - r) ** 2 for b, r in zip(beat, reconstructed)]
mean_err    = float(np.mean(pointwise))

# Soft anomaly score 0-100
if train_errors is not None:
    a_score = int(np.mean(train_errors < error) * 100)
else:
    a_score = int(min(error_pct, 100))


# &#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;
#  HERO HEADER
# &#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;
st.markdown("""
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

if upload_source:
    st.markdown('<p style="text-align:right; margin-top:-20px;"><span class="pw-badge" style="color:#f59e0b;border-color:#f59e0b44;background:#1a0d00;">&#128194; UPLOADED SIGNAL</span></p>', unsafe_allow_html=True)

if upload_warning:
    st.markdown(f"""
    <div class="pw-chip pw-chip-warn" style="margin-bottom:16px;">
        &#8505;&nbsp; {upload_warning}
    </div>
    """, unsafe_allow_html=True)


# &#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;
#  STATUS BANNER
# &#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;
if is_anomaly:
    st.markdown(f"""
    <div class="pw-status pw-status-anomaly">
        <div class="pw-status-icon">&#128680;</div>
        <div>
            <div class="pw-status-title-anomaly">ANOMALY DETECTED</div>
            <div class="pw-status-sub">
                Reconstruction error {error:.6f} is {error_pct:.1f}% of threshold --
                beat pattern outside the learned normal distribution
            </div>
        </div>
        <div style="margin-left:auto; text-align:center; flex-shrink:0; padding-left:24px;">
            <div style="font-family:'JetBrains Mono',monospace; font-size:10px; color:#5a7a9a; letter-spacing:2px; margin-bottom:4px;">ANOMALY SCORE</div>
            <div style="font-family:'JetBrains Mono',monospace; font-size:36px; font-weight:700; color:#ff2244; line-height:1;">{a_score}<span style="font-size:16px; color:#5a2a2a;">/100</span></div>
        </div>
        <div class="pw-status-bg-text">ANOMALY</div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="pw-status pw-status-normal">
        <div class="pw-status-icon">&#10003;</div>
        <div>
            <div class="pw-status-title-normal">NORMAL BEAT</div>
            <div class="pw-status-sub">
                Reconstruction error {error:.6f} is {error_pct:.1f}% of threshold --
                beat matches the learned normal pattern
            </div>
        </div>
        <div style="margin-left:auto; text-align:center; flex-shrink:0; padding-left:24px;">
            <div style="font-family:'JetBrains Mono',monospace; font-size:10px; color:#5a7a9a; letter-spacing:2px; margin-bottom:4px;">ANOMALY SCORE</div>
            <div style="font-family:'JetBrains Mono',monospace; font-size:36px; font-weight:700; color:#00ff88; line-height:1;">{a_score}<span style="font-size:16px; color:#1a4a2a;">/100</span></div>
        </div>
        <div class="pw-status-bg-text">NORMAL</div>
    </div>
    """, unsafe_allow_html=True)


# &#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;
#  DIAGNOSTICS ROW  (4 cards)
# &#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;
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
#  LIVE ECG SIMULATION  (Task 4 - Phase 2)
# ═══════════════════════════════════════════════════════════════
st.markdown('<div class="pw-section">&#9654;&nbsp; LIVE ECG SIMULATION</div>', unsafe_allow_html=True)

st.markdown("""
<div class="pw-card" style="margin-bottom:12px; font-size:12px;">
    Streams the selected beat point-by-point, simulating a real-time ECG monitor.
    The anomaly score updates live as each timestep arrives.
</div>
""", unsafe_allow_html=True)

sim_col1, sim_col2 = st.columns([1, 4])
with sim_col1:
    sim_speed = st.select_slider(
        "Speed",
        options=["Slow", "Normal", "Fast"],
        value="Normal",
        help="Controls how fast the ECG streams"
    )
    run_sim = st.button("&#9654; Start Simulation", use_container_width=True,
                        help="Stream the current beat beat-by-beat")

speed_map = {"Slow": 0.04, "Normal": 0.02, "Fast": 0.005}
delay = speed_map[sim_speed]

if run_sim:
    sim_placeholder = st.empty()
    score_placeholder = st.empty()

    streamed_beat = []
    streamed_x    = []

    accent_sim = "#ff2244" if is_anomaly else "#00ff88"

    for i, val in enumerate(beat):
        streamed_beat.append(val)
        streamed_x.append(i)

        # Partial anomaly score: compare partial mean error to threshold
        if i > 0:
            partial_recon  = reconstructed[:i+1]
            partial_beat   = beat[:i+1]
            partial_errors = [(b - r) ** 2 for b, r in zip(partial_beat, partial_recon)]
            partial_error  = float(np.mean(partial_errors))
            if train_errors is not None:
                live_score = int(np.mean(train_errors < partial_error) * 100)
            else:
                live_score = int(min((partial_error / threshold) * 100, 100))
        else:
            live_score = 0

        # Build the live chart
        fig_sim = go.Figure()

        # Shaded region under curve
        fig_sim.add_trace(go.Scatter(
            x=streamed_x,
            y=streamed_beat,
            fill='tozeroy',
            fillcolor='rgba(0,200,255,0.04)',
            line=dict(color='rgba(0,0,0,0)'),
            hoverinfo='skip', showlegend=False
        ))

        # Live ECG line
        fig_sim.add_trace(go.Scatter(
            x=streamed_x,
            y=streamed_beat,
            name="Live ECG",
            line=dict(color="#00c8ff", width=2.5),
            hovertemplate="t=%{x} | amp=%{y:.4f}<extra>Live</extra>"
        ))

        # Scanning cursor dot
        fig_sim.add_trace(go.Scatter(
            x=[i], y=[val],
            mode='markers',
            marker=dict(color="#00c8ff", size=8, symbol="circle",
                        line=dict(color="#ffffff", width=1.5)),
            showlegend=False, hoverinfo='skip'
        ))

        # Wave region shading
        for x0, x1, fc, label, fc2 in [
            (0,   40,  "#0a1628", "P",   "#60a5fa"),
            (60,  100, "#061408", "QRS", "#4ade80"),
            (110, 160, "#160606", "T",   "#f87171"),
        ]:
            fig_sim.add_vrect(x0=x0, x1=x1, fillcolor=fc, opacity=0.4, line_width=0,
                              annotation_text=label, annotation_position="top left",
                              annotation_font=dict(color=fc2, size=9, family="JetBrains Mono"))

        fig_sim.update_layout(
            height=280,
            plot_bgcolor="#04060e",
            paper_bgcolor="#060810",
            font=dict(color="#3a5a7a", family="JetBrains Mono", size=10),
            xaxis=dict(
                title="Timestep", gridcolor="#0a1422", zerolinecolor="#0a1422",
                range=[0, 187],
                title_font=dict(size=10), tickfont=dict(size=9)
            ),
            yaxis=dict(
                title="Amplitude (norm.)", gridcolor="#0a1422", zerolinecolor="#0a1422",
                title_font=dict(size=10), tickfont=dict(size=9)
            ),
            showlegend=False,
            margin=dict(l=55, r=20, t=20, b=50),
        )

        sim_placeholder.plotly_chart(fig_sim, use_container_width=True)

        # Live score bar
        bar_color = "#ff2244" if live_score > 70 else "#f59e0b" if live_score > 40 else "#00ff88"
        score_placeholder.markdown(f"""
        <div style="display:flex; align-items:center; gap:16px; padding:10px 16px;
                    background:#060810; border:1px solid #0d1f35; border-radius:8px;
                    margin-bottom:8px; font-family:'JetBrains Mono', monospace;">
            <span style="color:#3a5a7a; font-size:11px; white-space:nowrap;">ANOMALY SCORE</span>
            <div style="flex:1; background:#0a1422; border-radius:4px; height:8px; overflow:hidden;">
                <div style="width:{live_score}%; height:100%; background:{bar_color};
                            border-radius:4px; transition:width 0.1s;"></div>
            </div>
            <span style="color:{bar_color}; font-size:16px; font-weight:700;
                         min-width:52px; text-align:right;">{live_score}/100</span>
            <span style="font-size:10px; color:#3a5a7a; min-width:60px;">
                t={i+1}/187
            </span>
        </div>
        """, unsafe_allow_html=True)

        time.sleep(delay)

    # Final verdict after stream completes
    verdict_color = "#ff2244" if is_anomaly else "#00ff88"
    verdict_label = "&#9888; ANOMALY DETECTED" if is_anomaly else "&#10003; NORMAL BEAT"
    score_placeholder.markdown(f"""
    <div style="display:flex; align-items:center; gap:16px; padding:12px 20px;
                background:#060810; border:2px solid {verdict_color}44;
                border-radius:8px; margin-bottom:8px;
                font-family:'JetBrains Mono', monospace;
                box-shadow: 0 0 20px {verdict_color}22;">
        <span style="color:#3a5a7a; font-size:11px; white-space:nowrap;">FINAL RESULT</span>
        <div style="flex:1; background:#0a1422; border-radius:4px; height:8px; overflow:hidden;">
            <div style="width:{a_score}%; height:100%; background:{verdict_color}; border-radius:4px;"></div>
        </div>
        <span style="color:{verdict_color}; font-size:16px; font-weight:700;
                     min-width:52px; text-align:right;">{a_score}/100</span>
        <span style="color:{verdict_color}; font-size:13px; font-weight:700;">
            {verdict_label}
        </span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  ECG CHART  (Original vs Reconstructed)
# ═══════════════════════════════════════════════════════════════
chart_label = "UPLOADED ECG SIGNAL -- ORIGINAL vs RECONSTRUCTED" if upload_source else "ECG SIGNAL -- ORIGINAL vs RECONSTRUCTED"
st.markdown(f'<div class="pw-section">{chart_label}</div>', unsafe_allow_html=True)

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
    <span class="pw-pill pw-pill-p">P-WAVE &#183; 0&#8211;40</span>
    <span class="pw-pill pw-pill-qrs">QRS COMPLEX &#183; 60&#8211;100</span>
    <span class="pw-pill pw-pill-t">T-WAVE &#183; 110&#8211;160</span>
</div>
""", unsafe_allow_html=True)


# &#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;
#  EXPLAINABILITY HEATMAP  (Task 5 - Phase 2)
#  Color-coded ECG waveform + per-timestep error bar + clinical interpretation
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="pw-section">EXPLAINABILITY HEATMAP -- WHERE IS THE ANOMALY?</div>',
            unsafe_allow_html=True)

# ── Normalise pointwise errors 0-1 for colour mapping ──
p_min  = min(pointwise)
p_max  = max(pointwise)
p_range = p_max - p_min if p_max != p_min else 1e-9
norm_err = [(e - p_min) / p_range for e in pointwise]   # 0 = low, 1 = high error

def err_to_rgb(n):
    """Map 0-1 error to cyan(low) -> amber(mid) -> crimson(high)."""
    if n < 0.5:
        t = n * 2
        r = int(0   + t * 255)
        g = int(200 + t * (159 - 200))
        b = int(255 + t * (0   - 255))
    else:
        t = (n - 0.5) * 2
        r = int(255)
        g = int(159 + t * (34  - 159))
        b = int(0)
    return f"rgb({r},{g},{b})"

col_heat, col_interp = st.columns([3, 2])

with col_heat:
    fig_heat = go.Figure()

    # ── Heatmap: colour-coded ECG waveform (segment by segment) ──
    for i in range(len(beat) - 1):
        fig_heat.add_trace(go.Scatter(
            x=[x[i], x[i+1]],
            y=[beat[i], beat[i+1]],
            mode="lines",
            line=dict(color=err_to_rgb(norm_err[i]), width=3),
            hovertemplate=f"t={x[i]}<br>amp={beat[i]:.4f}<br>err={pointwise[i]:.6f}<extra></extra>",
            showlegend=False
        ))

    # ── Error bar chart underneath (secondary y-axis) ──
    bar_colors = [err_to_rgb(n) for n in norm_err]
    fig_heat.add_trace(go.Bar(
        x=x, y=pointwise,
        marker_color=bar_colors,
        opacity=0.55,
        yaxis="y2",
        hovertemplate="t=%{x}<br>err=%{y:.6f}<extra>Sq.Error</extra>",
        showlegend=False
    ))

    # Mean error line on bar axis
    fig_heat.add_hline(
        y=mean_err, line_dash="dash", line_color="#f59e0b", line_width=1,
        annotation_text=f"mean {mean_err:.5f}",
        annotation_position="top right",
        annotation_font=dict(color="#f59e0b", size=9, family="JetBrains Mono"),
        yref="y2"
    )

    # Wave region shading
    for x0, x1, fc, label, fc2 in [
        (0,   40,  "#0a1628", "P-wave", "#60a5fa"),
        (60,  100, "#061408", "QRS",    "#4ade80"),
        (110, 160, "#160606", "T-wave", "#f87171"),
    ]:
        fig_heat.add_vrect(x0=x0, x1=x1, fillcolor=fc, opacity=0.45, line_width=0,
                           annotation_text=label, annotation_position="top left",
                           annotation_font=dict(color=fc2, size=9, family="JetBrains Mono"))

    fig_heat.update_layout(
        height=300,
        plot_bgcolor="#04060e", paper_bgcolor="#060810",
        font=dict(color="#3a5a7a", family="JetBrains Mono", size=9),
        xaxis=dict(title="Timestep", gridcolor="#0a1422", zerolinecolor="#0a1422",
                   tickfont=dict(size=8)),
        yaxis=dict(title="ECG Amplitude", gridcolor="#0a1422", zerolinecolor="#0a1422",
                   tickfont=dict(size=8), domain=[0.35, 1.0]),
        yaxis2=dict(title="Sq. Error", gridcolor="#0a1422", zerolinecolor="#0a1422",
                    tickfont=dict(size=8), domain=[0.0, 0.30],
                    showgrid=False),
        showlegend=False,
        margin=dict(l=55, r=20, t=20, b=50),
        bargap=0.02
    )

    # Colour scale legend
    st.plotly_chart(fig_heat, use_container_width=True)
    st.markdown("""
    <div style="display:flex; align-items:center; gap:8px; font-family:'JetBrains Mono',monospace;
                font-size:10px; color:#3a5a7a; margin-top:-8px; margin-bottom:8px;">
        <span>LOW ERROR</span>
        <div style="flex:1; height:6px; border-radius:3px;
                    background:linear-gradient(to right, rgb(0,200,255), rgb(255,159,0), rgb(255,34,0));"></div>
        <span>HIGH ERROR</span>
    </div>
    """, unsafe_allow_html=True)

with col_interp:
    high_pts = [i for i, e in enumerate(pointwise) if e > mean_err]
    p_hits   = [p for p in high_pts if 0   <= p <= 40]
    qrs_hits = [p for p in high_pts if 60  <= p <= 100]
    t_hits   = [p for p in high_pts if 110 <= p <= 160]

    # Top-3 hottest timesteps
    top3 = sorted(range(len(pointwise)), key=lambda i: pointwise[i], reverse=True)[:3]

    st.markdown('<div class="pw-section" style="margin-top:4px;">CLINICAL INTERPRETATION</div>',
                unsafe_allow_html=True)

    if is_anomaly:
        findings = []
        if qrs_hits: findings.append(("&#9889;", "QRS Complex anomaly", "possible ventricular arrhythmia"))
        if p_hits:   findings.append(("&#12336;", "P-Wave anomaly",      "possible atrial abnormality"))
        if t_hits:   findings.append(("&#127754;", "T-Wave anomaly",     "possible ischemia / repolarization"))

        if findings:
            for icon, region, detail in findings:
                st.markdown(f"""
                <div class="pw-chip pw-chip-warn" style="margin-bottom:6px;">
                    {icon}&nbsp; <strong>{region}</strong><br/>
                    <span style="font-size:10px; opacity:0.8;">{detail}</span>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown('<div class="pw-chip pw-chip-warn">&#9888;&nbsp; Anomaly detected -- diffuse error across beat</div>',
                        unsafe_allow_html=True)
    else:
        st.markdown('<div class="pw-chip pw-chip-ok">&#10003;&nbsp; All wave segments within normal range</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class="pw-card" style="margin-top:8px;">
            The model reconstructed this beat with <strong>low error</strong>
            across all cardiac phases. No clinical abnormality indicated.
        </div>
        """, unsafe_allow_html=True)

    # Hottest timesteps callout
    st.markdown(f"""
    <div class="pw-card" style="margin-top:10px; font-size:11px;">
        <strong style="color:#f59e0b;">&#128293; Highest-error timesteps:</strong><br/>
        {"  ".join([f'<span style="color:#ff2244;">t={t}</span>' for t in top3])}<br/><br/>
        <strong>Heatmap key:</strong> cyan = low error &rarr; amber &rarr; crimson = high error.<br/><br/>
        Threshold = <span class="accent">{threshold:.6f}</span>
        &nbsp;({metrics.get("threshold_percentile", 95)}th pct).<br/>
        Trained on <strong>normal beats only</strong>.
    </div>
    """, unsafe_allow_html=True)


# &#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;
#  MODEL PERFORMANCE  (ROC + Confusion Matrix side by side)
# &#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;
st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown('<div class="pw-section">MODEL PERFORMANCE</div>', unsafe_allow_html=True)

# Performance metric row
pm1, pm2, pm3, pm4 = st.columns(4)
perf = [
    (pm1, "Precision",   f"{metrics['precision']*100:.2f}%", "green",  "green"),
    (pm2, "Recall",      f"{metrics['recall']*100:.2f}%",    "amber" if metrics['recall'] < 0.60 else "green",  "amber" if metrics['recall'] < 0.60 else "green"),
    (pm3, "F1 Score",    f"{metrics['f1']*100:.2f}%",        "cyan",   "cyan"),
    (pm4, "ROC-AUC",     f"{metrics['auc']:.4f}",            "green",  "green"),
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


# &#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;
#  METRIC EXPLANATION  (2 cards)
# &#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;
ex1, ex2 = st.columns(2)
with ex1:
    st.markdown(f"""
    <div class="pw-card">
        <strong>Why is Recall {metrics['recall']*100:.0f}%?</strong><br/><br/>
        A deliberate clinical tradeoff. In cardiac screening, a
        <strong>false positive</strong> -- flagging a healthy patient
        as sick -- triggers unnecessary panic and expensive tests.<br/><br/>
        PulseWatcher optimises <strong>precision first</strong>.
        When it raises an alert, it is correct
        <span class="accent">{metrics['precision']*100:.0f}% of the time</span>.
        Recall can be improved by lowering the detection threshold -- a tunable parameter
        depending on clinical context.
    </div>
    """, unsafe_allow_html=True)

with ex2:
    st.markdown(f"""
    <div class="pw-card">
        <strong>What does ROC-AUC {metrics["auc"]:.4f} mean?</strong><br/><br/>
        AUC measures the model's ability to <strong>distinguish
        normal from anomalous beats</strong> across all thresholds.<br/><br/>
        Random classifier &nbsp;-> <span style='color:#3a5a7a;'>0.50</span><br/>
        PulseWatcher &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-> <span class="accent">{metrics["auc"]:.4f}</span><br/>
        Perfect classifier &nbsp;-> <span style='color:#00ff88;'>1.00</span><br/><br/>
        Strong discriminative power -- without ever seeing
        a single anomalous beat during training.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  DOWNLOADABLE PDF REPORT  (Task 6 - Phase 2)
# ═══════════════════════════════════════════════════════════════
st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown('<div class="pw-section">&#11015;&nbsp; DOWNLOADABLE REPORT</div>',
            unsafe_allow_html=True)

col_pdf1, col_pdf2 = st.columns([2, 3])
with col_pdf1:
    beat_label = beat_type if not upload_source else "Uploaded CSV"
    pdf_bytes  = generate_pdf_report(
        beat, reconstructed, error, threshold, is_anomaly,
        a_score, pointwise, metrics, beat_label, upload_source
    )
    fname = f"pulsewatcher_{'anomaly' if is_anomaly else 'normal'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    st.download_button(
        label="&#11015; Download PDF Report",
        data=pdf_bytes,
        file_name=fname,
        mime="application/pdf",
        use_container_width=True,
        help="One-click PDF: beat index, classification, reconstruction error, clinical note"
    )

with col_pdf2:
    verdict_col = "#ff2244" if is_anomaly else "#00ff88"
    st.markdown(f"""
    <div class="pw-card" style="font-size:11px; line-height:1.7;">
        <strong>Report includes:</strong><br/>
        Beat classification &nbsp;&#183;&nbsp;
        Reconstruction error vs threshold &nbsp;&#183;&nbsp;
        Anomaly score (0-100)<br/>
        Top-5 highest-error timesteps &nbsp;&#183;&nbsp;
        Wave-region analysis (P / QRS / T)<br/>
        Clinical interpretation &nbsp;&#183;&nbsp;
        Full model performance metrics<br/><br/>
        Current result:&nbsp;
        <span style="color:{verdict_col}; font-weight:700;">
            {'ANOMALY' if is_anomaly else 'NORMAL'}
        </span>
        &nbsp;&#183;&nbsp; Anomaly score: <span style="color:{verdict_col};">{a_score}/100</span>
    </div>
    """, unsafe_allow_html=True)



# &#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;&#9552;
st.markdown("""
<div class="pw-footer">
    PulseWatcher &nbsp;&#183;&nbsp; MIT-BIH Arrhythmia Database
    &nbsp;&#183;&nbsp; LSTM Autoencoder &nbsp;&#183;&nbsp; Unsupervised Anomaly Detection
    &nbsp;&#183;&nbsp; B.Tech Final Year Project &#183; May 2026
</div>
""", unsafe_allow_html=True)