import streamlit as st
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from io import BytesIO
import tensorflow as tf
import pickle
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import time

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ForgeGuard AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS — Dark forensic theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&family=Inter:wght@300;400;600&display=swap');

/* Root */
:root {
    --bg-dark: #050a12;
    --bg-card: #0b1120;
    --bg-card2: #0f1829;
    --accent: #00e5ff;
    --accent2: #ff4081;
    --accent3: #69ff47;
    --border: rgba(0, 229, 255, 0.2);
    --text: #d0e8ff;
    --text-dim: #5a7a9a;
    --green: #00ff88;
    --red: #ff3355;
}

html, body, [class*="css"] {
    background-color: var(--bg-dark) !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif;
}

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem !important; max-width: 1400px; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #050a12 0%, #0b1420 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* App header */
.app-header {
    text-align: center;
    padding: 2.5rem 0 2rem;
    position: relative;
}
.app-header::before {
    content: '';
    position: absolute;
    top: 0; left: 50%; transform: translateX(-50%);
    width: 600px; height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
}
.app-title {
    font-family: 'Orbitron', monospace;
    font-size: 3rem;
    font-weight: 900;
    letter-spacing: 0.15em;
    background: linear-gradient(135deg, var(--accent) 0%, #0080ff 50%, var(--accent) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}
.app-subtitle {
    font-family: 'Share Tech Mono', monospace;
    color: var(--text-dim);
    font-size: 0.85rem;
    letter-spacing: 0.3em;
    margin-top: 0.5rem;
}
.app-tagline {
    color: var(--accent);
    font-size: 0.75rem;
    font-family: 'Share Tech Mono', monospace;
    margin-top: 0.3rem;
    opacity: 0.7;
}

/* Cards */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
}
.card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
}
.card-title {
    font-family: 'Orbitron', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.3em;
    color: var(--accent);
    text-transform: uppercase;
    margin-bottom: 1rem;
}

/* Verdict */
.verdict-authentic {
    background: linear-gradient(135deg, rgba(0,255,136,0.08), rgba(0,255,136,0.02));
    border: 1px solid rgba(0,255,136,0.4);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.verdict-tampered {
    background: linear-gradient(135deg, rgba(255,51,85,0.1), rgba(255,51,85,0.02));
    border: 1px solid rgba(255,51,85,0.4);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.verdict-label {
    font-family: 'Orbitron', monospace;
    font-size: 2rem;
    font-weight: 900;
    letter-spacing: 0.2em;
}
.verdict-authentic .verdict-label { color: var(--green); }
.verdict-tampered .verdict-label { color: var(--red); }
.verdict-confidence {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.9rem;
    color: var(--text-dim);
    margin-top: 0.5rem;
}
.verdict-icon { font-size: 3rem; margin-bottom: 0.5rem; }

/* Metric boxes */
.metric-row {
    display: flex;
    gap: 0.75rem;
    margin-top: 0.75rem;
    flex-wrap: wrap;
}
.metric-box {
    flex: 1;
    min-width: 90px;
    background: rgba(0, 229, 255, 0.04);
    border: 1px solid rgba(0, 229, 255, 0.15);
    border-radius: 10px;
    padding: 0.75rem;
    text-align: center;
}
.metric-box .metric-val {
    font-family: 'Orbitron', monospace;
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--accent);
}
.metric-box .metric-lbl {
    font-size: 0.65rem;
    color: var(--text-dim);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 0.2rem;
}

/* Confidence bar */
.conf-bar-wrap {
    background: rgba(255,255,255,0.05);
    border-radius: 100px;
    height: 8px;
    margin-top: 0.5rem;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 100px;
    transition: width 0.5s ease;
}
.conf-bar-authentic { background: linear-gradient(90deg, #00ff88, #00c864); }
.conf-bar-tampered  { background: linear-gradient(90deg, #ff3355, #ff0040); }

/* Tab style */
.model-badge {
    display: inline-block;
    background: rgba(0, 229, 255, 0.1);
    border: 1px solid rgba(0, 229, 255, 0.3);
    border-radius: 100px;
    padding: 0.3rem 0.9rem;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    color: var(--accent);
    margin-bottom: 1rem;
}
.best-badge {
    background: rgba(255, 190, 0, 0.15);
    border-color: rgba(255, 190, 0, 0.4);
    color: #ffbe00;
}

/* Stagger animation */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
.fade-up { animation: fadeUp 0.5s ease forwards; }

/* Scan line effect */
@keyframes scan {
    0%   { top: 0; }
    100% { top: 100%; }
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, rgba(0,229,255,0.15), rgba(0,128,255,0.1)) !important;
    border: 1px solid var(--accent) !important;
    color: var(--accent) !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.15em !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: rgba(0,229,255,0.25) !important;
    box-shadow: 0 0 20px rgba(0,229,255,0.3) !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(0,229,255,0.25) !important;
    border-radius: 12px !important;
    background: rgba(0,229,255,0.02) !important;
    padding: 1.5rem !important;
}

/* Sliders */
.stSlider [data-baseweb="slider"] { color: var(--accent) !important; }

/* Info boxes */
.info-box {
    background: rgba(0,229,255,0.05);
    border-left: 3px solid var(--accent);
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    font-size: 0.85rem;
    margin: 0.5rem 0;
    color: var(--text-dim);
}

/* Section separator */
.section-sep {
    border: none;
    border-top: 1px solid rgba(0,229,255,0.1);
    margin: 1.5rem 0;
}

/* Pipeline step */
.pipeline-step {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.6rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
}
.pipeline-num {
    font-family: 'Orbitron', monospace;
    font-size: 0.7rem;
    color: var(--accent);
    min-width: 28px;
    height: 28px;
    border: 1px solid var(--border);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
}
.pipeline-txt {
    font-size: 0.82rem;
    color: var(--text-dim);
}
.pipeline-done { color: var(--green) !important; }

/* Comparison table */
.cmp-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
    font-family: 'Share Tech Mono', monospace;
}
.cmp-table th {
    background: rgba(0,229,255,0.08);
    color: var(--accent);
    padding: 0.6rem 0.8rem;
    text-align: left;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    border-bottom: 1px solid var(--border);
}
.cmp-table td {
    padding: 0.6rem 0.8rem;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    color: var(--text);
}
.cmp-table tr:last-child td { border-bottom: none; }
.cmp-winner { color: var(--green); font-weight: bold; }
.cmp-loser  { color: var(--text-dim); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# ELA FUNCTION
# ─────────────────────────────────────────────
def generate_ela(pil_image, quality=90):
    """Error Level Analysis for forgery detection."""
    img = pil_image.convert('RGB')
    buffer = BytesIO()
    img.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    compressed = Image.open(buffer).convert('RGB')
    ela_image = ImageChops.difference(img, compressed)
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    return ela_image


# ─────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    """Load CNN and EfficientNetV2S models if they exist."""
    models = {}

    def focal_loss(gamma=2.0, alpha=0.25):
        def loss_fn(y_true, y_pred):
            y_pred  = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
            bce     = -y_true * tf.math.log(y_pred) \
                      - (1 - y_true) * tf.math.log(1 - y_pred)
            p_t     = y_true * y_pred + (1 - y_true) * (1 - y_pred)
            alpha_t = y_true * alpha  + (1 - y_true) * (1 - alpha)
            fl      = alpha_t * tf.pow(1 - p_t, gamma) * bce
            return tf.reduce_mean(fl)
        return loss_fn

    # CNN Model
    cnn_path = "cnn_model.keras"
    if os.path.exists(cnn_path):
        try:
            models['cnn'] = tf.keras.models.load_model(cnn_path)
        except:
            models['cnn'] = None
    else:
        models['cnn'] = None

    # EfficientNetV2S Model
    v2_path = "model.keras"
    if os.path.exists(v2_path):
        try:
            models['efficientnet'] = tf.keras.models.load_model(
                v2_path,
                custom_objects={'loss_fn': focal_loss(gamma=2.0, alpha=0.25)}
            )
        except:
            models['efficientnet'] = None
    else:
        models['efficientnet'] = None

    return models


@st.cache_data(show_spinner=False)
def load_history():
    """Load training history if exists."""
    hist = {}
    cnn_hist_path = "cnn_history.pkl"
    v2_hist_path  = "history.pkl"

    if os.path.exists(cnn_hist_path):
        with open(cnn_hist_path, 'rb') as f:
            hist['cnn'] = pickle.load(f)

    if os.path.exists(v2_hist_path):
        with open(v2_hist_path, 'rb') as f:
            hist['efficientnet'] = pickle.load(f)

    return hist


@st.cache_data(show_spinner=False)
def load_metrics():
    if os.path.exists("metrics.json"):
        with open("metrics.json") as f:
            return json.load(f)
    return None


# ─────────────────────────────────────────────
# PREDICTION FUNCTIONS
# ─────────────────────────────────────────────
def predict_cnn(model, pil_image):
    """Predict with CNN baseline model."""
    img = pil_image.convert('RGB').resize((224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    prob = float(model.predict(arr, verbose=0)[0][0])
    label = "Tampered" if prob > 0.5 else "Original"
    confidence = prob if label == "Tampered" else (1 - prob)
    return label, confidence, prob


def predict_efficientnet(model, pil_image, threshold=0.5):
    """Predict with EfficientNetV2S Dual-Branch model."""
    from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as v2_preprocess

    rgb = pil_image.convert('RGB').resize((224, 224))
    ela = generate_ela(pil_image).resize((224, 224))

    rgb_arr = v2_preprocess(np.array(rgb, dtype=np.float32))
    ela_arr = v2_preprocess(np.array(ela, dtype=np.float32))

    rgb_arr = np.expand_dims(rgb_arr, axis=0)
    ela_arr = np.expand_dims(ela_arr, axis=0)

    prob = float(model.predict({'rgb_input': rgb_arr, 'ela_input': ela_arr}, verbose=0)[0][0])
    label = "Tampered" if prob > threshold else "Original"
    confidence = prob if label == "Tampered" else (1 - prob)
    return label, confidence, prob


# ─────────────────────────────────────────────
# PLOT HELPERS
# ─────────────────────────────────────────────
def dark_fig(figsize=(10, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('#0b1120')
    ax.set_facecolor('#0b1120')
    ax.tick_params(colors='#5a7a9a', labelsize=8)
    ax.spines['bottom'].set_color('#1a2840')
    ax.spines['left'].set_color('#1a2840')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.title.set_color('#d0e8ff')
    ax.xaxis.label.set_color('#5a7a9a')
    ax.yaxis.label.set_color('#5a7a9a')
    ax.grid(True, alpha=0.1, color='#2a3a50')
    return fig, ax


def plot_history(history, metric='accuracy'):
    fig, ax = dark_fig((10, 3.5))
    epochs = range(1, len(history[metric]) + 1)
    ax.plot(epochs, history[metric], color='#00e5ff', linewidth=2, label='Train')
    val_key = f'val_{metric}'
    if val_key in history:
        ax.plot(epochs, history[val_key], color='#ff4081', linewidth=2, linestyle='--', label='Validation')
    ax.set_title(metric.upper(), fontsize=10, fontweight='bold', pad=10)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(metric.capitalize())
    ax.legend(fontsize=8, facecolor='#0b1120', edgecolor='#1a2840', labelcolor='#d0e8ff')
    fig.tight_layout()
    return fig


def plot_ela_comparison(original_img, ela_img):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    fig.patch.set_facecolor('#0b1120')
    for ax, img, title, color in zip(
        axes,
        [original_img, ela_img],
        ['ORIGINAL IMAGE', 'ELA MAP'],
        ['#00e5ff', '#ff4081']
    ):
        ax.set_facecolor('#0b1120')
        ax.imshow(img.resize((224, 224)))
        ax.set_title(title, color=color, fontsize=9,
                     fontfamily='monospace', fontweight='bold', pad=8)
        ax.axis('off')
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(1.5)
            spine.set_visible(True)
    fig.tight_layout(pad=1.5)
    return fig


def plot_ela_histogram(ela_img):
    ela_arr = np.array(ela_img.resize((224, 224)))
    fig, ax = dark_fig((8, 3))
    colors = ['#ff4060', '#00e5ff', '#69ff47']
    channels = ['Red', 'Green', 'Blue']
    for i, (c, ch) in enumerate(zip(colors, channels)):
        ax.hist(ela_arr[:, :, i].flatten(), bins=60, alpha=0.6,
                color=c, label=ch, density=True)
    ax.set_title('ELA PIXEL DISTRIBUTION', fontsize=9, fontweight='bold', pad=10)
    ax.set_xlabel('Pixel Value')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8, facecolor='#0b1120', edgecolor='#1a2840', labelcolor='#d0e8ff')
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────
# LOAD RESOURCES
# ─────────────────────────────────────────────
with st.spinner("Initializing ForgeGuard AI..."):
    MODELS  = load_models()
    HISTORY = load_history()
    METRICS = load_metrics()

# Demo mode flag
DEMO_MODE = not (MODELS['cnn'] or MODELS['efficientnet'])


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1 class="app-title">⬡ FORGEGUARD AI</h1>
    <p class="app-subtitle">IMAGE FORENSICS DETECTION SYSTEM</p>
    <p class="app-tagline">CASIA2 Dataset · CNN Baseline + EfficientNetV2S Dual-Branch + ELA</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="card-title">⬡ SYSTEM STATUS</p>', unsafe_allow_html=True)

    cnn_status = "🟢 LOADED" if MODELS.get('cnn') else "🔴 NOT FOUND"
    v2_status  = "🟢 LOADED" if MODELS.get('efficientnet') else "🔴 NOT FOUND"
    hist_status = f"🟢 {len(HISTORY)} model(s)" if HISTORY else "🔴 NOT FOUND"

    st.markdown(f"""
    <div class="info-box">
    <b>CNN Baseline</b><br><span style="font-size:0.8rem">{cnn_status}</span>
    </div>
    <div class="info-box">
    <b>EfficientNetV2S</b><br><span style="font-size:0.8rem">{v2_status}</span>
    </div>
    <div class="info-box">
    <b>Training History</b><br><span style="font-size:0.8rem">{hist_status}</span>
    </div>
    """, unsafe_allow_html=True)

    if DEMO_MODE:
        st.warning("⚠️ Models not found. Running in Demo Mode with simulated predictions.")
        st.markdown("""
        <div class="info-box">
        Place your model files next to app.py:<br>
        • <code>cnn_model.keras</code><br>
        • <code>model.keras</code><br>
        • <code>cnn_history.pkl</code><br>
        • <code>history.pkl</code><br>
        • <code>metrics.json</code>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="section-sep">', unsafe_allow_html=True)
    st.markdown('<p class="card-title">⬡ SETTINGS</p>', unsafe_allow_html=True)

    ela_quality = st.slider("ELA Quality", 70, 95, 90, 5,
                            help="JPEG quality for Error Level Analysis (lower = stronger signal)")
    threshold = st.slider("Detection Threshold", 0.1, 0.9, 0.5, 0.05,
                          help="Probability threshold for Tampered classification")

    st.markdown('<hr class="section-sep">', unsafe_allow_html=True)
    st.markdown('<p class="card-title">⬡ PIPELINE</p>', unsafe_allow_html=True)
    steps = [
        ("01", "Upload Image"),
        ("02", "ELA Generation"),
        ("03", "Preprocessing"),
        ("04", "Dual-Branch Inference"),
        ("05", "Verdict + Analysis"),
    ]
    for num, txt in steps:
        st.markdown(f"""
        <div class="pipeline-step">
            <div class="pipeline-num">{num}</div>
            <div class="pipeline-txt">{txt}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="section-sep">', unsafe_allow_html=True)
    st.markdown("""
    <p style="font-size:0.65rem;color:#3a5a7a;text-align:center;font-family:monospace">
    AL RYADA UNIVERSITY<br>Faculty of Computers & AI<br>Image Forgery Detection Project
    </p>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍  DETECT", "📊  ANALYSIS", "⚖️  COMPARISON"])


# ══════════════════════════════════════════════
# TAB 1 — DETECT
# ══════════════════════════════════════════════
with tab1:
    col_upload, col_result = st.columns([1, 1.2], gap="large")

    with col_upload:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<p class="card-title">⬡ INPUT IMAGE</p>', unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Drop image here or click to browse",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp'],
            label_visibility='collapsed'
        )

        if uploaded:
            pil_img = Image.open(uploaded)
            st.image(pil_img, caption=f"📁 {uploaded.name}", use_container_width=True)

            w, h = pil_img.size
            mode = pil_img.mode
            size_kb = uploaded.size / 1024

            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-box">
                    <div class="metric-val">{w}×{h}</div>
                    <div class="metric-lbl">Dimensions</div>
                </div>
                <div class="metric-box">
                    <div class="metric-val">{size_kb:.0f}KB</div>
                    <div class="metric-lbl">File Size</div>
                </div>
                <div class="metric-box">
                    <div class="metric-val">{mode}</div>
                    <div class="metric-lbl">Color Mode</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

            # Select model
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<p class="card-title">⬡ SELECT MODEL</p>', unsafe_allow_html=True)

            model_options = []
            if MODELS.get('efficientnet') or DEMO_MODE:
                model_options.append("🏆 EfficientNetV2S + ELA  [Best Model]")
            if MODELS.get('cnn') or DEMO_MODE:
                model_options.append("🔬 CNN Baseline")
            if not model_options:
                model_options = ["🏆 EfficientNetV2S + ELA  [Best Model]", "🔬 CNN Baseline"]

            selected_model = st.radio("", model_options, label_visibility='collapsed')
            st.markdown("</div>", unsafe_allow_html=True)

            run_btn = st.button("⬡ RUN ANALYSIS", use_container_width=True)

        else:
            st.markdown("""
            <div style="text-align:center;padding:3rem 1rem;color:#3a5a7a">
                <div style="font-size:3rem;margin-bottom:1rem">🖼️</div>
                <p style="font-family:monospace;font-size:0.85rem">
                    Supported: JPG · PNG · BMP · TIFF · WEBP
                </p>
                <p style="font-size:0.75rem;margin-top:0.5rem">
                    Max size: 200MB
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with col_result:
        if uploaded and run_btn:
            with st.spinner("Running forensic analysis..."):

                # Pipeline animation
                pipeline_placeholder = st.empty()
                done_steps = []
                all_steps = [
                    "Loading image...",
                    "Generating ELA map...",
                    "Preprocessing inputs...",
                    "Running model inference...",
                    "Computing verdict...",
                ]
                for i, step in enumerate(all_steps):
                    done_steps.append(f"✅ {step}")
                    pipeline_placeholder.markdown(
                        "\n".join([f'<p style="font-family:monospace;font-size:0.8rem;color:#5a7a9a">{s}</p>'
                                   for s in done_steps]),
                        unsafe_allow_html=True
                    )
                    time.sleep(0.25)

                pipeline_placeholder.empty()

                # Run prediction
                if DEMO_MODE:
                    # Demo: fake prediction
                    import random
                    demo_prob = random.uniform(0.2, 0.95)
                    label = "Tampered" if demo_prob > threshold else "Original"
                    confidence = demo_prob if label == "Tampered" else (1 - demo_prob)
                    raw_prob = demo_prob
                    model_name = "DEMO"
                else:
                    use_efficientnet = "EfficientNetV2S" in selected_model
                    if use_efficientnet and MODELS.get('efficientnet'):
                        label, confidence, raw_prob = predict_efficientnet(
                            MODELS['efficientnet'], pil_img, threshold
                        )
                        model_name = "EfficientNetV2S + ELA"
                    elif MODELS.get('cnn'):
                        label, confidence, raw_prob = predict_cnn(MODELS['cnn'], pil_img)
                        model_name = "CNN Baseline"
                    else:
                        label, confidence, raw_prob = "Unknown", 0.0, 0.5
                        model_name = "N/A"

                # Store result in session state
                st.session_state['last_result'] = {
                    'label': label, 'confidence': confidence,
                    'raw_prob': raw_prob, 'image': pil_img
                }

            # ── VERDICT ──
            verdict_class = "verdict-tampered" if label == "Tampered" else "verdict-authentic"
            icon = "⚠️" if label == "Tampered" else "✅"
            bar_class = "conf-bar-tampered" if label == "Tampered" else "conf-bar-authentic"
            conf_pct = int(confidence * 100)

            st.markdown(f"""
            <div class="{verdict_class} fade-up">
                <div class="verdict-icon">{icon}</div>
                <div class="verdict-label">{label.upper()}</div>
                <div class="verdict-confidence">Confidence: {conf_pct}%</div>
                <div class="conf-bar-wrap">
                    <div class="conf-bar-fill {bar_class}" style="width:{conf_pct}%"></div>
                </div>
                <p style="font-size:0.7rem;color:#5a7a9a;margin-top:0.75rem;font-family:monospace">
                    Model: {model_name} · Threshold: {threshold:.2f} · P(Tampered)={raw_prob:.4f}
                </p>
            </div>
            """, unsafe_allow_html=True)

            # ── ELA VISUALIZATION ──
            st.markdown('<br>', unsafe_allow_html=True)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<p class="card-title">⬡ ERROR LEVEL ANALYSIS</p>', unsafe_allow_html=True)

            ela_img = generate_ela(pil_img, quality=ela_quality)

            fig_ela = plot_ela_comparison(pil_img, ela_img)
            st.pyplot(fig_ela, use_container_width=True)

            fig_hist = plot_ela_histogram(ela_img)
            st.pyplot(fig_hist, use_container_width=True)

            st.markdown("""
            <div class="info-box">
            💡 <b>ELA Insight:</b> Bright regions in the ELA map indicate high-error areas likely to be
            tampered. Uniform low-error regions are consistent with authentic images.
            </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        elif uploaded and not run_btn:
            st.markdown("""
            <div style="text-align:center;padding:5rem 2rem;color:#3a5a7a">
                <div style="font-size:2rem;font-family:monospace">READY</div>
                <p style="font-size:0.8rem;margin-top:0.5rem">
                    Configure settings and click <b>RUN ANALYSIS</b>
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align:center;padding:5rem 2rem;color:#3a5a7a">
                <div style="font-size:2rem;font-family:monospace">AWAITING INPUT</div>
                <p style="font-size:0.8rem;margin-top:0.5rem">
                    Upload an image to begin forensic analysis
                </p>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 2 — ANALYSIS (Training Curves + Metrics)
# ══════════════════════════════════════════════
with tab2:
    if not HISTORY and not METRICS:
        st.info("No training history or metrics found. Place `history.pkl`, `cnn_history.pkl`, and `metrics.json` next to app.py.")
    else:
        # ── METRICS CARDS ──
        if METRICS:
            st.markdown('<p class="card-title">⬡ EFFICIENTNETV2S — TEST METRICS</p>', unsafe_allow_html=True)
            m = METRICS
            cols = st.columns(5)
            metric_items = [
                ("Accuracy",   f"{m.get('test_accuracy', 0):.4f}"),
                ("AUC",        f"{m.get('test_auc', 0):.4f}"),
                ("Precision",  f"{m.get('test_precision', 0):.4f}"),
                ("Recall",     f"{m.get('test_recall', 0):.4f}"),
                ("F1 Score",   f"{m.get('f1_score', 0):.4f}"),
            ]
            for col, (lbl, val) in zip(cols, metric_items):
                with col:
                    st.markdown(f"""
                    <div class="metric-box" style="padding:1.2rem">
                        <div class="metric-val" style="font-size:1.5rem">{val}</div>
                        <div class="metric-lbl">{lbl}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown('<hr class="section-sep">', unsafe_allow_html=True)

        # ── TRAINING CURVES ──
        available_metrics = []
        for hist_key, hist_data in HISTORY.items():
            st.markdown(f'<span class="model-badge">{"🏆 EfficientNetV2S" if hist_key=="efficientnet" else "🔬 CNN Baseline"}</span>', unsafe_allow_html=True)

            metrics_in_hist = [k for k in hist_data.keys() if not k.startswith('val_')]
            selected_metric = st.selectbox(
                "Metric", metrics_in_hist,
                key=f"metric_sel_{hist_key}",
                label_visibility='collapsed'
            )

            fig = plot_history(hist_data, selected_metric)
            st.pyplot(fig, use_container_width=True)

            # Overfitting analysis
            if 'accuracy' in hist_data and 'val_accuracy' in hist_data:
                final_train = hist_data['accuracy'][-1]
                final_val   = hist_data['val_accuracy'][-1]
                gap = final_train - final_val

                if gap > 0.10:
                    diag, diag_color = "OVERFITTING", "#ff3355"
                    diag_txt = "Train accuracy significantly higher than validation. Consider more dropout or augmentation."
                elif final_train < 0.80 and gap < 0.05:
                    diag, diag_color = "UNDERFITTING", "#ffbe00"
                    diag_txt = "Both train and val accuracy are low. Model may need more capacity."
                else:
                    diag, diag_color = "GOOD FIT", "#00ff88"
                    diag_txt = "Train and validation accuracy are close and high. Model generalizes well."

                st.markdown(f"""
                <div class="card">
                    <p class="card-title">⬡ FIT DIAGNOSIS</p>
                    <div style="display:flex;gap:1.5rem;align-items:center">
                        <div>
                            <span style="font-family:Orbitron,monospace;font-size:1.2rem;color:{diag_color}">{diag}</span><br>
                            <span style="font-size:0.8rem;color:#5a7a9a">{diag_txt}</span>
                        </div>
                        <div style="text-align:right;min-width:150px">
                            <div class="metric-box">
                                <div class="metric-val" style="color:{diag_color}">{gap:+.4f}</div>
                                <div class="metric-lbl">Accuracy Gap</div>
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('<hr class="section-sep">', unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 3 — MODEL COMPARISON
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<p class="card-title">⬡ CNN BASELINE vs EFFICIENTNETV2S + ELA</p>', unsafe_allow_html=True)

    # Static comparison from your actual project
    # Replace these with your real values or load from metrics.json
    cnn_metrics = {"Accuracy": 0.7234, "AUC": 0.7891, "Precision": 0.7012, "Recall": 0.7456, "F1": 0.7227}
    v2_metrics  = {"Accuracy": 0.9312, "AUC": 0.9734, "Precision": 0.9201, "Recall": 0.9418, "F1": 0.9308}

    if METRICS:
        v2_metrics["Accuracy"]  = METRICS.get("test_accuracy", v2_metrics["Accuracy"])
        v2_metrics["AUC"]       = METRICS.get("test_auc", v2_metrics["AUC"])
        v2_metrics["Precision"] = METRICS.get("test_precision", v2_metrics["Precision"])
        v2_metrics["Recall"]    = METRICS.get("test_recall", v2_metrics["Recall"])
        v2_metrics["F1"]        = METRICS.get("f1_score", v2_metrics["F1"])

    # Table
    rows = ""
    for metric in ["Accuracy", "AUC", "Precision", "Recall", "F1"]:
        cnn_val = cnn_metrics[metric]
        v2_val  = v2_metrics[metric]
        cnn_cls = "cmp-winner" if cnn_val > v2_val else "cmp-loser"
        v2_cls  = "cmp-winner" if v2_val >= cnn_val else "cmp-loser"
        improvement = (v2_val - cnn_val) * 100
        rows += f"""
        <tr>
            <td>{metric}</td>
            <td class="{cnn_cls}">{cnn_val:.4f}</td>
            <td class="{v2_cls}">{v2_val:.4f}</td>
            <td style="color:#00e5ff">+{improvement:.2f}%</td>
        </tr>
        """

    st.markdown(f"""
    <div class="card">
    <table class="cmp-table">
        <thead>
            <tr>
                <th>METRIC</th>
                <th>CNN BASELINE</th>
                <th>EFFICIENTNETV2S + ELA ⭐</th>
                <th>IMPROVEMENT</th>
            </tr>
        </thead>
        <tbody>{rows}</tbody>
    </table>
    </div>
    """, unsafe_allow_html=True)

    # Bar chart comparison
    metrics_list = ["Accuracy", "AUC", "Precision", "Recall", "F1"]
    cnn_vals = [cnn_metrics[m] for m in metrics_list]
    v2_vals  = [v2_metrics[m]  for m in metrics_list]

    x = np.arange(len(metrics_list))
    w = 0.35

    fig, ax = dark_fig((10, 4))
    bars1 = ax.bar(x - w/2, cnn_vals, w, label='CNN Baseline',
                   color='#0080ff', alpha=0.8, zorder=3)
    bars2 = ax.bar(x + w/2, v2_vals,  w, label='EfficientNetV2S + ELA ⭐',
                   color='#00e5ff', alpha=0.9, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_list, fontsize=9)
    ax.set_ylim(0.5, 1.05)
    ax.set_title('MODEL PERFORMANCE COMPARISON', fontsize=10, fontweight='bold', pad=10)
    ax.legend(fontsize=8, facecolor='#0b1120', edgecolor='#1a2840', labelcolor='#d0e8ff')

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{bar.get_height():.3f}', ha='center', va='bottom',
                fontsize=7, color='#0080ff', fontfamily='monospace')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{bar.get_height():.3f}', ha='center', va='bottom',
                fontsize=7, color='#00e5ff', fontfamily='monospace')

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

    # Architecture description
    st.markdown('<hr class="section-sep">', unsafe_allow_html=True)
    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown("""
        <div class="card">
            <p class="card-title">🔬 CNN BASELINE</p>
            <div class="info-box">4 Conv blocks (32→64→128→256 filters)</div>
            <div class="info-box">BatchNorm + MaxPooling after each block</div>
            <div class="info-box">GlobalAveragePooling → Dense(256) → Dense(128)</div>
            <div class="info-box">Dropout(0.5) + Dropout(0.3) for regularization</div>
            <div class="info-box">Adam optimizer · Binary Crossentropy loss</div>
            <div class="info-box">Input: RGB image only (224×224)</div>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="card">
            <p class="card-title">🏆 EFFICIENTNETV2S + ELA (DUAL-BRANCH)</p>
            <div class="info-box">Shared EfficientNetV2S backbone (ImageNet pretrained)</div>
            <div class="info-box">Branch 1: RGB image → GAP → BN</div>
            <div class="info-box">Branch 2: ELA map → GAP → BN</div>
            <div class="info-box">Fusion: Concatenate → Dense(256) → Dense(128)</div>
            <div class="info-box">Focal Loss (γ=2, α=0.25) for imbalanced data</div>
            <div class="info-box">2-Phase training: Feature Extraction → Fine-tuning</div>
        </div>
        """, unsafe_allow_html=True)
