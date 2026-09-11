import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="centered",
)

# ----------------------------------------------------------------------------
# Custom CSS (background, colors, fonts, cards)
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #1f2937 0%, #111827 50%, #0f172a 100%);
        color: #f1f5f9;
    }

    /* Title */
    .app-title {
        text-align: center;
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .app-subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 1.8rem;
    }

    /* Card container for inputs */
    .input-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.4rem 1.6rem 0.6rem 1.6rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    .section-header {
        font-size: 1.05rem;
        font-weight: 700;
        color: #93c5fd;
        margin-bottom: 0.6rem;
        border-left: 4px solid #60a5fa;
        padding-left: 0.6rem;
    }

    /* Labels */
    label, .stSelectbox label, .stNumberInput label {
        color: #e2e8f0 !important;
        font-weight: 600 !important;
    }

    /* Predict button */
    div.stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        color: white;
        font-weight: 700;
        font-size: 1.1rem;
        padding: 0.7rem 0;
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.45);
        transition: transform 0.15s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(99, 102, 241, 0.6);
    }

    /* Result box */
    .price-box {
        text-align: center;
        background: linear-gradient(135deg, #059669, #10b981);
        border-radius: 18px;
        padding: 1.8rem 1rem;
        margin-top: 1.5rem;
        box-shadow: 0 8px 26px rgba(16, 185, 129, 0.35);
    }
    .price-label {
        color: #d1fae5;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    .price-value {
        color: white;
        font-size: 3.2rem;
        font-weight: 900;
        letter-spacing: 1px;
    }
    .price-note {
        color: #d1fae5;
        font-size: 0.85rem;
        margin-top: 0.4rem;
    }

    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Load model artifact
# ----------------------------------------------------------------------------
artifact = joblib.load("laptop_price_model.pkl")
model = artifact["model"]
scaler = artifact["scaler"]
feature_columns = artifact["feature_columns"]
cpu_rank_mapping = artifact["cpu_rank_mapping"]
storage_rank_mapping = artifact["storage_rank_mapping"]
ohe_cat_cols = artifact["ohe_cat_cols"]
final_num_cols = artifact["final_num_cols"]

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown('<div class="app-title">💻 Laptop Price Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Isi spesifikasi laptop di bawah ini, lalu klik '
    '<b>Predict Price</b> untuk melihat estimasi harganya.</div>',
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Dropdown options berdasarkan nilai unik pada data training
# ----------------------------------------------------------------------------
RAM_OPTIONS = [2, 4, 6, 8, 12, 16, 24, 32, 64]
RES_W_OPTIONS = [1366, 1440, 1600, 1920, 2160, 2256, 2304, 2560, 2736, 2880, 3200, 3840]
CPU_GHZ_OPTIONS = sorted([
    0.9, 1.0, 1.1, 1.2, 1.3, 1.44, 1.5, 1.6, 1.8, 1.9, 1.92,
    2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9,
    3.0, 3.1, 3.2, 3.6,
])

# ----------------------------------------------------------------------------
# Input: Spesifikasi Utama
# ----------------------------------------------------------------------------
st.markdown('<div class="input-card">', unsafe_allow_html=True)
st.markdown('<div class="section-header">🔧 Spesifikasi Utama</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    ram = st.selectbox("RAM (GB)", RAM_OPTIONS, index=RAM_OPTIONS.index(8))
with col2:
    res_w = st.selectbox("Resolusi Width (px)", RES_W_OPTIONS, index=RES_W_OPTIONS.index(1920))
with col3:
    cpu_ghz = st.selectbox("CPU GHz", CPU_GHZ_OPTIONS, index=CPU_GHZ_OPTIONS.index(2.5))

col4, col5 = st.columns(2)
with col4:
    cpu_series = st.selectbox("CPU Series", list(cpu_rank_mapping.keys()))
with col5:
    memory_type = st.selectbox("Storage Type", list(storage_rank_mapping.keys()))

st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Input: Tipe & Brand
# ----------------------------------------------------------------------------
st.markdown('<div class="input-card">', unsafe_allow_html=True)
st.markdown('<div class="section-header">🏷️ Tipe & Brand</div>', unsafe_allow_html=True)

col6, col7 = st.columns(2)
with col6:
    typename = st.selectbox(
        "Type",
        ["Notebook", "Ultrabook", "Gaming", "Netbook", "Workstation", "2 in 1 Convertible"],
    )
with col7:
    company = st.selectbox(
        "Company", ["Dell", "HP", "Lenovo", "Asus", "Apple", "Acer", "MSI"]
    )

col8, col9, col10 = st.columns(3)
with col8:
    gpu_brand = st.selectbox("GPU Brand", ["Intel", "Nvidia", "AMD"])
with col9:
    cpu_brand = st.selectbox("CPU Brand", ["Intel", "AMD", "Samsung"])
with col10:
    opsys = st.selectbox("OS", ["Windows", "macOS", "Linux", "No OS", "Chrome OS"])

st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Predict
# ----------------------------------------------------------------------------
if st.button("🔮 Predict Price"):
    row = {col: 0 for col in feature_columns}
    row["Ram_GB"] = ram
    row["Res_W"] = res_w
    row["Cpu_GHz"] = cpu_ghz
    row["Cpu_Series_Rank"] = cpu_rank_mapping[cpu_series]
    row["Memory_Type_Rank"] = storage_rank_mapping[memory_type]

    for col in feature_columns:
        if col == f"TypeName_{typename}":
            row[col] = 1
        if col == f"Gpu_Brand_{gpu_brand}":
            row[col] = 1
        if col == f"OpSys_Mapping_{opsys}":
            row[col] = 1
        if col == f"Cpu_Brand_{cpu_brand}":
            row[col] = 1
        if col == f"Company_{company}":
            row[col] = 1

    X_input = pd.DataFrame([row])[feature_columns]
    X_input_scaled = scaler.transform(X_input)

    pred_log = model.predict(X_input_scaled)
    pred_price = np.expm1(pred_log)[0]

    st.markdown(
        f"""
        <div class="price-box">
            <div class="price-label">💰 Estimasi Harga Laptop</div>
            <div class="price-value">€{pred_price:,.2f}</div>
            <div class="price-note">Berdasarkan spesifikasi yang kamu pilih</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
