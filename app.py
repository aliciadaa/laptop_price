import streamlit as st
import pandas as pd
import numpy as np
import joblib

artifact = joblib.load("laptop_price_model.pkl")
model = artifact["model"]
scaler = artifact["scaler"]
feature_columns = artifact["feature_columns"]
cpu_rank_mapping = artifact["cpu_rank_mapping"]
storage_rank_mapping = artifact["storage_rank_mapping"]
ohe_cat_cols = artifact["ohe_cat_cols"]
final_num_cols = artifact["final_num_cols"]

st.title("Laptop Price Predictor")

ram = st.number_input("RAM (GB)", 2, 64, 8)
res_w = st.number_input("Resolusi Width (px)", 800, 4000, 1920)
cpu_ghz = st.number_input("CPU GHz", 1.0, 5.0, 2.5)
cpu_series = st.selectbox("CPU Series", list(cpu_rank_mapping.keys()))
memory_type = st.selectbox("Storage Type", list(storage_rank_mapping.keys()))
typename = st.selectbox("Type", ["Notebook", "Ultrabook", "Gaming", "Netbook", "Workstation", "2 in 1 Convertible"])
gpu_brand = st.selectbox("GPU Brand", ["Intel", "Nvidia", "AMD"])
opsys = st.selectbox("OS", ["Windows", "macOS", "Linux", "No OS", "Chrome OS"])
cpu_brand = st.selectbox("CPU Brand", ["Intel", "AMD", "Samsung"])
company = st.selectbox("Company", ["Dell", "HP", "Lenovo", "Asus", "Apple", "Acer", "MSI"])

if st.button("Predict Price"):
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

    st.success(f"Estimasi harga: €{pred_price:,.2f}")
