import re

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻")

FEATURES = ["Company", "TypeName", "Ram", "Weight", "Touchscreen", "IPS",
            "PPI", "Cpu_brand", "HDD", "SSD", "Gpu_brand", "os"]
CAT_COLS = ["Company", "TypeName", "Cpu_brand", "Gpu_brand", "os"]


def cpu_brand(x):
    x = str(x).lower()
    if "i7" in x:
        return "Intel i7"
    if "i5" in x:
        return "Intel i5"
    if "i3" in x:
        return "Intel i3"
    if "intel" in x:
        return "Other Intel"
    return "AMD"


def os_cat(x):
    x = str(x)
    if "Windows" in x:
        return "Windows"
    if x in ["macOS", "Mac OS X"]:
        return "Mac"
    return "Others"


def parse_storage(mem, kinds):
    """Memory string se storage ki total GB nikalta hai (GB/TB, 1.0TB sab handle)."""
    total = 0.0
    for part in str(mem).split("+"):
        m = re.match(r"\s*([\d.]+)\s*(GB|TB)\s*(.*)", part.strip(), re.I)
        if m and any(k in m.group(3).lower() for k in kinds):
            val = float(m.group(1))
            if m.group(2).upper() == "TB":
                val *= 1000
            total += val
    return int(total)


@st.cache_resource
def train_model():
    df = pd.read_csv("laptop_data.csv")
    df = df.drop(columns=[c for c in df.columns if "Unnamed" in c])

    df["Ram"] = (df["Ram"].astype(str).str.replace("GB", "", regex=False)
                 .str.strip().astype(float).astype(int))
    df["Weight"] = (df["Weight"].astype(str).str.replace("kg", "", regex=False)
                    .str.strip().astype(float))

    sr = df["ScreenResolution"].astype(str)
    df["Touchscreen"] = sr.str.contains("Touchscreen").astype(int)
    df["IPS"] = sr.str.contains("IPS").astype(int)
    res = sr.str.extract(r"(\d+)x(\d+)").astype(float)
    df["PPI"] = (((res[0] ** 2 + res[1] ** 2) ** 0.5) / df["Inches"].astype(float)).round(2)

    df["Cpu_brand"] = df["Cpu"].apply(cpu_brand)

    df["HDD"] = df["Memory"].apply(lambda s: parse_storage(s, ["hdd", "hybrid"]))
    df["SSD"] = df["Memory"].apply(lambda s: parse_storage(s, ["ssd", "flash"]))

    df["Gpu_brand"] = df["Gpu"].astype(str).str.split().str[0]
    df = df[df["Gpu_brand"] != "ARM"]
    df["os"] = df["OpSys"].apply(os_cat)

    df = df.dropna(subset=FEATURES + ["Price"])

    X = df[FEATURES]
    y = np.log(df["Price"])

    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS)],
        remainder="passthrough",
    )
    pipe = Pipeline([("pre", pre), ("model", LinearRegression())])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    metrics = {
        "r2": r2_score(y_test, pred),
        "mae_inr": mean_absolute_error(np.exp(y_test), np.exp(pred)),
        "rows": len(df),
    }
    return pipe, X, metrics


pipe, ref, metrics = train_model()

st.title("💻 Laptop Price Predictor")
st.caption("Linear Regression model | multiple features, target: Price")

company = st.selectbox("Brand", sorted(ref["Company"].unique()))
type_ = st.selectbox("Type", sorted(ref["TypeName"].unique()))
ram = st.selectbox("RAM (GB)", [2, 4, 6, 8, 12, 16, 24, 32, 64], index=3)
weight = st.number_input("Weight (kg)", 0.5, 5.0, 1.5)
touch = st.selectbox("Touchscreen", ["No", "Yes"])
ips = st.selectbox("IPS Display", ["No", "Yes"])
screen_size = st.number_input("Screen size (inches)", 10.0, 18.0, 15.6)
resolution = st.selectbox("Resolution", [
    "1920x1080", "1366x768", "1600x900", "3840x2160", "3200x1800",
    "2880x1800", "2560x1600", "2560x1440", "2304x1440"])
cpu = st.selectbox("CPU", sorted(ref["Cpu_brand"].unique()))
hdd = st.selectbox("HDD (GB)", [0, 128, 256, 500, 1000, 2000])
ssd = st.selectbox("SSD (GB)", [0, 8, 16, 32, 64, 128, 256, 512, 1000], index=6)
gpu = st.selectbox("GPU", sorted(ref["Gpu_brand"].unique()))
os_ = st.selectbox("OS", sorted(ref["os"].unique()))

if st.button("Predict Price"):
    x_res, y_res = map(int, resolution.split("x"))
    ppi = ((x_res ** 2 + y_res ** 2) ** 0.5) / screen_size
    row = pd.DataFrame([{
        "Company": company, "TypeName": type_, "Ram": ram, "Weight": weight,
        "Touchscreen": int(touch == "Yes"), "IPS": int(ips == "Yes"),
        "PPI": ppi, "Cpu_brand": cpu, "HDD": hdd, "SSD": ssd,
        "Gpu_brand": gpu, "os": os_,
    }])
    price = np.exp(pipe.predict(row)[0])
    st.success(f"Estimated Price: ₹{int(price):,}")

with st.expander("Model performance"):
    st.write(f"Training rows: {metrics['rows']}")
    st.write(f"R² score: {metrics['r2']:.3f}")
    st.write(f"MAE: ₹{int(metrics['mae_inr']):,}")
