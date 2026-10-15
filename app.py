import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻")

FEATURES = ["Company", "TypeName", "Ram", "Weight", "Touchscreen", "IPS",
            "PPI", "Cpu_brand", "HDD", "SSD", "Gpu_brand", "os"]
CAT_COLS = ["Company", "TypeName", "Cpu_brand", "Gpu_brand", "os"]


def cpu_brand(x):
    x = x.lower()
    if "i7" in x: return "Intel i7"
    if "i5" in x: return "Intel i5"
    if "i3" in x: return "Intel i3"
    if "intel" in x: return "Other Intel"
    return "AMD"


def os_cat(x):
    if "Windows" in x: return "Windows"
    if x in ["macOS", "Mac OS X"]: return "Mac"
    return "Others"


@st.cache_resource
def train_model():
    df = pd.read_csv("laptop_data.csv")
    df = df.drop(columns=[c for c in df.columns if "Unnamed" in c])

    df["Ram"] = df["Ram"].str.replace("GB", "").astype(int)
    df["Weight"] = df["Weight"].str.replace("kg", "").astype(float)

    df["Touchscreen"] = df["ScreenResolution"].str.contains("Touchscreen").astype(int)
    df["IPS"] = df["ScreenResolution"].str.contains("IPS").astype(int)
    res = df["ScreenResolution"].str.extract(r"(\d+)x(\d+)").astype(int)
    df["PPI"] = (((res[0] ** 2 + res[1] ** 2) ** 0.5) / df["Inches"]).round(2)

    df["Cpu_brand"] = df["Cpu"].apply(cpu_brand)

    mem = df["Memory"].str.replace("GB", "").str.replace("TB", "000")
    df["HDD"] = mem.apply(lambda s: sum(int(p.split()[0]) for p in s.split("+") if "HDD" in p))
    df["SSD"] = mem.apply(lambda s: sum(int(p.split()[0]) for p in s.split("+") if "SSD" in p))

    df["Gpu_brand"] = df["Gpu"].str.split().str[0]
    df = df[df["Gpu_brand"] != "ARM"]
    df["os"] = df["OpSys"].apply(os_cat)

    X = df[FEATURES]
    y = np.log(df["Price"])

    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), CAT_COLS)],
        remainder="passthrough",
    )
    pipe = Pipeline([("pre", pre), ("model", LinearRegression())])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    metrics = {"r2": r2_score(y_test, pred), "mae": mean_absolute_error(y_test, pred)}
    return pipe, X, metrics


pipe, ref, metrics = train_model()

st.title("💻 Laptop Price Predictor")
st.caption("Linear Regression model | features multiple, target: Price")

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
hdd = st.selectbox("HDD (GB)", [0, 128, 256, 512, 1000, 2000])
ssd = st.selectbox("SSD (GB)", [0, 8, 128, 256, 512, 1000], index=3)
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
    st.write(f"R² score: {metrics['r2']:.3f}")
    st.write(f"MAE (log scale): {metrics['mae']:.3f}")
