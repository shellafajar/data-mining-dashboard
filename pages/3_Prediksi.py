import streamlit as st
import pandas as pd
import gdown
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

st.title("Prediksi Status Gizi Balita")

# --- Load Dataset ---
@st.cache_data
def load_data():
    file_path = "data/data_balita.csv"
    if not os.path.exists(file_path):
        url = "https://drive.google.com/uc?id=1lo2WUzf-fsgRj7v8DAdr4yOjXgX9vqd0"
        os.makedirs("data", exist_ok=True)
        gdown.download(url, file_path, quiet=False)
    return pd.read_csv(file_path)

df = load_data()

# Normalisasi nilai Jenis Kelamin agar konsisten
df["Jenis Kelamin"] = df["Jenis Kelamin"].str.strip().str.lower()
jenis_kelamin_radio = st.radio("Jenis Kelamin", ("Laki-laki", "Perempuan"))
jk_input = jenis_kelamin_radio.strip().lower()

# --- Preprocessing ---
le_gender = LabelEncoder()
df["Jenis Kelamin"] = le_gender.fit_transform(df["Jenis Kelamin"])

le_status = LabelEncoder()
df["Status Gizi"] = le_status.fit_transform(df["Status Gizi"])

X = df[["Umur (bulan)", "Tinggi Badan (cm)", "Jenis Kelamin"]]
y = df["Status Gizi"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

st.subheader("Masukkan Data Balita:")
umur = st.slider("Umur (bulan)", 0, 60, 24)
tinggi = st.number_input("Tinggi Badan (cm)", min_value=30.0, max_value=130.0, value=75.0)

# Konversi input
try:
    jk_encoded = le_gender.transform([jk_input])[0]
    data_input = pd.DataFrame({
        "Umur (bulan)": [umur],
        "Tinggi Badan (cm)": [tinggi],
        "Jenis Kelamin": [jk_encoded]
    })

    if st.button("Prediksi"):
        pred = model.predict(data_input)[0]
        hasil = le_status.inverse_transform([pred])[0]
        st.success(f"Prediksi Status Gizi: *{hasil}*")
except ValueError as e:
    st.error("Gagal mengenali nilai jenis kelamin. Pastikan nilainya sesuai dengan data.")
