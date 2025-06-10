import streamlit as st
import pandas as pd
import gdown
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

st.title("Modeling & Evaluasi - Status Gizi Balita")

# --- Load Data ---
@st.cache_data
def load_data():
    file_path = "data/data_balita.csv"
    if not os.path.exists(file_path):
        url = "https://drive.google.com/uc?id=1lo2WUzf-fsgRj7v8DAdr4yOjXgX9vqd0"
        os.makedirs("data", exist_ok=True)
        gdown.download(url, file_path, quiet=False)
    return pd.read_csv(file_path)

df = load_data()

# --- Pra-pemrosesan ---
st.subheader("Pre Processing Data")

# Encode fitur kategorik
df_encoded = df.copy()
le_gender = LabelEncoder()
df_encoded["Jenis Kelamin"] = le_gender.fit_transform(df["Jenis Kelamin"])
le_status = LabelEncoder()
df_encoded["Status Gizi"] = le_status.fit_transform(df["Status Gizi"])

X = df_encoded[["Umur (bulan)", "Tinggi Badan (cm)", "Jenis Kelamin"]]
y = df_encoded["Status Gizi"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.write("Jumlah data latih:", X_train.shape[0])
st.write("Jumlah data uji:", X_test.shape[0])

# --- Pelatihan Model ---
st.subheader("Pelatihan Model: Random Forest")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Evaluasi ---
y_pred = model.predict(X_test)

st.markdown("Akurasi & Laporan Klasifikasi")
report = classification_report(y_test, y_pred, target_names=le_status.classes_, output_dict=True)
df_report = pd.DataFrame(report).transpose()
st.dataframe(df_report.style.highlight_max(axis=0))

# --- Confusion Matrix ---
st.markdown("Confusion Matrix")
fig, ax = plt.subplots()
disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=le_status.classes_)
disp.plot(ax=ax, cmap="Blues")
st.pyplot(fig)
