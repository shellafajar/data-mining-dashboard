import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import gdown

# Judul
st.title("EDA - Status Gizi Balita")

# Unduh dataset dari Google Drive jika belum ada
@st.cache_data
def load_data():
    file_path = "data/data_balita.csv"
    if not os.path.exists(file_path):
        url = "https://drive.google.com/uc?id=1lo2WUzf-fsgRj7v8DAdr4yOjXgX9vqd0"
        os.makedirs("data", exist_ok=True)
        gdown.download(url, file_path, quiet=False)
    return pd.read_csv(file_path)

df = load_data()

# Ringkasan data
st.subheader("Ringkasan Dataset")
st.write(df.shape)
st.dataframe(df.head())

# Distribusi Status Gizi
st.subheader("Distribusi Status Gizi")
st.bar_chart(df["Status Gizi"].value_counts())

# Distribusi umur
st.subheader("Distribusi Umur (bulan)")
fig1, ax1 = plt.subplots()
sns.histplot(df["Umur (bulan)"], kde=True, ax=ax1)
st.pyplot(fig1)

# Distribusi tinggi badan
st.subheader("Distribusi Tinggi Badan (cm)")
fig2, ax2 = plt.subplots()
sns.histplot(df["Tinggi Badan (cm)"], kde=True, color="green", ax=ax2)
st.pyplot(fig2)

# Boxplot tinggi badan per status gizi
st.subheader("Tinggi Badan per Status Gizi")
fig3, ax3 = plt.subplots()
sns.boxplot(data=df, x="Status Gizi", y="Tinggi Badan (cm)", palette="Set2", ax=ax3)
plt.xticks(rotation=15)
st.pyplot(fig3)
