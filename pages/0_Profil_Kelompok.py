import streamlit as st
import pandas as pd

st.title("Profil Kelompok Data Mining")

st.markdown("""
Informasi anggota kelompok yang mengerjakan proyek *Dashboard Prediksi Status Gizi Balita*.
""")

# Data anggota kelompok
data_kelompok = [
    {"Nama": "Nisrina Khoridatun Nabilah", "NIM": "4101422092"},
    {"Nama": "Shella Fajar Cahyani", "NIM": "4101422065"},
    {"Nama": "Enik Duwi Nur Cahyanti", "NIM": "4101422076"},
    {"Nama": "Karennina Metta Kurniawan", "NIM": "2304030029"},
]

df_kelompok = pd.DataFrame(data_kelompok)

# Tampilkan sebagai tabel
st.subheader("Daftar Anggota Kelompok")
st.table(df_kelompok)

# Tampilkan sebagai kartu (opsional, bisa dihapus jika cukup tabel)
st.subheader("Profil Singkat")

for anggota in data_kelompok:
    with st.container():
        st.markdown(f"*{anggota['Nama']}*  \nNIM: {anggota['NIM']}")
        st.markdown("---")
