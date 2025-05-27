# === Libraries ====

import streamlit as st
import pandas as pd 
import numpy as np 


# === Read the data ===
data = pd.read_csv("C:\Users\areba\Downloads\Topo2025_prices_USDA.csv")

# === Initial configuration ===
st.set_page_config(
    layout = "wide",
    page_title = "Fruits a Topoligical Analysis for Intellica"
    
)

# === Title and page description ==
st.markdown("<h1 style='font-size: 60px;'>🍓 Relación Precio-Volumen de Frutitas 🍓</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='font-size: 25px;'>Analizar a través del tiempo los cambios significativos en relación al precio y volumen de distintas frutas</h1>", unsafe_allow_html=True)

# === Sidebar with instructions ===

with st.sidebar:
    st.image("https://raw.githubusercontent.com/SaraRiveraM/TDA-para-Intellica-/Sara/Images/intelica-open-graph.jpg")
    st.markdown("<h1 style='font-size: 20px;'>Elija la fruta a analizar y posteriormente elija el tipo de datos a visualizar:</h1>", unsafe_allow_html=True)
    fruta = st.radio("Seleccione una fruta:", ["Aguacate", "Chile", "Fresa", "Frambuesa"])

# === TÍTULO PRINCIPAL ===
st.markdown(f"<h1 style='font-size: 50px;'>🍓 Análisis Topológico: Relación Volumen-Precio - {fruta}</h1>", unsafe_allow_html=True)

# --- MAPEO A ARCHIVOS ---
archivos = {
    #"Aguacate": "mapper_aguacate.html",
    #"Chile": "mapper_chile.html",
    #"Fresa": "mapper_fresa.html",
    "Frambuesa": "https://raw.githubusercontent.com/SaraRiveraM/TDA-para-Intellica-/blob/Sara/Web_Page/TDA_Analysis/mercado_agricola_mapper_mejorado_frambuesa.html"
}

archivo_html = archivos.get(fruta)
if archivo_html:
    with open(archivo_html, "r") as file:
        html_content = file.read()
    st.components.v1.html(html_content, height=800, width=1200)

