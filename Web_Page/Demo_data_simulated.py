# === Libraries ====

import streamlit as st
import pandas as pd 
import numpy as np 


# === Read the data ===


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
    st.image("https://raw.githubusercontent.com/SaraRiveraM/TDA-para-Intellica-/Sara/Images/intelica-open-graph.jpg")  # Imagen en formato RAW
    st.markdown("<h1 style='font-size: 20px;'>Elija la fruta a analizar y posteriormente elija el tipo de datos a visualizar:</h1>", unsafe_allow_html=True)
    st.write("") 

    # Selector de fruta
    fruta = st.radio(
        "Seleccione una fruta:",
        ["Aguacate", "Chile", "Fresa", "Frambuesa"]
    )

