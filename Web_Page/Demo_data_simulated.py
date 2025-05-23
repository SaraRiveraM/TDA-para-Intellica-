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
st.markdown("<h1 style='font-size: 60px;'>Relación Precio-Volumen de Frutitas</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='font-size: 25px;'>Analizar a través del tiempo los cambios significativos en relación al precio y volumen de distintas frutas</h1>", unsafe_allow_html=True)

# === Sidebar with instructions ===

with st.sidebar:
    # st.image("C:/Users/52452/Downloads/PISA.png")  # Cambia esto por la ubicación correcta de tu imagen
    st.markdown("<h1 style='font-size: 20px;'>Instrucciones de uso de la calculadora:</h1>", unsafe_allow_html=True)
    st.write("Para predecir cuál es el estado estable y la vida promedio de un cliente o producto se ingresa el ID de alguno de estos.") 
