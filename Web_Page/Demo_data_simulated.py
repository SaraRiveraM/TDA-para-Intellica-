# === Libraries ====

import streamlit as st
import pandas as pd 
import numpy as np 
import requests
from datetime import datetime


# === Read the data ===

data = pd.read_csv("C:/Users/areba/Downloads/Topo2025_prices_USDA.csv")

# === Initial configuration ===
st.set_page_config(
    layout = "wide",
    page_title = "Fruits a Topoligical Analysis for Intellica"
    
)

# === Title and page description ==
st.markdown("<h1 style='font-size: 60px;'> Intelica </h1>", unsafe_allow_html=True)
st.markdown("<h1 style='font-size: 25px;'>🫐 Esta página busca analizar a través del tiempo los cambios significativos en relación al precio y volumen de la zarzamora y de la mora azul por medio del análisis topológoco de características🫐</h1>", unsafe_allow_html=True)

# === Sidebar with instructions ===

with st.sidebar:
    st.image("https://raw.githubusercontent.com/SaraRiveraM/TDA-para-Intellica-/main/Images/images.png")
    st.markdown("<h1 style='font-size: 20px;'>Elija la fruta a analizar y posteriormente elija el tipo de datos a visualizar:</h1>", unsafe_allow_html=True)
    fruta = st.radio("Seleccione una fruta:", ["Zarzamora", "Mora Azul"])

# === TÍTULO PRINCIPAL ===
st.markdown(f"<h1 style='font-size: 40px;'>💲Análisis Topológico: Relación Volumen-Precio - {fruta} 💹</h1>", unsafe_allow_html=True)

# === MAPEO A ARCHIVOS ===
html_files = {
    "Mora Azul": "https://raw.githubusercontent.com/SaraRiveraM/TDA-para-Intellica-/main/TDA_Analysis/mercado_agricola_mapper_mejorado_frambuesa.html",
    # "Zarzamora": "URL_DEL_ARCHIVO_HTML_PARA_ZARZAMORA"  # Añade la URL correcta
}

html_url = html_files.get(fruta)
if html_url:
    try:
        response = requests.get(html_url)
        if response.status_code == 200:
            with st.spinner("Cargando visualización..."):
                st.components.v1.html(response.text, height=800, width=1200, scrolling=True)
        else:
            st.error(f"No se pudo cargar el archivo HTML. Código de error: {response.status_code}")
    except Exception as e:
        st.error(f"Ocurrió un error al cargar el archivo: {str(e)}")
else:
    st.warning("No hay visualización disponible para la fruta seleccionada")

# === Consulta de precio y volumen con base en la fecha y la fruta ===
def consultar_precio_volumen(fecha, fruta):
    if fruta == "Zarzamora":
        df = data[data['commodity'] == 'Blackberries']
    else:
        df = data[data['commodity'] == 'Blueberries']
    
    df['report_date'] = pd.to_datetime(df['date'])
    df = df[df['report_date'] == fecha]
    
    if not df.empty:
        precio_bajo = df['low_price'].values[0]
        precio_alto = df['high_price'].values[0]
        variacion = df['variety'].values[0]
        return precio_bajo, precio_alto, variacion
    else:
        return None, None