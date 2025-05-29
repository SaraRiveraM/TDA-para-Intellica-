# === Libraries ===
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# === Initial configuration ===
st.set_page_config(
    layout="wide",
    page_title="Fruits: a Topological Analysis for Intelica"
)

# === Read the data ===
@st.cache_data
def load_data():
    return pd.read_csv("C:/Users/52452/Downloads/Topo2025_prices_USDA.csv")

data = load_data()

# === Title and page description ===
st.markdown("<h1 style='font-size: 60px;'> Intelica </h1>", unsafe_allow_html=True)
st.markdown("<h1 style='font-size: 25px;'>🫐 Esta página busca analizar a través del tiempo los cambios significativos en relación al precio y volumen de la zarzamora y de la mora azul por medio del análisis topológico de características 🫐</h1>", unsafe_allow_html=True)

# === Sidebar with instructions ===
with st.sidebar:
    st.image("https://raw.githubusercontent.com/SaraRiveraM/TDA-para-Intellica-/main/Images/images.png")
    st.markdown("<h1 style='font-size: 20px;'>Elija la fruta a analizar:</h1>", unsafe_allow_html=True)
    fruta = st.radio("Seleccione una fruta:", ["Zarzamora", "Mora Azul"])

# === TÍTULO PRINCIPAL ===
st.markdown(f"<h1 style='font-size: 40px;'>💲 Análisis Topológico: Relación Volumen-Precio - {fruta} 💹</h1>", unsafe_allow_html=True)

# === Selector de fecha y precios ===
st.markdown("---")  # Línea separadora
st.subheader("🔍 Consulta histórica de precios")

# Convertir la columna de fecha a datetime
data['report_date'] = pd.to_datetime(data['report_date'])

# Obtener fechas únicas
fechas_disponibles = data['report_date'].dt.date.unique()

# Selector de fecha
fecha_seleccionada = st.selectbox(
    "Seleccione una fecha:",
    options=sorted(fechas_disponibles, reverse=True),
    index=0
)

# Consultar datos
if fruta == "Zarzamora":
    df_filtrado = data[(data['commodity'] == 'Blackberries') & 
                      (data['report_date'].dt.date == fecha_seleccionada)]
else:
    df_filtrado = data[(data['commodity'] == 'Blueberries') & 
                      (data['report_date'].dt.date == fecha_seleccionada)]

# Mostrar resultados
if not df_filtrado.empty:
    st.success("📊 Datos encontrados:")
    
    # Crear columnas para mejor presentación
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Precio mínimo", f"${df_filtrado['low_price'].values[0]:.2f}")
    
    with col2:
        st.metric("Precio máximo", f"${df_filtrado['high_price'].values[0]:.2f}")
    
    with col3:
        st.metric("Volumen", f"{df_filtrado['volume'].values[0]:,}")
    
    # Mostrar variedad si existe
    if 'variety' in df_filtrado.columns:
        st.write(f"**Variedad:** {df_filtrado['variety'].values[0]}")
else:
    st.warning("No se encontraron datos para la fecha seleccionada")

# === Visualización HTML (Comentada) ===
st.markdown("---")
st.subheader("📊 Visualización Topológica de Datos")

# Comentado hasta resolver el problema de carga del HTML
"""
# Mapeo a archivos HTML (requiere configuración adecuada)
html_files = {
    "Mora Azul": "URL_DEL_ARCHIVO_HTML",
    "Zarzamora": "URL_DEL_ARCHIVO_HTML"
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
"""

# === Visualización de series temporales ===
st.markdown("---")
st.subheader("📈 Evolución histórica de precios")

if fruta == "Zarzamora":
    df_historico = data[data['commodity'] == 'Blackberries']
else:
    df_historico = data[data['commodity'] == 'Blueberries']

if not df_historico.empty:
    df_historico = df_historico.sort_values('report_date')
    
    # Gráfico de líneas para precios
    st.line_chart(df_historico.set_index('report_date')[['low_price', 'high_price']])
    
    # Gráfico de barras para volumen
    st.bar_chart(df_historico.set_index('report_date')['volume'])
else:
    st.warning("No hay datos históricos disponibles")

# === Análisis estacional ===
st.markdown("---")
st.subheader("📅 Análisis estacional")

if not df_historico.empty:
    df_historico['month'] = df_historico['report_date'].dt.month_name()
    monthly_avg = df_historico.groupby('month')[['low_price', 'high_price']].mean().reindex([
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ])
    
    st.write("Promedio mensual de precios:")
    st.bar_chart(monthly_avg)
    
    # Estadísticas adicionales
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Precio máximo histórico", f"${monthly_avg['high_price'].max():.2f}")
        st.metric("Mes con mayor precio", monthly_avg['high_price'].idxmax())
    with col2:
        st.metric("Precio mínimo histórico", f"${monthly_avg['low_price'].min():.2f}")
        st.metric("Mes con menor precio", monthly_avg['low_price'].idxmin())
    
    st.write("""
    ### Interpretación:
    - Los meses con precios más altos indican menor disponibilidad
    - Los meses con precios bajos pueden indicar temporada de cosecha
    - La diferencia entre precios altos y bajos muestra la volatilidad del mercado
    """)

# === Mapa de calor de precios ===
if st.checkbox("Mostrar mapa de calor por meses y años"):
    st.markdown("---")
    st.subheader("🌡️ Mapa de calor de precios")
    
    df_heatmap = df_historico.copy()
    df_heatmap['year'] = df_heatmap['date'].dt.year
    
    pivot_table = df_heatmap.pivot_table(
        values='high_price',
        index='month',
        columns='year',
        aggfunc='mean'
    ).reindex([
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ])
    
    st.dataframe(pivot_table.style.background_gradient(cmap='YlOrRd'))

# === Footer ===
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px;">
    <p>© 2023 Intelica - Análisis de Datos Agrícolas</p>
    <p>Contacto: info@intelica.com | Tel: +123456789</p>
</div>
""", unsafe_allow_html=True)