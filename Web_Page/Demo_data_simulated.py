# === Libraries ===
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from ripser import Rips
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import matplotlib.pyplot as plt
from persim import plot_diagrams
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from gtda.time_series import SlidingWindow
from gtda.diagrams import PersistenceEntropy, Scaler
from gtda.homology import VietorisRipsPersistence
from gtda.metaestimators import CollectionTransformer
from gtda.pipeline import Pipeline
from gtda.time_series import SlidingWindow
from gtda.time_series import TakensEmbedding
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from ripser import Rips
import seaborn as sns 

# === Initial configuration ===
st.set_page_config(
    layout="wide",
    page_title="Fruits: a Topological Analysis for Intelica"
)

# === Load Data ===
@st.cache_data
def load_data():
    return pd.read_csv("C:/Users/52452/Downloads/df_fruits.csv")

data = load_data()
data['report_date'] = pd.to_datetime(data['report_date'])

# === Sidebar ===
with st.sidebar:
    st.image("https://raw.githubusercontent.com/SaraRiveraM/TDA-para-Intellica-/main/Images/images.png")
    st.markdown("<h1 style='font-size: 20px;'>Elija la fruta a analizar:</h1>", unsafe_allow_html=True)
    fruta = st.radio("Seleccione una fruta:", ["Zarzamora", "Mora Azul"])

fruta_dict = {
    "Zarzamora": "Blackberries",
    "Mora Azul": "Blueberries"
}

# === Tabs ===
tab1, tab2 = st.tabs([
    f"🧪 Análisis exploratorio de los precios de la {fruta}",
    "🧠 Análisis topológico (diagramas de persistencia)"
])

# ========================
# === TAB 1 - Exploración
# ========================
with tab1:
    st.markdown(f"<h1 style='font-size: 40px;'>💲 Análisis Topológico: Relación de los Cambios Abruptos de los Precios de la - {fruta} 💹</h1>", unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("🔍 Consulta histórica de precios")

    data['year'] = data['report_date'].dt.year
    data['month'] = data['report_date'].dt.month
    data['day'] = data['report_date'].dt.day

    st.markdown("### 📅 Seleccione una fecha:")
    col1, col2, col3 = st.columns(3)
    with col1:
        años_disponibles = sorted(data['year'].unique(), reverse=True)
        año_seleccionado = st.selectbox("Año", años_disponibles)
    with col2:
        meses_disponibles = sorted(data[data['year'] == año_seleccionado]['month'].unique())
        mes_seleccionado = st.selectbox("Mes", meses_disponibles, format_func=lambda x: datetime(1900, x, 1).strftime('%B'))
    with col3:
        días_disponibles = sorted(data[(data['year'] == año_seleccionado) & (data['month'] == mes_seleccionado)]['day'].unique(), reverse=True)
        día_seleccionado = st.selectbox("Día", días_disponibles)

    fecha_seleccionada = datetime(año_seleccionado, mes_seleccionado, día_seleccionado).date()
    st.write(f"📌 Fecha seleccionada: `{fecha_seleccionada}`")

    df_filtrado = data[(data['commodity'] == fruta_dict[fruta]) & 
                       (data['report_date'].dt.date == fecha_seleccionada)]

    if not df_filtrado.empty:
        st.success("📊 Datos encontrados:")

    # Series temporales
    st.markdown("---")
    st.subheader("📈 Evolución histórica de precios")

    df_historico = data[data['commodity'] == fruta_dict[fruta]]
    if not df_historico.empty:
        df_historico = df_historico.sort_values('report_date')
        st.line_chart(df_historico.set_index('report_date')[['price']])
    else:
        st.warning("No hay datos históricos disponibles")

    # Análisis estacional
    st.markdown("---")
    st.subheader("📅 Análisis estacional")

    if not df_historico.empty:
        df_historico['month'] = df_historico['report_date'].dt.month_name()
        monthly_avg = df_historico.groupby('month')[['price']].mean().reindex([
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ])
        st.write("Promedio mensual de precios:")
        st.bar_chart(monthly_avg)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Precio máximo histórico", f"${monthly_avg['price'].max():.2f}")
            st.metric("Mes con mayor precio", monthly_avg['price'].idxmax())
        with col2:
            st.metric("Precio mínimo histórico", f"${monthly_avg['price'].min():.2f}")
            st.metric("Mes con menor precio", monthly_avg['price'].idxmin())

        st.markdown("""
        ### Interpretación:
        - Los meses con precios más altos indican menor disponibilidad
        - Los precios bajos pueden coincidir con temporadas de cosecha
        - La diferencia muestra la volatilidad del mercado
        """)

    # Heatmap
    if st.checkbox("Mostrar mapa de calor por meses y años"):
        st.markdown("---")
        st.subheader("🌡️ Mapa de calor de precios")

        df_heatmap = df_historico.copy()
        df_heatmap['year'] = df_heatmap['report_date'].dt.year
        df_heatmap['month'] = df_heatmap['report_date'].dt.month_name()

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

# ========================
# === TAB 2 - Topología
# ========================
# ========================
# === TAB 2 - Topología
# ========================
with tab2:
    st.markdown("## 🔺 Análisis Topológico")
    st.write("Esta sección explora patrones complejos en la evolución de los precios mediante herramientas de Topología Computacional.")

    # === Leer datasets individuales ===
    data_m = pd.read_csv("C:/Users/52452/Downloads/blueberry_prices.csv")
    data_b = pd.read_csv("C:/Users/52452/Downloads/blackberry_prices.csv")

    # === Usar la fruta seleccionada desde el sidebar ===
    fruta_dict = {
        "Zarzamora": "Blackberries",
        "Mora Azul": "Blueberries"
    }

    df_topo = data_b if fruta == "Zarzamora" else data_m

    st.markdown(f"### 🍇 Fruta seleccionada: `{fruta}`")

    # Elegir tipo de precio
    columna_precio = st.selectbox("Seleccione el tipo de precio:", ['price', 'low_price', 'high_price'])
    serie = df_topo[columna_precio].dropna().values

    # Selección de método
    metodo = st.radio("Seleccione el método topológico:", ["Takens Embedding", "Sliding Windows", "Rips Diagram"], horizontal=True)

    if metodo == "Takens Embedding":
        st.markdown("#### 🔹 Takens Embedding con Persistencia")

        embedder = TakensEmbedding(time_delay=5, dimension=5, stride=2)
        batch_pca = CollectionTransformer(PCA(n_components=3), n_jobs=-1)
        persistence = VietorisRipsPersistence(homology_dimensions=[0, 1, 2], n_jobs=-1)
        scaling = Scaler()
        entropy = PersistenceEntropy(normalize=True, nan_fill_value=-10)

        pipeline = Pipeline([
            ("embedder", embedder),
            ("pca", batch_pca),
            ("persistence", persistence),
            ("scaling", scaling),
            ("entropy", entropy)
        ])

        X = serie.reshape(-1, 1)
        entropies = pipeline.fit_transform(X)
        st.line_chart(entropies.flatten())

    elif metodo == "Sliding Windows":
        st.markdown("#### 🔹 Sliding Windows + Persistencia")

        pipeline = Pipeline([
            ("window", CollectionTransformer(SlidingWindow(size=30, stride=10))),
            ("pca", CollectionTransformer(PCA(n_components=3), n_jobs=-1)),
            ("persistence", VietorisRipsPersistence(homology_dimensions=[0, 1, 2], n_jobs=-1)),
            ("scaling", Scaler()),
            ("entropy", PersistenceEntropy(normalize=True, nan_fill_value=-10))
        ])

        X = serie.reshape(-1, 1)
        entropies = pipeline.fit_transform(X)
        st.line_chart(entropies.flatten())

    elif metodo == "Rips Diagram":
        st.markdown("#### 🔹 Diagrama de Persistencia (Rips) clásico")

        def calcular_persistencia(X, maxdim=2):
            X_2d = np.array(X).reshape(-1, 1)
            return Rips(maxdim=maxdim).fit_transform(X_2d)

        X = serie.reshape(-1, 1)
        diagrams = calcular_persistencia(X)

        fig, ax = plt.subplots()
        plot_diagrams(diagrams, ax=ax, show=False)
        ax.set_title("Diagramas de Persistencia")
        st.pyplot(fig)


# === Footer ===
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px;">
    <p>© 2023 Intelica - Análisis de Datos Agrícolas</p>
    <p>Contacto: info@intelica.com | Tel: +123456789</p>
</div>
""", unsafe_allow_html=True)
