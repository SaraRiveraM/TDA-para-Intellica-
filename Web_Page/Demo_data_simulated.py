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
from gtda.metaestimators import CollectionTransformer
from gtda.pipeline import Pipeline
from gtda.time_series import SlidingWindow
from gtda.time_series import TakensEmbedding
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from ripser import Rips
from sklearn.preprocessing import FunctionTransformer
from gtda.homology import VietorisRipsPersistence
import seaborn as sns 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler


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

    data['day'] = data['report_date'].dt.day
    data['month'] = data['report_date'].dt.month
    data['year'] = data['report_date'].dt.year

    st.markdown("### 📅 Seleccione una fecha:")
    col1, col2, col3 = st.columns(3)

    with col1:
        días_disponibles = sorted(data['day'].unique())
        día_seleccionado = st.selectbox("Día", días_disponibles)

    with col2:
        meses_disponibles = sorted(
            data[data['day'] == día_seleccionado]['month'].unique()
        )
        mes_seleccionado = st.selectbox(
            "Mes",
            meses_disponibles,
            format_func=lambda x: datetime(1900, x, 1).strftime('%B')
        )

    with col3:
        años_disponibles = sorted(
            data[(data['day'] == día_seleccionado) & (data['month'] == mes_seleccionado)]['year'].unique(),
            reverse=True
        )
        año_seleccionado = st.selectbox("Año", años_disponibles)

            

        fecha_seleccionada = datetime(año_seleccionado, mes_seleccionado, día_seleccionado).date()
        st.write(f"📌 Fecha seleccionada: `{fecha_seleccionada}`")

        df_filtrado = data[(data['commodity'] == fruta_dict[fruta]) & 
                        (data['report_date'].dt.date == fecha_seleccionada)]

    if not df_filtrado.empty:
        st.success("📊 Datos encontrados:")

    # =============== 📅 Análisis Estacional ===============
    st.markdown("---")
    st.subheader("📅 Análisis estacional")

    # Mapeo manual de meses en español si locale falla
    month_translation = {
        'January': 'Enero', 'February': 'Febrero', 'March': 'Marzo',
        'April': 'Abril', 'May': 'Mayo', 'June': 'Junio',
        'July': 'Julio', 'August': 'Agosto', 'September': 'Septiembre',
        'October': 'Octubre', 'November': 'Noviembre', 'December': 'Diciembre'
    }

    # Solo si hay datos
    if not data.empty:
        data['month'] = data['report_date'].dt.month_name().map(month_translation)

        orden_meses = [
            'Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
            'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'
        ]

        # === 📊 Promedio mensual ===
        monthly_avg = data.groupby('month')[['price']].mean().reindex(orden_meses)
        st.write("Promedio mensual de precios:")
        st.bar_chart(monthly_avg)

        # === 📈 Métricas destacadas ===
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Precio máximo histórico", f"${monthly_avg['price'].max():.2f}")
            st.metric("Mes con mayor precio", monthly_avg['price'].idxmax())
        with col2:
            st.metric("Precio mínimo histórico", f"${monthly_avg['price'].min():.2f}")
            st.metric("Mes con menor precio", monthly_avg['price'].idxmin())

        # === 🧠 Interpretación ===
        st.markdown("""
        ### Interpretación:
        - Los meses con precios más altos indican menor disponibilidad
        - Los precios bajos pueden coincidir con temporadas de cosecha
        - La diferencia muestra la volatilidad del mercado
        """)

        # =============== 🌡️ Heatmap ===============
        if st.checkbox("Mostrar mapa de calor por meses y años"):
            st.markdown("---")
            st.subheader("🌡️ Mapa de calor de precios")

            df_heatmap = data.copy()
            df_heatmap['year'] = df_heatmap['report_date'].dt.year
            df_heatmap['month'] = df_heatmap['report_date'].dt.month_name().map(month_translation)

            pivot_table = df_heatmap.pivot_table(
                values='price',
                index='month',
                columns='year',
                aggfunc='mean'
            ).reindex(orden_meses)

            st.dataframe(pivot_table.style.background_gradient(cmap='YlOrRd'))


# ========================
# === TAB 2 - Topología
# ========================
# Custom transformer for collection handling
class CollectionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, estimator, n_jobs=None):
        self.estimator = estimator
        self.n_jobs = n_jobs
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.array([self.estimator.fit_transform(x) for x in X])

# === Tab 2 - Topología ===
with tab2:
    st.header("🔺 Análisis Topológico de Series Temporales")
    
    # === Data Loading ===
    fruta_nombre = fruta_dict[fruta]
    if fruta_nombre == "Blueberries":
        data_f = pd.read_csv("C:/Users/52452/Downloads/blueberry_prices.csv")
    else:
        data_f = pd.read_csv("C:/Users/52452/Downloads/blackberry_prices.csv")
    
    data_f['report_date'] = pd.to_datetime(data_f['report_date'])
    data_f = data_f.sort_values("report_date")
    serie = data_f["price"].values.reshape(-1, 1)
    
    # === Parameter Configuration ===
    with st.expander("⚙️ Configuración de Parámetros", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            embedding_dim = st.slider("Dimensión Embedding", 2, 5, 3, key='embed_dim')
            embedding_delay = st.slider("Time Delay", 1, 10, 2, key='embed_delay')
            te_stride = st.slider("Stride Embedding", 1, 5, 1, key='te_stride')
        with col2:
            window_size = st.slider("Tamaño Ventana", 10, 60, 30, key='window_size')
            sw_stride = st.slider("Stride Ventana", 1, 20, 5, key='sw_stride')
            max_homology = st.slider("Dimensión Homológica Máxima", 1, 3, 2, key='max_hom')
    
    # Convertir a lista las dimensiones homológicas
    homology_dims = list(range(max_homology + 1))  # Conversión explícita a lista
    
    # === Pipeline Definitions ===
    # 1. Takens Embedding Pipeline
    te_pipeline = Pipeline([
        ("embedding", TakensEmbedding(
            time_delay=embedding_delay,
            dimension=embedding_dim,
            stride=te_stride
        )),
        ("pca", CollectionTransformer(PCA(n_components=3)) if embedding_dim > 3 else ("passthrough", "passthrough")),
        ("persistence", VietorisRipsPersistence(
            homology_dimensions=homology_dims  # Usamos la lista convertida
        )),
        ("scaling", Scaler()),
        ("entropy", PersistenceEntropy(normalize=True))
    ], memory=None)
    
    # 2. Sliding Window Pipeline
    sw_pipeline = Pipeline([
        ("window", SlidingWindow(
            size=window_size,
            stride=sw_stride
        )),
        ("pca", CollectionTransformer(PCA(n_components=3))),
        ("persistence", VietorisRipsPersistence(
            homology_dimensions=homology_dims  # Usamos la lista convertida
        )),
        ("scaling", Scaler()),
        ("entropy", PersistenceEntropy(normalize=True))
    ], memory=None)
    
    # 3. Direct Rips Pipeline - Versión corregida
    def calcular_persistencia(X, maxdim):
        X_2d = np.array(X).reshape(-1, 1)
        homology_dims = list(range(maxdim + 1))  # Conversión a lista aquí también
        rips = VietorisRipsPersistence(homology_dimensions=homology_dims)
        return rips.fit_transform([X_2d])[0]
    
    rips_pipeline = Pipeline([
        ('persistence', FunctionTransformer(
            calcular_persistencia,
            kw_args={'maxdim': max_homology}
        ))
    ])
    
    # === Visualization in Tabs ===
    te_tab, sw_tab, rips_tab = st.tabs([
        "🔄 Takens Embedding", 
        "📊 Sliding Windows", 
        "🔷 Diagramas Persistencia"
    ])
    
    with rips_tab:
        st.subheader("Diagramas de Persistencia")
        try:
            with st.spinner("Generando diagramas..."):
                diagrams = rips_pipeline.fit_transform(serie)
                
                fig, ax = plt.subplots(figsize=(8, 8))
                #plot_diagram(diagrams, ax=ax)
                ax.set_title(f'Diagrama de Persistencia - {fruta}')
                ax.set_xlabel('Nacimiento')
                ax.set_ylabel('Muerte')
                ax.grid(True, linestyle='--', alpha=0.6)
                st.pyplot(fig)
                
                st.markdown("""
                **Interpretación:**
                - Puntos lejos de la diagonal = características persistentes
                - Color indica dimensión homológica:
                  - Azul: Componentes conexas (H0)
                  - Naranja: Bucles (H1)
                  - Verde: Cavidades (H2)
                """)
        except Exception as e:
            st.error(f"Error en diagramas de persistencia: {str(e)}")
            st.info("Verifica que los datos tengan suficiente variabilidad")
    
    # === Data Summary ===
    st.sidebar.markdown("---")
    st.sidebar.subheader("Resumen de Datos")
    st.sidebar.write(f"📅 Rango temporal: {data_f['report_date'].min().date()} a {data_f['report_date'].max().date()}")
    st.sidebar.write(f"🔢 Puntos de datos: {len(serie)}")


# === Footer ===
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px;">
    <p>© 2023 Intelica - Análisis de Datos Agrícolas</p>
    <p>Contacto: info@intelica.com | Tel: +123456789</p>
</div>
""", unsafe_allow_html=True)
