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
with tab2:
    
    st.header("🔺 Análisis Topológico de Series Temporales")

    # === Lectura según fruta seleccionada ===
    fruta_nombre = fruta_dict[fruta]

    if fruta_nombre == "Blueberries":
        data_f = pd.read_csv("C:/Users/52452/Downloads/blueberry_prices.csv")
    else:
        data_f = pd.read_csv("C:/Users/52452/Downloads/blackberry_prices.csv")

    data_f['report_date'] = pd.to_datetime(data_f['report_date'])
    data_f = data_f.sort_values("report_date")
    serie = data_f["price"].values.reshape(-1, 1)

    # === Mostrar serie original ===
    st.subheader("📉 Serie de Precios")
    st.line_chart(data_f.set_index("report_date")["price"])

    # ===========================
    # === Pipelines definidos ===
    # ===========================


    # --- Takens Embedding pipeline ---
    embedding_dimension = 5
    embedding_time_delay = 5
    stride = 2

    embedder = TakensEmbedding(time_delay=embedding_time_delay,
                               dimension=embedding_dimension,
                               stride=stride)

    batch_pca = CollectionTransformer(PCA(n_components=3), n_jobs=-1)
    persistence = VietorisRipsPersistence(homology_dimensions=[0, 1, 2], n_jobs=-1)
    scaling = Scaler()
    entropy = PersistenceEntropy(normalize=True, nan_fill_value=-10)

    steps_te = [
        ("embedder", embedder),
        ("pca", batch_pca),
        ("persistence", persistence),
        ("scaling", scaling),
        ("entropy", entropy)
    ]

    topological_transfomer_te = Pipeline(steps_te)

    # --- Sliding Window pipeline ---
    window_size = 30
    stride = 10

    steps_sw = [
        ("window", CollectionTransformer(SlidingWindow(size=window_size, stride=stride))),
        ("pca", CollectionTransformer(PCA(n_components=3), n_jobs=-1)),
        ("persistence", VietorisRipsPersistence(homology_dimensions=[0, 1, 2], n_jobs=-1)),
        ("scaling", Scaler()),
        ("entropy", PersistenceEntropy(normalize=True, nan_fill_value=-10))
    ]

    topological_transformer_sw = Pipeline(steps_sw)

    # --- Rips clásico pipeline ---
    def calcular_persistencia(X, maxdim=2):
        X_2d = np.array(X).reshape(-1, 1)
        rips = VietorisRipsPersistence(homology_dimensions=range(maxdim + 1))
        return rips.fit_transform([X_2d])[0]  # Devuelve solo el primer diagrama


    homology_persistence_pipeline = Pipeline([
        ('persistencia', FunctionTransformer(
            calcular_persistencia,
            kw_args={'maxdim': 2}
        ))
    ])

    # ===============================
    # === Resultados de análisis ===
    # ===============================

    st.subheader("🔹 Takens Embedding")
    resultado_te = topological_transfomer_te.fit_transform(serie)
    st.line_chart(resultado_te)

    st.subheader("🔸 Sliding Windows")
    resultado_sw = topological_transformer_sw.fit_transform(serie)
    st.line_chart(resultado_sw)

    st.subheader("🔻 Diagrama de Persistencia - Rips directo")
    diagrams = homology_persistence_pipeline.fit_transform(serie)


    from gtda.plotting import plot_diagram

    fig, ax = plt.subplots()
    rips.plot(diagrams, ax=ax, show=False)
    ax.set_title(f'Diagramas de Persistencia para {fruta}')
    st.pyplot(fig)


# === Footer ===
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px;">
    <p>© 2023 Intelica - Análisis de Datos Agrícolas</p>
    <p>Contacto: info@intelica.com | Tel: +123456789</p>
</div>
""", unsafe_allow_html=True)
