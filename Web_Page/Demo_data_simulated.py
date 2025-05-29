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
                values='high_price',
                index='month',
                columns='year',
                aggfunc='mean'
            ).reindex(orden_meses)

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

    st.subheader("📉 Serie de Precios")
    st.line_chart(data_f.set_index("report_date")["price"])

    # Tabs internos para cada pipeline
    tab_te, tab_sw, tab_rips = st.tabs(["Takens Embedding", "Sliding Windows", "Rips Directo"])

    with tab_te:
        st.subheader("🔹 Takens Embedding")

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
        resultado_te = topological_transfomer_te.fit_transform(serie)
        te_df = pd.DataFrame(resultado_te, columns=['Entropía_0', 'Entropía_1', 'Entropía_2'])
        st.line_chart(te_df)
        st.caption(f"""
        Evolución de la entropía de persistencia por dimensión topológica (0=componentes, 1=bucles, 2=cavidades).
        Embedding: dim={embedding_dimension}, delay={embedding_time_delay}, stride={stride}
        """)

    with tab_sw:
        st.subheader("🔸 Sliding Windows")

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
        resultado_sw = topological_transformer_sw.fit_transform(serie)
        sw_df = pd.DataFrame(resultado_sw, columns=['Entropía_0', 'Entropía_1', 'Entropía_2'])
        st.line_chart(sw_df)
        st.caption(f"""
        Evolución temporal de características topológicas (ventana={window_size}, stride={stride}).
        Las fluctuaciones indican cambios en la estructura topológica subyacente.
        """)

    with tab_rips:
        st.subheader("🔻 Diagrama de Persistencia - Rips directo")

        def calcular_persistencia(X, maxdim=2):
            X_2d = np.array(X).reshape(-1, 1)
            rips = VietorisRipsPersistence(homology_dimensions=range(maxdim + 1))
            return rips.fit_transform([X_2d])[0]

        homology_persistence_pipeline = Pipeline([
            ('persistencia', FunctionTransformer(
                calcular_persistencia,
                kw_args={'maxdim': 2}
            ))
        ])

        diagrams = homology_persistence_pipeline.fit_transform(serie)

        from gtda.plotting import plot_diagram
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_diagram(diagrams, ax=ax)
        ax.set_title(f'Diagrama de Persistencia para {fruta}')
        ax.set_xlabel('Tiempo de nacimiento')
        ax.set_ylabel('Tiempo de muerte')
        ax.grid(True, linestyle='--', alpha=0.6)

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, ['Dimensión 0', 'Dimensión 1', 'Dimensión 2'],
                  title='Dimensión Homológica',
                  bbox_to_anchor=(1.05, 1), 
                  loc='upper left')

        st.pyplot(fig, bbox_inches='tight', use_container_width=True)
        st.caption("""
        Diagrama que muestra los ciclos topológicos (puntos) y su persistencia.
        Puntos lejos de la diagonal representan características topológicas persistentes.
        """)


# === Footer ===
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px;">
    <p>© 2023 Intelica - Análisis de Datos Agrícolas</p>
    <p>Contacto: info@intelica.com | Tel: +123456789</p>
</div>
""", unsafe_allow_html=True)
