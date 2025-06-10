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
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy.signal import find_peaks


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
    fruta = st.radio("Seleccione una fruta:", ["zarzamora", "mora azul"])

fruta_dict = {
    "zarzamora": "Blackberries",
    "mora azul": "Blueberries"
}

# === Tabs ===
tab1, tab2 = st.tabs([
    f"🧪 Análisis exploratorio de los precios de la {fruta}",
    f"🧠 Análisis topológico de {fruta}"
])

# ========================
# === TAB 1 - Exploración
# ========================
with tab1:
    st.markdown(f"<h1 style='font-size: 40px;'>💲 Análisis Topológico: Relación de los Cambios Abruptos de los Precios de la  {fruta} 💹</h1>", unsafe_allow_html=True)
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
        
        
    # === Mostrar serie original ===
    st.subheader("📉 Serie de Precios")
    st.line_chart(data.set_index("report_date")["price"])

    # =============== 📅 Análisis Estacional ===============
    st.markdown("----")
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

# Función TDA transformation
def tda_transformation(X, method='TE'): 
    if method == 'TE':
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
        steps_te = [("embedder", embedder),
                ("pca", batch_pca),
                ("persistence", persistence),
                ("scaling", scaling),
                ("entropy", entropy)]
        topological_transformer_te = Pipeline(steps_te)
        return topological_transformer_te.fit_transform(X)
    if method == "SW": 
        # Parámetros
        window_size = 30
        stride = 10
        # Pasos del pipeline
        steps_sw = [
            ("window", CollectionTransformer(SlidingWindow(size=window_size, stride=stride))),
            ("pca", CollectionTransformer(PCA(n_components=3), n_jobs=-1)),
            ("persistence", VietorisRipsPersistence(homology_dimensions=[0, 1, 2], n_jobs=-1)),
            ("scaling", Scaler()),
            ("entropy", PersistenceEntropy(normalize=True, nan_fill_value=-10))
        ]
        topological_transformer_sw = Pipeline(steps_sw)
        return topological_transformer_sw.fit_transform(X)

# Función para plot persistent homology
def plot_persistent_homology(x, method="TE", embedding_dimension=5, embedding_time_delay=5, stride=2, homology_dimensions=[0, 1, 2], window_size=5):
    """
    Calcula y plotea el diagrama de persistencia (H0, H1, H2) de una serie de tiempo.
    Parámetros:
    - x: np.array, serie de tiempo unidimensional.
    - embedding_dimension: dimensión del embedding de Takens.
    - embedding_time_delay: delay entre componentes del embedding.
    - stride: stride del embedding.
    - homology_dimensions: lista de dimensiones homológicas a calcular.
    """
    if method == "TE":
        # Takens embedding
        takens = SingleTakensEmbedding(time_delay=embedding_time_delay,
                                    dimension=embedding_dimension,
                                    stride=stride)
        X_embedded = takens.fit_transform(x)
        # Vietoris-Rips
        vr = VietorisRipsPersistence(homology_dimensions=homology_dimensions)
        X_embedded_batch = X_embedded[None, :, :]  # reshape para batch
        diagrams = vr.fit_transform(X_embedded_batch)
        # Plot
        fig = plot_diagram(diagrams[0])
        return fig
    if method == "SW": 
        # Sliding window embedding
        sliding = SlidingWindow(size=window_size, stride=stride)
        X_embedded = sliding.fit_transform(x)
        # Vietoris-Rips
        vr = VietorisRipsPersistence(homology_dimensions=homology_dimensions)
        X_embedded_batch = X_embedded[None, :, :]  # reshape para batch
        diagrams = vr.fit_transform(X_embedded_batch)
        # Plot
        fig = plot_diagram(diagrams[0])
        return fig

# Función para preparar datos para CNN usando TDA
def prepare_cnn_data_with_tda(serie, method='SW'):
    """
    Prepara los datos de la serie temporal usando transformación TDA para el modelo CNN
    """
    # Aplicar transformación TDA
    X_tda = tda_transformation([serie.flatten()], method=method)
    return X_tda

# Función para cargar y aplicar el modelo CNN
@st.cache_resource
def load_cnn_model():
    """
    Carga el modelo CNN desde el archivo .keras
    """
    try:
        from tensorflow.keras.models import load_model
        model = load_model("C:/Users/52452/Downloads/modelo_sw.keras")
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

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
    
    # Agregar columnas de año y mes para filtrado
    data_f['year'] = data_f['report_date'].dt.year
    data_f['month'] = data_f['report_date'].dt.month
    data_f['year_month'] = data_f['report_date'].dt.to_period('M')
    
    st.subheader(f"📊 Análisis de precios de {fruta_nombre}")
    
    # === Selección de período ===
    col1, col2 = st.columns(2)
    
    with col1:
        # Seleccionar año
        years_available = sorted(data_f['year'].unique())
        selected_year = st.selectbox("Selecciona el año:", years_available)
    
    with col2:
        # Seleccionar mes
        months_dict = {
            1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
            5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
            9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
        }
        
        # Filtrar meses disponibles para el año seleccionado
        months_available = sorted(data_f[data_f['year'] == selected_year]['month'].unique())
        month_options = {month: months_dict[month] for month in months_available}
        
        selected_month = st.selectbox("Selecciona el mes:", 
                                    options=list(month_options.keys()),
                                    format_func=lambda x: month_options[x])
    
    # === Filtrar datos según selección ===
    data_filtered = data_f[(data_f['year'] == selected_year) & (data_f['month'] == selected_month)]
    
    if len(data_filtered) == 0:
        st.warning("No hay datos disponibles para el período seleccionado.")
    else:
        # Mostrar información del período seleccionado
        st.info(f"📅 Período seleccionado: {months_dict[selected_month]} {selected_year} "
                f"({len(data_filtered)} registros)")
        
        # Preparar serie temporal
        serie = data_filtered["price"].values.reshape(-1, 1)
        
        # === Visualización de la serie temporal ===
        st.subheader("📈 Serie Temporal del Período Seleccionado")
        
        fig_serie = plt.figure(figsize=(12, 6))
        plt.plot(data_filtered['report_date'], data_filtered['price'], 
                marker='o', linewidth=2, markersize=4)
        plt.title(f'Precios de {fruta_nombre} - {months_dict[selected_month]} {selected_year}')
        plt.xlabel('Fecha')
        plt.ylabel('Precio ($)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_serie)
        
        # === Análisis Topológico ===
        st.subheader("🔺 Análisis de Persistencia Topológica")
        
        # Selección del método TDA
        tda_method = st.selectbox("Selecciona el método TDA:", 
                                 options=['TE', 'SW'], 
                                 format_func=lambda x: 'Takens Embedding' if x == 'TE' else 'Sliding Window')
        
        if len(serie) >= 10:  # Mínimo de datos necesarios
            try:
                # Mostrar diagrama de persistencia
                st.subheader("📊 Diagrama de Persistencia")
                
                with st.spinner("Calculando diagrama de persistencia..."):
                    fig_persistence = plot_persistent_homology(serie.flatten(), method=tda_method)
                    st.plotly_chart(fig_persistence, use_container_width=True)
                
                # === Análisis con CNN usando TDA ===
                st.subheader("🤖 Predicción de Cambios Abruptos con CNN + TDA")
                
                # Cargar modelo CNN
                model = load_cnn_model()
                
                if model is not None:
                    with st.spinner("Aplicando transformación TDA y prediciendo..."):
                        # Preparar datos para CNN usando TDA
                        X_tda = prepare_cnn_data_with_tda(serie, method=tda_method)
                        
                        # Verificar si tenemos datos válidos
                        if X_tda is not None and len(X_tda) > 0:
                            # Normalizar datos TDA
                            from sklearn.preprocessing import MinMaxScaler
                            scaler = MinMaxScaler()
                            
                            # Reshape para normalización
                            X_tda_flat = X_tda.reshape(-1, 1)
                            X_tda_scaled = scaler.fit_transform(X_tda_flat)
                            
                            # Preparar para CNN - ajustar dimensiones según tu modelo
                            # Asumiendo que el modelo espera secuencias, creamos ventanas de las características TDA
                            if len(X_tda_scaled) >= 3:  # Mínimo para crear una secuencia
                                X_cnn_input = X_tda_scaled.reshape(1, -1, 1)  # (samples, timesteps, features)
                                
                                try:
                                    # Realizar predicción
                                    prediction = model.predict(X_cnn_input)
                                    
                                    # Mostrar resultado principal
                                    prob_cambio = float(prediction[0][0]) if prediction.ndim > 1 else float(prediction[0])
                                    
                                    # Métricas principales
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        st.metric("🎯 Probabilidad de Cambio Abrupto", 
                                                f"{prob_cambio:.2%}")
                                    
                                    with col2:
                                        risk_level = "Alto" if prob_cambio > 0.7 else "Medio" if prob_cambio > 0.4 else "Bajo"
                                        color = "🔴" if prob_cambio > 0.7 else "🟡" if prob_cambio > 0.4 else "🟢"
                                        st.metric(f"{color} Nivel de Riesgo", risk_level)
                                    
                                    with col3:
                                        confidence = "Alta" if abs(prob_cambio - 0.5) > 0.3 else "Media" if abs(prob_cambio - 0.5) > 0.15 else "Baja"
                                        st.metric("📊 Confianza", confidence)
                                    
                                    # Visualización de características TDA
                                    st.subheader("🔍 Características Topológicas Extraídas")
                                    
                                    fig_tda = plt.figure(figsize=(12, 6))
                                    
                                    # Plot de características TDA
                                    plt.subplot(2, 1, 1)
                                    plt.plot(X_tda_scaled.flatten(), 'b-', marker='o', linewidth=2, markersize=4)
                                    plt.title(f'Características TDA - Método: {tda_method}')
                                    plt.ylabel('Valor Normalizado')
                                    plt.grid(True, alpha=0.3)
                                    
                                    # Plot de la serie original para comparación
                                    plt.subplot(2, 1, 2)
                                    plt.plot(data_filtered['report_date'], data_filtered['price'], 
                                            'g-', marker='o', linewidth=2, markersize=4)
                                    plt.title(f'Serie Original - {fruta_nombre}')
                                    plt.xlabel('Fecha')
                                    plt.ylabel('Precio ($)')
                                    plt.xticks(rotation=45)
                                    plt.grid(True, alpha=0.3)
                                    
                                    plt.tight_layout()
                                    st.pyplot(fig_tda)
                                    
                                    # === Interpretación de resultados ===
                                    st.subheader("📋 Interpretación de Resultados")
                                    
                                    interpretacion = f"""
                                    **Análisis TDA para {fruta_nombre} en {months_dict[selected_month]} {selected_year}:**
                                    
                                    **Método utilizado:** {tda_method} ({'Takens Embedding' if tda_method == 'TE' else 'Sliding Window'})
                                    
                                    **Características topológicas detectadas:**
                                    - Número de características extraídas: {len(X_tda_scaled)}
                                    - Rango de valores: [{X_tda_scaled.min():.3f}, {X_tda_scaled.max():.3f}]
                                    - Variabilidad topológica: {X_tda_scaled.std():.3f}
                                    
                                    **Predicción del modelo:**
                                    - Probabilidad de cambio abrupto: {prob_cambio:.2%}
                                    - Clasificación de riesgo: {risk_level}
                                    """
                                    
                                    st.markdown(interpretacion)
                                    
                                    # Recomendaciones basadas en el resultado
                                    if prob_cambio > 0.7:
                                        st.error("""
                                        ⚠️ **ALERTA DE ALTO RIESGO**
                                        
                                        Las características topológicas indican una alta probabilidad de cambios abruptos en los precios:
                                        - Implementar estrategias de cobertura inmediatamente
                                        - Monitorear el mercado con mayor frecuencia
                                        - Considerar reducir posiciones de riesgo
                                        """)
                                    elif prob_cambio > 0.4:
                                        st.warning("""
                                        ⚡ **PRECAUCIÓN MODERADA**
                                        
                                        Se detecta volatilidad en las características topológicas:
                                        - Mantener vigilancia sobre las condiciones del mercado
                                        - Preparar estrategias de contingencia
                                        - Evaluar la diversificación del portafolio
                                        """)
                                    else:
                                        st.success("""
                                        ✅ **CONDICIONES ESTABLES**
                                        
                                        Las características topológicas sugieren estabilidad:
                                        - Ambiente favorable para operaciones regulares
                                        - Riesgo de volatilidad extrema relativamente bajo
                                        - Continuar con estrategias normales de trading
                                        """)
                                    
                                except Exception as e:
                                    st.error(f"Error en la predicción del modelo: {e}")
                                    st.info("Verifica que el modelo sea compatible con las características TDA extraídas.")
                            
                            else:
                                st.warning("Las características TDA extraídas son insuficientes para el modelo CNN.")
                        
                        else:
                            st.error("No se pudieron extraer características TDA de la serie temporal.")
                
                else:
                    st.error("No se pudo cargar el modelo CNN. Verifica que el archivo 'modelo_sw.keras' esté en la ruta correcta.")
                    
            except Exception as e:
                st.error(f"Error en el análisis topológico: {e}")
                st.info("Verifica que tengas instaladas todas las librerías necesarias (giotto-tda, etc.)")
        
        else:
            st.warning("Se necesitan al menos 10 puntos de datos para el análisis topológico.")
        
        # === Información adicional sobre TDA ===
        with st.expander("ℹ️ Información sobre Análisis Topológico de Datos (TDA)"):
            st.markdown("""
            **¿Qué es TDA?**
            
            El Análisis Topológico de Datos extrae características geométricas y topológicas de los datos que son invariantes 
            a deformaciones continuas, capturando la "forma" subyacente de los datos.
            
            **Métodos implementados:**
            
            - **Takens Embedding (TE)**: Reconstruye el espacio de estados de un sistema dinámico a partir de la serie temporal
            - **Sliding Window (SW)**: Crea embeddings usando ventanas deslizantes de la serie temporal
            
            **Ventajas para análisis financiero:**
            
            - Captura patrones no lineales complejos
            - Robusto ante ruido en los datos
            - Identifica cambios estructurales en la dinámica del mercado
            - Proporciona características invariantes para clasificación
            """)
        
        # === Comparación de métodos ===
        if st.checkbox("🔄 Comparar métodos TDA"):
            st.subheader("📊 Comparación: Takens Embedding vs Sliding Window")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Takens Embedding:**")
                try:
                    with st.spinner("Calculando TE..."):
                        X_te = prepare_cnn_data_with_tda(serie, method='TE')
                        if X_te is not None:
                            st.write(f"- Características extraídas: {len(X_te[0])}")
                            st.write(f"- Rango: [{X_te.min():.3f}, {X_te.max():.3f}]")
                except:
                    st.write("Error al calcular TE")
            
            with col2:
                st.write("**Sliding Window:**")
                try:
                    with st.spinner("Calculando SW..."):
                        X_sw = prepare_cnn_data_with_tda(serie, method='SW')
                        if X_sw is not None:
                            st.write(f"- Características extraídas: {len(X_sw[0])}")
                            st.write(f"- Rango: [{X_sw.min():.3f}, {X_sw.max():.3f}]")
                except:
                    st.write("Error al calcular SW")
                    
                    # Mostrar resultados
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # Probabilidad promedio de cambio abrupto
                        avg_prob = np.mean(predictions)
                        st.metric("Probabilidad Promedio de Cambio Abrupto", 
                                f"{avg_prob:.2%}")
                    
                    with col2:
                        # Máxima probabilidad
                        max_prob = np.max(predictions)
                        st.metric("Máxima Probabilidad", f"{max_prob:.2%}")
                    
                    with col3:
                        # Número de alertas (probabilidad > 0.7)
                        alerts = np.sum(predictions > 0.7)
                        st.metric("Alertas de Alto Riesgo", f"{alerts}")
                    
                    # Gráfico de predicciones
                    st.subheader("📊 Evolución de Predicciones de Cambios Abruptos")
                    
                    fig_pred = plt.figure(figsize=(12, 8))
                    
                    # Subplot 1: Precios originales
                    plt.subplot(2, 1, 1)
                    plt.plot(data_filtered['report_date'].iloc[window_size-1:], 
                            data_filtered['price'].iloc[window_size-1:], 
                            'b-', label='Precio Real', linewidth=2)
                    plt.title(f'Precios de {fruta_nombre}')
                    plt.ylabel('Precio ($)')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    
                    # Subplot 2: Predicciones de cambios abruptos
                    plt.subplot(2, 1, 2)
                    dates_pred = data_filtered['report_date'].iloc[window_size-1:len(predictions)+window_size-1]
                    plt.plot(dates_pred, predictions, 'r-', label='Prob. Cambio Abrupto', linewidth=2)
                    plt.axhline(y=0.5, color='orange', linestyle='--', label='Umbral Medio (50%)')
                    plt.axhline(y=0.7, color='red', linestyle='--', label='Umbral Alto (70%)')
                    plt.fill_between(dates_pred, predictions.flatten(), alpha=0.3, color='red')
                    plt.title('Predicciones de Cambios Abruptos')
                    plt.xlabel('Fecha')
                    plt.ylabel('Probabilidad')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig_pred)
                    
                    # === Análisis detallado ===
                    st.subheader("📋 Análisis Detallado")
                    
                    # Crear DataFrame con resultados
                    results_df = pd.DataFrame({
                        'Fecha': dates_pred,
                        'Precio': data_filtered['price'].iloc[window_size-1:len(predictions)+window_size-1].values,
                        'Probabilidad_Cambio': predictions.flatten(),
                        'Nivel_Riesgo': pd.cut(predictions.flatten(), 
                                             bins=[0, 0.3, 0.7, 1.0], 
                                             labels=['Bajo', 'Medio', 'Alto'])
                    })
                    
                    # Mostrar tabla con resultados
                    st.dataframe(results_df.style.format({
                        'Precio': '${:.2f}',
                        'Probabilidad_Cambio': '{:.2%}'
                    }))
                    
                    # === Resumen ejecutivo ===
                    st.subheader("📈 Resumen Ejecutivo")
                    
                    high_risk_days = len(results_df[results_df['Nivel_Riesgo'] == 'Alto'])
                    medium_risk_days = len(results_df[results_df['Nivel_Riesgo'] == 'Medio'])
                    low_risk_days = len(results_df[results_df['Nivel_Riesgo'] == 'Bajo'])
                    
                    st.markdown(f"""
                    **Análisis para {fruta_nombre} en {months_dict[selected_month]} {selected_year}:**
                    
                    - 🔴 **Días de Alto Riesgo:** {high_risk_days} ({high_risk_days/len(results_df)*100:.1f}%)
                    - 🟡 **Días de Riesgo Medio:** {medium_risk_days} ({medium_risk_days/len(results_df)*100:.1f}%)
                    - 🟢 **Días de Bajo Riesgo:** {low_risk_days} ({low_risk_days/len(results_df)*100:.1f}%)
                    
                    **Recomendaciones:**
                    """)
                    
                    if avg_prob > 0.7:
                        st.error("⚠️ **ALTO RIESGO:** Se detecta alta probabilidad de cambios abruptos. Se recomienda monitoreo constante y estrategias de cobertura.")
                    elif avg_prob > 0.4:
                        st.warning("⚡ **RIESGO MODERADO:** Volatilidad detectada. Se sugiere precaución en las operaciones.")
                    else:
                        st.success("✅ **BAJO RIESGO:** Período relativamente estable para operaciones.")
                    
                except Exception as e:
                    st.error(f"Error al realizar predicciones: {e}")
                    st.info("Verifica que el modelo sea compatible con las dimensiones de los datos.")
            
                else:
                    st.warning(f"No hay suficientes datos para el análisis. Se necesitan al menos {window_size} puntos de datos.")
        
        else:
            st.error("No se pudo cargar el modelo CNN. Verifica que el archivo 'modelo_sw.keras' esté en la ruta correcta.")
            
        # === Análisis Topológico Adicional (opcional) ===
        if st.checkbox("Mostrar Análisis Topológico Detallado"):
            st.subheader("🔺 Análisis de Persistencia Topológica")
            
            try:
                # Aquí puedes agregar análisis topológico usando gudhi o giotto-tda
                # Por ejemplo, diagramas de persistencia
                st.info("Análisis topológico en desarrollo. Implementar con bibliotecas como gudhi o giotto-tda.")
                
                # Ejemplo básico de análisis de forma
                from scipy.signal import find_peaks
                
                # Encontrar picos y valles
                peaks, _ = find_peaks(serie.flatten(), height=np.mean(serie))
                valleys, _ = find_peaks(-serie.flatten(), height=-np.mean(serie))
                
                st.write(f"📊 **Características topológicas básicas:**")
                st.write(f"- Número de picos detectados: {len(peaks)}")
                st.write(f"- Número de valles detectados: {len(valleys)}")
                st.write(f"- Volatilidad (desviación estándar): ${np.std(serie):.2f}")
                
            except Exception as e:
                st.warning(f"Error en análisis topológico: {e}")
    
    
# === Footer ===
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px;">
    <p>© 2023 Intelica - Análisis de Datos Agrícolas</p>
    <p>Contacto: info@intelica.com | Tel: +123456789</p>
</div>
""", unsafe_allow_html=True)
