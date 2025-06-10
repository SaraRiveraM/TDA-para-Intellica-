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
from gtda.plotting import plot_diagram
import tempfile
import requests



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
    fruta = st.radio("Seleccione una fruta:", ["Zarzamora", "Mora azul"])

fruta_dict = {
    "Zarzamora": "Blackberries",
    "Mora azul": "Blueberries"
}

# === Tabs ===
tab1, tab2 = st.tabs([
    f"💵 Análisis exploratorio de los precios de la {fruta}",
    f"🧠 Análisis topológico de la {fruta}"
])

# ========================
# === TAB 1 - Exploración
# ========================
with tab1:
    st.markdown(f"<h1 style='font-size: 40px;'>💲 Análisis Topológico: Relación de los Cambios Abruptos de los Precios de la  {fruta} 💹</h1>", unsafe_allow_html=True)
    st.markdown("---")
    st.mardown("Con esta app, se busca hacer un análisis de preciuos de distintas frutas, para que en un futuro, se pueda detectar si hay cambios abruptos o ciclos en los datos.")
    st.mardown("👈 **Despliega el sidebar para cambiar de fruta a consultar.**")
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
    """
    Aplica transformación TDA a una serie temporal
    """
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
        
    elif method == "SW": 
        # Parámetros
        window_size = min(30, len(X[0]) // 3)  # Ajustar tamaño de ventana
        stride = max(1, window_size // 3)  # Ajustar stride
        
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
    else:
        raise ValueError("Method debe ser 'TE' o 'SW'")

# Función para plot persistent homology
def plot_persistent_homology(x, method="TE", embedding_dimension=5, embedding_time_delay=5, stride=2, homology_dimensions=[0, 1, 2], window_size=5):
    """
    Calcula y plotea el diagrama de persistencia (H0, H1, H2) de una serie de tiempo.
    """
    try:
        if method == "TE":
            # Takens embedding
            takens = TakensEmbedding(time_delay=embedding_time_delay,
                                   dimension=embedding_dimension,
                                   stride=stride)
            X_embedded = takens.fit_transform(x.reshape(-1, 1))
            
        elif method == "SW": 
            # Sliding window embedding
            window_size = min(window_size, len(x) // 3)  # Ajustar tamaño
            sliding = SlidingWindow(size=window_size, stride=stride)
            X_embedded = sliding.fit_transform(x.reshape(-1, 1))
            
        else:
            raise ValueError("Method debe ser 'TE' o 'SW'")
        
        # Verificar que tenemos suficientes puntos
        if len(X_embedded) < 3:
            raise ValueError("No hay suficientes puntos después del embedding")
        
        # Vietoris-Rips persistence
        vr = VietorisRipsPersistence(homology_dimensions=homology_dimensions)
        diagrams = vr.fit_transform(X_embedded)
        
        # Plot del diagrama de persistencia
        fig = plot_diagram(diagrams[0])
        return fig
        
    except Exception as e:
        st.error(f"Error en plot_persistent_homology: {e}")
        return None

# Función para preparar datos para CNN usando TDA
def prepare_cnn_data_with_tda(serie, method='SW'):
    """
    Prepara los datos de la serie temporal usando transformación TDA para el modelo CNN
    """
    try:
        # Verificar que la serie tenga suficientes datos
        if len(serie.flatten()) < 10:
            raise ValueError("Serie temporal muy corta para análisis TDA")
            
        # Aplicar transformación TDA
        X_tda = tda_transformation([serie.flatten()], method=method)
        
        # Verificar que obtuvimos resultados válidos
        if X_tda is None or len(X_tda) == 0:
            raise ValueError("La transformación TDA no produjo resultados")
            
        return X_tda
        
    except Exception as e:
        st.error(f"Error en TDA transformation: {e}")
        return None

# Función para cargar y aplicar el modelo CNN
@st.cache_resource
def load_cnn_model():
    """
    Carga el modelo CNN con manejo robusto de errores
    """
    try:
        url = "https://raw.githubusercontent.com/SaraRiveraM/TDA-para-Intellica-/main/Web_Page/Models/modelo_sw.keras"
        
        # Verificar conectividad
        response = requests.get(url, timeout=30)
        
        if response.status_code != 200:
            raise Exception(f"Error al descargar el modelo. Código de estado: {response.status_code}")

        # Crear archivo temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as tmp_file:
            tmp_file.write(response.content)
            tmp_model_path = tmp_file.name

        # Verificar que el archivo se escribió correctamente
        import os
        if not os.path.exists(tmp_model_path) or os.path.getsize(tmp_model_path) == 0:
            raise Exception("El archivo del modelo no se guardó correctamente")

        # Cargar modelo con diferentes métodos
        try:
            model = tf.keras.models.load_model(tmp_model_path)
        except Exception as e1:
            try:
                # Intentar cargar sin compilar
                model = tf.keras.models.load_model(tmp_model_path, compile=False)
            except Exception as e2:
                raise Exception(f"No se pudo cargar el modelo. Error 1: {e1}, Error 2: {e2}")
        
        # Limpiar archivo temporal
        try:
            os.unlink(tmp_model_path)
        except:
            pass  # No es crítico si no se puede eliminar
            
        return model
        
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

def make_cnn_prediction(model, X_tda, method='SW'):
    """
    Realiza predicción con el modelo CNN manejando diferentes formatos de entrada
    """
    try:
        if X_tda is None or len(X_tda) == 0:
            raise ValueError("Datos TDA inválidos")
            
        # Normalizar datos TDA
        scaler = MinMaxScaler()
        X_tda_flat = X_tda.reshape(-1, 1)
        X_tda_scaled = scaler.fit_transform(X_tda_flat)
        
        # Preparar entrada para el modelo
        if len(X_tda_scaled) >= 3:
            # Diferentes formatos según la arquitectura del modelo
            input_formats = [
                X_tda_scaled.reshape(1, -1, 1),  # (batch, sequence, features)
                X_tda_scaled.reshape(1, -1),     # (batch, features)
                X_tda_scaled.T.reshape(1, -1, 1), # Transpuesta
            ]
            
            prediction = None
            for i, input_format in enumerate(input_formats):
                try:
                    prediction = model.predict(input_format, verbose=0)
                    break
                except Exception as e:
                    if i == len(input_formats) - 1:  # Último intento
                        raise e
                    continue
            
            if prediction is None:
                raise ValueError("No se pudo hacer la predicción con ningún formato de entrada")
                
            # Extraer probabilidad
            prob_cambio = float(prediction[0][0]) if prediction.ndim > 1 else float(prediction[0])
            prob_cambio = max(0.0, min(1.0, prob_cambio))  # Clamp entre 0 y 1
            
            return prob_cambio, X_tda_scaled
            
        else:
            raise ValueError("Características TDA insuficientes")
            
    except Exception as e:
        st.error(f"Error en la predicción: {e}")
        return None, None

# Función para mostrar resultados de análisis TDA
def display_tda_results(prob_cambio, X_tda_scaled, tda_method, fruta_nombre, selected_month, selected_year, months_dict):
    """
    Muestra los resultados del análisis TDA de forma organizada
    """
    if prob_cambio is None:
        st.error("No se pudo realizar la predicción")
        return
        
    # Métricas principales
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🎯 Probabilidad de Cambio Abrupto", f"{prob_cambio:.2%}")
    
    with col2:
        risk_level = "Alto" if prob_cambio > 0.7 else "Medio" if prob_cambio > 0.4 else "Bajo"
        color = "🔴" if prob_cambio > 0.7 else "🟡" if prob_cambio > 0.4 else "🟢"
        st.metric(f"{color} Nivel de Riesgo", risk_level)
    
    with col3:
        confidence = "Alta" if abs(prob_cambio - 0.5) > 0.3 else "Media" if abs(prob_cambio - 0.5) > 0.15 else "Baja"
        st.metric("📊 Confianza", confidence)
    
    # Visualización de características TDA
    st.subheader("🔍 Características Topológicas Extraídas")
    
    fig_tda = plt.figure(figsize=(12, 4))
    plt.plot(X_tda_scaled.flatten(), 'b-', marker='o', linewidth=2, markersize=4)
    plt.title(f'Características TDA - Método: {tda_method}')
    plt.xlabel('Índice de Característica')
    plt.ylabel('Valor Normalizado')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_tda)
    
    # Interpretación de resultados
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
    
    # Recomendaciones
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
    
# Función principal de análisis TDA (para integrar en tu código de Streamlit)
def run_tda_analysis(serie, tda_method, fruta_nombre, selected_month, selected_year, months_dict):
    """
    Función principal que ejecuta todo el análisis TDA
    """
    try:
        # Mostrar diagrama de persistencia
        st.subheader("📊 Diagrama de Persistencia")
        
        with st.spinner("Calculando diagrama de persistencia..."):
            fig_persistence = plot_persistent_homology(serie.flatten(), method=tda_method)
            if fig_persistence:
                st.plotly_chart(fig_persistence, use_container_width=True)
            else:
                st.warning("No se pudo generar el diagrama de persistencia")
        
        # Análisis con CNN usando TDA
        st.subheader("🤖 Predicción de Cambios Abruptos con CNN + TDA")
        
        # Cargar modelo CNN
        model = load_cnn_model()
        
        if model is not None:
            with st.spinner("Aplicando transformación TDA y prediciendo..."):
                # Preparar datos para CNN usando TDA
                X_tda = prepare_cnn_data_with_tda(serie, method=tda_method)
                
                if X_tda is not None:
                    # Realizar predicción
                    prob_cambio, X_tda_scaled = make_cnn_prediction(model, X_tda, tda_method)
                    
                    if prob_cambio is not None:
                        # Mostrar resultados
                        display_tda_results(prob_cambio, X_tda_scaled, tda_method, 
                                          fruta_nombre, selected_month, selected_year, months_dict)
                    else:
                        st.error("No se pudo realizar la predicción con el modelo CNN")
                else:
                    st.error("No se pudieron extraer características TDA de la serie temporal")
        else:
            st.error("No se pudo cargar el modelo CNN desde GitHub")
            
    except Exception as e:
        st.error(f"Error en el análisis TDA: {e}")
        st.info("Verifica que tengas instaladas todas las librerías necesarias (giotto-tda, tensorflow, etc.)")

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
        
        # Ejecutar análisis TDA
        if len(serie) >= 10:  # Mínimo de datos necesarios
            try:
                # Ejecutar análisis TDA completo usando la función principal
                run_tda_analysis(serie, tda_method, fruta_nombre, selected_month, selected_year, months_dict)
                
            except Exception as e:
                st.error(f"Error en el análisis topológico: {e}")
                st.info("Verifica que tengas instaladas todas las librerías necesarias (giotto-tda, tensorflow, etc.)")
                
                # Mostrar detalles del error en modo debug
                with st.expander("🔍 Detalles del error (Debug)"):
                    import traceback
                    st.code(traceback.format_exc())
        
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
            
            **Interpretación de resultados:**
            
            - **Probabilidad > 70%**: Alto riesgo de cambios abruptos
            - **Probabilidad 40-70%**: Riesgo moderado, precaución recomendada
            - **Probabilidad < 40%**: Condiciones relativamente estables
            """)
        
        # === Comparación de métodos TDA ===
        if st.checkbox("🔄 Comparar métodos TDA"):
            st.subheader("📊 Comparación: Takens Embedding vs Sliding Window")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Takens Embedding:**")
                try:
                    with st.spinner("Calculando TE..."):
                        X_te = prepare_cnn_data_with_tda(serie, method='TE')
                        if X_te is not None and len(X_te) > 0:
                            st.write(f"✅ Características extraídas: {len(X_te[0])}")
                            st.write(f"📊 Rango: [{X_te.min():.3f}, {X_te.max():.3f}]")
                            st.write(f"📈 Desviación estándar: {X_te.std():.3f}")
                            
                            # Mostrar algunas estadísticas adicionales
                            te_stats = {
                                "Media": X_te.mean(),
                                "Mediana": np.median(X_te),
                                "Varianza": X_te.var()
                            }
                            
                            for stat_name, stat_value in te_stats.items():
                                st.write(f"• {stat_name}: {stat_value:.3f}")
                        else:
                            st.write("❌ Error al calcular TE")
                            
                except Exception as e:
                    st.write(f"❌ Error al calcular TE: {str(e)[:50]}...")
            
            with col2:
                st.write("**Sliding Window:**")
                try:
                    with st.spinner("Calculando SW..."):
                        X_sw = prepare_cnn_data_with_tda(serie, method='SW')
                        if X_sw is not None and len(X_sw) > 0:
                            st.write(f"✅ Características extraídas: {len(X_sw[0])}")
                            st.write(f"📊 Rango: [{X_sw.min():.3f}, {X_sw.max():.3f}]")
                            st.write(f"📈 Desviación estándar: {X_sw.std():.3f}")
                            
                            # Mostrar algunas estadísticas adicionales
                            sw_stats = {
                                "Media": X_sw.mean(),
                                "Mediana": np.median(X_sw),
                                "Varianza": X_sw.var()
                            }
                            
                            for stat_name, stat_value in sw_stats.items():
                                st.write(f"• {stat_name}: {stat_value:.3f}")
                        else:
                            st.write("❌ Error al calcular SW")
                            
                except Exception as e:
                    st.write(f"❌ Error al calcular SW: {str(e)[:50]}...")
            
            # Comparación visual si ambos métodos funcionan
            try:
                if st.button("📈 Mostrar comparación visual"):
                    with st.spinner("Generando comparación visual..."):
                        fig_comparison = plt.figure(figsize=(14, 8))
                        
                        # Subplot para TE
                        plt.subplot(2, 1, 1)
                        X_te_comp = prepare_cnn_data_with_tda(serie, method='TE')
                        if X_te_comp is not None:
                            scaler_te = MinMaxScaler()
                            X_te_scaled = scaler_te.fit_transform(X_te_comp.reshape(-1, 1)).flatten()
                            plt.plot(X_te_scaled, 'b-', marker='o', linewidth=2, markersize=3, label='Takens Embedding')
                            plt.title('Características TDA - Takens Embedding')
                            plt.ylabel('Valor Normalizado')
                            plt.grid(True, alpha=0.3)
                            plt.legend()
                        
                        # Subplot para SW
                        plt.subplot(2, 1, 2)
                        X_sw_comp = prepare_cnn_data_with_tda(serie, method='SW')
                        if X_sw_comp is not None:
                            scaler_sw = MinMaxScaler()
                            X_sw_scaled = scaler_sw.fit_transform(X_sw_comp.reshape(-1, 1)).flatten()
                            plt.plot(X_sw_scaled, 'r-', marker='s', linewidth=2, markersize=3, label='Sliding Window')
                            plt.title('Características TDA - Sliding Window')
                            plt.xlabel('Índice de Característica')
                            plt.ylabel('Valor Normalizado')
                            plt.grid(True, alpha=0.3)
                            plt.legend()
                        
                        plt.tight_layout()
                        st.pyplot(fig_comparison)
                        
            except Exception as e:
                st.error(f"Error en la comparación visual: {e}")
        
        # === Estadísticas de la serie temporal ===
        with st.expander("📊 Estadísticas de la serie temporal"):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📈 Precio Máximo", f"${data_filtered['price'].max():.2f}")
            
            with col2:
                st.metric("📉 Precio Mínimo", f"${data_filtered['price'].min():.2f}")
            
            with col3:
                st.metric("📊 Precio Promedio", f"${data_filtered['price'].mean():.2f}")
            
            with col4:
                st.metric("📋 Volatilidad", f"{data_filtered['price'].std():.2f}")
            
            # Información adicional
            st.markdown("---")
            price_change = data_filtered['price'].iloc[-1] - data_filtered['price'].iloc[0] if len(data_filtered) > 1 else 0
            price_change_pct = (price_change / data_filtered['price'].iloc[0] * 100) if len(data_filtered) > 1 and data_filtered['price'].iloc[0] != 0 else 0
            
            st.write(f"**Cambio en el período:** ${price_change:.2f} ({price_change_pct:+.1f}%)")
            st.write(f"**Número de observaciones:** {len(data_filtered)}")
            st.write(f"**Rango de fechas:** {data_filtered['report_date'].min().strftime('%Y-%m-%d')} a {data_filtered['report_date'].max().strftime('%Y-%m-%d')}")
        
        # === Configuración avanzada ===
        with st.expander("⚙️ Configuración avanzada TDA"):
            st.markdown("**Parámetros para Takens Embedding:**")
            
            col1, col2 = st.columns(2)
            with col1:
                custom_dimension = st.slider("Dimensión del embedding", 2, 10, 5)
                custom_delay = st.slider("Time delay", 1, 10, 5)
            
            with col2:
                custom_stride = st.slider("Stride", 1, 5, 2)
            
            st.markdown("**Parámetros para Sliding Window:**")
            max_window = min(30, len(serie) // 2)
            custom_window = st.slider("Tamaño de ventana", 3, max_window, min(10, max_window))
            
            if st.button("🔄 Aplicar configuración personalizada"):
                st.info("Aplicando configuración personalizada...")
                # Aquí podrías llamar a las funciones con parámetros personalizados
                # run_tda_analysis_custom(serie, tda_method, custom_params, ...)
                st.success("Configuración aplicada (funcionalidad en desarrollo)")
    
# === Footer ===
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px;">
    <p>© 2023 Intelica - Análisis de Datos Agrícolas</p>
    <p>Contacto: info@intelica.com | Tel: +123456789</p>
</div>
""", unsafe_allow_html=True)
