import numpy as np 
import pandas as pd 
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from gtda.time_series import SlidingWindow
from gtda.diagrams import PersistenceEntropy, Scaler
from gtda.homology import VietorisRipsPersistence
from gtda.metaestimators import CollectionTransformer
from gtda.pipeline import Pipeline
from gtda.time_series import SlidingWindow
from gtda.time_series import TakensEmbedding, SingleTakensEmbedding
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from ripser import Rips
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.preprocessing import FunctionTransformer
from ripser import Rips
from kmapper import KeplerMapper, Cover
from sklearn.cluster import KMeans
import kmapper as km
from scipy.signal import find_peaks
from gtda.plotting import plot_diagram


def group_series(df, feature): 
    # Inicializamos listas
    X = []
    y = []

    # Agrupamos por serie_id
    for serie_id, group in df.groupby('serie_id'):
        # Ordenamos por dia por si acaso
        group = group.sort_values('fecha')
        
        # Extraemos la serie temporal (precio_frambuesa)
        serie_precio = group[feature].values
        
        # Guardamos en X
        X.append(serie_precio)
        
    
        etiqueta = group['has_market_change'].iloc[0]
        y.append(etiqueta)

    
    return X, y

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
    

def mapper(df, features=['precio_frambuesa', 'volumen_frambuesa'], n_cubes = 4, perc_overlap = 0.43, n_clusters = 3): 
    
    X_mapper = df[features].values


    # Crear objeto Mapper
    mapper = KeplerMapper(verbose=1)


    # Aplicar Mapper
    graph = mapper.map(X_mapper, 
                    cover=Cover(n_cubes=n_cubes, perc_overlap=perc_overlap),
                    clusterer=KMeans(n_clusters=n_clusters))

    # Visualizar
    mapper.visualize(graph, 
                    path_html="mercado_agricola_mapper.html",
                    title="Análisis Topológico de Mercado Agrícola")



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
        fig.show()

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
        fig.show()

       

