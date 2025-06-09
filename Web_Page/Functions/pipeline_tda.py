import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from ripser import Rips
from kmapper import KeplerMapper, Cover
from sklearn.cluster import KMeans
import kmapper as km
from scipy.signal import find_peaks
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
import json
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

class DataLoader(BaseEstimator, TransformerMixin):
    """Carga y prepara los datos iniciales"""
    def __init__(self, fruits=None, features=None):
        self.fruits = fruits
        self.features = features
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, path):
        df = pd.read_csv(path)
        try:
            df_fruits = df[df['commodity'].isin(self.fruits)][self.features]
            scaler = StandardScaler()
            X = scaler.fit_transform(df_fruits)
            return X
        except Exception as e:
            print(f"Error: {e}")
            raise NameError(f"{self.features} o {self.fruits} no se encuentran en el dataframe")

class TopologicalAnalysis(BaseEstimator, TransformerMixin):
    """Realiza análisis topológico de datos"""
    def __init__(self, plot_persistence=True):
        self.plot_persistence = plot_persistence
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if self.plot_persistence:
            self._plot_persistence(X)
        return X
    
    def _plot_persistence(self, X):
        rips = Rips()
        diagrams = rips.fit_transform(X)
        rips.plot(diagrams, show=False)
        plt.title('Diagramas de Persistencia para Mercado Agrícola')
        plt.show()

class MapperTransformer(BaseEstimator, TransformerMixin):
    """Aplica el algoritmo Mapper para reducción de dimensionalidad"""
    def __init__(self, n_cubes=4, perc_overlap=0.43, n_clusters=3):
        self.n_cubes = n_cubes
        self.perc_overlap = perc_overlap
        self.n_clusters = n_clusters
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        mapper = KeplerMapper(verbose=1)
        graph = mapper.map(
            X,
            cover=Cover(n_cubes=self.n_cubes, perc_overlap=self.perc_overlap),
            clusterer=KMeans(n_clusters=self.n_clusters)
        )
        mapper.visualize(
            graph,
            path_html="mercado_agricola_mapper.html",
            title="Análisis Topológico de Mercado Agrícola"
        )
        return graph

class TimeSeriesAnalyzer(BaseEstimator, TransformerMixin):
    """Analiza series temporales y detecta puntos de inflexión"""
    def __init__(self, price_columns):
        self.price_columns = price_columns
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, data):
        peaks = {}
        for producto in self.price_columns:
            pks, _ = find_peaks(np.gradient(data[producto]), height=0.5)
            peaks[producto] = pks
            
            plt.figure()
            plt.plot(data[producto], color="orchid")
            plt.scatter(pks, data[producto].iloc[pks], c='maroon', label='Puntos de inflexión')
            plt.xlabel('Tiempo')
            plt.title(f"Puntos de inflexión en {producto}")
            plt.show()
        
        return peaks

class ChangeClassifier(BaseEstimator, TransformerMixin):
    """Clasifica cambios abruptos en los datos"""
    def __init__(self, feature_columns, test_size=100):
        self.feature_columns = feature_columns
        self.test_size = test_size
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, data, peaks):
        data['cambio_abrupto'] = 0
        for p in peaks:
            data.loc[p, 'cambio_abrupto'] = 1
            
        X = data[self.feature_columns]
        y = data['cambio_abrupto']
        
        X_train, X_test = X[:-self.test_size], X[-self.test_size:]
        y_train, y_test = y[:-self.test_size], y[-self.test_size:]
        
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        print(f"Precisión en detección de cambios: {score:.2f}")
        
        return model

class PriceVolumeAnalyzer(BaseEstimator, TransformerMixin):
    """Analiza la relación precio-volumen"""
    def __init__(self, products):
        self.products = products
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, data):
        results = {}
        models = {}
        
        for producto in self.products:
            X = data[[f'volumen_{producto}', 'temperatura', 'humedad']]
            y = data[f'precio_{producto}']
            
            model = LinearRegression().fit(X, y)
            results[producto] = {
                'coef_volumen': model.coef_[0],
                'score': model.score(X, y)
            }
            models[producto] = model
            
        print("Relación precio-volumen:")
        print(json.dumps(results, indent=2))
        
        return models

class VisualizationPipeline:
    """Pipeline para visualización de resultados"""
    def __init__(self, products):
        self.products = products
        
    def visualize_relationships(self, data, models):
        sns.set(style="whitegrid")
        plt.figure(figsize=(15, 10))
        
        for i, producto in enumerate(self.products, 1):
            X = data[[f'volumen_{producto}']].values
            y = data[f'precio_{producto}'].values
            model = models[producto]
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            
            plt.subplot(2, 2, i)
            sns.scatterplot(x=data[f'volumen_{producto}'], y=data[f'precio_{producto}'], 
                          alpha=0.6, label='Datos reales')
            plt.plot(X, y_pred, color='red', linewidth=2, 
                   label=f'Regresión (R²={r2:.2f})')
            plt.title(f'Relación Volumen-Precio: {producto.capitalize()}')
            plt.xlabel(f'Volumen de {producto}')
            plt.ylabel(f'Precio de {producto}')
            plt.legend()
            
            equation = f'y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}'
            plt.text(0.05, 0.95, equation, transform=plt.gca().transAxes,
                   fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
            
        plt.tight_layout()
        plt.show()

# Pipeline principal
class AgriculturalMarketPipeline:
    def __init__(self, config):
        self.config = config
        
    def run(self, data_path):
        # 1. Carga de datos
        data_loader = DataLoader(fruits=self.config['fruits'], features=self.config['features'])
        X = data_loader.transform(data_path)
        
        # 2. Análisis topológico
        topo_analysis = TopologicalAnalysis(plot_persistence=True)
        X = topo_analysis.transform(X)
        
        # 3. Mapper algorithm
        mapper = MapperTransformer(
            n_cubes=self.config.get('n_cubes', 4),
            perc_overlap=self.config.get('perc_overlap', 0.43),
            n_clusters=self.config.get('n_clusters', 3)
        )
        graph = mapper.transform(X)
        
        # Cargar datos completos para análisis temporal
        full_data = pd.read_csv(data_path)
        
        # 4. Análisis de series temporales
        ts_analyzer = TimeSeriesAnalyzer(price_columns=self.config['price_columns'])
        peaks = ts_analyzer.transform(full_data)
        
        # 5. Clasificación de cambios
        classifier = ChangeClassifier(feature_columns=self.config['classifier_features'])
        model = classifier.transform(full_data, peaks)
        
        # 6. Análisis precio-volumen
        pv_analyzer = PriceVolumeAnalyzer(products=self.config['products'])
        models = pv_analyzer.transform(full_data)
        
        # 7. Visualización
        visualizer = VisualizationPipeline(products=self.config['products'])
        visualizer.visualize_relationships(full_data, models)
        
        return {
            'graph': graph,
            'peaks': peaks,
            'classifier': model,
            'price_models': models
        }

# Configuración
config = {
    'fruits': ['frambuesa', 'aguacate', 'chile'],
    'features': ['precio', 'volumen', 'temperatura', 'humedad'],
    'price_columns': ['precio_frambuesa', 'precio_aguacate', 'precio_chile'],
    'classifier_features': ['precio', 'volumen', 'temperatura', 'humedad'],
    'products': ['frambuesa', 'aguacate', 'chile'],
    'n_cubes': 4,
    'perc_overlap': 0.43,
    'n_clusters': 3
}

# Ejecutar pipeline
pipeline = AgriculturalMarketPipeline(config)
results = pipeline.run('datos_mercado_agricola.csv')