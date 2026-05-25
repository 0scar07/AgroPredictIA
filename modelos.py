"""
AgroPredict IA — Módulo: Modelos ML
Gestión, creación y persistencia de modelos de red neuronal (Keras).
Modelos: Rendimiento (regresión) y Riesgo de Plaga (regresión).
"""

import os
import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import MinMaxScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODEL_DIR, exist_ok=True)


class ModelManager:
    """
    Gestiona la creación, guardado y carga de modelos de ML.
    Soporta modelos de Keras (red neuronal) y scikit-learn (Random Forest, Gradient Boosting).
    """

    # ── Arquitecturas disponibles ─────────────────────────────────────────────
    ARQUITECTURAS = {
        "red_neuronal": "Red Neuronal (Keras)",
        "random_forest": "Random Forest (sklearn)",
        "gradient_boosting": "Gradient Boosting (sklearn)",
    }

    def __init__(self):
        self.modelos: dict = {}          # {nombre: modelo}
        self.scalers: dict = {}          # {nombre: scaler}
        self.metadata: dict = {}         # {nombre: {features, target, metricas, ...}}

    # ── Construcción de red neuronal ──────────────────────────────────────────

    def construir_red_neuronal(
        self,
        input_dim: int,
        n_capas: int = 2,
        dropout: float = 0.1,
        activacion: str = "relu",
    ) -> "keras.Sequential":
        """
        Construye una red neuronal totalmente conectada para regresión.

        Args:
            input_dim:  Número de variables de entrada.
            n_capas:    Número de capas ocultas (1, 2 o 3).
            dropout:    Tasa de dropout entre capas.
            activacion: Función de activación ('relu', 'tanh', 'elu').

        Returns:
            Modelo Keras compilado (pérdida MSE, optimizador Adam).
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow/Keras no está instalado. Usa: pip install tensorflow")

        mapeo_capas = {
            1: [64],
            2: [128, 64],
            3: [256, 128, 64],
        }
        unidades = mapeo_capas.get(n_capas, [128, 64])

        modelo = keras.Sequential(name="agropredict_nn")

        for i, units in enumerate(unidades):
            if i == 0:
                modelo.add(keras.layers.Dense(
                    units, input_shape=(input_dim,),
                    activation=activacion,
                    kernel_initializer="glorot_uniform",
                    name=f"dense_{i+1}"
                ))
            else:
                modelo.add(keras.layers.Dense(
                    units, activation=activacion,
                    name=f"dense_{i+1}"
                ))
            if dropout > 0 and i < len(unidades) - 1:
                modelo.add(keras.layers.Dropout(dropout, name=f"dropout_{i+1}"))

        # Capa de salida: neurona única (regresión, sin activación)
        modelo.add(keras.layers.Dense(1, name="output"))

        modelo.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="mse",
            metrics=["mae"]
        )
        return modelo

    def construir_random_forest(self, n_estimators: int = 100, max_depth: int = None):
        """Crea un modelo Random Forest para regresión."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn no instalado. Usa: pip install scikit-learn")
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )

    def construir_gradient_boosting(self, n_estimators: int = 100, lr: float = 0.1):
        """Crea un modelo Gradient Boosting para regresión."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn no instalado. Usa: pip install scikit-learn")
        return GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=lr,
            max_depth=4,
            random_state=42
        )

    # ── Persistencia ──────────────────────────────────────────────────────────

    def guardar_modelo(self, nombre: str, modelo, scaler=None, metadata: dict = None):
        """
        Guarda el modelo en disco.
        - Keras:   formato SavedModel en models/{nombre}/
        - sklearn: formato joblib en models/{nombre}.joblib
        """
        self.modelos[nombre]  = modelo
        self.scalers[nombre]  = scaler
        self.metadata[nombre] = metadata or {}

        path = os.path.join(MODEL_DIR, nombre)

        if TF_AVAILABLE and isinstance(modelo, keras.Model):
            modelo.save(path + "_keras")
        elif SKLEARN_AVAILABLE:
            try:
                import joblib
                joblib.dump({"modelo": modelo, "scaler": scaler, "meta": metadata},
                            path + ".joblib")
            except ImportError:
                pass  # joblib opcional

    def cargar_modelo(self, nombre: str):
        """Carga un modelo desde disco y lo almacena en memoria."""
        path = os.path.join(MODEL_DIR, nombre)

        keras_path = path + "_keras"
        joblib_path = path + ".joblib"

        if TF_AVAILABLE and os.path.exists(keras_path):
            modelo = keras.models.load_model(keras_path)
            self.modelos[nombre] = modelo
            return modelo

        if os.path.exists(joblib_path):
            try:
                import joblib
                obj = joblib.load(joblib_path)
                self.modelos[nombre] = obj["modelo"]
                self.scalers[nombre] = obj.get("scaler")
                self.metadata[nombre] = obj.get("meta", {})
                return obj["modelo"]
            except ImportError:
                return None

        return None

    def listar_modelos(self) -> list:
        """Devuelve lista de modelos guardados en disco."""
        modelos = []
        for f in os.listdir(MODEL_DIR):
            nombre = f.replace("_keras", "").replace(".joblib", "")
            if nombre not in [m["nombre"] for m in modelos]:
                meta = self.metadata.get(nombre, {})
                modelos.append({
                    "nombre":   nombre,
                    "metricas": meta.get("metricas", {}),
                    "features": meta.get("features", []),
                    "target":   meta.get("target", ""),
                    "tipo":     meta.get("tipo", "desconocido"),
                })
        return modelos

    def eliminar_modelo(self, nombre: str) -> bool:
        """Elimina un modelo de disco y memoria."""
        import shutil
        eliminado = False
        keras_path  = os.path.join(MODEL_DIR, nombre + "_keras")
        joblib_path = os.path.join(MODEL_DIR, nombre + ".joblib")

        if os.path.exists(keras_path):
            shutil.rmtree(keras_path); eliminado = True
        if os.path.exists(joblib_path):
            os.remove(joblib_path); eliminado = True

        self.modelos.pop(nombre, None)
        self.scalers.pop(nombre, None)
        self.metadata.pop(nombre, None)
        return eliminado

    # ── Utilidades de normalización ───────────────────────────────────────────

    @staticmethod
    def crear_scaler() -> "MinMaxScaler":
        """Retorna un nuevo MinMaxScaler de sklearn."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn requerido para normalización avanzada")
        return MinMaxScaler()

    @staticmethod
    def normalizar_manual(valores: list) -> tuple:
        """
        Normalización min-max sin sklearn.
        Returns:
            (valores_normalizados, min_val, max_val)
        """
        arr = np.array(valores, dtype=float)
        mn, mx = arr.min(), arr.max()
        if mx == mn:
            return arr * 0, mn, mx
        return (arr - mn) / (mx - mn), mn, mx

    @staticmethod
    def desnormalizar(valor: float, mn: float, mx: float) -> float:
        """Invierte la normalización min-max."""
        return float(valor * (mx - mn) + mn)

    # ── Resumen de modelo ─────────────────────────────────────────────────────

    def resumen(self, nombre: str) -> dict:
        """Devuelve un resumen del modelo cargado."""
        modelo = self.modelos.get(nombre)
        if modelo is None:
            return {"error": f"Modelo '{nombre}' no encontrado"}

        info = {"nombre": nombre, "tipo": type(modelo).__name__}

        if TF_AVAILABLE and isinstance(modelo, keras.Model):
            info["parametros"]   = modelo.count_params()
            info["capas"]        = len(modelo.layers)
            info["arquitectura"] = [
                {"nombre": l.name, "tipo": type(l).__name__,
                 "unidades": getattr(l, "units", None)}
                for l in modelo.layers
            ]

        info["metricas"] = self.metadata.get(nombre, {}).get("metricas", {})
        return info
