"""
AgroPredict IA — Módulo: Entrenamiento
Lógica completa de entrenamiento de modelos ML a partir de datos CSV.
Soporta: red neuronal (Keras), Random Forest y Gradient Boosting.
Genera métricas: R², MAE, RMSE, historial de pérdida por época.
"""

import numpy as np
from modelos import ModelManager

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class TrainingEngine:
    """
    Motor de entrenamiento para modelos agronómicos.

    Flujo:
        1. Recibe lista de filas (dicts del CSV) y configuración.
        2. Prepara y normaliza el dataset.
        3. Entrena uno o dos modelos (rendimiento y/o plaga).
        4. Calcula métricas sobre el conjunto de prueba.
        5. Guarda los modelos vía ModelManager.
        6. Retorna resumen con métricas e historial de pérdida.
    """

    # Mapeos automáticos de nombre de columna → variable
    FEATURE_KEYS = {
        "temperatura": ["temp", "temperatura", "temperature"],
        "humedad":     ["humedad", "hum", "humidity"],
        "lluvia_mm":   ["lluvia", "rain", "precipitation", "lluvia_mm"],
        "ndvi":        ["ndvi"],
        "humedad_suelo": ["humedad_suelo", "soil", "hsuelo", "soil_moisture"],
    }
    TARGET_KEYS = {
        "rendimiento": ["rendimiento", "rend", "yield", "produccion"],
        "plaga":       ["plaga", "riesgo", "risk", "plague"],
    }

    def __init__(self):
        self.manager = ModelManager()
        self.historial_loss: dict = {}       # {nombre_modelo: [loss por época]}
        self.predicciones_train: dict = {}   # {nombre_modelo: (real, predicho)}

    # ── Punto de entrada principal ────────────────────────────────────────────

    def entrenar(self, rows: list, config: dict) -> dict:
        """
        Entrena los modelos según la configuración enviada desde el frontend.

        Args:
            rows:   Lista de dicts (cada fila del CSV subido).
            config: {
                "tipo_modelo": "red_neuronal"|"random_forest"|"gradient_boosting",
                "epocas":       int   (solo redes neuronales),
                "lr":           float (solo redes neuronales),
                "n_capas":      int   (1-3, solo redes neuronales),
                "batch_size":   int   (solo redes neuronales),
                "n_estimators": int   (solo sklearn),
                "test_size":    float (fracción para prueba, default 0.2),
                "feat_cols":    dict  {variable: columna_csv}  (opcional, para mapeo manual),
                "target_rend":  str   (columna CSV de rendimiento, opcional),
                "target_plaga": str   (columna CSV de plaga, opcional),
            }

        Returns:
            Dict con métricas, historial de pérdida e información de modelos entrenados.
        """
        if not rows:
            return {"error": "No hay datos para entrenar"}

        columnas = list(rows[0].keys())

        # ── 1. Mapear columnas ──────────────────────────────────────────────
        feat_cols = self._mapear_features(columnas, config.get("feat_cols", {}))
        target_rend  = config.get("target_rend")  or self._encontrar_col(columnas, self.TARGET_KEYS["rendimiento"])
        target_plaga = config.get("target_plaga") or self._encontrar_col(columnas, self.TARGET_KEYS["plaga"])

        if not feat_cols:
            return {"error": "No se encontraron columnas de variables de entrada válidas"}
        if not target_rend and not target_plaga:
            return {"error": "No se encontró columna objetivo (rendimiento o plaga)"}

        # ── 2. Preparar dataset ─────────────────────────────────────────────
        ds = self._preparar_dataset(rows, feat_cols, target_rend, target_plaga)
        if isinstance(ds, str):
            return {"error": ds}

        tipo         = config.get("tipo_modelo", "red_neuronal")
        test_size    = float(config.get("test_size", 0.2))
        resultado    = {"modelos_entrenados": [], "metricas": {}, "historial": {}, "features": feat_cols}

        # ── 3. Entrenar modelo de Rendimiento ───────────────────────────────
        if target_rend and ds["y_rend"] is not None:
            info_rend = self._entrenar_modelo(
                X=ds["X_norm"], y=ds["y_rend"],
                nombre="agropredict_rendimiento",
                tipo=tipo, config=config,
                test_size=test_size,
                y_raw=ds["y_rend_raw"],
                y_norm_params=ds["rend_norm"]
            )
            resultado["modelos_entrenados"].append("rendimiento")
            resultado["metricas"]["rendimiento"] = info_rend["metricas"]
            resultado["historial"]["rendimiento"] = info_rend.get("historial", [])

        # ── 4. Entrenar modelo de Plaga ─────────────────────────────────────
        if target_plaga and ds["y_plaga"] is not None:
            info_plaga = self._entrenar_modelo(
                X=ds["X_norm"], y=ds["y_plaga"],
                nombre="agropredict_plaga",
                tipo=tipo, config=config,
                test_size=test_size,
                y_raw=ds["y_plaga_raw"],
                y_norm_params=ds["plaga_norm"]
            )
            resultado["modelos_entrenados"].append("plaga")
            resultado["metricas"]["plaga"] = info_plaga["metricas"]
            resultado["historial"]["plaga"] = info_plaga.get("historial", [])

        resultado["n_muestras"] = ds["n"]
        resultado["n_features"] = len(feat_cols)
        resultado["tipo_modelo"] = tipo
        return resultado

    # ── Preparación del dataset ───────────────────────────────────────────────

    def _preparar_dataset(self, rows: list, feat_cols: list, target_rend: str, target_plaga: str) -> dict:
        """
        Filtra filas válidas, extrae matrices X e y, aplica normalización min-max.
        """
        filas_validas = [
            r for r in rows
            if all(self._es_numerico(r.get(c)) for c in feat_cols)
            and (not target_rend  or self._es_numerico(r.get(target_rend)))
            and (not target_plaga or self._es_numerico(r.get(target_plaga)))
        ]

        if len(filas_validas) < 10:
            return f"Solo {len(filas_validas)} filas válidas. Se necesitan al menos 10."

        X_raw = np.array([[float(r[c]) for c in feat_cols] for r in filas_validas])

        # Normalizar cada columna de entrada individualmente
        X_norm = np.zeros_like(X_raw)
        feat_params = []
        for i in range(X_raw.shape[1]):
            col = X_raw[:, i]
            mn, mx = col.min(), col.max()
            X_norm[:, i] = 0 if mx == mn else (col - mn) / (mx - mn)
            feat_params.append({"col": feat_cols[i], "min": mn, "max": mx})

        # Targets
        y_rend, rend_norm, y_rend_raw = None, {}, None
        if target_rend:
            y_rend_raw = np.array([float(r[target_rend]) for r in filas_validas])
            y_norm, mn, mx = ModelManager.normalizar_manual(y_rend_raw)
            y_rend = y_norm
            rend_norm = {"min": float(mn), "max": float(mx)}

        y_plaga, plaga_norm, y_plaga_raw = None, {}, None
        if target_plaga:
            y_plaga_raw = np.array([float(r[target_plaga]) for r in filas_validas])
            y_norm, mn, mx = ModelManager.normalizar_manual(y_plaga_raw)
            y_plaga = y_norm
            plaga_norm = {"min": float(mn), "max": float(mx)}

        return {
            "X_norm": X_norm, "feat_params": feat_params,
            "y_rend": y_rend, "y_rend_raw": y_rend_raw, "rend_norm": rend_norm,
            "y_plaga": y_plaga, "y_plaga_raw": y_plaga_raw, "plaga_norm": plaga_norm,
            "n": len(filas_validas),
        }

    # ── Entrenamiento individual ──────────────────────────────────────────────

    def _entrenar_modelo(
        self, X: np.ndarray, y: np.ndarray,
        nombre: str, tipo: str, config: dict,
        test_size: float, y_raw: np.ndarray, y_norm_params: dict
    ) -> dict:
        """Entrena un solo modelo y retorna sus métricas e historial."""

        # Split train/test
        if SKLEARN_AVAILABLE:
            X_tr, X_te, y_tr, y_te, y_tr_raw, y_te_raw = train_test_split(
                X, y, y_raw, test_size=test_size, random_state=42)
        else:
            split = int(len(X) * (1 - test_size))
            X_tr, X_te = X[:split], X[split:]
            y_tr, y_te = y[:split], y[split:]
            y_tr_raw, y_te_raw = y_raw[:split], y_raw[split:]

        historial = []

        if tipo == "red_neuronal":
            if not TF_AVAILABLE:
                return {"error": "TensorFlow no disponible", "metricas": {}}

            epocas     = int(config.get("epocas", 80))
            lr         = float(config.get("lr", 0.01))
            n_capas    = int(config.get("n_capas", 2))
            batch_size = int(config.get("batch_size", 16))

            modelo = self.manager.construir_red_neuronal(
                input_dim=X_tr.shape[1],
                n_capas=n_capas
            )
            # Recompila con el lr especificado
            modelo.compile(optimizer=keras.optimizers.Adam(lr), loss="mse", metrics=["mae"])

            class LossLogger(keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    historial.append(float(logs.get("loss", 0)))

            modelo.fit(
                X_tr, y_tr.reshape(-1, 1),
                epochs=epocas,
                batch_size=batch_size,
                verbose=0,
                callbacks=[LossLogger()],
                shuffle=True
            )

            y_pred_norm = modelo.predict(X_te, verbose=0).flatten()
            y_pred = ModelManager.desnormalizar(y_pred_norm, y_norm_params["min"], y_norm_params["max"])

        else:
            # sklearn models
            if tipo == "random_forest":
                modelo = self.manager.construir_random_forest(
                    n_estimators=int(config.get("n_estimators", 100))
                )
            else:
                modelo = self.manager.construir_gradient_boosting(
                    n_estimators=int(config.get("n_estimators", 100)),
                    lr=float(config.get("lr", 0.1))
                )
            modelo.fit(X_tr, y_tr_raw)
            y_pred = modelo.predict(X_te)
            # Simular historial para compatibilidad con frontend
            historial = [float(np.mean((y_pred - y_te_raw)**2))] * 10

        # ── Métricas ────────────────────────────────────────────────────────
        metricas = self._calcular_metricas(y_te_raw, y_pred)

        # Guardar modelo
        self.manager.guardar_modelo(
            nombre=nombre, modelo=modelo,
            metadata={"metricas": metricas, "target": nombre.split("_")[-1], "tipo": tipo}
        )

        return {"metricas": metricas, "historial": historial}

    # ── Métricas ──────────────────────────────────────────────────────────────

    @staticmethod
    def _calcular_metricas(y_real: np.ndarray, y_pred: np.ndarray) -> dict:
        """Calcula R², MAE y RMSE."""
        y_real = np.array(y_real, dtype=float)
        y_pred = np.array(y_pred, dtype=float)

        if SKLEARN_AVAILABLE:
            r2   = float(r2_score(y_real, y_pred))
            mae  = float(mean_absolute_error(y_real, y_pred))
            rmse = float(np.sqrt(mean_squared_error(y_real, y_pred)))
        else:
            # Cálculo manual
            media  = y_real.mean()
            ss_res = np.sum((y_real - y_pred) ** 2)
            ss_tot = np.sum((y_real - media) ** 2)
            r2   = float(1 - ss_res / ss_tot) if ss_tot != 0 else 1.0
            mae  = float(np.mean(np.abs(y_real - y_pred)))
            rmse = float(np.sqrt(np.mean((y_real - y_pred) ** 2)))

        return {
            "r2":   round(max(0.0, min(1.0, r2)), 4),
            "r2_pct": round(max(0.0, min(100.0, r2 * 100)), 2),
            "mae":  round(mae, 4),
            "rmse": round(rmse, 4),
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _mapear_features(self, columnas: list, manual: dict) -> list:
        """Detecta automáticamente columnas de entrada numéricas útiles."""
        encontradas = []
        for var, claves in self.FEATURE_KEYS.items():
            col = manual.get(var) or self._encontrar_col(columnas, claves)
            if col:
                encontradas.append(col)
        return encontradas

    @staticmethod
    def _encontrar_col(columnas: list, claves: list) -> str | None:
        for c in columnas:
            if any(k in c.lower() for k in claves):
                return c
        return None

    @staticmethod
    def _es_numerico(v) -> bool:
        try:
            float(v)
            return True
        except (TypeError, ValueError):
            return False
