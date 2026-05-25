"""
AgroPredict IA — Módulo: Predicciones
"""

import numpy as np
from modelos import ModelManager

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class PredictionEngine:

    # ─────────────────────────────────────────────
    # RANGOS POR CULTIVO
    # ─────────────────────────────────────────────

    RANGOS_CULTIVO = {

        "maiz": {
            "t": (18, 30),
            "h": (50, 80),
            "ndvi_opt": 0.70
        },

        "arroz": {
            "t": (20, 35),
            "h": (70, 95),
            "ndvi_opt": 0.75
        },

        "papa": {
            "t": (10, 22),
            "h": (60, 85),
            "ndvi_opt": 0.68
        },

        "platano": {
            "t": (20, 32),
            "h": (65, 90),
            "ndvi_opt": 0.78
        },

        "cafe": {
            "t": (17, 26),
            "h": (70, 85),
            "ndvi_opt": 0.72
        },

        "default": {
            "t": (18, 30),
            "h": (55, 85),
            "ndvi_opt": 0.65
        }
    }

    # ─────────────────────────────────────────────
    # RENDIMIENTO BASE
    # ─────────────────────────────────────────────

    RENDIMIENTO_BASE = {

        "maiz": 6.5,
        "arroz": 5.5,
        "papa": 22.0,
        "platano": 25.0,
        "cafe": 2.8,
        "default": 5.0

    }

    # ─────────────────────────────────────────────
    # INIT
    # ─────────────────────────────────────────────

    def __init__(self):

        self.manager = ModelManager()

        self._modelo_rend = None
        self._modelo_plaga = None

        self._cargar_modelos()

    # ─────────────────────────────────────────────
    # CARGAR MODELOS
    # ─────────────────────────────────────────────

    def _cargar_modelos(self):

        self._modelo_rend = self.manager.cargar_modelo(
            "agropredict_rendimiento"
        )

        self._modelo_plaga = self.manager.cargar_modelo(
            "agropredict_plaga"
        )

    # ─────────────────────────────────────────────
    # PREDICCIÓN PRINCIPAL
    # ─────────────────────────────────────────────

    def predecir(self, datos: dict):

        temp = float(datos.get("temperatura", 25))
        humedad = float(datos.get("humedad", 70))
        lluvia = float(datos.get("lluvia_mm", 80))
        ndvi = float(datos.get("ndvi", 0.6))
        hsuelo = float(datos.get("humedad_suelo", 50))

        cultivo_raw = str(
            datos.get("cultivo", "")
        ).lower()

        cultivo = self._normalizar_cultivo(cultivo_raw)

        print("======== DATOS PREDICCIÓN ========")
        print("TEMP:", temp)
        print("HUMEDAD:", humedad)
        print("LLUVIA:", lluvia)
        print("NDVI:", ndvi)
        print("HSUELO:", hsuelo)
        print("CULTIVO:", cultivo)

        # ─────────────────────────
        # MODELO IA
        # ─────────────────────────

        if self._modelo_rend or self._modelo_plaga:

            resultado = self._predecir_con_modelo(
                temp,
                humedad,
                lluvia,
                ndvi,
                hsuelo,
                cultivo
            )

            resultado["modo"] = "modelo_entrenado"

        # ─────────────────────────
        # HEURÍSTICO
        # ─────────────────────────

        else:

            resultado = self._predecir_heuristico(
                temp,
                humedad,
                lluvia,
                ndvi,
                hsuelo,
                cultivo
            )

            resultado["modo"] = "heuristico"

        resultado["recomendaciones"] = self._generar_recomendaciones(
            resultado,
            cultivo,
            temp,
            humedad,
            lluvia,
            ndvi
        )

        resultado["nivel_riesgo"] = self._nivel_riesgo(
            resultado["plaga"]
        )

        resultado["estres_hidrico"] = self._estres_hidrico(
            humedad,
            hsuelo,
            lluvia
        )

        return resultado

    # ─────────────────────────────────────────────
    # PREDICCIÓN CON IA
    # ─────────────────────────────────────────────

    def _predecir_con_modelo(
        self,
        temp,
        humedad,
        lluvia,
        ndvi,
        hsuelo,
        cultivo
    ):

        x = np.array([
            [temp, humedad, lluvia, ndvi, hsuelo]
        ])

        print("INPUT MODELO:", x)

        rend = None
        plaga = None

        # ─────────────────────────
        # RENDIMIENTO
        # ─────────────────────────

        if self._modelo_rend:

            try:

                if TF_AVAILABLE and isinstance(
                    self._modelo_rend,
                    keras.Model
                ):

                    rend_norm = float(
                        self._modelo_rend.predict(
                            x,
                            verbose=0
                        )[0][0]
                    )

                else:

                    rend_norm = float(
                        self._modelo_rend.predict(x)[0]
                    )

                print("REND NORM:", rend_norm)

                meta = self.manager.metadata.get(
                    "agropredict_rendimiento",
                    {}
                )

                mn = meta.get("min", 0)
                mx = meta.get("max", 25)

                rend = ModelManager.desnormalizar(
                    rend_norm,
                    mn,
                    mx
                )

                rend = max(0.0, rend)

            except Exception as e:

                print("ERROR RENDIMIENTO:", e)

        # ─────────────────────────
        # PLAGA
        # ─────────────────────────

        if self._modelo_plaga:

            try:

                if TF_AVAILABLE and isinstance(
                    self._modelo_plaga,
                    keras.Model
                ):

                    plaga_norm = float(
                        self._modelo_plaga.predict(
                            x,
                            verbose=0
                        )[0][0]
                    )

                else:

                    plaga_norm = float(
                        self._modelo_plaga.predict(x)[0]
                    )

                print("PLAGA NORM:", plaga_norm)

                meta = self.manager.metadata.get(
                    "agropredict_plaga",
                    {}
                )

                mn = meta.get("min", 0)
                mx = meta.get("max", 100)

                plaga = ModelManager.desnormalizar(
                    plaga_norm,
                    mn,
                    mx
                )

                plaga = max(
                    0.0,
                    min(100.0, plaga)
                )

            except Exception as e:

                print("ERROR PLAGA:", e)

        # ─────────────────────────
        # FALLBACK
        # ─────────────────────────

        heur = self._predecir_heuristico(
            temp,
            humedad,
            lluvia,
            ndvi,
            hsuelo,
            cultivo
        )

        if rend is None:
            rend = heur["rendimiento"]

        if plaga is None:
            plaga = heur["plaga"]

        rend = round(rend, 2)
        plaga = round(plaga, 1)

        conf = self._calcular_confianza_modelo(
            temp,
            humedad,
            lluvia,
            ndvi
        )

        return {

            "rendimiento": rend,
            "plaga": plaga,
            "confianza": conf

        }

    # ─────────────────────────────────────────────
    # HEURÍSTICO
    # ─────────────────────────────────────────────

    def _predecir_heuristico(
        self,
        temp,
        humedad,
        lluvia,
        ndvi,
        hsuelo,
        cultivo
    ):

        rangos = self.RANGOS_CULTIVO.get(
            cultivo,
            self.RANGOS_CULTIVO["default"]
        )

        base = self.RENDIMIENTO_BASE.get(
            cultivo,
            self.RENDIMIENTO_BASE["default"]
        )

        # ─────────────────────────
        # FACTOR TEMPERATURA
        # ─────────────────────────

        t_min, t_max = rangos["t"]

        t_opt = (t_min + t_max) / 2

        dist_temp = abs(temp - t_opt) / (
            (t_max - t_min) / 2
        )

        factor_temp = max(
            0.3,
            1.0 - dist_temp * 0.6
        )

        # ─────────────────────────
        # FACTOR HUMEDAD
        # ─────────────────────────

        h_min, h_max = rangos["h"]

        if h_min <= humedad <= h_max:

            factor_hum = 1.0

        elif humedad < h_min:

            factor_hum = max(
                0.4,
                humedad / h_min
            )

        else:

            factor_hum = max(
                0.5,
                1.0 - (
                    (humedad - h_max) / h_max * 0.5
                )
            )

        # ─────────────────────────
        # FACTOR NDVI
        # ─────────────────────────

        ndvi_opt = rangos["ndvi_opt"]

        factor_ndvi = min(
            1.0,
            ndvi / ndvi_opt
        ) ** 0.5

        # ─────────────────────────
        # FACTOR LLUVIA
        # ─────────────────────────

        if lluvia < 20:

            factor_lluvia = 0.6

        elif lluvia < 60:

            factor_lluvia = 0.8 + lluvia / 300

        elif lluvia <= 130:

            factor_lluvia = 1.0

        else:

            factor_lluvia = max(
                0.6,
                1.0 - (
                    (lluvia - 130) / 500
                )
            )

        # ─────────────────────────
        # RENDIMIENTO
        # ─────────────────────────

        rendimiento = (
            base *
            factor_temp *
            factor_hum *
            factor_ndvi *
            factor_lluvia
        )

        ruido = np.random.normal(
            0,
            base * 0.05
        )

        rendimiento = round(
            max(
                0.0,
                rendimiento + ruido
            ),
            2
        )

        # ─────────────────────────
        # PLAGA — fórmula lineal calibrada
        # ─────────────────────────

        plaga_score = (
            max(0, (temp   - 20))  * 2.0   +   # >20°C aumenta riesgo
            max(0, (humedad - 60)) * 0.5   -   # >60% humedad aumenta riesgo
            lluvia * 0.08                  -   # lluvia lava esporas
            hsuelo * 0.15                  +   # suelo húmedo reduce riesgo
            (1 - ndvi) * 30                    # planta débil = más vulnerable
        )

        plaga = plaga_score + np.random.normal(0, 2)

        plaga = round(
            max(
                0.0,
                min(100.0, plaga)
            ),
            1
        )

        # ─────────────────────────
        # CONFIANZA
        # ─────────────────────────

        confianza = self._calcular_confianza_heuristica(
            temp,
            humedad,
            lluvia,
            ndvi,
            cultivo
        )

        return {

            "rendimiento": rendimiento,
            "plaga": plaga,
            "confianza": confianza

        }

    # ─────────────────────────────────────────────
    # CONFIANZA MODELO
    # ─────────────────────────────────────────────

    def _calcular_confianza_modelo(
        self,
        temp,
        humedad,
        lluvia,
        ndvi
    ):

        scores = [

            1.0 if 15 <= temp <= 38 else 0.7,

            1.0 if 30 <= humedad <= 98 else 0.7,

            1.0 if 0 <= lluvia <= 200 else 0.75,

            1.0 if 0.2 <= ndvi <= 0.95 else 0.8

        ]

        return round(

            min(
                95,
                max(
                    55,
                    sum(scores) /
                    len(scores) * 90
                )
            ),

            1
        )

    # ─────────────────────────────────────────────
    # CONFIANZA HEURÍSTICA
    # ─────────────────────────────────────────────

    def _calcular_confianza_heuristica(
        self,
        temp,
        humedad,
        lluvia,
        ndvi,
        cultivo
    ):

        rangos = self.RANGOS_CULTIVO.get(
            cultivo,
            self.RANGOS_CULTIVO["default"]
        )

        t_min, t_max = rangos["t"]

        h_min, h_max = rangos["h"]

        en_rango = sum([

            t_min <= temp <= t_max,

            h_min <= humedad <= h_max,

            20 <= lluvia <= 150,

            0.35 <= ndvi <= 0.92

        ])

        return round(

            50 + en_rango * 8 +

            np.random.uniform(-3, 3),

            1
        )

    # ─────────────────────────────────────────────
    # UTILIDADES
    # ─────────────────────────────────────────────

    @staticmethod
    def _nivel_riesgo(plaga):

        if plaga > 70:
            return "CRÍTICO"

        if plaga > 45:
            return "ALTO"

        if plaga > 25:
            return "MEDIO"

        return "BAJO"

    @staticmethod
    def _estres_hidrico(
        humedad,
        hsuelo,
        lluvia
    ):

        score = (

            (humedad / 100 * 0.4) +

            (hsuelo / 100 * 0.4) +

            (min(lluvia, 120) / 120 * 0.2)

        )

        if score < 0.35:
            return "Severo"

        if score < 0.55:
            return "Moderado"

        if score < 0.75:
            return "Leve"

        return "Sin estrés"

    @staticmethod
    def _normalizar_cultivo(nombre):

        mapa = {

            "maiz": "maiz",
            "maíz": "maiz",
            "corn": "maiz",

            "arroz": "arroz",
            "rice": "arroz",

            "papa": "papa",
            "potato": "papa",

            "platano": "platano",
            "plátano": "platano",
            "banana": "platano",

            "cafe": "cafe",
            "café": "cafe",
            "coffee": "cafe"

        }

        for clave, valor in mapa.items():

            if clave in nombre:

                return valor

        return "default"

    # ─────────────────────────────────────────────
    # RECOMENDACIONES
    # ─────────────────────────────────────────────

    def _generar_recomendaciones(
        self,
        resultado,
        cultivo,
        temp,
        humedad,
        lluvia,
        ndvi
    ):

        recs = []

        plaga = resultado["plaga"]

        rend = resultado["rendimiento"]

        if plaga > 70:

            recs.append(
                "🔴 Riesgo crítico de plaga."
            )

        elif plaga > 45:

            recs.append(
                "🟠 Riesgo alto de plaga."
            )

        elif plaga > 25:

            recs.append(
                "🟡 Riesgo medio de plaga."
            )

        if temp > 34:

            recs.append(
                "🌡️ Temperatura alta."
            )

        elif temp < 15:

            recs.append(
                "❄️ Temperatura baja."
            )

        if humedad > 90:

            recs.append(
                "💧 Humedad excesiva."
            )

        elif humedad < 40:

            recs.append(
                "💧 Humedad baja."
            )

        if lluvia < 20:

            recs.append(
                "🌧️ Poca lluvia."
            )

        elif lluvia > 150:

            recs.append(
                "🌧️ Exceso de lluvia."
            )

        if ndvi < 0.4:

            recs.append(
                "🛰️ NDVI bajo."
            )

        base_rend = self.RENDIMIENTO_BASE.get(
            cultivo,
            5.0
        )

        if rend < base_rend * 0.6:

            recs.append(
                "📉 Rendimiento bajo."
            )

        if not recs:

            recs.append(
                "✅ Condiciones favorables."
            )

        return recs