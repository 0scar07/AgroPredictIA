"""
AgroPredict IA — Módulo: Gráficas
Generación de visualizaciones con matplotlib/seaborn.
Cada función produce una imagen PNG codificada en Base64 para enviar al frontend.

Gráficas disponibles:
    - rendimiento_cultivos     : Barras de rendimiento por cultivo
    - temperatura_humedad      : Dispersión temperatura vs humedad
    - riesgo_plaga_distribucion: Donut de distribución de riesgo
    - perdida_entrenamiento    : Curva de pérdida (loss) por época
    - real_vs_predicho         : Scatter real vs predicho (R²)
    - ndvi_historico           : Línea NDVI mensual por parcela
    - comparativa_parcelas     : Barras comparativas multi-métrica
"""

import io
import base64
import numpy as np

import matplotlib
matplotlib.use("Agg")   # backend sin GUI para servidor
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", palette="Greens")
    SEABORN = True
except ImportError:
    SEABORN = False

# Paleta corporativa AgroPredict
VERDE_OSCURO  = "#1B5E20"
VERDE_MEDIO   = "#2E7D32"
VERDE_CLARO   = "#66BB6A"
VERDE_SUAVE   = "#C8E6C9"
AMBER         = "#F59E0B"
ROJO          = "#DC2626"
NARANJA       = "#EA580C"
AZUL          = "#0EA5E9"
FONDO         = "#F5F7F2"
TEXTO         = "#1C1917"
MUTED         = "#78716C"

PALETA = [VERDE_CLARO, AZUL, AMBER, ROJO, "#BC8CFF", NARANJA]


class GraficaEngine:
    """
    Motor de generación de gráficas para AgroPredict IA.
    Todas las gráficas se devuelven como string Base64 (PNG) para el frontend.
    """

    def generar(self, tipo: str, datos: dict) -> str:
        """
        Dispatcher principal.

        Args:
            tipo:  Identificador del tipo de gráfica.
            datos: Datos específicos para esa gráfica.

        Returns:
            String Base64 del PNG generado.
        """
        generadores = {
            "rendimiento_cultivos":       self.rendimiento_cultivos,
            "temperatura_humedad":        self.temperatura_humedad,
            "riesgo_plaga_distribucion":  self.riesgo_plaga_distribucion,
            "perdida_entrenamiento":      self.perdida_entrenamiento,
            "real_vs_predicho":           self.real_vs_predicho,
            "ndvi_historico":             self.ndvi_historico,
            "comparativa_parcelas":       self.comparativa_parcelas,
        }
        fn = generadores.get(tipo)
        if fn is None:
            return self._error_imagen(f"Tipo de gráfica desconocido: {tipo}")
        try:
            return fn(datos)
        except Exception as e:
            return self._error_imagen(str(e))

    # ═══════════════════════════════════════════════════════════════════════
    # GRÁFICAS INDIVIDUALES
    # ═══════════════════════════════════════════════════════════════════════

    def rendimiento_cultivos(self, datos: dict) -> str:
        """
        Barras verticales: rendimiento promedio (t/ha) por cultivo.
        datos: {"cultivos": [...], "valores": [...]}
        """
        cultivos = datos.get("cultivos", ["Maíz", "Arroz", "Papa", "Plátano", "Café"])
        valores  = datos.get("valores",  [3.2,    4.1,    18.5,  22.4,     2.1  ])

        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor(FONDO)
        ax.set_facecolor(FONDO)

        barras = ax.bar(cultivos, valores, color=PALETA[:len(cultivos)],
                        width=0.55, edgecolor="white", linewidth=1.2, zorder=3)
        ax.set_grid_on = True
        ax.yaxis.grid(True, color="#E2E5DF", zorder=0)
        ax.set_axisbelow(True)

        for bar, val in zip(barras, valores):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{val:.1f}", ha="center", va="bottom",
                    fontsize=11, fontweight="bold", color=TEXTO)

        ax.set_title("Rendimiento promedio por cultivo", fontsize=14,
                     fontweight="bold", color=TEXTO, pad=14)
        ax.set_ylabel("t/ha", fontsize=11, color=MUTED)
        ax.tick_params(colors=MUTED, labelsize=10)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#E2E5DF")

        plt.tight_layout()
        return self._to_base64(fig)

    def temperatura_humedad(self, datos: dict) -> str:
        """
        Dispersión temperatura vs humedad coloreada por nivel de riesgo de plaga.
        datos: {"temperaturas": [...], "humedades": [...], "riesgos": [...]}
        """
        temps   = np.array(datos.get("temperaturas", np.random.uniform(18, 36, 60)))
        hums    = np.array(datos.get("humedades",    np.random.uniform(40, 95, 60)))
        riesgos = np.array(datos.get("riesgos",      np.random.uniform(0, 100, 60)))

        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor(FONDO)
        ax.set_facecolor(FONDO)

        sc = ax.scatter(temps, hums, c=riesgos, cmap="RdYlGn_r",
                        s=60, alpha=0.75, edgecolors="white", linewidths=0.5, zorder=3)
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Riesgo de plaga (%)", color=MUTED, fontsize=10)
        cbar.ax.tick_params(colors=MUTED, labelsize=9)

        ax.yaxis.grid(True, color="#E2E5DF", zorder=0); ax.xaxis.grid(True, color="#E2E5DF", zorder=0)
        ax.set_axisbelow(True)
        ax.set_title("Temperatura vs Humedad — Riesgo de plaga", fontsize=13,
                     fontweight="bold", color=TEXTO, pad=12)
        ax.set_xlabel("Temperatura (°C)", fontsize=11, color=MUTED)
        ax.set_ylabel("Humedad (%)", fontsize=11, color=MUTED)
        ax.tick_params(colors=MUTED, labelsize=10)
        ax.spines[["top", "right"]].set_visible(False)

        plt.tight_layout()
        return self._to_base64(fig)

    def riesgo_plaga_distribucion(self, datos: dict) -> str:
        """
        Donut de distribución de niveles de riesgo de plaga.
        datos: {"bajo": n, "medio": n, "alto": n, "critico": n}
        """
        vals    = [datos.get("bajo",   30), datos.get("medio",  25),
                   datos.get("alto",   25), datos.get("critico", 20)]
        labels  = ["Bajo", "Medio", "Alto", "Crítico"]
        colores = [VERDE_CLARO, AMBER, NARANJA, ROJO]

        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor(FONDO)
        ax.set_facecolor(FONDO)

        wedges, texts, autotexts = ax.pie(
            vals, labels=labels, colors=colores,
            autopct="%1.1f%%", pctdistance=0.82,
            wedgeprops={"width": 0.52, "edgecolor": "white", "linewidth": 2},
            startangle=90, textprops={"color": TEXTO, "fontsize": 11}
        )
        for at in autotexts:
            at.set(fontsize=10, fontweight="bold", color="white")

        ax.set_title("Distribución de riesgo de plaga", fontsize=13,
                     fontweight="bold", color=TEXTO, pad=14)
        plt.tight_layout()
        return self._to_base64(fig)

    def perdida_entrenamiento(self, datos: dict) -> str:
        """
        Curva de pérdida (loss) por época durante el entrenamiento.
        datos: {"historial_rend": [...], "historial_plaga": [...], "epocas": int}
        """
        hist_rend  = datos.get("historial_rend",  [])
        hist_plaga = datos.get("historial_plaga", [])
        n_epocas   = datos.get("epocas", max(len(hist_rend), len(hist_plaga), 1))
        epocas     = list(range(1, n_epocas + 1))

        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor(FONDO)
        ax.set_facecolor(FONDO)

        if hist_rend:
            ax.plot(epocas[:len(hist_rend)], hist_rend,
                    color=VERDE_CLARO, linewidth=2, label="Rendimiento", zorder=3)
            ax.fill_between(epocas[:len(hist_rend)], hist_rend,
                            alpha=0.12, color=VERDE_CLARO)
        if hist_plaga:
            ax.plot(epocas[:len(hist_plaga)], hist_plaga,
                    color=AMBER, linewidth=2, label="Plaga", zorder=3)
            ax.fill_between(epocas[:len(hist_plaga)], hist_plaga,
                            alpha=0.12, color=AMBER)

        ax.yaxis.grid(True, color="#E2E5DF", zorder=0)
        ax.set_axisbelow(True)
        ax.set_title("Curva de pérdida durante el entrenamiento", fontsize=13,
                     fontweight="bold", color=TEXTO, pad=12)
        ax.set_xlabel("Época", fontsize=11, color=MUTED)
        ax.set_ylabel("Loss (MSE)", fontsize=11, color=MUTED)
        ax.tick_params(colors=MUTED, labelsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        if hist_rend or hist_plaga:
            ax.legend(fontsize=10, framealpha=0.8)

        plt.tight_layout()
        return self._to_base64(fig)

    def real_vs_predicho(self, datos: dict) -> str:
        """
        Scatter de valores reales vs predichos con línea ideal.
        datos: {"real": [...], "predicho": [...], "r2": float}
        """
        real     = np.array(datos.get("real",     []))
        predicho = np.array(datos.get("predicho", []))
        r2       = datos.get("r2", None)

        if len(real) == 0:
            real     = np.random.uniform(2, 22, 50)
            predicho = real + np.random.normal(0, 1.5, 50)

        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor(FONDO)
        ax.set_facecolor(FONDO)

        ax.scatter(real, predicho, color=VERDE_MEDIO, alpha=0.65,
                   s=55, edgecolors="white", linewidths=0.5, zorder=3)

        mn, mx = min(real.min(), predicho.min()), max(real.max(), predicho.max())
        margin = (mx - mn) * 0.05
        linea  = [mn - margin, mx + margin]
        ax.plot(linea, linea, color=ROJO, linestyle="--", linewidth=1.5,
                label="Predicción ideal", zorder=4)

        ax.yaxis.grid(True, color="#E2E5DF", zorder=0)
        ax.xaxis.grid(True, color="#E2E5DF", zorder=0)
        ax.set_axisbelow(True)

        titulo = "Real vs Predicho"
        if r2 is not None:
            titulo += f"  —  R² = {r2:.3f}"
        ax.set_title(titulo, fontsize=13, fontweight="bold", color=TEXTO, pad=12)
        ax.set_xlabel("Valor real", fontsize=11, color=MUTED)
        ax.set_ylabel("Valor predicho", fontsize=11, color=MUTED)
        ax.tick_params(colors=MUTED, labelsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=10, framealpha=0.8)

        plt.tight_layout()
        return self._to_base64(fig)

    def ndvi_historico(self, datos: dict) -> str:
        """
        Líneas de NDVI mensual por parcela.
        datos: {"meses": [...], "series": [{"label": str, "valores": [...]}]}
        """
        meses  = datos.get("meses",  ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep"])
        series = datos.get("series", [
            {"label": "La Esperanza", "valores": [0.65,0.72,0.68,0.61,0.57,0.54,0.54,0.52,0.50]},
            {"label": "El Progreso",  "valores": [0.70,0.72,0.69,0.65,0.55,0.51,0.49,0.48,0.47]},
            {"label": "San José",     "valores": [0.30,0.45,0.58,0.68,0.72,0.75,0.71,0.70,0.68]},
            {"label": "La Palma",     "valores": [0.60,0.67,0.73,0.76,0.79,0.78,0.78,0.77,0.76]},
        ])

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor(FONDO)
        ax.set_facecolor(FONDO)

        for i, s in enumerate(series):
            vals   = s["valores"][:len(meses)]
            color  = PALETA[i % len(PALETA)]
            ax.plot(meses[:len(vals)], vals, marker="o", markersize=5,
                    linewidth=2, color=color, label=s["label"], zorder=3)

        ax.axhline(0.5, color=MUTED, linestyle=":", linewidth=1, alpha=0.6)
        ax.yaxis.grid(True, color="#E2E5DF", zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylim(0.2, 1.0)
        ax.set_title("NDVI histórico por parcela — Sentinel-2", fontsize=13,
                     fontweight="bold", color=TEXTO, pad=12)
        ax.set_ylabel("NDVI", fontsize=11, color=MUTED)
        ax.tick_params(colors=MUTED, labelsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=10, framealpha=0.85, loc="upper left")

        plt.tight_layout()
        return self._to_base64(fig)

    def comparativa_parcelas(self, datos: dict) -> str:
        """
        Barras agrupadas para comparar múltiples métricas entre parcelas.
        datos: {"parcelas": [...], "metricas": {"Rendimiento": [...], "Plaga %": [...]}}
        """
        parcelas = datos.get("parcelas", ["La Esperanza","El Progreso","San José","La Palma"])
        metricas = datos.get("metricas", {
            "Rendimiento (t/ha)": [3.2, 4.1, 18.5, 22.4],
            "Riesgo plaga (%)":   [89,  55,  14,   31  ],
        })

        n_grupos  = len(parcelas)
        n_metrics = len(metricas)
        ancho     = 0.8 / n_metrics
        x         = np.arange(n_grupos)

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor(FONDO)
        ax.set_facecolor(FONDO)

        for i, (metrica, vals) in enumerate(metricas.items()):
            offset = (i - n_metrics / 2 + 0.5) * ancho
            ax.bar(x + offset, vals[:n_grupos], width=ancho * 0.9,
                   color=PALETA[i % len(PALETA)], label=metrica,
                   edgecolor="white", linewidth=0.8, zorder=3)

        ax.yaxis.grid(True, color="#E2E5DF", zorder=0)
        ax.set_axisbelow(True)
        ax.set_xticks(x)
        ax.set_xticklabels([p.split("—")[0].strip() for p in parcelas],
                            fontsize=10, color=TEXTO)
        ax.set_title("Comparativa de métricas por parcela", fontsize=13,
                     fontweight="bold", color=TEXTO, pad=12)
        ax.tick_params(colors=MUTED, labelsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=10, framealpha=0.85)

        plt.tight_layout()
        return self._to_base64(fig)

    # ═══════════════════════════════════════════════════════════════════════
    # UTILIDADES
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _to_base64(fig) -> str:
        """Convierte figura matplotlib a PNG Base64."""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    @staticmethod
    def _error_imagen(mensaje: str) -> str:
        """Genera imagen de error cuando la gráfica falla."""
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.text(0.5, 0.5, f"⚠️ Error al generar gráfica\n{mensaje}",
                ha="center", va="center", fontsize=11, color=ROJO,
                transform=ax.transAxes)
        ax.axis("off")
        return GraficaEngine._to_base64(fig)
