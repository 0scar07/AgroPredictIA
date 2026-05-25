# 🌱 AgroPredict IA — v2.0

Plataforma de inteligencia artificial para predicción agronómica.
Arquitectura separada: **HTML puro** (frontend) + **Python** (modelos, entrenamiento, gráficas, backend).

---

## 📁 Estructura del proyecto

```
AgroPredictIA/
│
├── app.py              ← Backend Flask (auth, BD, rutas API)
├── modelos.py          ← Gestión de modelos ML (Keras / sklearn)
├── entrenamiento.py    ← Lógica de entrenamiento (R², MAE, RMSE)
├── predicciones.py     ← Motor de inferencia (con modelo o heurístico)
├── graficas.py         ← Gráficas matplotlib/seaborn → Base64 PNG
├── requirements.txt    ← Dependencias Python
│
├── static/
│   ├── index.html      ← Login + Register
│   └── app.html        ← Aplicación principal (solo HTML/CSS/JS)
│
├── models/             ← Modelos entrenados guardados (auto-creado)
└── agropredict.db      ← Base de datos SQLite (auto-creada)
```

---

## 🚀 Instalación y ejecución

### 1. Instalar dependencias base

```bash
pip install flask flask-cors numpy matplotlib seaborn
```

### 2. (Opcional) Instalar dependencias ML

```bash
# Para redes neuronales
pip install tensorflow

# Para Random Forest y Gradient Boosting
pip install scikit-learn joblib
```

### 3. Ejecutar el servidor

```bash
python app.py
```

### 4. Abrir en el navegador

```
http://localhost:5000
```

---

## 🔐 Login y registro

- **Registro**: Crea una cuenta en `/` → "Registrarse"
- **Login**: Inicia sesión con tu cuenta
- **Demo**: Botón "Acceder con cuenta demo" (usuario: `demo@agropredict.co`, clave: `demo1234`)
- **Base de datos**: SQLite (`agropredict.db`) — se crea automáticamente

### Tablas SQLite
| Tabla | Descripción |
|-------|-------------|
| `usuarios` | Registro de usuarios (nombre, email, password hash SHA-256) |
| `predicciones_log` | Historial de todas las predicciones generadas |
| `parcelas` | Parcelas registradas por usuario |
| `sensores_log` | Lecturas de sensores IoT por parcela |

---

## 🧠 Módulos Python

### `modelos.py` — ModelManager
- Construye redes neuronales Keras (1-3 capas, dropout, activación configurable)
- Construye modelos sklearn (Random Forest, Gradient Boosting)
- Guarda/carga modelos desde disco (formato SavedModel o joblib)
- Normalización min-max manual o con sklearn MinMaxScaler

### `entrenamiento.py` — TrainingEngine
- Mapeo automático de columnas CSV a variables (temperatura, humedad, ndvi…)
- Split train/test configurable
- Métricas: **R²**, **MAE**, **RMSE**
- Historial de pérdida por época (compatible con Keras callbacks)
- Soporta: Red Neuronal, Random Forest, Gradient Boosting

### `predicciones.py` — PredictionEngine
- Usa modelos entrenados si están disponibles
- Fallback a modelo **heurístico agronómico** calibrado
- Predicción individual o por lote (`predecir_lote`)
- Genera recomendaciones automáticas basadas en condiciones

### `graficas.py` — GraficaEngine
- 7 tipos de gráficas matplotlib/seaborn
- Retorna PNG en Base64 para renderizar en el frontend
- Paleta corporativa verde AgroPredict

---

## 📡 API Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| POST | `/api/register` | Registro de usuario |
| POST | `/api/login` | Login |
| POST | `/api/logout` | Cerrar sesión |
| GET  | `/api/me` | Info del usuario actual |
| POST | `/api/predecir` | Generar predicción (guarda en BD) |
| GET  | `/api/historial-predicciones` | Historial del usuario |
| POST | `/api/entrenar` | Entrenar modelos con datos CSV |
| POST | `/api/grafica/<tipo>` | Generar gráfica matplotlib |
| GET  | `/api/csv-variables` | Lista de CSV disponibles |
| GET  | `/api/descargar-csv/<variable>` | Descargar CSV específico |

---

## 📥 Datasets CSV disponibles

| Variable | Columnas | Descripción |
|----------|----------|-------------|
| `temperatura` | fecha, parcela, cultivo, temperatura, humedad, lluvia_mm | Datos climáticos por parcela |
| `humedad` | fecha, parcela, cultivo, humedad, lluvia_mm, humedad_suelo | Humedad ambiental y del suelo |
| `rendimiento` | fecha, parcela, cultivo, temperatura, humedad, ndvi, rendimiento_t_ha | Rendimiento por cultivo |
| `plaga` | fecha, parcela, cultivo, temperatura, humedad, lluvia_mm, humedad_suelo, riesgo_plaga_pct, alerta | Riesgo de plaga |
| `ndvi` | fecha, parcela, cultivo, ndvi, ndvi_interpretacion, cobertura_nubosa_pct | Índice vegetal Sentinel-2 |
| `lluvia_mm` | fecha, zona, departamento, lluvia_mm, dias_lluvia, intensidad | Precipitación por zonas |

---

## 🔧 Modo demo (sin backend)

La app funciona **sin Flask** con funcionalidad reducida:
- Login/registro simulado (redirige a `app.html` directamente)
- Predicciones con valores simulados
- Entrenamiento con historial de pérdida simulado
- Gráficas muestran mensaje de "backend no disponible"
- Descarga CSV muestra alerta explicativa

Para funcionalidad completa, ejecuta `python app.py`.

---

## 👥 Créditos

**Proyecto AgroPredict IA** — Baena, Barrios, Llanos, Lázaro, Montes  
Universidad — Entrega 2 IA · Colombia 2025
