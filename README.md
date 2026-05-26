# 🌱 AgroPredict IA — v2.0

Sistema inteligente de predicción agronómica desarrollado con **Flask + Machine Learning + HTML/CSS/JavaScript** para el análisis, monitoreo y entrenamiento de modelos aplicados a cultivos agrícolas.

---

# 🚀 Características principales

✅ Predicciones agronómicas en tiempo real  
✅ Entrenamiento de modelos personalizados con CSV  
✅ Dashboard interactivo  
✅ Registro e inicio de sesión  
✅ Historial de predicciones  
✅ Generación de gráficas automáticas  
✅ Compatible con SQLite y PostgreSQL  
✅ Arquitectura frontend + backend desacoplada  
✅ Modo demo sin backend  

---

# 🧩 Tecnologías utilizadas

## Frontend
- HTML5
- CSS3
- JavaScript Vanilla

## Backend
- Python 3
- Flask
- Flask-CORS

## Machine Learning
- TensorFlow / Keras
- Scikit-Learn
- NumPy
- Pandas

## Visualización
- Matplotlib
- Seaborn

## Base de datos
- SQLite
- PostgreSQL (Supabase)

---

# 📁 Estructura del proyecto

```bash
AgroPredictIA/
│
├── app.py                 # Backend Flask principal
├── modelos.py             # Gestión de modelos ML
├── entrenamiento.py       # Entrenamiento y métricas
├── predicciones.py        # Motor de inferencia
├── graficas.py            # Generación de gráficas
├── requirements.txt       # Dependencias
│
├── static/
│   ├── index.html         # Login / Register
│   └── app.html           # Dashboard principal
│
├── models/                # Modelos entrenados
├── datasets/              # CSV agrícolas
└── agropredict.db         # Base de datos SQLite
```

---

# ⚙️ Instalación

## 1️⃣ Clonar repositorio

```bash
git clone https://github.com/0scar07/AgroPredictIA.git
cd AgroPredictIA
```

---

## 2️⃣ Crear entorno virtual (Opcional)

### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac
```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3️⃣ Instalar dependencias

```bash
pip install -r requirements.txt
```

---

# ▶️ Ejecutar proyecto

```bash
python app.py
```

Servidor disponible en:

```bash
http://localhost:5000
```

---

# 🔐 Sistema de autenticación

La plataforma incluye:

- Registro de usuarios
- Inicio de sesión
- Manejo de sesiones
- Logout
- Persistencia de usuarios en base de datos

---

# 🧠 Machine Learning

AgroPredict IA permite entrenar distintos modelos:

| Modelo | Estado |
|--------|--------|
| Redes Neuronales | ✅ |
| Random Forest | ✅ |
| Gradient Boosting | ✅ |
| Modelos heurísticos | ✅ |

---

# 📊 Métricas utilizadas

El sistema calcula automáticamente:

- R² Score
- MAE
- RMSE
- Loss History
- Accuracy estimada

---

# 📈 Funcionalidades IA

## Predicción agrícola

El sistema puede estimar:

- Rendimiento esperado
- Riesgo de plagas
- Estrés hídrico
- Nivel NDVI
- Confianza del modelo

---

## Entrenamiento con CSV

Se pueden cargar datasets personalizados con variables como:

```text
temperatura
humedad
lluvia_mm
ndvi
humedad_suelo
rendimiento
```

---

# 📡 API Endpoints

| Método | Endpoint | Función |
|---|---|---|
| POST | `/api/register` | Registrar usuario |
| POST | `/api/login` | Iniciar sesión |
| POST | `/api/logout` | Cerrar sesión |
| GET | `/api/me` | Datos del usuario |
| POST | `/api/predecir` | Generar predicción |
| POST | `/api/entrenar` | Entrenar modelo |
| GET | `/api/historial-predicciones` | Historial |
| POST | `/api/grafica/<tipo>` | Generar gráfica |
| GET | `/api/parcelas` | Obtener parcelas |
| POST | `/api/parcelas` | Crear parcela |

---

# 🗄️ Base de datos

## SQLite (local)

Archivo:

```bash
agropredict.db
```

## PostgreSQL (Supabase)

Compatible mediante:

```python
psycopg2
```

---

# 📥 Datasets incluidos

| Dataset | Descripción |
|---|---|
| temperatura.csv | Variables climáticas |
| humedad.csv | Humedad ambiental |
| rendimiento.csv | Producción agrícola |
| ndvi.csv | Índices de vegetación |
| plaga.csv | Riesgo de plagas |
| lluvia_mm.csv | Precipitación |

---

# 🖥️ Frontend

El frontend fue construido completamente en:

- HTML puro
- CSS puro
- JavaScript Vanilla

Sin frameworks externos.

---

# 🔧 Modo Demo

La aplicación puede funcionar sin backend:

✅ Login simulado  
✅ Navegación funcional  
✅ Predicciones mock  
✅ Entrenamiento visual  
✅ Dashboard demostrativo  

---

# ☁️ Deploy

Compatible con:

- Render
- Railway
- Azure
- Vercel (Frontend)
- Supabase

---

# ⚠️ Notas importantes

- Algunos modelos requieren TensorFlow instalado
- El primer entrenamiento puede tardar varios segundos
- En Render Free puede existir cold start
- El proyecto puede ejecutarse completamente en local

---

# 👨‍💻 Autores

### Proyecto AgroPredict IA

Desarrollado por:

- Baena
- Barrios
- Llanos
- Lázaro
- Montes

📍 Colombia — 2025

---

# 🔗 Repositorio oficial

https://github.com/0scar07/AgroPredictIA
