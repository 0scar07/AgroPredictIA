"""
AgroPredict IA — Backend Flask
Base de datos: PostgreSQL (Supabase)
"""

from flask import Flask, request, jsonify, send_file, session, redirect
from flask_cors import CORS
import os, io, csv, hashlib

import psycopg2
from psycopg2.extras import RealDictCursor

from modelos      import ModelManager
from predicciones import PredictionEngine
from entrenamiento import TrainingEngine
from graficas     import GraficaEngine

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = "agropredict_secret_2024"
CORS(app, supports_credentials=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONEXIÓN POSTGRESQL — SUPABASE
# ══════════════════════════════════════════════════════════════════════════════

DB_URL = "postgresql://postgres:AgroPredict2026!@db.vkygwebfnzrxbmpdzwti.supabase.co:5432/postgres"

def get_db():
    conn = psycopg2.connect(DB_URL, cursor_factory=RealDictCursor)
    return conn

def init_db():
    conn = get_db()
    cur  = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS usuarios (
            id        SERIAL PRIMARY KEY,
            nombre    TEXT    NOT NULL,
            email     TEXT    UNIQUE NOT NULL,
            password  TEXT    NOT NULL,
            creado_en TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS predicciones_log (
            id           SERIAL PRIMARY KEY,
            usuario_id   INTEGER NOT NULL REFERENCES usuarios(id),
            parcela      TEXT,
            cultivo      TEXT,
            temperatura  REAL,
            humedad      REAL,
            lluvia_mm    REAL,
            ndvi         REAL,
            rendimiento  REAL,
            plaga        REAL,
            confianza    REAL,
            creado_en    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS parcelas (
            id          SERIAL PRIMARY KEY,
            usuario_id  INTEGER NOT NULL REFERENCES usuarios(id),
            nombre      TEXT,
            cultivo     TEXT,
            hectareas   REAL,
            fase        TEXT,
            estado      TEXT DEFAULT 'normal',
            creado_en   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS sensores_log (
            id            SERIAL PRIMARY KEY,
            usuario_id    INTEGER NOT NULL REFERENCES usuarios(id),
            parcela_id    INTEGER,
            temperatura   REAL,
            humedad       REAL,
            lluvia_mm     REAL,
            ndvi          REAL,
            humedad_suelo REAL,
            timestamp     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    cur.close()
    conn.close()
    print("✅ Tablas creadas/verificadas en Supabase")

# ══════════════════════════════════════════════════════════════════════════════
# AUTH
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/register", methods=["POST"])
def register():
    data     = request.json
    nombre   = data.get("nombre", "").strip()
    email    = data.get("email",  "").strip().lower()
    password = data.get("password", "")
    if not nombre or not email or not password:
        return jsonify({"ok": False, "msg": "Todos los campos son requeridos"}), 400
    pw_hash = hashlib.sha256(password.encode()).hexdigest()
    try:
        conn = get_db(); cur = conn.cursor()
        cur.execute("INSERT INTO usuarios (nombre, email, password) VALUES (%s,%s,%s) RETURNING id",
                    (nombre, email, pw_hash))
        uid = cur.fetchone()["id"]
        conn.commit(); cur.close(); conn.close()
        session["user_id"]   = uid
        session["user_name"] = nombre
        return jsonify({"ok": True, "msg": "Cuenta creada exitosamente", "nombre": nombre})
    except psycopg2.errors.UniqueViolation:
        return jsonify({"ok": False, "msg": "El correo ya está registrado"}), 409
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)}), 500

@app.route("/api/login", methods=["POST"])
def login():
    data     = request.json
    email    = data.get("email",    "").strip().lower()
    password = data.get("password", "")
    pw_hash  = hashlib.sha256(password.encode()).hexdigest()
    try:
        conn = get_db(); cur = conn.cursor()
        cur.execute("SELECT * FROM usuarios WHERE email=%s AND password=%s", (email, pw_hash))
        user = cur.fetchone(); cur.close(); conn.close()
        if not user:
            return jsonify({"ok": False, "msg": "Credenciales inválidas"}), 401
        session["user_id"]   = user["id"]
        session["user_name"] = user["nombre"]
        return jsonify({"ok": True, "nombre": user["nombre"], "id": user["id"]})
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)}), 500

@app.route("/api/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"ok": True})

@app.route("/api/me")
def me():
    uid = session.get("user_id")
    if not uid:
        return jsonify({"ok": False}), 401
    try:
        conn = get_db(); cur = conn.cursor()
        cur.execute("SELECT id, nombre, email, creado_en FROM usuarios WHERE id=%s", (uid,))
        u = cur.fetchone(); cur.close(); conn.close()
        return jsonify({"ok": True, "user": dict(u)})
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)}), 500

# ══════════════════════════════════════════════════════════════════════════════
# MOTORES IA
# ══════════════════════════════════════════════════════════════════════════════

engine_pred  = PredictionEngine()
engine_model = ModelManager()
engine_train = TrainingEngine()
engine_graf  = GraficaEngine()

# ══════════════════════════════════════════════════════════════════════════════
# PREDICCIONES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/predecir", methods=["POST"])
def predecir():
    uid = session.get("user_id")
    if not uid:
        return jsonify({"ok": False, "msg": "No autenticado"}), 401
    data      = request.json
    resultado = engine_pred.predecir(data)
    try:
        conn = get_db(); cur = conn.cursor()
        cur.execute("""INSERT INTO predicciones_log
            (usuario_id,parcela,cultivo,temperatura,humedad,lluvia_mm,ndvi,rendimiento,plaga,confianza)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (uid, data.get("parcela",""), data.get("cultivo",""),
             data.get("temperatura"), data.get("humedad"), data.get("lluvia_mm"),
             data.get("ndvi"), resultado.get("rendimiento"),
             resultado.get("plaga"), resultado.get("confianza")))
        conn.commit(); cur.close(); conn.close()
    except Exception as e:
        print(f"⚠️ Error guardando predicción: {e}")
    return jsonify({"ok": True, **resultado})

@app.route("/api/historial-predicciones")
def historial_predicciones():
    uid = session.get("user_id")
    if not uid:
        return jsonify({"ok": False}), 401
    try:
        conn = get_db(); cur = conn.cursor()
        cur.execute("SELECT * FROM predicciones_log WHERE usuario_id=%s ORDER BY creado_en DESC LIMIT 50", (uid,))
        rows = cur.fetchall(); cur.close(); conn.close()
        return jsonify({"ok": True, "data": [dict(r) for r in rows]})
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)}), 500

# ══════════════════════════════════════════════════════════════════════════════
# ENTRENAMIENTO
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/entrenar", methods=["POST"])
def entrenar():
    uid = session.get("user_id")
    if not uid:
        return jsonify({"ok": False}), 401
    data   = request.json
    result = engine_train.entrenar(data.get("rows", []), data.get("config", {}))
    return jsonify({"ok": True, **result})

# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICAS
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/grafica/<tipo>", methods=["POST"])
def grafica(tipo):
    uid = session.get("user_id")
    if not uid:
        return jsonify({"ok": False}), 401
    img_b64 = engine_graf.generar(tipo, request.json)
    return jsonify({"ok": True, "imagen": img_b64})

# ══════════════════════════════════════════════════════════════════════════════
# PARCELAS
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/parcelas", methods=["GET"])
def get_parcelas():
    uid = session.get("user_id")
    if not uid:
        return jsonify({"ok": False}), 401
    try:
        conn = get_db(); cur = conn.cursor()
        cur.execute("SELECT * FROM parcelas WHERE usuario_id=%s ORDER BY creado_en DESC", (uid,))
        rows = cur.fetchall(); cur.close(); conn.close()
        return jsonify({"ok": True, "data": [dict(r) for r in rows]})
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)}), 500

@app.route("/api/parcelas", methods=["POST"])
def create_parcela():
    uid = session.get("user_id")
    if not uid:
        return jsonify({"ok": False}), 401
    data = request.json
    try:
        conn = get_db(); cur = conn.cursor()
        cur.execute("""INSERT INTO parcelas (usuario_id,nombre,cultivo,hectareas,fase,estado)
            VALUES (%s,%s,%s,%s,%s,%s) RETURNING id""",
            (uid, data.get("nombre"), data.get("cultivo"),
             data.get("hectareas"), data.get("fase"), data.get("estado","normal")))
        pid = cur.fetchone()["id"]
        conn.commit(); cur.close(); conn.close()
        return jsonify({"ok": True, "id": pid})
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)}), 500

# ══════════════════════════════════════════════════════════════════════════════
# SERVIR PÁGINAS
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return app.send_static_file("index.html")

@app.route("/app")
def app_page():
    if not session.get("user_id"):
        return redirect("/")
    return app.send_static_file("app.html")

# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    init_db()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
