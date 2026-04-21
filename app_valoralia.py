# -*- coding: utf-8 -*-
# Valoralia Systems. Aplicación web productiva
# Autora: María Luisa Ros Bolea
# TFM. Máster en Inteligencia Artificial y Big Data. CEU San Pablo
#
# ARQUITECTURA BLINDADA CONTRA VERSIONES:
# - XGBoost se carga con xgb.Booster() (API nativa, NO toca scikit-learn)
# - El preprocesador se reproduce a mano con numpy puro desde un JSON
# - No hay dependencia runtime de scikit-learn. Inmune a cambios de versiones.

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

# ==============================================================================
# 1. Configuración de la página y paleta corporativa
# ==============================================================================
st.set_page_config(
    page_title="Valoralia Systems",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }
    h1, h2, h3, h4, label, .stMarkdown p, .stMarkdown li {
        color: #f8fafc !important;
        font-family: 'Inter', 'Helvetica Neue', sans-serif;
    }
    .stSelectbox label, .stNumberInput label, .stFileUploader label, .stTextInput label {
        color: #cbd5e1 !important;
        font-weight: 500;
    }
    .stSelectbox div[data-baseweb="select"] span,
    .stNumberInput input, .stTextInput input,
    .stFileUploader div {
        color: #0f172a !important;
    }
    .caja-entrada {
        background: rgba(248, 250, 252, 0.04);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(248, 250, 252, 0.08);
        border-radius: 12px;
        padding: 2rem;
        margin-bottom: 1.5rem;
    }
    .titulo-v {
        border-left: 6px solid #b91c1c;
        padding-left: 20px;
        margin-bottom: 16px;
    }
    .stButton > button {
        background: #b91c1c !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 700 !important;
        height: 52px;
        width: 100%;
        text-transform: uppercase;
        font-size: 1rem;
        letter-spacing: 1px;
    }
    .stButton > button:hover { background: #991b1b !important; }
    .bloque-info {
        background: rgba(185, 28, 28, 0.1);
        border-left: 3px solid #b91c1c;
        padding: 16px 20px;
        border-radius: 6px;
        margin-top: 16px;
    }
    .resultado-precio {
        background: #0f172a;
        border-top: 4px solid #b91c1c;
        padding: 32px;
        border-radius: 10px;
        text-align: center;
        margin-top: 24px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# 2. Carga de artefactos. Sin dependencia de scikit-learn en runtime
# ==============================================================================
@st.cache_resource(show_spinner="Cargando el modelo Valoralia...")
def cargar_artefactos():
    # 2.1. XGBoost con API nativa. No toca scikit-learn
    booster = xgb.Booster()
    booster.load_model("modelo_xgb.json")

    # 2.2. Parámetros del preprocesador desde JSON (imputer, scaler, OHE)
    with open("preprocesador_params.json", "r", encoding="utf-8") as f:
        params = json.load(f)

    # Convierto listas a arrays numpy para operaciones rápidas
    params["imputer_medianas"] = np.array(params["imputer_medianas"], dtype=np.float32)
    params["scaler_mean"] = np.array(params["scaler_mean"], dtype=np.float32)
    params["scaler_scale"] = np.array(params["scaler_scale"], dtype=np.float32)

    # 2.3. Medianas PCA para fallback (joblib puro, sin sklearn dentro)
    try:
        medianas_pca = joblib.load("medianas_pca.pkl")
    except Exception:
        medianas_pca = {}

    return booster, params, medianas_pca


try:
    booster, params, medianas_pca = cargar_artefactos()
except Exception as err:
    st.error(f"Fallo al cargar los artefactos: {err}")
    st.stop()


# ==============================================================================
# 3. Transformación manual (reemplaza al ColumnTransformer de scikit-learn)
# ==============================================================================
def transformar_entrada(fila_dict, params):
    """Reproduce a mano la transformación que hacía el ColumnTransformer del NB04.

    Bloque 1. Variables numéricas: imputación por mediana + estandarización.
    Bloque 2. Variables categóricas: One-Hot Encoding con handle_unknown='ignore'.

    Devuelve un array numpy de 33.503 columnas listo para el XGBoost.
    """
    cols_num = params["cols_num"]
    cols_cat = params["cols_cat_orden"]
    medianas = params["imputer_medianas"]
    mean = params["scaler_mean"]
    scale = params["scaler_scale"]
    cat_usadas = params["ohe_categorias_usadas"]
    cat_ignoradas = params["ohe_n_cols_ignoradas"]

    # --- Bloque numérico ---
    vec_num = np.zeros(len(cols_num), dtype=np.float32)
    for i, col in enumerate(cols_num):
        valor = fila_dict.get(col, None)
        if valor is None or (isinstance(valor, float) and np.isnan(valor)):
            valor = medianas[i]
        vec_num[i] = float(valor)
    vec_num = (vec_num - mean) / scale

    # --- Bloque categórico ---
    partes_cat = []
    for col in cols_cat:
        if col in cat_usadas:
            # Columna importante: construyo el one-hot real
            categorias = cat_usadas[col]
            valor = str(fila_dict.get(col, ""))
            one_hot = np.zeros(len(categorias), dtype=np.float32)
            if valor in categorias:
                idx = categorias.index(valor)
                one_hot[idx] = 1.0
            # Si valor no está en categorías, queda todo a cero (handle_unknown='ignore')
            partes_cat.append(one_hot)
        else:
            # Columna ignorada (url, titulo, descripcion, etc.): todo a cero
            # porque el valor "-" que ponemos no coincide con ninguna categoría aprendida
            n = cat_ignoradas[col]
            partes_cat.append(np.zeros(n, dtype=np.float32))

    vec_cat = np.concatenate(partes_cat)
    vector_final = np.concatenate([vec_num, vec_cat]).astype(np.float32)
    return vector_final.reshape(1, -1)


# ==============================================================================
# 4. Catálogos de categorías que el modelo conoce (39 zonas + 5 tipos)
# ==============================================================================
ZONAS_MODELO = {
    "Madrid. Centro":                   "Centro",
    "Madrid. Salamanca":                "Salamanca",
    "Madrid. Chamberi":                 "Chamberi",
    "Madrid. Retiro":                   "Retiro",
    "Madrid. Chamartin":                "Chamartin",
    "Madrid. Tetuan":                   "Tetuan",
    "Madrid. Fuencarral":               "Fuencarral",
    "Madrid. Moncloa":                  "Moncloa",
    "Madrid. Ciudad Lineal":            "Ciudad_Lineal",
    "Madrid. Latina":                   "Latina",
    "Madrid. Carabanchel":              "Carabanchel",
    "Madrid. Usera":                    "Usera",
    "Madrid. Villaverde":               "Villaverde",
    "Madrid. Puente de Vallecas":       "Puente_Vallecas",
    "Madrid. Villa de Vallecas":        "Villa_Vallecas",
    "Madrid. Vicalvaro":                "Vicalvaro",
    "Madrid. San Blas":                 "San_Blas",
    "Madrid. Moratalaz":                "Moratalaz",
    "Pozuelo de Alarcon":               "Pozuelo",
    "Las Rozas":                        "Las_Rozas",
    "Boadilla del Monte":               "Boadilla",
    "Majadahonda":                      "Majadahonda",
    "Tres Cantos":                      "Tres_Cantos",
    "San Sebastian de los Reyes":       "SS_Reyes",
    "Alcobendas":                       "Alcobendas",
    "Colmenar Viejo":                   "Colmenar_Viejo",
    "Alcala de Henares":                "Alcala_Henares",
    "Torrejon de Ardoz":                "Torrejon",
    "Coslada":                          "Coslada",
    "Rivas-Vaciamadrid":                "Rivas",
    "Arganda del Rey":                  "Arganda",
    "Getafe":                           "Getafe",
    "Leganes":                          "Leganes",
    "Alcorcon":                         "Alcorcon",
    "Mostoles":                         "Mostoles",
    "Fuenlabrada":                      "Fuenlabrada",
    "Parla":                            "Parla",
    "Pinto":                            "Pinto",
    "Villaviciosa de Odon":             "Villaviciosa_Odon",
}

TIPOS_INMUEBLE = {
    "Piso":    "piso",
    "Atico":   "atico",
    "Duplex":  "duplex",
    "Estudio": "estudio",
    "Chalet":  "chalet",
}

MAPA_TERNARIO = {"Si": 1, "No": 0, "Desconocido": -1}


# ==============================================================================
# 5. Interfaz
# ==============================================================================
st.markdown(
    "<div class='titulo-v'><h1>Valoralia Systems</h1>"
    "<p style='color:#cbd5e1; font-size: 1.05rem;'>Valoración automática de inmuebles residenciales en Madrid. "
    "Modelo híbrido XGBoost con 50 componentes visuales PCA extraídas con ResNet50.</p></div>",
    unsafe_allow_html=True,
)

st.markdown("<div class='caja-entrada'>", unsafe_allow_html=True)
st.subheader("Ubicación y dimensiones del inmueble")
col_a, col_b, col_c = st.columns(3)
with col_a:
    zona_etiqueta = st.selectbox("Zona geográfica", options=list(ZONAS_MODELO.keys()), index=0)
    codigo_postal = st.number_input("Código postal", min_value=28001, max_value=28999, value=28001, step=1)
with col_b:
    tipo_etiqueta = st.selectbox("Tipo de inmueble", options=list(TIPOS_INMUEBLE.keys()), index=0)
    superficie_m2 = st.number_input("Superficie útil (m²)", min_value=20, max_value=1000, value=85, step=5)
with col_c:
    planta = st.number_input("Planta", min_value=0, max_value=30, value=3, step=1)
    num_imagenes_manual = st.number_input(
        "Número de fotos disponibles", min_value=0, max_value=30, value=8, step=1,
        help="Si subes fotos abajo, se sobrescribe con el conteo real."
    )
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='caja-entrada'>", unsafe_allow_html=True)
st.subheader("Distribución y características")
col_d, col_e, col_f = st.columns(3)
with col_d:
    habitaciones = st.number_input("Habitaciones", min_value=0, max_value=10, value=2, step=1)
    banos = st.number_input("Baños", min_value=1, max_value=max(1, habitaciones + 2), value=1, step=1)
with col_e:
    ascensor = st.selectbox("Ascensor", options=["Si", "No", "Desconocido"], index=0)
    terraza = st.selectbox("Terraza", options=["Si", "No", "Desconocido"], index=2)
with col_f:
    garaje = st.selectbox("Garaje", options=["Si", "No", "Desconocido"], index=2)
    calefaccion = st.selectbox("Calefacción", options=["Si", "No", "Desconocido"], index=0)

estado_opciones = {"Reformado": 1, "Original": 0, "A reformar": 0, "Desconocido": -1}
estado_etiqueta = st.selectbox("Estado de reforma", options=list(estado_opciones.keys()), index=3)
estado_reforma = estado_opciones[estado_etiqueta]
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='caja-entrada'>", unsafe_allow_html=True)
st.subheader("Aportación visual al modelo híbrido")
fotos_subidas = st.file_uploader(
    "Sube las fotografías del interior (opcional)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    help="El TFM demuestra que las componentes PCA visuales aportan el 27,2% de la importancia predictiva.",
)

if fotos_subidas:
    st.markdown(
        f"""<div class='bloque-info'>
            <p style='color: #f8fafc; font-weight: 600; margin: 0; letter-spacing: 1px;'>
            FOTOGRAFÍAS RECIBIDAS: {len(fotos_subidas)}
            </p>
            <p style='color: #cbd5e1; font-size: 0.9rem; margin-top: 6px; margin-bottom: 0;'>
            Esta versión web utiliza las medianas PCA del mercado como fallback para mantener ligero el contenedor.
            El procesamiento visual completo con ResNet50 y PCA (entrenado sobre más de 100.000 fotografías en el NB03)
            está disponible en el repositorio académico del TFM.
            </p>
        </div>""",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        "<p style='color: #94a3b8; font-size: 0.9rem; margin-top: 12px;'>"
        "Sin fotografías: el modelo utiliza las medianas PCA del conjunto de entrenamiento como fallback.</p>",
        unsafe_allow_html=True,
    )
st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# 6. Predicción
# ==============================================================================
if st.button("Generar valoración de mercado"):
    zona_valor = ZONAS_MODELO[zona_etiqueta]
    tipo_valor = TIPOS_INMUEBLE[tipo_etiqueta]

    # Construyo la fila con las 70 claves que el modelo conoce
    fila = {
        "url": "-",
        "fecha_scraping": "-",
        "titulo": "-",
        "url_imagen_principal": "-",
        "urls_imagenes": "-",
        "descripcion": "-",
        "fuente": "pisos.com",
        "zona_scraping": zona_valor,
        "tipo_inmueble": tipo_valor,
        "superficie_m2": superficie_m2,
        "habitaciones": habitaciones,
        "banos": banos,
        "planta": planta,
        "num_imagenes": len(fotos_subidas) if fotos_subidas else num_imagenes_manual,
        "codigo_postal": codigo_postal,
        "ascensor": MAPA_TERNARIO[ascensor],
        "terraza": MAPA_TERNARIO[terraza],
        "garaje": MAPA_TERNARIO[garaje],
        "calefaccion": MAPA_TERNARIO[calefaccion],
        "estado_reforma": estado_reforma,
    }
    # Las 50 componentes PCA visuales: fallback con medianas del mercado
    for i in range(1, 51):
        clave = f"pca_{i}"
        fila[clave] = float(medianas_pca.get(clave, 0.0))

    try:
        # Transformación manual (reemplaza al ColumnTransformer)
        vector = transformar_entrada(fila, params)

        # Predicción con XGBoost Booster nativo
        dmatrix = xgb.DMatrix(vector)
        prediccion_log = booster.predict(dmatrix)[0]
        precio_estimado = float(np.expm1(prediccion_log))

        precio_m2 = precio_estimado / superficie_m2
        precio_formateado = "{:,.0f}".format(precio_estimado).replace(",", ".")
        m2_formateado = "{:,.0f}".format(precio_m2).replace(",", ".")

        st.markdown(
            f"""<div class='resultado-precio'>
                <p style='color: #b91c1c; font-weight: 700; letter-spacing: 2px; margin-bottom: 8px; font-size: 0.9rem;'>
                VALORACIÓN ESTIMADA VALORALIA
                </p>
                <p style='font-size: 3.2rem; font-weight: 800; color: #f8fafc; margin: 0;'>
                {precio_formateado} EUR
                </p>
                <p style='color: #cbd5e1; font-size: 1rem; margin-top: 8px;'>
                Equivalente a {m2_formateado} EUR/m² en {zona_etiqueta}
                </p>
                <p style='color: #94a3b8; font-size: 0.8rem; margin-top: 16px; margin-bottom: 0;'>
                Margen de error típico del modelo: MAE 210.799 EUR · MAPE 20,09% · R² logarítmico 0,9146
                </p>
            </div>""",
            unsafe_allow_html=True,
        )

    except Exception as err:
        st.error(f"Fallo en el pipeline de predicción: {err}")

# Pie
st.markdown(
    "<p style='text-align: center; color: #64748b; font-size: 0.8rem; padding: 24px 0 12px 0;'>"
    "Valoralia Systems · TFM María Luisa Ros Bolea · CEU San Pablo · 2026</p>",
    unsafe_allow_html=True,
)
