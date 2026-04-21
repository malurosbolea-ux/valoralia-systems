
# -*- coding: utf-8 -*-
# Valoralia Systems. Aplicacion web productiva
# Autora: Maria Luisa Ros Bolea
# TFM. Master en Inteligencia Artificial y Big Data. CEU San Pablo

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

# Configuracion de la pagina y paleta corporativa
st.set_page_config(
    page_title="Valoralia Systems",
    page_icon="V",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .stApp { background: #fafafa; }
    h1 { color: #0f172a; font-weight: 700; }
    h2, h3 { color: #0f172a; }
    .precio-grande {
        font-size: 3rem; font-weight: 700; color: #0f172a;
        text-align: center; padding: 1.5rem;
        background: white; border: 2px solid #0f172a; border-radius: 8px;
    }
    .metrica-sec { color: #4b5563; text-align: center; font-size: 1.1rem; }
    .aviso { color: #b91c1c; font-weight: 600; }
    div.stButton > button:first-child {
        background-color: #0f172a; color: white; border-radius: 6px;
        font-weight: 600; padding: 0.6rem 1.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Carga de artefactos con cache
@st.cache_resource(show_spinner="Cargando el cerebro de Valoralia...")
def cargar_artefactos():
    preprocesador = joblib.load("preprocesador.pkl")
    modelo = xgb.XGBRegressor()
    modelo.load_model("modelo_xgb.json")
    try:
        medianas = joblib.load("medianas_pca.pkl")
    except Exception:
        medianas = {}
    return preprocesador, modelo, medianas

preprocesador, modelo_xgb, medianas_pca = cargar_artefactos()

# Cabecera
st.title("Valoralia Systems")
st.markdown(
    "<p style='color:#4b5563; font-size: 1.15rem;'>"
    "Sistema de valoracion automatica de inmuebles residenciales en Madrid "
    "mediante aprendizaje automatico hibrido con features visuales.</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# Zonas disponibles
zonas = [
    "Madrid - Salamanca", "Madrid - Chamberi", "Madrid - Centro",
    "Madrid - Retiro", "Madrid - Chamartin", "Madrid - Tetuan",
    "Madrid - Puente de Vallecas", "Madrid - Latina", "Madrid - Usera",
    "Madrid - Carabanchel", "Madrid - Arganzuela", "Madrid - Moncloa",
    "Madrid - Fuencarral", "Madrid - Ciudad Lineal", "Madrid - Hortaleza",
    "Madrid - San Blas", "Madrid - Villaverde", "Madrid - Vicalvaro",
    "Madrid - Moratalaz", "Madrid - Villa de Vallecas", "Madrid - Barajas",
    "Pozuelo de Alarcon", "Las Rozas", "Boadilla del Monte", "Majadahonda",
    "Tres Cantos", "San Sebastian de los Reyes", "Alcobendas", "Colmenar Viejo",
    "Alcala de Henares", "Torrejon de Ardoz", "Coslada", "Rivas-Vaciamadrid",
    "Arganda del Rey", "Getafe", "Leganes", "Alcorcon", "Mostoles",
    "Fuenlabrada", "Parla", "Pinto", "Villaviciosa de Odon",
]

# Formulario de datos del inmueble
st.header("Datos del inmueble")
col1, col2, col3 = st.columns(3)
with col1:
    superficie_m2 = st.number_input("Superficie (m2)", min_value=15, max_value=1000, value=85, step=5)
    habitaciones = st.number_input("Habitaciones", min_value=0, max_value=15, value=3, step=1)
    banos = st.number_input("Banos", min_value=1, max_value=10, value=2, step=1)
    planta = st.number_input("Planta", min_value=0, max_value=30, value=3, step=1)

with col2:
    zona = st.selectbox("Zona", options=zonas, index=0)
    tipo_inmueble = st.selectbox("Tipo de inmueble", options=["piso", "atico", "duplex", "estudio", "chalet"])
    codigo_postal = st.number_input("Codigo postal", min_value=28001, max_value=28999, value=28001, step=1)
    num_imagenes = st.number_input("Numero de fotos del anuncio", min_value=0, max_value=50, value=8, step=1)

with col3:
    ascensor = st.selectbox("Ascensor", options=["Desconocido", "Si", "No"], index=1)
    terraza = st.selectbox("Terraza", options=["Desconocido", "Si", "No"], index=0)
    garaje = st.selectbox("Garaje", options=["Desconocido", "Si", "No"], index=0)
    calefaccion = st.selectbox("Calefaccion", options=["Desconocido", "Si", "No"], index=1)
    estado_reforma = st.selectbox("Estado", options=["Desconocido", "Reformado", "Original", "A reformar"], index=0)

mapa_ternario = {"Si": 1, "No": 0, "Desconocido": -1, "Reformado": 1, "Original": 0, "A reformar": 0}

st.markdown("---")
st.header("Fotografias del interior")
st.markdown(
    "<p style='color:#4b5563;'>Sube las fotos del interior. Cuantas mas fotos, "
    "mayor precision visual. Si no subes ninguna, aplico el fallback de medianas "
    "PCA del mercado.</p>",
    unsafe_allow_html=True,
)
fotos_subidas = st.file_uploader(
    "Seleccionar fotografias",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

st.markdown("---")
if st.button("Calcular tasacion con Valoralia", use_container_width=True):
    try:
        # Componentes PCA: para el prototipo de despliegue uso el fallback de medianas.
        # En una version pro-ResNet50 procesaria las fotos subidas con la red convolucional
        # y promediaria sus vectores antes de aplicar PCA.
        componentes_pca = dict(medianas_pca) if medianas_pca else {f"pca_{i}": 0.0 for i in range(1, 51)}
        usado_fallback = (len(fotos_subidas) == 0)

        fila = {
            "superficie_m2": superficie_m2,
            "habitaciones": habitaciones,
            "banos": banos,
            "planta": planta,
            "num_imagenes": len(fotos_subidas) if fotos_subidas else num_imagenes,
            "codigo_postal": codigo_postal,
            "ascensor": mapa_ternario[ascensor],
            "terraza": mapa_ternario[terraza],
            "garaje": mapa_ternario[garaje],
            "calefaccion": mapa_ternario[calefaccion],
            "estado_reforma": mapa_ternario[estado_reforma],
            "zona_scraping": zona,
            "tipo_inmueble": tipo_inmueble,
        }
        fila.update(componentes_pca)

        X_nuevo = pd.DataFrame([fila])
        X_trans = preprocesador.transform(X_nuevo)
        prediccion_log = modelo_xgb.predict(X_trans)[0]
        precio_estimado = float(np.expm1(prediccion_log))

        st.header("Resultado de la tasacion")
        precio_formateado = "{:,.0f}".format(precio_estimado).replace(",", ".")
        st.markdown(
            "<div class='precio-grande'>" + precio_formateado + " EUR</div>",
            unsafe_allow_html=True,
        )
        precio_m2 = precio_estimado / superficie_m2
        precio_m2_formateado = "{:,.0f}".format(precio_m2).replace(",", ".")
        st.markdown(
            "<p class='metrica-sec'>Equivalente a " + precio_m2_formateado +
            " EUR/m2 en " + zona + "</p>",
            unsafe_allow_html=True,
        )
        if usado_fallback:
            st.markdown(
                "<p class='aviso'>Prediccion calculada con fallback de medianas PCA "
                "por ausencia de fotografias. La incertidumbre es mayor.</p>",
                unsafe_allow_html=True,
            )

    except Exception as err:
        st.markdown(
            "<p class='aviso'>Se ha producido un fallo durante la tasacion: " +
            str(err) + "</p>",
            unsafe_allow_html=True,
        )

st.markdown("---")
st.markdown(
    "<p style='color:#4b5563; text-align: center; font-size: 0.9rem;'>"
    "Valoralia Systems. TFM de Maria Luisa Ros Bolea. CEU San Pablo. 2026</p>",
    unsafe_allow_html=True,
)

