# -*- coding: utf-8 -*-
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
import sklearn
import sklearn.utils

# ==============================================================================
# 0. PARCHE DE INGENIERÍA (OBLIGATORIO PARA EVITAR EL ERROR __sklearn_tags__)
# ==============================================================================
if not hasattr(sklearn.base.BaseEstimator, "__sklearn_tags__"):
    def __sklearn_tags__(self):
        from sklearn.utils import _tags
        return _tags._DEFAULT_TAGS
    sklearn.base.BaseEstimator.__sklearn_tags__ = __sklearn_tags__

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
    .caja-entrada {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2rem;
        border-radius: 0.75rem;
    }
    .stButton>button {
        background-color: #b91c1c !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        font-weight: 700 !important;
        width: 100%;
        border-radius: 0.5rem !important;
    }
    .resultado-precio {
        background: #0f172a;
        border: 2px solid #b91c1c;
        padding: 2.5rem;
        border-radius: 1rem;
        text-align: center;
        margin-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==============================================================================
# 2. Carga de artefactos (Cerebro del sistema)
# ==============================================================================
@st.cache_resource
def cargar_artefactos():
    # Cargo los archivos exactos de mi TFM
    preprocesador = joblib.load("preprocesador.pkl")
    modelo = xgb.XGBRegressor()
    modelo.load_model("modelo_xgb.json")
    try:
        medianas = joblib.load("medianas_pca.pkl")
    except Exception:
        medianas = {}
    return preprocesador, modelo, medianas

try:
    preprocesador, modelo_xgb, medianas_pca = cargar_artefactos()
except Exception as e:
    st.error(f"Fallo al cargar los artefactos: {e}")
    st.stop()

# ==============================================================================
# 3. Interfaz de usuario
# ==============================================================================
st.markdown("<h1 style='color: #f8fafc; font-weight: 800; margin-bottom: 0;'>Valoralia Systems</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #94a3b8; font-size: 1.1rem; margin-bottom: 2.5rem;'>Motor predictivo híbrido basado en visión artificial (ResNet50)</p>", unsafe_allow_html=True)

with st.container():
    st.markdown("<div class='caja-entrada'>", unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.subheader("Ubicación y dimensiones")
        
        # Mapeo de zonas reales sin tildes (Exacto como el entrenamiento)
        mapeo_zonas = {
            "Madrid. Chamberi": "Chamberi", "Madrid. Chamartin": "Chamartin", 
            "Madrid. Tetuan": "Tetuan", "Madrid. Salamanca": "Salamanca",
            "Madrid. Retiro": "Retiro", "Madrid. Centro": "Centro",
            "Madrid. Puente de Vallecas": "Puente_Vallecas", "Madrid. Ciudad Lineal": "Ciudad_Lineal",
            "Madrid. Moncloa": "Moncloa", "Madrid. Latina": "Latina",
            "Madrid. Carabanchel": "Carabanchel", "Madrid. Usera": "Usera",
            "Madrid. Fuencarral": "Fuencarral", "Madrid. Hortaleza": "Hortaleza",
            "Madrid. San Blas": "San_Blas", "Madrid. Tetuan": "Tetuan",
            "Madrid. Villa de Vallecas": "Villa_Vallecas", "Madrid. Vicalvaro": "Vicalvaro",
            "Madrid. Villaverde": "Villaverde", "Madrid. Moratalaz": "Moratalaz",
            "Madrid. Barajas": "Barajas", "Pozuelo de Alarcon": "Pozuelo",
            "Alcobendas": "Alcobendas", "San Sebastian de los Reyes": "SS_Reyes",
            "Majadahonda": "Majadahonda", "Las Rozas": "Las_Rozas"
        }
        
        zona_sel = st.selectbox("Zona de Madrid", list(mapeo_zonas.keys()))
        zona_key = mapeo_zonas[zona_sel]
        cp = st.text_input("Código postal", "28001")
        superficie = st.number_input("Superficie útil (m²)", 20, 1000, 85)
        habitaciones = st.number_input("Habitaciones", 1, 15, 2)
        
    with col2:
        st.subheader("Características del activo")
        banos = st.number_input("Baños", 1, habitaciones+2, 1)
        planta = st.number_input("Planta", -2, 50, 2)
        tipo = st.selectbox("Tipo de inmueble", ['piso', 'atico', 'duplex', 'estudio', 'chalet'])
        reforma = st.selectbox("Estado", ["Buen estado", "A reformar", "Obra nueva"])
        
    st.markdown("<hr style='opacity: 0.1; margin: 2rem 0;'>", unsafe_allow_html=True)
    
    st.subheader("Amenidades y Visual")
    c1, c2, c3, c4 = st.columns(4)
    with c1: asc = st.selectbox("Ascensor", ["Sí", "No", "Desconocido"])
    with c2: ter = st.selectbox("Terraza", ["Sí", "No", "Desconocido"])
    with c3: gar = st.selectbox("Garaje", ["Sí", "No", "Desconocido"])
    with c4: cal = st.selectbox("Calefacción", ["Sí", "No", "Desconocido"])

    fotos = st.file_uploader("Adjuntar fotos para análisis visual", accept_multiple_files=True)
    n_fotos = len(fotos) if fotos else 0
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    if st.button("GENERAR VALORACIÓN PROFESIONAL"):
        # Mapeos numéricos (Ternario de Miguel)
        m_reforma = {"A reformar": -1, "Buen estado": 0, "Obra nueva": 1}
        m_bool = {"Sí": 1, "No": 0, "Desconocido": -1}
        
        # Construcción de las 70 columnas exactas
        datos = {
            'superficie_m2': superficie, 'habitaciones': habitaciones, 'banos': banos, 
            'planta': planta, 'zona_scraping': zona_key, 'codigo_postal': cp, 
            'descripcion': "-", 'tipo_inmueble': tipo, 'fuente': "pisos.com", 
            'estado_reforma': m_reforma[reforma], 'ascensor': m_bool[asc], 
            'terraza': m_bool[ter], 'garaje': m_bool[gar], 'calefaccion': m_bool[cal], 
            'num_imagenes': n_fotos, 'url': "-", 'fecha_scraping': "-", 'titulo': "-", 
            'url_imagen_principal': "-", 'urls_imagenes': "-"
        }
        
        df = pd.DataFrame([datos])
        # Añado las 50 PCA visuales
        for i in range(1, 51):
            df[f'pca_{i}'] = medianas_pca.get(f'pca_{i}', 0.0)
            
        try:
            X_trans = preprocesador.transform(df)
            p_log = modelo_xgb.predict(X_trans)[0]
            precio = np.expm1(p_log)
            
            st.markdown(
                f"""<div class='resultado-precio'>
                    <p style='color: #b91c1c; font-weight: 700; letter-spacing: 2px;'>VALORACIÓN ESTIMADA VALORALIA</p>
                    <p style='font-size: 3.5rem; font-weight: 800; color: #f8fafc; margin: 0;'>{precio:,.0f} EUR</p>
                    <p style='color: #94a3b8; font-size: 0.9rem; margin-top: 15px;'>MAE: 210.799 EUR | MAPE: 20,09% | R²: 0,9146</p>
                </div>""", unsafe_allow_html=True)
            st.balloons()
            
        except Exception as err:
            st.error(f"Fallo matemático en la predicción: {err}")

st.markdown("<p style='text-align: center; color: #475569; padding: 20px;'>© 2026 Valoralia Systems | Proyecto TFM</p>", unsafe_allow_html=True)
