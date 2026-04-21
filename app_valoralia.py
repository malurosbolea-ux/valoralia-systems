import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb

# ==============================================================================
# IDENTIDAD CORPORATIVA VALORALIA (Azul Marino, Rojo, Gris)
# ==============================================================================
st.set_page_config(
    page_title="Valoralia Systems | Real Estate Intelligence",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Inyección de CSS: Fondo con edificio, opacidad y paleta corporativa
st.markdown(
    """
    <style>
    /* Fondo con imagen de arquitectura y capa azul marino translúcida */
    .stApp {
        background: linear-gradient(rgba(15, 23, 42, 0.9), rgba(30, 41, 59, 0.95)), 
                    url('https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?q=80&w=2070');
        background-size: cover;
        background-attachment: fixed;
    }
    
    /* Tarjetas con efecto cristal y bordes grisáceos */
    .caja-premium {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 12px;
        padding: 35px;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.3);
        margin-bottom: 25px;
    }
    
    /* Tipografía y Colores Corporativos */
    h1 {
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        text-align: left;
        border-left: 5px solid #e11d48; /* Rojo Corporativo */
        padding-left: 15px;
    }
    
    label { color: #cbd5e1 !important; font-weight: 500 !important; }
    
    /* Botón de acción Rojo Valoralia */
    .stButton>button {
        background: #e11d48 !important;
        color: white !important;
        border: none !important;
        border-radius: 4px !important;
        font-weight: 700 !important;
        height: 50px;
        width: 100%;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background: #be123c !important;
        box-shadow: 0 0 20px rgba(225, 29, 72, 0.4);
    }

    /* Caja de resultado Azul Marino con acento Rojo */
    .resultado-caja {
        background: #0f172a;
        border-top: 4px solid #e11d48;
        padding: 30px;
        border-radius: 0 0 10px 10px;
        text-align: center;
    }
    
    .precio-final {
        font-size: 3.5rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Iconos Vectoriales (SVG) en Gris y Rojo (Sin emojis)
svg_pin = '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#e11d48" stroke-width="2"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"></path><circle cx="12" cy="10" r="3"></circle></svg>'
svg_chart = '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><line x1="3" y1="9" x2="21" y2="9"></line><line x1="9" y1="21" x2="9" y2="9"></line></svg>'

# ==============================================================================
# CARGA DE DATOS REALES (RIGOR TÉCNICO)
# ==============================================================================
@st.cache_resource
def cargar_activos():
    # Cargo mis archivos originales. Si no están, la app avisa, no inventa.
    prepro = joblib.load('preprocesador.pkl')
    model = xgb.XGBRegressor()
    model.load_model('modelo_xgb.json')
    meds_pca = joblib.load('medianas_pca.pkl')
    return prepro, model, meds_pca

try:
    preprocesador, modelo_xgb, medianas_pca = cargar_activos()
except Exception as e:
    st.error(f"Error de conexión con el núcleo de datos: {e}")
    st.stop()

# ==============================================================================
# INTERFAZ PROFESIONAL Y LIMITACIONES ANTI-BROMAS
# ==============================================================================
st.markdown("<h1>Valoralia Systems</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #94a3b8; margin-left: 20px;'>Soluciones Predictivas para el Sector Inmobiliario de Lujo</p>", unsafe_allow_html=True)

with st.container():
    st.markdown("<div class='caja-premium'>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown(f"<h3>{svg_pin} Parámetros de Ubicación</h3>", unsafe_allow_html=True)
        distrito = st.selectbox("Distrito Municipal", ["Centro", "Salamanca", "Chamberí", "Retiro", "Chamartín", "Tetuán", "Fuencarral-El Pardo", "Moncloa-Aravaca", "Latina", "Carabanchel", "Usera", "Puente de Vallecas", "Moratalaz", "Ciudad Lineal", "Hortaleza", "Villaverde", "Villa de Vallecas", "Vicálvaro", "San Blas-Canillejas", "Barajas", "Arganzuela", "Pozuelo de Alarcón"])
        superficie = st.number_input("Superficie Total (m²)", min_value=25, max_value=900, value=85)
        habitaciones = st.number_input("Dormitorios", min_value=1, max_value=12, value=2)
        
    with col2:
        st.markdown(f"<h3>{svg_chart} Análisis de Activo</h3>", unsafe_allow_html=True)
        
        # LÍMITE REAL: No permito más baños que habitaciones + 2 (evita datos basura)
        max_banos = habitaciones + 2
        banos = st.number_input("Cuartos de Baño", min_value=1, max_value=max_banos, value=1)
        
        planta = st.number_input("Altura (Planta)", min_value=0, max_value=45, value=2)
        estado = st.selectbox("Certificación de Estado", ["A reformar", "Buen estado", "Obra nueva"])

    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("Generar Valoración de Mercado"):
        # Construyo la fila con las 71 columnas que el modelo exige
        datos = {
            'superficie_m2': superficie,
            'habitaciones': habitaciones,
            'banos': banos,
            'planta': planta,
            'zona_scraping': distrito,
            'descripcion': estado,
            'tipo_inmueble': 'Pisos',
            'fuente': 'Venta'
        }
        
        # Relleno visual con las medianas reales de mi entrenamiento (Fallback honesto)
        for i in range(1, 51):
            c_pca = f'pca_{i}'
            datos[c_pca] = medianas_pca.get(c_pca, 0.0)
            
        df_scoring = pd.DataFrame([datos])
        
        try:
            # Transformación y Predicción
            X_scoring = preprocesador.transform(df_scoring)
            p_log = modelo_xgb.predict(X_scoring)[0]
            valoralia_index = np.expm1(p_log)
            
            st.markdown(
                f"""
                <div class='resultado-caja'>
                    <p style='color: #e11d48; font-weight: 700; letter-spacing: 3px;'>TASACIÓN ESTIMADA</p>
                    <p class='precio-final'>{valoralia_index:,.0f} €</p>
                    <p style='color: #94a3b8; font-size: 0.9rem;'>Basado en el análisis de 7.970 activos reales en Madrid</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
            st.balloons()
            
        except Exception as e:
            st.error(f"Fallo en el pipeline de datos: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

# Footer Corporativo Limpio
st.markdown(
    "<div style='text-align: center; color: #475569; font-size: 0.8rem; padding: 20px;'>"
    "© 2026 Valoralia Systems | Proprietary AI Model | Madrid, Spain"
    "</div>", 
    unsafe_allow_html=True
)
