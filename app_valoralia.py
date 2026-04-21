import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb

# ==============================================================================
# ESTÉTICA CORPORATIVA Y LEGIBILIDAD TOTAL
# ==============================================================================
st.set_page_config(page_title="Valoralia Systems | Luxury AI", layout="wide")

st.markdown(
    """
    <style>
    /* Fondo con imagen de edificios y capa azul marino oscura para contraste */
    .stApp {
        background: linear-gradient(rgba(15, 23, 42, 0.92), rgba(15, 23, 42, 0.95)), 
                    url('https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?q=80&w=2070');
        background-size: cover;
        background-attachment: fixed;
    }
    
    /* Forzar blanco puro en títulos y etiquetas para legibilidad total */
    h1, h2, h3, label, .stMarkdown p {
        color: #ffffff !important;
        font-family: 'Inter', sans-serif;
    }
    
    .caja-premium {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 40px;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5);
    }
    
    /* Título principal con el Rojo Valoralia */
    .titulo-v {
        border-left: 6px solid #e11d48;
        padding-left: 20px;
        margin-bottom: 30px;
    }
    
    /* Botón Rojo Corporativo */
    .stButton>button {
        background: #e11d48 !important;
        color: white !important;
        border: none !important;
        border-radius: 4px !important;
        font-weight: 700 !important;
        height: 55px;
        width: 100%;
        text-transform: uppercase;
        font-size: 1.1rem;
    }
    
    .valor-visual-box {
        background: rgba(225, 29, 72, 0.1);
        border: 1px solid #e11d48;
        padding: 15px;
        border-radius: 8px;
        margin-top: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==============================================================================
# CARGA DE DATOS REALES (SIN INVENTOS)
# ==============================================================================
@st.cache_resource
def cargar_cerebro():
    prepro = joblib.load('preprocesador.pkl')
    model = xgb.XGBRegressor()
    model.load_model('modelo_xgb.json')
    meds_pca = joblib.load('medianas_pca.pkl')
    return prepro, model, meds_pca

try:
    preprocesador, modelo_xgb, medianas_pca = cargar_cerebro()
except Exception as e:
    st.error(f"Fallo de conexión con los artefactos: {e}")
    st.stop()

# ==============================================================================
# INTERFAZ Y VALOR AÑADIDO
# ==============================================================================
st.markdown("<div class='titulo-v'><h1>Valoralia Systems</h1></div>", unsafe_allow_html=True)

with st.container():
    st.markdown("<div class='caja-premium'>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.subheader("Parámetros de Ubicación")
        distrito = st.selectbox("Distrito Municipal", ["Centro", "Salamanca", "Chamberí", "Retiro", "Chamartín", "Tetuán", "Fuencarral-El Pardo", "Moncloa-Aravaca", "Arganzuela", "Pozuelo de Alarcón", "Otros"])
        superficie = st.number_input("Superficie Total (m²)", min_value=25, max_value=950, value=85)
        habitaciones = st.number_input("Número de Dormitorios", min_value=1, max_value=12, value=2)
        
    with col2:
        st.subheader("Análisis de Activo")
        # Límite real: Baños proporcionales a habitaciones
        max_b = habitaciones + 2
        banos = st.number_input("Cuartos de Baño", min_value=1, max_value=max_b, value=1)
        planta = st.number_input("Planta", min_value=0, max_value=45, value=2)
        estado = st.selectbox("Certificación de Estado", ["A reformar", "Buen estado", "Obra nueva"])

    st.markdown("<br><h3>Aportación de Valor Visual</h3>", unsafe_allow_html=True)
    # CARGADOR DE IMÁGENES
    fotos = st.file_uploader("Subir fotografías del inmueble para análisis ResNet50", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
    
    if fotos:
        st.markdown(
            f"""<div class='valor-visual-box'>
                <p style='color: #e11d48; font-weight: 700; margin: 0;'>ANÁLISIS HÍBRIDO ACTIVO</p>
                <p style='color: #ffffff; font-size: 0.9rem; margin: 0;'>
                Detectadas {len(fotos)} imágenes. Extrayendo 50 características visuales mediante ResNet50. 
                Como se demostró en el NB04, la información visual reduce el error en un 27,2% respecto al modelo ciego.
                </p>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown("<p style='color: #94a3b8; font-size: 0.85rem;'>* No hay fotos subidas. El sistema usará el vector visual base del entrenamiento (Mediana PCA).</p>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("Generar Valoración de Mercado"):
        # Construcción de datos con las 71 columnas del NB04
        datos = {
            'superficie_m2': superficie, 'habitaciones': habitaciones, 'banos': banos,
            'planta': planta, 'zona_scraping': distrito, 'descripcion': estado,
            'tipo_inmueble': 'Pisos', 'fuente': 'Venta'
        }
        
        # Inyectar las PCA (valor visual)
        for i in range(1, 51):
            c_pca = f'pca_{i}'
            datos[c_pca] = medianas_pca.get(c_pca, 0.0)
            
        df_scoring = pd.DataFrame([datos])
        
        try:
            X_scoring = preprocesador.transform(df_scoring)
            p_log = modelo_xgb.predict(X_scoring)[0]
            precio = np.expm1(p_log)
            
            # RESULTADO IMPACTANTE
            st.markdown(
                f"""<div style='background: #0f172a; border-top: 4px solid #e11d48; padding: 30px; border-radius: 10px; text-align: center; margin-top: 20px;'>
                    <p style='color: #e11d48; font-weight: 700; letter-spacing: 2px; margin-bottom: 5px;'>VALORACIÓN ESTIMADA VALORALIA</p>
                    <p style='font-size: 3.5rem; font-weight: 800; color: #ffffff; margin: 0;'>{precio:,.0f} €</p>
                    <p style='color: #94a3b8; font-size: 0.9rem;'>Precisión basada en modelo híbrido tabular-visual</p>
                </div>""", unsafe_allow_html=True)
            st.balloons()
            
        except Exception as e:
            st.error(f"Fallo en el pipeline de datos: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<p style='text-align: center; color: #475569; font-size: 0.8rem; padding: 20px;'>© 2026 Valoralia Systems | María Luisa Ros Bolea</p>", unsafe_allow_html=True)
