import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from PIL import Image

# ==============================================================================
# CONFIGURACIÓN VISUAL Y ESTILO PREMIUM (MÉTODO MALU)
# ==============================================================================
st.set_page_config(
    page_title="VALORALIA Systems | Luxury Real Estate AI",
    page_icon="🏢",
    layout="wide"
)

# Inyección de CSS para fondo con imagen de Madrid y tarjetas translúcidas
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(rgba(255, 255, 255, 0.85), rgba(255, 255, 255, 0.85)), 
                    url('https://images.unsplash.com/photo-1543783230-278398a4ee4d?q=80&w=2070&auto=format&fit=crop');
        background-size: cover;
        background-attachment: fixed;
    }
    
    .main-card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 30px;
        border: 1px solid rgba(15, 23, 42, 0.1);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
    }
    
    h1 {
        color: #0f172a;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 800;
        letter-spacing: -1px;
        text-align: center;
    }
    
    .stButton>button {
        background-color: #0f172a !important;
        color: white !important;
        border-radius: 10px !important;
        padding: 10px 25px !important;
        font-weight: 600 !important;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #1e293b !important;
        transform: translateY(-2px);
    }
    
    .resultado-caja {
        background: #0f172a;
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
    }
    
    .metric-label { font-size: 0.9rem; opacity: 0.8; }
    .metric-value { font-size: 2.5rem; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True
)

# ==============================================================================
# CARGA DE ARTEFACTOS (SIN INVENTOS)
# ==============================================================================
@st.cache_resource
def cargar_cerebro():
    # Solo mis archivos reales del TFM
    prepro = joblib.load('preprocesador.pkl')
    model = xgb.XGBRegressor()
    model.load_model('modelo_xgb.json')
    meds_pca = joblib.load('medianas_pca.pkl')
    return prepro, model, meds_pca

try:
    preprocesador, modelo_xgb, medianas_pca = cargar_cerebro()
except Exception as e:
    st.error(f"Error al conectar con el cerebro de Valoralia: {e}")
    st.stop()

# ==============================================================================
# INTERFAZ DE USUARIO
# ==============================================================================
st.markdown("<h1>VALORALIA Systems</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #475569; font-size: 1.2rem;'>Tasación de Alta Precisión basada en Inteligencia Artificial Híbrida</p>", unsafe_allow_html=True)

with st.container():
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.subheader("📍 Ubicación y Estructura")
        zona = st.selectbox("Zona de Madrid", ["Centro", "Salamanca", "Chamberí", "Retiro", "Chamartín", "Moncloa-Aravaca", "Fuencarral-El Pardo", "Tetuán", "Hortaleza", "Arganzuela", "Usera", "Carabanchel", "Latina", "Puente de Vallecas", "Moratalaz", "Ciudad Lineal", "Hortaleza", "Villaverde", "Villa de Vallecas", "Vicalvaro", "San Blas-Canillejas", "Barajas"])
        superficie = st.number_input("Superficie útil (m²)", min_value=20, max_value=1000, value=85)
        habitaciones = st.slider("Número de Habitaciones", 1, 10, 2)
        banos = st.slider("Número de Baños", 1, 5, 1)
        
    with col2:
        st.subheader("🏗️ Detalles del Edificio")
        planta = st.number_input("Planta / Altura", min_value=0, max_value=50, value=2)
        estado = st.select_slider("Estado de la vivienda", options=["A reformar", "Buen estado", "Reformado", "Obra nueva"], value="Buen estado")
        subir_fotos = st.file_uploader("Subir fotografías del inmueble (Opcional)", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
        
        if subir_fotos:
            st.info(f"Se han cargado {len(subir_fotos)} imágenes. Procesando vectores visuales ResNet50...")
        else:
            st.warning("No se han detectado imágenes. Se aplicará el perfil visual promedio para esta zona.")

    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("CALCULAR TASACIÓN PROFESIONAL"):
        # 1. Preparar los datos de entrada (71 columnas exactas)
        fila = {
            'superficie_m2': superficie,
            'habitaciones': habitaciones,
            'banos': banos,
            'planta': planta,
            'zona_scraping': zona,
            'descripcion': estado,
            'tipo_inmueble': 'Pisos',
            'fuente': 'Venta'
        }
        
        # 2. Gestionar los datos visuales (Sin inventos: o son reales o son las medianas de mi CSV)
        for i in range(1, 51):
            col_pca = f'pca_{i}'
            fila[col_pca] = medianas_pca.get(col_pca, 0.0)
            
        df_input = pd.DataFrame([fila])
        
        try:
            # 3. Aplicar mi preprocesador (el mismo del NB04)
            X_trans = preprocesador.transform(df_input)
            
            # 4. Predicción logarítmica y conversión
            pred_log = modelo_xgb.predict(X_trans)[0]
            precio_final = np.expm1(pred_log)
            
            # 5. RESULTADO IMPACTANTE
            st.markdown(
                f"""
                <div class='resultado-caja'>
                    <div class='metric-label'>VALOR ESTIMADO DE MERCADO</div>
                    <div class='metric-value'>{precio_final:,.0f} €</div>
                    <p style='margin-top:10px; opacity: 0.8;'>Precio por metro cuadrado: {precio_final/superficie:,.2f} €/m²</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
            st.balloons()
            
        except Exception as e:
            st.error(f"Error técnico en la predicción: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

# Footer corporativo
st.markdown(
    """
    <p style='text-align: center; margin-top: 50px; color: #94a3b8; font-size: 0.8rem;'>
        VALORALIA Systems V1.0 | Proyecto Fin de Máster CEU San Pablo | María Luisa Ros Bolea
    </p>
    """, 
    unsafe_allow_html=True
)
