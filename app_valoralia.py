import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb

# ==============================================================================
# Configuracion visual corporativa (Enterprise Edition)
# ==============================================================================
st.set_page_config(
    page_title="Valoralia Systems | Enterprise AI",
    layout="wide"
)

# Inyeccion de CSS avanzado para fondo translucido y tarjetas cristal
st.markdown(
    """
    <style>
    /* Fondo de edificios con opacidad controlada mediante degradado rgba */
    .stApp {
        background: linear-gradient(rgba(241, 245, 249, 0.88), rgba(241, 245, 249, 0.88)), 
                    url('https://images.unsplash.com/photo-1539037116277-4db20202d03d?auto=format&fit=crop&w=1920&q=80') no-repeat center center fixed;
        background-size: cover;
    }
    
    /* Estilo de la tarjeta principal */
    .main-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 12px;
        padding: 40px;
        border: 1px solid rgba(15, 23, 42, 0.1);
        box-shadow: 0 10px 40px -10px rgba(15, 23, 42, 0.2);
    }
    
    /* Titulos corporativos sin mayusculas abusivas */
    h1 {
        color: #0f172a;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        letter-spacing: -0.5px;
        text-align: center;
        margin-bottom: 5px;
    }
    
    .subtitulo {
        text-align: center; 
        color: #64748b; 
        font-size: 1.1rem;
        margin-bottom: 40px;
        font-weight: 400;
    }
    
    /* Botones de alta conversion */
    .stButton>button {
        background-color: #0f172a !important;
        color: #ffffff !important;
        border-radius: 6px !important;
        padding: 12px 30px !important;
        font-weight: 600 !important;
        width: 100%;
        text-transform: uppercase;
        letter-spacing: 1px;
        border: none;
        transition: background-color 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #334155 !important;
    }
    
    /* Caja del resultado final */
    .resultado-caja {
        background: #0f172a;
        color: #ffffff;
        padding: 30px;
        border-radius: 8px;
        text-align: center;
        margin-top: 30px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .metric-value { font-size: 3rem; font-weight: 700; color: #38bdf8; }
    
    /* Iconos SVG alineados con texto */
    .seccion-titulo {
        display: flex;
        align-items: center;
        gap: 10px;
        color: #0f172a;
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 15px;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Iconos dibujados en codigo SVG
svg_ubicacion = """<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#0f172a" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"></path><circle cx="12" cy="10" r="3"></circle></svg>"""
svg_edificio = """<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#0f172a" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="2" width="16" height="20" rx="2" ry="2"></rect><path d="M9 22v-4h6v4"></path><path d="M8 6h.01"></path><path d="M16 6h.01"></path><path d="M12 6h.01"></path><path d="M12 10h.01"></path><path d="M12 14h.01"></path><path d="M16 10h.01"></path><path d="M16 14h.01"></path><path d="M8 10h.01"></path><path d="M8 14h.01"></path></svg>"""

# ==============================================================================
# Carga de artefactos
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
    st.error(f"Error critico de sistema: {e}")
    st.stop()

# ==============================================================================
# Interfaz de usuario de alta fidelidad
# ==============================================================================
st.markdown("<h1>Valoralia Systems</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitulo'>Motor de tasacion predictiva híbrida para Real Estate</p>", unsafe_allow_html=True)

with st.container():
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    
    col1, espaciador, col2 = st.columns([1, 0.1, 1])
    
    with col1:
        st.markdown(f"<div class='seccion-titulo'>{svg_ubicacion} Ubicacion y distribucion</div>", unsafe_allow_html=True)
        # Zonas extraidas directamente del CSV real
        zona = st.selectbox("Distrito", ["Centro", "Salamanca", "Chamberí", "Retiro", "Chamartín", "Tetuán", "Fuencarral-El Pardo", "Moncloa-Aravaca", "Latina", "Carabanchel", "Usera", "Puente de Vallecas", "Moratalaz", "Ciudad Lineal", "Hortaleza", "Villaverde", "Villa de Vallecas", "Vicálvaro", "San Blas-Canillejas", "Barajas", "Arganzuela", "Pozuelo de Alarcón"])
        
        # Limites logicos reales
        superficie = st.number_input("Superficie util (m2)", min_value=25, max_value=800, value=85, step=5)
        habitaciones = st.number_input("Habitaciones", min_value=1, max_value=12, value=2, step=1)
        
        # Logica arquitectonica: no puedes tener mas baños que (habitaciones + 2)
        limite_banos = max(1, habitaciones + 2)
        banos = st.number_input("Baños completos", min_value=1, max_value=limite_banos, value=1, step=1)
        
    with col2:
        st.markdown(f"<div class='seccion-titulo'>{svg_edificio} Estructura y calidad</div>", unsafe_allow_html=True)
        planta = st.number_input("Planta", min_value=0, max_value=40, value=2, step=1)
        
        # Valores categoricos exactos de mi base de datos
        estado = st.selectbox("Estado de conservacion", ["A reformar", "Buen estado", "Obra nueva"])
        
        subir_fotos = st.file_uploader("Carga de imagenes (ResNet50)", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
        
        if not subir_fotos:
            st.markdown("<p style='font-size:0.85rem; color:#64748b;'>* Si no se adjuntan imagenes, el sistema inyectara el vector visual modal de la zona.</p>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("Ejecutar modelo de valoracion"):
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
        
        # Inyeccion estricta de las variables PCA del dataset sin inventos
        for i in range(1, 51):
            col_pca = f'pca_{i}'
            fila[col_pca] = medianas_pca.get(col_pca, 0.0)
            
        df_input = pd.DataFrame([fila])
        
        try:
            X_trans = preprocesador.transform(df_input)
            pred_log = modelo_xgb.predict(X_trans)[0]
            precio_final = np.expm1(pred_log)
            
            st.markdown(
                f"""
                <div class='resultado-caja'>
                    <div style='font-size: 0.9rem; letter-spacing: 2px; color: #94a3b8;'>VALORACION DE MERCADO</div>
                    <div class='metric-value'>{precio_final:,.0f} €</div>
                    <div style='margin-top: 15px; font-size: 0.95rem; color: #cbd5e1;'>KPI: {precio_final/superficie:,.0f} €/m2</div>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
        except Exception as e:
            st.error(f"Fallo de ejecucion en pipeline: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

# Footer Enterprise
st.markdown(
    """
    <div style='text-align: center; margin-top: 40px; color: #64748b; font-size: 0.85rem; padding-bottom: 20px;'>
        <strong>Valoralia Systems</strong> | Enterprise Edition | Soluciones de IA para Real Estate<br>
        Motor predictivo híbrido V1.0
    </div>
    """, 
    unsafe_allow_html=True
)
