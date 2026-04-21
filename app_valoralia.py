import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb

# ==============================================================================
# CONFIGURACIÓN VISUAL IMPACTANTE Y PREMIUM
# ==============================================================================
st.set_page_config(
    page_title="Valoralia Systems",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Inyección de CSS para fondo de edificio oscuro (opacidad controlada) y colores vibrantes
st.markdown(
    """
    <style>
    /* Fondo con imagen de edificios y capa azul marino oscura para dar contraste premium */
    .stApp {
        background-image: linear-gradient(rgba(15, 23, 42, 0.85), rgba(15, 23, 42, 0.95)), 
                          url('https://images.unsplash.com/photo-1543783230-278398a4ee4d?q=80&w=2070');
        background-size: cover;
        background-attachment: fixed;
    }
    
    /* Forzar que todos los textos base sean blancos para que resalten */
    h1, h2, h3, p, label, .stMarkdown, .stText {
        color: #f8fafc !important;
    }
    
    /* Tarjeta de cristal translúcido (Glassmorphism) */
    .caja-cristal {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 20px;
        padding: 40px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.6);
        margin-bottom: 20px;
    }
    
    /* Título principal con color dorado corporativo */
    .titulo-principal {
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        background: -webkit-linear-gradient(45deg, #f59e0b, #fbbf24);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }
    
    .subtitulo {
        text-align: center;
        font-size: 1.2rem;
        color: #cbd5e1 !important;
        margin-bottom: 40px;
        font-weight: 300;
    }
    
    /* Botón gigante de llamada a la acción con gradiente vibrante */
    .stButton>button {
        background: linear-gradient(90deg, #d97706, #f59e0b) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 800 !important;
        font-size: 1.3rem !important;
        padding: 15px !important;
        width: 100%;
        box-shadow: 0 4px 15px rgba(245, 158, 11, 0.4);
        transition: transform 0.2s ease;
    }
    
    .stButton>button:hover {
        transform: scale(1.02);
    }
    
    /* Cajas de selección y números oscurecidas para encajar en el diseño */
    .stSelectbox>div>div, .stNumberInput>div>div {
        background-color: rgba(15, 23, 42, 0.6) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }
    
    /* Caja de resultado final impactante en verde esmeralda */
    .caja-resultado {
        background: rgba(16, 185, 129, 0.15);
        border: 2px solid #10b981;
        border-radius: 15px;
        padding: 30px;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.2);
    }
    
    .precio-final {
        font-size: 4rem;
        font-weight: 800;
        color: #34d399;
        margin: 0;
        line-height: 1;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Dibujos vectoriales SVG con color corporativo dorado (Nada de emoticonos)
icono_ubicacion = """<svg style="vertical-align: bottom; margin-right: 8px;" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#fbbf24" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"></path><circle cx="12" cy="10" r="3"></circle></svg>"""
icono_edificio = """<svg style="vertical-align: bottom; margin-right: 8px;" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#fbbf24" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="2" width="16" height="20" rx="2" ry="2"></rect><path d="M9 22v-4h6v4"></path><path d="M8 6h.01"></path><path d="M16 6h.01"></path><path d="M12 6h.01"></path><path d="M12 10h.01"></path><path d="M12 14h.01"></path><path d="M16 10h.01"></path><path d="M16 14h.01"></path><path d="M8 10h.01"></path><path d="M8 14h.01"></path></svg>"""

# ==============================================================================
# CARGA DE ARTEFACTOS MATEMÁTICOS (SIN INVENTOS)
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
    st.error(f"Fallo crítico al cargar la inteligencia artificial: {e}")
    st.stop()

# ==============================================================================
# INTERFAZ Y LIMITACIONES ANTI-BROMAS
# ==============================================================================
st.markdown("<div class='titulo-principal'>Valoralia Systems</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitulo'>Motor Predictivo Híbrido de Alta Precisión para Real Estate</div>", unsafe_allow_html=True)

st.markdown("<div class='caja-cristal'>", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown(f"<h3>{icono_ubicacion} Ubicación y Espacio</h3>", unsafe_allow_html=True)
    # Lista estricta del CSV original
    zona = st.selectbox("Distrito de Madrid", ["Centro", "Salamanca", "Chamberí", "Retiro", "Chamartín", "Tetuán", "Fuencarral-El Pardo", "Moncloa-Aravaca", "Latina", "Carabanchel", "Usera", "Puente de Vallecas", "Moratalaz", "Ciudad Lineal", "Hortaleza", "Villaverde", "Villa de Vallecas", "Vicálvaro", "San Blas-Canillejas", "Barajas", "Arganzuela", "Pozuelo de Alarcón"])
    
    # Límites lógicos absolutos para no romper el modelo
    superficie = st.number_input("Superficie útil (m²)", min_value=20, max_value=1200, value=85, step=5)
    habitaciones = st.number_input("Habitaciones", min_value=1, max_value=15, value=2, step=1)
    
with col2:
    st.markdown(f"<h3>{icono_edificio} Estructura y Calidad</h3>", unsafe_allow_html=True)
    
    # Límite anti-bromas: No puedes tener 7000 baños. Como mucho, las habitaciones que tengas más 2 de cortesía.
    tope_banos = max(1, habitaciones + 2)
    banos = st.number_input("Baños completos", min_value=1, max_value=tope_banos, value=1, step=1)
    
    planta = st.number_input("Planta del inmueble (0 = Bajo)", min_value=-2, max_value=50, value=2, step=1)
    
    # Estados exactos del CSV
    estado = st.selectbox("Estado de conservación", ["A reformar", "Buen estado", "Obra nueva"])

st.markdown("</div>", unsafe_allow_html=True)

# Botón de tasación
if st.button("CALCULAR TASACIÓN DE MERCADO"):
    
    # Preparo los datos exactos que pide el modelo sin inventarme ni una sola columna
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
    
    # Inyecto las medianas reales de mi CSV para rellenar la parte de las fotos y que el modelo no explote
    for i in range(1, 51):
        col_pca = f'pca_{i}'
        fila[col_pca] = medianas_pca.get(col_pca, 0.0)
        
    df_input = pd.DataFrame([fila])
    
    try:
        # Traducción y predicción estricta
        X_trans = preprocesador.transform(df_input)
        pred_log = modelo_xgb.predict(X_trans)[0]
        precio_final = np.expm1(pred_log)
        
        # Resultado con diseño espectacular verde brillante
        st.markdown(
            f"""
            <div class='caja-resultado'>
                <p style='font-size: 1.2rem; font-weight: 600; color: #a7f3d0; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 5px;'>Valor Estimado</p>
                <p class='precio-final'>{precio_final:,.0f} €</p>
                <p style='margin-top: 15px; color: #6ee7b7; font-size: 1.1rem;'>Ratio de mercado: {precio_final/superficie:,.0f} €/m²</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        st.balloons()
        
    except Exception as e:
        st.error(f"Fallo en la ejecución matemática: {e}")
