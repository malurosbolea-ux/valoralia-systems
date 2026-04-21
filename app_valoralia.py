import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

# Configuración de la página
st.set_page_config(page_title="Valoralia Systems", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(rgba(15, 23, 42, 0.95), rgba(15, 23, 42, 0.95)), 
                    url('https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?q=80&w=2070');
        background-size: cover;
        background-attachment: fixed;
    }
    h1, h2, h3, label, p { color: #f8fafc !important; font-family: 'Inter', sans-serif; }
    .caja-premium {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 12px;
        padding: 35px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    .titulo-v { border-left: 5px solid #e11d48; padding-left: 15px; margin-bottom: 25px; }
    .stButton>button {
        background: #e11d48 !important; color: white !important; border: none !important;
        height: 50px; width: 100%; font-weight: 700 !important; text-transform: uppercase;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def cargar_motor():
    prepro = joblib.load('preprocesador.pkl')
    model = xgb.XGBRegressor()
    model.load_model('modelo_xgb.json')
    meds_pca = joblib.load('medianas_pca.pkl')
    return prepro, model, meds_pca

try:
    preprocesador, modelo_xgb, medianas_pca = cargar_motor()
except Exception as e:
    st.error(f"Error de sistema: {e}")
    st.stop()

st.markdown("<div class='titulo-v'><h1>Valoralia Systems</h1></div>", unsafe_allow_html=True)

with st.container():
    st.markdown("<div class='caja-premium'>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.subheader("Ubicación y dimensiones")
        zonas_reales = ['Centro', 'Salamanca', 'Chamberi', 'Retiro', 'Chamartin', 'Tetuan', 
                        'Fuencarral', 'Moncloa', 'Latina', 'Carabanchel', 'Usera', 
                        'Puente_Vallecas', 'Moratalaz', 'Ciudad_Lineal', 'Hortaleza', 
                        'Villaverde', 'Villa_Vallecas', 'Vicalvaro', 'San_Blas', 'Barajas', 
                        'Pozuelo', 'SS_Reyes', 'Alcobendas', 'Majadahonda', 'Las_Rozas']
        zona = st.selectbox("Zona", zonas_reales)
        codigo_postal = st.text_input("Código postal", value="28001")
        superficie = st.number_input("Superficie (m²)", min_value=20, max_value=1000, value=85)
        habitaciones = st.number_input("Habitaciones", min_value=1, max_value=15, value=2)
        
    with col2:
        st.subheader("Características del inmueble")
        banos = st.number_input("Baños", min_value=1, max_value=habitaciones+2, value=1)
        planta = st.number_input("Planta", min_value=-2, max_value=50, value=2)
        tipo_inmueble = st.selectbox("Tipo de inmueble", ['piso', 'atico', 'duplex', 'estudio', 'chalet'])
        estado_reforma_texto = st.selectbox("Estado de la reforma", ["A reformar", "Buen estado", "Obra nueva"])
        
    st.markdown("---")
    st.subheader("Amenidades")
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a: ascensor_texto = st.selectbox("Ascensor", ["Sí", "No", "Desconocido"])
    with col_b: terraza_texto = st.selectbox("Terraza", ["Sí", "No", "Desconocido"])
    with col_c: garaje_texto = st.selectbox("Garaje", ["Sí", "No", "Desconocido"])
    with col_d: calefaccion_texto = st.selectbox("Calefacción", ["Sí", "No", "Desconocido"])

    st.markdown("---")
    st.subheader("Análisis visual")
    fotos = st.file_uploader("Adjuntar fotografías (solo conteo en versión demo)", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
    num_imagenes = len(fotos) if fotos else 0
    
    st.markdown("""
        <p style='color: #94a3b8; font-size: 0.85rem;'>
        * Nota de transparencia: Debido a las limitaciones computacionales del despliegue en la nube, 
        esta versión de la aplicación no ejecuta la red neuronal ResNet50 en tiempo real. Se inyectará el vector 
        visual promedio (medianas PCA) extraído durante la fase de entrenamiento, ajustando el modelo con la cantidad 
        de imágenes proporcionadas.
        </p>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("Calcular tasación de mercado"):
        
        # Mapeo de diccionarios estricto paso a paso
        mapeo_ternario = {"Sí": 1, "No": 0, "Desconocido": -1}
        mapeo_estado = {"A reformar": -1, "Buen estado": 0, "Obra nueva": 1}
        
        ascensor = mapeo_ternario[ascensor_texto]
        terraza = mapeo_ternario[terraza_texto]
        garaje = mapeo_ternario[garaje_texto]
        calefaccion = mapeo_ternario[calefaccion_texto]
        estado_reforma = mapeo_estado[estado_reforma_texto]

        datos_entrada = {
            'superficie_m2': superficie,
            'habitaciones': habitaciones,
            'banos': banos,
            'planta': planta,
            'zona_scraping': zona,
            'codigo_postal': codigo_postal,
            'descripcion': 'Inmueble estandar en venta en Madrid.', 
            'tipo_inmueble': tipo_inmueble,
            'fuente': 'pisos.com',
            'estado_reforma': estado_reforma,
            'ascensor': ascensor,
            'terraza': terraza,
            'garaje': garaje,
            'calefaccion': calefaccion,
            'num_imagenes': num_imagenes
        }
        
        df_base = pd.DataFrame([datos_entrada])
        
        for i in range(1, 51):
            col_pca = f'pca_{i}'
            df_base[col_pca] = medianas_pca.get(col_pca, 0.0)
            
        try:
            X_trans = preprocesador.transform(df_base)
            prediccion_log = modelo_xgb.predict(X_trans)[0]
            precio_estimado = np.expm1(prediccion_log)
            
            st.markdown(
                f"""
                <div style='background: #0f172a; border-top: 4px solid #e11d48; padding: 25px; border-radius: 8px; text-align: center; margin-top: 20px;'>
                    <p style='color: #e11d48; font-weight: 700; letter-spacing: 2px; margin-bottom: 5px;'>VALORACIÓN ESTIMADA</p>
                    <p style='font-size: 3.5rem; font-weight: 800; color: #ffffff; margin: 0;'>{precio_estimado:,.0f} €</p>
                    <p style='color: #94a3b8; font-size: 0.9rem;'>Ratio: {precio_estimado/superficie:,.0f} €/m²</p>
                </div>
                """, unsafe_allow_html=True
            )
            
        except Exception as err:
            st.error(f"Fallo en la ejecución matemática: {err}")

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #475569; font-size: 0.8rem; padding: 20px;'>© 2026 Valoralia Systems | Proyecto Fin de Máster</p>", unsafe_allow_html=True)
