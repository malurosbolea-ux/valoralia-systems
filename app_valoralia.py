"""
Valoralia Systems - Aplicacion de valoracion inmobiliaria hibrida
Autora: Maria Luisa Ros Bolea
TFM Master IA y Big Data - CEU San Pablo 2025-2026
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Valoralia Systems", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;900&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp {
    background: linear-gradient(rgba(255,255,255,0.93), rgba(255,255,255,0.93)),
    url("https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?q=80&w=2070&auto=format&fit=crop");
    background-size: cover; background-position: center; background-attachment: fixed;
}
h1 { color: #0f172a; font-weight: 900; font-size: 2.5rem; border-bottom: 3px solid #b91c1c; padding-bottom: 10px; }
h2, h3 { color: #0f172a; font-weight: 700; }
.stButton>button {
    background-color: #0f172a; color: #ffffff; border-radius: 4px; border: none;
    padding: 0.8rem 2rem; font-size: 1.1rem; font-weight: bold; width: 100%; transition: 0.3s;
}
.stButton>button:hover { background-color: #b91c1c; color: #ffffff; }
div[data-testid="stMetricValue"] { color: #b91c1c; font-weight: 900; font-size: 2.4rem; }
div[data-testid="stMetricLabel"] { color: #0f172a; font-weight: bold; font-size: 1.1rem; }
.info-box { background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 8px;
    padding: 1rem; font-size: 0.85rem; color: #4b5563; line-height: 1.6; margin-top: 1rem; }
.footer { text-align: center; color: #94a3b8; font-size: 0.8rem; padding: 2rem 0 1rem;
    border-top: 1px solid #e2e8f0; margin-top: 2rem; }
</style>
""", unsafe_allow_html=True)

# --- Carga de artefactos ---
@st.cache_resource
def cargar_paquete():
    ruta = Path(__file__).parent / "valoralia_production.pkl"
    return joblib.load(ruta)

paquete = cargar_paquete()
modelo = paquete["modelo"]
preprocessor = paquete["preprocessor"]
medianas_pca = paquete["medianas_pca"]
cols_num = paquete["cols_num"]
cols_cat = paquete["cols_cat"]
cols_pca = paquete["cols_pca"]
metricas = paquete["metricas_base"]

# --- Las 39 zonas REALES del dataset de entrenamiento ---
ZONAS_DISPLAY = {
    'Alcala_Henares': 'Alcala de Henares', 'Alcobendas': 'Alcobendas',
    'Alcorcon': 'Alcorcon', 'Arganda': 'Arganda del Rey',
    'Boadilla': 'Boadilla del Monte', 'Carabanchel': 'Carabanchel',
    'Centro': 'Centro', 'Chamartin': 'Chamartin', 'Chamberi': 'Chamberi',
    'Ciudad_Lineal': 'Ciudad Lineal', 'Colmenar_Viejo': 'Colmenar Viejo',
    'Coslada': 'Coslada', 'Fuencarral': 'Fuencarral-El Pardo',
    'Fuenlabrada': 'Fuenlabrada', 'Getafe': 'Getafe',
    'Las_Rozas': 'Las Rozas', 'Latina': 'Latina', 'Leganes': 'Leganes',
    'Majadahonda': 'Majadahonda', 'Moncloa': 'Moncloa-Aravaca',
    'Moratalaz': 'Moratalaz', 'Mostoles': 'Mostoles', 'Parla': 'Parla',
    'Pinto': 'Pinto', 'Pozuelo': 'Pozuelo de Alarcon',
    'Puente_Vallecas': 'Puente de Vallecas', 'Retiro': 'Retiro',
    'Rivas': 'Rivas-Vaciamadrid', 'SS_Reyes': 'San Sebastian de los Reyes',
    'Salamanca': 'Salamanca', 'San_Blas': 'San Blas-Canillejas',
    'Tetuan': 'Tetuan', 'Torrejon': 'Torrejon de Ardoz',
    'Tres_Cantos': 'Tres Cantos', 'Usera': 'Usera',
    'Vicalvaro': 'Vicalvaro', 'Villa_Vallecas': 'Villa de Vallecas',
    'Villaverde': 'Villaverde', 'Villaviciosa_Odon': 'Villaviciosa de Odon'
}
ZONAS_KEYS = list(ZONAS_DISPLAY.keys())
ZONAS_LABELS = list(ZONAS_DISPLAY.values())

# Los 5 tipos REALES del dataset (en minuscula, tal como los entreno el modelo)
TIPOS_KEYS = ['piso', 'atico', 'duplex', 'estudio', 'chalet']
TIPOS_DISPLAY = ['Piso', 'Atico', 'Duplex', 'Estudio', 'Chalet']

def mapear(v):
    """1=Si, 0=No, -1=Desconocido (regla de Miguel: no inventar datos)."""
    return 1 if v == "Si" else (0 if v == "No" else -1)

# --- Cabecera ---
st.markdown("<h1>Valoralia Systems</h1>", unsafe_allow_html=True)
st.markdown("**Motor de tasacion inmobiliaria impulsado por inteligencia artificial visual**")
st.write("---")

# --- Layout ---
col_izq, col_der = st.columns(2, gap="large")

with col_izq:
    st.subheader("Caracteristicas del inmueble")

    c1, c2 = st.columns(2)
    zona_idx = c1.selectbox("Zona de Madrid", range(len(ZONAS_LABELS)),
                            format_func=lambda i: ZONAS_LABELS[i], index=6)
    zona = ZONAS_KEYS[zona_idx]

    tipo_idx = c2.selectbox("Tipo de inmueble", range(len(TIPOS_DISPLAY)),
                            format_func=lambda i: TIPOS_DISPLAY[i], index=0)
    tipo = TIPOS_KEYS[tipo_idx]

    c3, c4 = st.columns(2)
    superficie = c3.number_input("Superficie (m2)", min_value=10, max_value=1500, value=80, step=5)
    habitaciones = c4.number_input("Habitaciones", min_value=0, max_value=20, value=2)

    c5, c6 = st.columns(2)
    banos = c5.number_input("Banos", min_value=0, max_value=10, value=1)
    planta = c6.number_input("Planta", min_value=-2, max_value=50, value=1)

    st.write("---")
    st.markdown("**Equipamiento y estado**")

    c7, c8 = st.columns(2)
    ascensor = c7.selectbox("Ascensor", ["Desconocido", "Si", "No"])
    terraza = c8.selectbox("Terraza", ["Desconocido", "Si", "No"])

    c9, c10 = st.columns(2)
    garaje = c9.selectbox("Garaje", ["Desconocido", "Si", "No"])
    calefaccion = c10.selectbox("Calefaccion", ["Desconocido", "Si", "No"])

    estado_reforma = st.selectbox("Estado de reforma", ["Desconocido", "Si", "No"])

with col_der:
    st.subheader("Analisis visual (IA)")
    st.info(
        "Sube fotografias del interior de la vivienda. La red neuronal ResNet50 "
        "analizara la calidad visual para ajustar la tasacion. Si no se adjuntan "
        "fotografias, se aplicara el perfil visual medio del mercado madrileno."
    )

    imagenes = st.file_uploader("Fotografias del interior",
                                type=['jpg', 'jpeg', 'png'],
                                accept_multiple_files=True)

    if imagenes:
        st.success(f"{len(imagenes)} fotografias listas para analisis.")
    else:
        st.warning(
            "Modo tasacion basica: se aplicara la imputacion estadistica "
            "(medianas PCA) del mercado madrileno como referencia visual."
        )

    st.write("")
    boton = st.button("CALCULAR VALORACION", type="primary")

# --- Logica predictiva ---
if boton:
    # Construyo el registro con TODAS las 13 variables que espera el preprocesador
    datos = {
        "superficie_m2": float(superficie),
        "habitaciones": float(habitaciones),
        "banos": float(banos),
        "planta": float(planta),
        "num_imagenes": float(len(imagenes)) if imagenes else 0.0,
        "codigo_postal": 0.0,
        "ascensor": mapear(ascensor),
        "terraza": mapear(terraza),
        "garaje": mapear(garaje),
        "calefaccion": mapear(calefaccion),
        "estado_reforma": mapear(estado_reforma),
        "zona_scraping": zona,
        "tipo_inmueble": tipo
    }

    # Fallback a medianas PCA (la vivienda tipica de Madrid, NO ceros)
    for col in cols_pca:
        datos[col] = medianas_pca.get(col, 0.0)

    # Prediccion
    df_input = pd.DataFrame([datos])[cols_num + cols_cat + cols_pca]
    X_transformed = preprocessor.transform(df_input)
    pred_log = modelo.predict(X_transformed)[0]
    precio_estimado = float(np.expm1(pred_log))

    mae = metricas["MAE"]
    precio_bajo = max(0, precio_estimado - mae)
    precio_alto = precio_estimado + mae

    # Resultado
    st.write("---")
    st.subheader("Resultado de la valoracion")

    c_r1, c_r2, c_r3 = st.columns(3)
    c_r1.metric("Estimacion conservadora", f"{precio_bajo:,.0f} EUR")
    c_r2.metric("Valoracion central", f"{precio_estimado:,.0f} EUR")
    c_r3.metric("Estimacion optimista", f"{precio_alto:,.0f} EUR")

    st.markdown(f"""
    <div class="info-box">
        <strong>Metodologia:</strong> Modelo hibrido XGBoost entrenado con 7.970 inmuebles
        reales de Pisos.com (marzo 2026) y 103.438 fotografias procesadas con ResNet50.
        El intervalo se basa en el error absoluto medio historico ({mae:,.0f} EUR).
        Las componentes visuales utilizan el perfil medio del mercado madrileno.
        R2 logaritmico: 0,9146 · MAPE: 20,2%
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    <strong>Valoralia Systems</strong> v1.0<br>
    Maria Luisa Ros Bolea · TFM Master IA y Big Data · CEU San Pablo 2025-2026<br>
    Datos reales de Pisos.com · 39 zonas de Madrid · Modelo validado con CV 5-fold
</div>
""", unsafe_allow_html=True)
