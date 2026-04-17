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
from PIL import Image

# Configuracion de pagina
st.set_page_config(
    page_title="Valoralia Systems | Tasacion Hibrida",
    layout="wide"
)

# Inyeccion de diseno corporativo (Comando 19996)
st.markdown("""
    <style>
    .main {background-color: #ffffff;}
    h1 {color: #0f172a; font-weight: 800; border-bottom: 2px solid #0f172a; padding-bottom: 10px;}
    h2, h3 {color: #4b5563;}
    .stButton>button {
        background-color: #0f172a;
        color: #ffffff;
        border-radius: 4px;
        border: none;
        padding: 0.6rem 2rem;
        font-weight: bold;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #b91c1c;
        color: #ffffff;
    }
    div[data-testid="stMetricValue"] {
        color: #b91c1c;
        font-weight: 800;
    }
    .stAlert {background-color: #f8f9fa; border-left-color: #0f172a;}
    </style>
    """, unsafe_allow_html=True)

# Cabecera
st.title("Valoralia Systems")
st.markdown("**Motor de tasacion inmobiliaria impulsado por Inteligencia Artificial Visual**")
st.markdown("---")

# Carga de artefactos
@st.cache_resource
def cargar_paquete():
    ruta = Path(__file__).parent / "valoralia_production.pkl"
    return joblib.load(ruta)

@st.cache_resource
def cargar_pipeline_visual():
    try:
        import torch
        import torchvision.models as models
        from torchvision.models import ResNet50_Weights
        import torchvision.transforms as transforms

        ruta_pca = Path(__file__).parent / "valoralia_pca_transformer.pkl"
        if not ruta_pca.exists():
            return None

        pca = joblib.load(ruta_pca)
        pesos = ResNet50_Weights.DEFAULT
        modelo_rn = models.resnet50(weights=pesos)
        
        modelo_extractor = torch.nn.Sequential(*(list(modelo_rn.children())[:-1]))
        modelo_extractor.eval()

        transformacion = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return pca, modelo_extractor, transformacion
    except Exception as e:
        st.warning(f"Aviso interno: Pipeline visual en modo fallback. Motivo: {e}")
        return None

try:
    paquete = cargar_paquete()
    modelo = paquete["modelo"]
    preprocessor = paquete["preprocesador"]
    medianas_pca = paquete["medianas_pca"]
    metricas = paquete["metricas_test"]
    cols_num = paquete["columnas_numericas"]
    cols_cat = paquete["columnas_categoricas"]
    cols_pca = paquete["columnas_pca"]
except FileNotFoundError:
    st.error("Error critico: Archivo valoralia_production.pkl no encontrado. Comprueba el repositorio.")
    st.stop()

# Estructura a dos columnas
col_izq, col_der = st.columns([1, 1], gap="large")

with col_izq:
    st.subheader("1. Caracteristicas Estructurales")
    
    c1, c2 = st.columns(2)
    superficie = c1.number_input("Superficie (m2)", min_value=15.0, value=90.0, step=5.0)
    habitaciones = c2.number_input("Habitaciones", min_value=0.0, value=3.0, step=1.0)
    
    c3, c4 = st.columns(2)
    banos = c3.number_input("Banos", min_value=0.0, value=2.0, step=1.0)
    planta = c4.number_input("Planta (-1 sotano, 0 bajo)", min_value=-1.0, value=2.0, step=1.0)

    st.markdown("##### Equipamiento")
    c5, c6 = st.columns(2)
    ascensor = c5.selectbox("Ascensor", options=[1, 0], format_func=lambda x: "Si" if x==1 else "No")
    terraza = c6.selectbox("Terraza", options=[1, 0], format_func=lambda x: "Si" if x==1 else "No")
    
    c7, c8 = st.columns(2)
    garaje = c7.selectbox("Garaje", options=[1, 0], format_func=lambda x: "Si" if x==1 else "No")
    estado_reforma = c8.selectbox("Estado Reforma", options=[1, 0], format_func=lambda x: "Reformado" if x==1 else "A reformar")
    
    zona = st.selectbox("Zona Geografica", options=["Madrid Capital", "Pozuelo de Alarcon", "Majadahonda", "Las Rozas"])

with col_der:
    st.subheader("2. Calidad Visual")
    st.info("Sube fotografias del interior para que la IA realice el ajuste fino de la tasacion basandose en los acabados y el estado de conservacion.")
    imagenes = st.file_uploader("Adjuntar imagenes (Opcional)", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
    
    if imagenes:
        st.success(f"{len(imagenes)} fotografia(s) cargada(s) correctamente.")
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    boton_calcular = st.button("EJECUTAR TASACION HIBRIDA")

# Logica de ejecucion
if boton_calcular:
    datos = {
        'superficie_m2': superficie,
        'habitaciones': habitaciones,
        'banos': banos,
        'planta': planta,
        'ascensor': ascensor,
        'estado_reforma': estado_reforma,
        'terraza': terraza,
        'garaje': garaje,
        'zona_scraping': zona,
        'tipo_inmueble': 'Piso'
    }

    pipeline = cargar_pipeline_visual()
    usa_fotos = False

    if imagenes and pipeline is not None:
        usa_fotos = True
        pca, modelo_extractor, transformacion = pipeline
        import torch
        
        vectores = []
        with st.spinner("Procesando vectores visuales con ResNet50..."):
            with torch.no_grad():
                for img_file in imagenes:
                    img = Image.open(img_file).convert('RGB')
                    img_t = transformacion(img).unsqueeze(0)
                    feat = modelo_extractor(img_t).flatten().numpy()
                    vectores.append(feat)
            
            vector_medio = np.mean(vectores, axis=0).reshape(1, -1)
            pca_feats = pca.transform(vector_medio)[0]
            
            for i, col in enumerate(cols_pca):
                datos[col] = pca_feats[i]
    else:
        # Fallback de seguridad exigido por el tribunal
        for col in cols_pca:
            datos[col] = medianas_pca.get(col, 0.0)

    # Prediccion
    df_input = pd.DataFrame([datos])[cols_num + cols_cat + cols_pca]
    X_transformed = preprocessor.transform(df_input)
    pred_log = modelo.predict(X_transformed)[0]
    precio_estimado = float(np.expm1(pred_log))

    mae = metricas.get("MAE", 215672) 
    precio_bajo = max(0, precio_estimado - mae)
    precio_alto = precio_estimado + mae

    # Resultados visuales
    st.markdown("---")
    st.subheader("3. Informe de Tasacion")
    
    if not usa_fotos:
        st.warning("Tasacion realizada en modo ciego. Se ha inyectado la mediana visual del mercado al no detectar fotografias validas.")
    
    c_res1, c_res2, c_res3 = st.columns(3)
    c_res1.metric("Escenario Conservador", f"{precio_bajo:,.0f} €".replace(",", "."))
    c_res2.metric("Valor Optimo de Mercado", f"{precio_estimado:,.0f} €".replace(",", "."))
    c_res3.metric("Escenario Alcista", f"{precio_alto:,.0f} €".replace(",", "."))
