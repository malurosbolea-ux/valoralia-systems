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

# Configuracion de pagina nativa
st.set_page_config(
    page_title="Valoralia Systems",
    layout="wide"
)

# Inyeccion de CSS limpio para el fondo de edificios y colores corporativos
st.markdown("""
    <style>
    /* Fondo con imagen de edificios reales y capa blanca al 93% de opacidad */
    .stApp {
        background: linear-gradient(rgba(255, 255, 255, 0.93), rgba(255, 255, 255, 0.93)), url("https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?q=80&w=2070&auto=format&fit=crop");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: #000000;
    }
    
    /* Tipografia y colores corporativos */
    h1 { color: #0f172a; font-weight: 900; font-size: 2.5rem; border-bottom: 3px solid #b91c1c; padding-bottom: 10px;}
    h2, h3 { color: #0f172a; font-weight: 700; }
    
    /* Boton de calculo premium */
    .stButton>button {
        background-color: #0f172a;
        color: #ffffff;
        border-radius: 4px;
        border: none;
        padding: 0.8rem 2rem;
        font-size: 1.1rem;
        font-weight: bold;
        width: 100%;
        transition: 0.3s;
    }
    .stButton>button:hover { background-color: #b91c1c; color: #ffffff; }
    
    /* Metricas rojas corporativas */
    div[data-testid="stMetricValue"] { color: #b91c1c; font-weight: 900; font-size: 2.4rem; }
    div[data-testid="stMetricLabel"] { color: #0f172a; font-weight: bold; font-size: 1.1rem; }
    </style>
    """, unsafe_allow_html=True)

# Cabecera
st.markdown("<h1>Valoralia Systems</h1>", unsafe_allow_html=True)
st.markdown("**Motor de tasación inmobiliaria impulsado por Inteligencia Artificial Visual**")
st.write("---")

# Carga de artefactos blindada
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
        return None

try:
    paquete = cargar_paquete()
    modelo = paquete.get("modelo", paquete.get("model"))
    preprocessor = paquete.get("preprocesador", paquete.get("preprocessor"))
    medianas_pca = paquete.get("medianas_pca", paquete.get("medianas", {}))
    metricas = paquete.get("metricas_test", paquete.get("metricas", {"MAE": 215672}))
    cols_num = paquete.get("columnas_numericas", paquete.get("num_cols", []))
    cols_cat = paquete.get("columnas_categoricas", paquete.get("cat_cols", []))
    cols_pca = paquete.get("columnas_pca", paquete.get("pca_cols", [f"pca_{i}" for i in range(1, 51)]))
    
    if preprocessor is None:
        st.error("Error crítico interno: No se ha encontrado el preprocesador en el paquete cargado.")
        st.stop()
        
except FileNotFoundError:
    st.error("Error crítico: Archivo valoralia_production.pkl no encontrado.")
    st.stop()

# Layout nativo de Streamlit
col_izq, col_der = st.columns(2, gap="large")

with col_izq:
    st.subheader("Parámetros Estructurales")
    
    c1, c2 = st.columns(2)
    superficie = c1.number_input("Superficie (m2)", min_value=15.0, value=90.0, step=5.0)
    habitaciones = c2.number_input("Habitaciones", min_value=0.0, value=3.0, step=1.0)
    
    c3, c4 = st.columns(2)
    banos = c3.number_input("Baños", min_value=0.0, value=2.0, step=1.0)
    planta = c4.number_input("Planta (-1 sótano, 0 bajo)", min_value=-1.0, value=2.0, step=1.0)

    st.write("---")
    st.markdown("**Equipamiento y Estado**")
    
    c5, c6 = st.columns(2)
    ascensor = c5.selectbox("Ascensor", options=[1, 0], format_func=lambda x: "Sí" if x==1 else "No")
    terraza = c6.selectbox("Terraza", options=[1, 0], format_func=lambda x: "Sí" if x==1 else "No")
    
    c7, c8 = st.columns(2)
    garaje = c7.selectbox("Garaje", options=[1, 0], format_func=lambda x: "Sí" if x==1 else "No")
    estado_reforma = c8.selectbox("Estado Reforma", options=[1, 0], format_func=lambda x: "Reformado" if x==1 else "A reformar")
    
    zona = st.selectbox("Zona Geográfica", options=["Madrid Capital", "Pozuelo de Alarcón", "Majadahonda", "Las Rozas"])

with col_der:
    st.subheader("Auditoría Visual (IA)")
    st.info("Adjunte fotografías del interior de la vivienda. La Red Neuronal ResNet50 analizará los píxeles para calibrar la tasación basándose en la estética y acabados.")
    
    imagenes = st.file_uploader("", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
    
    if imagenes:
        st.success(f"{len(imagenes)} documentos visuales listos para análisis.")
    else:
        st.warning("Modo tasación básica: Ante la ausencia de material visual, se aplicará la imputación estadística (Mediana) del mercado.")
        
    st.write("")
    st.write("")
    boton_calcular = st.button("PROCESAR TASACIÓN HÍBRIDA")

# Lógica del modelo predictivo (Intacta)
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
        with st.spinner("Analizando píxeles mediante ResNet50..."):
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
        for col in cols_pca:
            datos[col] = medianas_pca.get(col, 0.0)

    df_input = pd.DataFrame([datos])[cols_num + cols_cat + cols_pca]
    X_transformed = preprocessor.transform(df_input)
    pred_log = modelo.predict(X_transformed)[0]
    precio_estimado = float(np.expm1(pred_log))

    mae = metricas.get("MAE", 215672) 
    precio_bajo = max(0, precio_estimado - mae)
    precio_alto = precio_estimado + mae

    st.write("---")
    st.subheader("Dictamen Final de Tasación")
    
    c_res1, c_res2, c_res3 = st.columns(3)
    c_res1.metric("Escenario Conservador", f"{precio_bajo:,.0f} €".replace(",", "."))
    c_res2.metric("Valor Óptimo de Mercado", f"{precio_estimado:,.0f} €".replace(",", "."))
    c_res3.metric("Escenario Alcista", f"{precio_alto:,.0f} €".replace(",", "."))
