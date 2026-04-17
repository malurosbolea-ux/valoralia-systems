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

# Configuracion de pagina (Cero emojis por regla de proyecto)
st.set_page_config(
    page_title="Valoralia Systems",
    layout="wide"
)

# Inyeccion de CSS avanzado para diseno corporativo de alto impacto
st.markdown("""
    <style>
    /* Fondo principal y textos base */
    .stApp {
        background-color: #f8f9fa;
        color: #000000;
    }
    
    /* Titulos principales */
    h1 {
        color: #0f172a; 
        font-weight: 900; 
        font-size: 2.5rem;
        letter-spacing: -1px;
    }
    h2, h3 {
        color: #0f172a;
        font-weight: 700;
    }
    
    /* Subtitulos descriptivos */
    .subtitle-text {
        color: #4b5563;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        border-bottom: 2px solid #b91c1c;
        padding-bottom: 10px;
    }

    /* Tarjetas (Cards) para organizar la interfaz */
    .css-card {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 25px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 20px;
        border-top: 4px solid #0f172a;
    }

    /* Tarjeta de resultados (Impacto visual rojo) */
    .result-card {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 30px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        text-align: center;
        border-top: 6px solid #b91c1c;
        margin-top: 30px;
    }

    /* Estilo del boton principal */
    .stButton>button {
        background-color: #0f172a;
        color: #ffffff;
        border-radius: 6px;
        border: none;
        padding: 0.75rem 2.5rem;
        font-size: 1.1rem;
        font-weight: bold;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(15, 23, 42, 0.2);
    }
    .stButton>button:hover {
        background-color: #b91c1c;
        color: #ffffff;
        box-shadow: 0 6px 8px rgba(185, 28, 28, 0.3);
        transform: translateY(-2px);
    }

    /* Metricas de resultado */
    div[data-testid="stMetricValue"] {
        color: #b91c1c;
        font-weight: 900;
        font-size: 2.2rem;
    }
    div[data-testid="stMetricLabel"] {
        color: #4b5563;
        font-size: 1.1rem;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

# Encabezado corporativo
st.markdown("<h1>Valoralia Systems</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle-text'>Motor de tasacion inmobiliaria impulsado por Inteligencia Artificial Visual</div>", unsafe_allow_html=True)

# Carga de artefactos asegurada (Solucion al KeyError)
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
    # Blindaje contra el diccionario en ingles/espanol
    modelo = paquete.get("modelo", paquete.get("model"))
    preprocessor = paquete.get("preprocesador", paquete.get("preprocessor"))
    medianas_pca = paquete.get("medianas_pca", paquete.get("medianas", {}))
    metricas = paquete.get("metricas_test", paquete.get("metricas", {"MAE": 215672}))
    cols_num = paquete.get("columnas_numericas", paquete.get("num_cols", []))
    cols_cat = paquete.get("columnas_categoricas", paquete.get("cat_cols", []))
    cols_pca = paquete.get("columnas_pca", paquete.get("pca_cols", [f"pca_{i}" for i in range(1, 51)]))
    
    if preprocessor is None:
        st.error("Error critico: No se ha encontrado el preprocesador en el paquete.")
        st.stop()
        
except FileNotFoundError:
    st.error("Archivo valoralia_production.pkl no encontrado. Deteniendo ejecucion.")
    st.stop()

# Layout de la aplicacion en dos columnas
col_izq, col_der = st.columns([1, 1], gap="large")

with col_izq:
    st.markdown("<div class='css-card'>", unsafe_allow_html=True)
    st.markdown("<h3>Parametros Estructurales</h3>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    superficie = c1.number_input("Superficie (m2)", min_value=15.0, value=90.0, step=5.0)
    habitaciones = c2.number_input("Habitaciones", min_value=0.0, value=3.0, step=1.0)
    
    c3, c4 = st.columns(2)
    banos = c3.number_input("Banos", min_value=0.0, value=2.0, step=1.0)
    planta = c4.number_input("Planta (-1 sotano, 0 bajo)", min_value=-1.0, value=2.0, step=1.0)

    st.markdown("<hr style='margin: 15px 0; border-color: #e5e7eb;'>", unsafe_allow_html=True)
    
    c5, c6 = st.columns(2)
    ascensor = c5.selectbox("Ascensor", options=[1, 0], format_func=lambda x: "Si" if x==1 else "No")
    terraza = c6.selectbox("Terraza", options=[1, 0], format_func=lambda x: "Si" if x==1 else "No")
    
    c7, c8 = st.columns(2)
    garaje = c7.selectbox("Garaje", options=[1, 0], format_func=lambda x: "Si" if x==1 else "No")
    estado_reforma = c8.selectbox("Estado Reforma", options=[1, 0], format_func=lambda x: "Reformado" if x==1 else "A reformar")
    
    zona = st.selectbox("Zona Geografica", options=["Madrid Capital", "Pozuelo de Alarcon", "Majadahonda", "Las Rozas"])
    st.markdown("</div>", unsafe_allow_html=True)

with col_der:
    st.markdown("<div class='css-card'>", unsafe_allow_html=True)
    st.markdown("<h3>Auditoria Visual (IA)</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color: #4b5563;'>Adjunte fotografias del interior para calibrar la calidad estetica de los acabados.</p>", unsafe_allow_html=True)
    
    imagenes = st.file_uploader("", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
    
    if imagenes:
        st.success(f"{len(imagenes)} documentos visuales en memoria.")
    else:
        st.info("Modo tasacion basica. Se aplicara imputacion por mediana del mercado.")
        
    st.markdown("<br>", unsafe_allow_html=True)
    boton_calcular = st.button("PROCESAR TASACION")
    st.markdown("</div>", unsafe_allow_html=True)

# Motor predictivo
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

    # Integracion de caracteristicas visuales o Fallback
    if imagenes and pipeline is not None:
        usa_fotos = True
        pca, modelo_extractor, transformacion = pipeline
        import torch
        
        vectores = []
        with st.spinner("Analizando pixeles mediante ResNet50..."):
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
        # Aplicacion estricta de la regla de Miguel: no inventar datos. Se usan medianas.
        for col in cols_pca:
            datos[col] = medianas_pca.get(col, 0.0)

    # Transformacion predictiva
    df_input = pd.DataFrame([datos])[cols_num + cols_cat + cols_pca]
    X_transformed = preprocessor.transform(df_input)
    pred_log = modelo.predict(X_transformed)[0]
    precio_estimado = float(np.expm1(pred_log))

    mae = metricas.get("MAE", 215672) 
    precio_bajo = max(0, precio_estimado - mae)
    precio_alto = precio_estimado + mae

    # Renderizado de resultados
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    st.markdown("<h2>Dictamen de Valoracion</h2>", unsafe_allow_html=True)
    
    if not usa_fotos:
        st.markdown("<p style='color: #b91c1c; font-weight: bold;'>Aviso: Tasacion ejecutada con perfiles visuales neutros (Mediana).</p>", unsafe_allow_html=True)
    
    c_res1, c_res2, c_res3 = st.columns(3)
    c_res1.metric("Escenario Conservador", f"{precio_bajo:,.0f} €".replace(",", "."))
    c_res2.metric("Valor Optimo de Mercado", f"{precio_estimado:,.0f} €".replace(",", "."))
    c_res3.metric("Escenario Alcista", f"{precio_alto:,.0f} €".replace(",", "."))
    
    st.markdown("</div>", unsafe_allow_html=True)
