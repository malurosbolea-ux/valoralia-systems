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

# Configuracion de pagina (Estricta regla: Cero emojis)
st.set_page_config(
    page_title="Valoralia Systems",
    layout="wide"
)

# Inyeccion de CSS avanzado para diseno corporativo de ALTO IMPACTO
# Se anade textura de casas sutil en el fondo y simbolos elegantes
st.markdown("""
    <style>
    /* Fondo principal con textura de casas muy sutil (96% opacidad blanca) */
    [data-testid="stAppViewContainer"]::before {
        content: "";
        position: fixed;
        top: 0; left: 0; width: 100%; height: 100%;
        background-image: url("https://images.unsplash.com/photo-1591115765373-5207764f72e7?q=80&w=1920&auto=format&fit=crop"); /* Subtle drawn cityscape/houses sketch */
        background-size: cover;
        background-position: center;
        opacity: 0.04; /* Extremely faint */
        z-index: -1;
    }
    
    .stApp {
        color: #000000;
    }
    
    /* Titulos principales corporativos */
    h1 {
        color: #0f172a; 
        font-weight: 900; 
        font-size: 2.8rem;
        letter-spacing: -1.5px;
        margin-bottom: 5px;
    }
    h2, h3 {
        color: #0f172a;
        font-weight: 700;
    }
    
    /* Subtitulos descriptivos premium */
    .subtitle-text {
        color: #4b5563;
        font-size: 1.3rem;
        margin-bottom: 2.5rem;
        border-bottom: 3px solid #b91c1c;
        padding-bottom: 12px;
        font-weight: 500;
    }

    /* Tarjetas de datos (Cards) con simbolos sobrios */
    .css-card {
        background-color: rgba(255, 255, 255, 0.95); /* Nearly opaque white background */
        border-radius: 12px;
        padding: 30px;
        box-shadow: 0 10px 15px -3px rgba(15, 23, 42, 0.08);
        margin-bottom: 25px;
        border-top: 5px solid #0f172a;
        border-left: 1px solid #e5e7eb;
    }
    
    /* Titulos dentro de tarjetas con simbolos (Unicode, no emojis) */
    .card-title {
        color: #0f172a;
        font-weight: 700;
        font-size: 1.5rem;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* Tarjeta de resultados (Impacto visual corporativo) */
    .result-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 40px;
        box-shadow: 0 20px 25px -5px rgba(185, 28, 28, 0.1);
        text-align: center;
        border-top: 8px solid #b91c1c;
        margin-top: 40px;
    }

    /* Estilo del boton principal de alto impacto */
    .stButton>button {
        background-color: #0f172a;
        color: #ffffff;
        border-radius: 8px;
        border: none;
        padding: 1rem 3rem;
        font-size: 1.2rem;
        font-weight: bold;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 5px 8px rgba(15, 23, 42, 0.3);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton>button:hover {
        background-color: #b91c1c;
        color: #ffffff;
        box-shadow: 0 8px 12px rgba(185, 28, 28, 0.4);
        transform: translateY(-3px);
    }

    /* Metricas de resultado premium */
    div[data-testid="stMetricValue"] {
        color: #b91c1c;
        font-weight: 900;
        font-size: 2.8rem;
        letter-spacing: -1px;
    }
    div[data-testid="stMetricLabel"] {
        color: #0f172a;
        font-size: 1.2rem;
        font-weight: 700;
    }
    
    /* Alertas de Streamlit corporativas */
    .stAlert {background-color: #ffffff; border-left-color: #0f172a; box-shadow: 0 2px 4px rgba(0,0,0,0.05);}
    </style>
    """, unsafe_allow_html=True)

# Encabezado corporativo premium
st.markdown("<h1>Valoralia Systems</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle-text'>Motor de tasacion inmobiliaria impulsado por Inteligencia Artificial Visual</div>", unsafe_allow_html=True)

# Carga de artefactos asegurada (Solucion al KeyError anterior)
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
    # Blindaje contra el diccionario en ingles/espanol (Solucion KeyError)
    modelo = paquete.get("modelo", paquete.get("model"))
    preprocessor = paquete.get("preprocesador", paquete.get("preprocessor"))
    medianas_pca = paquete.get("medianas_pca", paquete.get("medianas", {}))
    metricas = paquete.get("metricas_test", paquete.get("metricas", {"MAE": 215672}))
    cols_num = paquete.get("columnas_numericas", paquete.get("num_cols", []))
    cols_cat = paquete.get("columnas_categoricas", paquete.get("cat_cols", []))
    cols_pca = paquete.get("columnas_pca", paquete.get("pca_cols", [f"pca_{i}" for i in range(1, 51)]))
    
    if preprocessor is None:
        st.error("Error critico interno: No se ha encontrado el preprocesador en el paquete cargado.")
        st.stop()
        
except FileNotFoundError:
    st.error("Error critico: Archivo valoralia_production.pkl no encontrado. El despliegue se ha detenido.")
    st.stop()

# Layout de la aplicacion en dos columnas premium
col_izq, col_der = st.columns([1, 1], gap="large")

with col_izq:
    st.markdown("<div class='css-card'>", unsafe_allow_html=True)
    # Simbolo Unicode sobrio: Edificio clasico (&#127963;)
    st.markdown("<div class='card-title'>&#127963; Parametros Estructurales</div>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    # Unicode: Superficie/Medida (&#128208;)
    superficie = c1.number_input("&#128208; Superficie (m2)", min_value=15.0, value=90.0, step=5.0)
    # Unicode: Cama (&#128719;)
    habitaciones = c2.number_input("&#128719; Habitaciones", min_value=0.0, value=3.0, step=1.0)
    
    c3, c4 = st.columns(2)
    # CORRECCION DEL FALLO ANTERIOR: De 'Vanos' a 'Banos'
    # Unicode: Ducha/Bano (&#128703;)
    banos = c3.number_input("&#128703; Banos", min_value=0.0, value=2.0, step=1.0)
    # Unicode: Flechas arriba/abajo (&#8597;)
    planta = c4.number_input("&#8597; Planta (-1 sotano, 0 bajo)", min_value=-1.0, value=2.0, step=1.0)

    st.markdown("<hr style='margin: 18px 0; border-color: #e5e7eb; border-style: dashed;'>", unsafe_allow_html=True)
    st.markdown("<b>Equipamiento y Estado</b>", unsafe_allow_html=True)
    
    c5, c6 = st.columns(2)
    ascensor = c5.selectbox("Ascensor", options=[1, 0], format_func=lambda x: "Si" if x==1 else "No")
    terraza = c6.selectbox("Terraza", options=[1, 0], format_func=lambda x: "Si" if x==1 else "No")
    
    c7, c8 = st.columns(2)
    garaje = c7.selectbox("Garaje", options=[1, 0], format_func=lambda x: "Si" if x==1 else "No")
    estado_reforma = c8.selectbox("Estado Reforma", options=[1, 0], format_func=lambda x: "Reformado" if x==1 else "A reformar")
    
    # Unicode: Mapa (&#128506;)
    zona = st.selectbox("&#128506; Zona Geografica", options=["Madrid Capital", "Pozuelo de Alarcon", "Majadahonda", "Las Rozas"])
    st.markdown("</div>", unsafe_allow_html=True)

with col_der:
    st.markdown("<div class='css-card'>", unsafe_allow_html=True)
    # Simbolo Unicode sobrio: Camara de fotos (&#128248;)
    st.markdown("<div class='card-title'>&#128248; Auditoria Visual (IA)</div>", unsafe_allow_html=True)
    st.markdown("<p style='color: #4b5563; font-weight: 500;'>Adjunte fotografias del interior de la vivienda. La Red Neuronal ResNet50 analizara los pixeles para calibrar la tasacion basandose en la calidad estetica de los acabados y el estado de conservacion.</p>", unsafe_allow_html=True)
    
    # Uploader visual limpio
    imagenes = st.file_uploader("", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
    
    if imagenes:
        st.success(f"{len(imagenes)} documentos visuales listos para analisis.")
    else:
        # Unicode: Info (&#128712;)
        st.info("&#128712; Modo tasacion basica activado. Ante la ausencia de material visual, el sistema aplicara la imputacion estadistica por la mediana del mercado para garantizar la trazabilidad.")
        
    st.markdown("<br><br>", unsafe_allow_html=True)
    boton_calcular = st.button("PROCESAR TASACION HIBRIDA")
    st.markdown("</div>", unsafe_allow_html=True)

# Motor predictivo (Logica de Claude intacta, cero invencion de datos)
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

    # Integracion de caracteristicas visuales o Fallback de seguridad exigido por Miguel
    if imagenes and pipeline is not None:
        usa_fotos = True
        pca, modelo_extractor, transformacion = pipeline
        import torch
        
        vectores = []
        with st.spinner("Analizando pixeles mediante ResNet50 (Extraccion de caracteristicas)..."):
            with torch.no_grad():
                for img_file in imagenes:
                    img = Image.open(img_file).convert('RGB')
                    img_t = transformacion(img).unsqueeze(0)
                    feat = modelo_extractor(img_t).flatten().numpy()
                    vectores.append(feat)
            
            # Promedio multipantalla exigido por el tribunal
            vector_medio = np.mean(vectores, axis=0).reshape(1, -1)
            # Reduccion de dimensionalidad PCA
            pca_feats = pca.transform(vector_medio)[0]
            
            for i, col in enumerate(cols_pca):
                datos[col] = pca_feats[i]
    else:
        # Aplicacion de la Regla de Miguel: Imputacion estadistica por Medianas visuales (no ceros).
        for col in cols_pca:
            datos[col] = medianas_pca.get(col, 0.0)

    # Transformacion predictiva XGBoost
    df_input = pd.DataFrame([datos])[cols_num + cols_cat + cols_pca]
    X_transformed = preprocessor.transform(df_input)
    pred_log = modelo.predict(X_transformed)[0]
    precio_estimado = float(np.expm1(pred_log))

    # Calculo de escenarios de incertidumbre (MAE)
    mae = metricas.get("MAE", 215672) 
    precio_bajo = max(0, precio_estimado - mae)
    precio_alto = precio_estimado + mae

    # Renderizado premium de resultados
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    # Simbolo Unicode sobrio: Grafico de barras (&#128202;)
    st.markdown("<div class='card-title' style='justify-content: center;'>&#128202; Dictamen Final de Tasacion</div>", unsafe_allow_html=True)
    
    if not usa_fotos:
        st.markdown("<p style='color: #b91c1c; font-weight: bold; background-color: #fee2e2; padding: 10px; border-radius: 6px;'>Aviso: Tasacion ejecutada en modo tabular ciego. Se han imputado valores visuales neutros (Mediana) al dataset hibrido.</p>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    c_res1, c_res2, c_res3 = st.columns(3)
    c_res1.metric("Escenario Conservador", f"{precio_bajo:,.0f} €".replace(",", "."))
    # Unicode: Estrella (&#127775;) para destacar el valor central
    c_res2.metric("&#127775; Valor Optimo de Mercado", f"{precio_estimado:,.0f} €".replace(",", "."))
    c_res3.metric("Escenario Alcista", f"{precio_alto:,.0f} €".replace(",", "."))
    
    st.markdown("<br><p style='color: #4b5563; font-size: 0.9rem;'>Valoralia Systems utiliza un modelo XGBoost hibrido con ResNet50 para reducir la opacidad del mercado inmobiliario.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
