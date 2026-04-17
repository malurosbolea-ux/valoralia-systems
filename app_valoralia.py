"""
Valoralia Systems - Aplicacion de valoracion inmobiliaria hibrida
Autora: Maria Luisa Ros Bolea
TFM Master IA y Big Data - CEU San Pablo 2025-2026

Esta aplicacion permite tasar viviendas en Madrid combinando datos
tabulares con analisis visual de fotografias del interior mediante
la arquitectura ResNet50 y reduccion de dimensionalidad PCA.
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from PIL import Image

# --- Configuracion de pagina ---
st.set_page_config(
    page_title="Valoralia Systems",
    page_icon="",
    layout="centered"
)

# --- Paleta Comando 19996 ---
AZUL_MARINO = "#0f172a"
GRIS_OSCURO = "#4b5563"

# --- Carga del paquete de produccion ---
@st.cache_resource
def cargar_paquete():
    """Carga el paquete de produccion con modelo, preprocesador y medianas."""
    ruta = Path(__file__).parent / "valoralia_production.pkl"
    return joblib.load(ruta)

@st.cache_resource
def cargar_pipeline_visual():
    """Carga el transformador PCA y el modelo ResNet50 para procesar fotos.
    Si no estan disponibles, devuelve None y la app usa medianas como fallback."""
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
        modelo_resnet = models.resnet50(weights=pesos)
        extractor = torch.nn.Sequential(*(list(modelo_resnet.children())[:-1]))
        extractor.eval()

        transformacion = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        return {
            "pca": pca,
            "extractor": extractor,
            "transformacion": transformacion,
            "torch": torch
        }
    except ImportError:
        return None
    except Exception:
        return None


def procesar_imagenes(imagenes_subidas, pipeline_visual):
    """Procesa una o varias fotografias con ResNet50 y devuelve los 50 componentes PCA.

    Promedia los vectores de todas las fotos (regla multipantalla del tutor)
    y aplica la transformacion PCA entrenada en NB03.

    Parameters
    ----------
    imagenes_subidas : list
        Lista de objetos UploadedFile de Streamlit.
    pipeline_visual : dict
        Diccionario con el extractor ResNet50, la transformacion y el PCA.

    Returns
    -------
    dict
        Diccionario con las 50 componentes PCA (pca_1 a pca_50).
    """
    extractor = pipeline_visual["extractor"]
    transformacion = pipeline_visual["transformacion"]
    pca = pipeline_visual["pca"]
    torch = pipeline_visual["torch"]

    vectores = []
    for img_file in imagenes_subidas:
        try:
            imagen = Image.open(img_file).convert("RGB")
            tensor = transformacion(imagen).unsqueeze(0)
            with torch.no_grad():
                vector = extractor(tensor).squeeze().numpy()
            vectores.append(vector)
        except Exception:
            continue

    if len(vectores) == 0:
        return None

    # Promedio de todos los vectores (regla multipantalla)
    vector_medio = np.mean(vectores, axis=0).reshape(1, -1)

    # Transformacion PCA (50 componentes)
    componentes = pca.transform(vector_medio)[0]

    return {f"pca_{i+1}": float(componentes[i]) for i in range(len(componentes))}


# --- Carga de artefactos ---
paquete = cargar_paquete()
modelo = paquete["modelo"]
preprocessor = paquete["preprocessor"]
medianas_pca = paquete["medianas_pca"]
cols_num = paquete["cols_num"]
cols_cat = paquete["cols_cat"]
cols_pca = paquete["cols_pca"]
metricas = paquete["metricas_base"]

pipeline_visual = cargar_pipeline_visual()
vision_disponible = pipeline_visual is not None

# --- Zonas y tipos ---
ZONAS = ['Alcala_Henares', 'Alcobendas', 'Alcorcon', 'Arganda', 'Boadilla',
         'Carabanchel', 'Centro', 'Chamartin', 'Chamberi', 'Ciudad_Lineal',
         'Colmenar_Viejo', 'Coslada', 'Fuencarral', 'Fuenlabrada', 'Getafe',
         'Las_Rozas', 'Latina', 'Leganes', 'Majadahonda', 'Moncloa',
         'Moratalaz', 'Mostoles', 'Parla', 'Pinto', 'Pozuelo',
         'Puente_Vallecas', 'Retiro', 'Rivas', 'SS_Reyes', 'Salamanca',
         'San_Blas', 'Tetuan', 'Torrejon', 'Tres_Cantos', 'Usera',
         'Vicalvaro', 'Villa_Vallecas', 'Villaverde', 'Villaviciosa_Odon']

TIPOS = ['atico', 'chalet', 'duplex', 'estudio', 'piso']

# --- Interfaz ---
st.title("Valoralia Systems")
st.markdown(
    "Sistema de valoracion inmobiliaria hibrido: combina datos estructurales "
    "con analisis visual de las fotografias del interior mediante inteligencia "
    "artificial (ResNet50 + PCA)."
)
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    zona = st.selectbox("Zona de Madrid", ZONAS)
    tipo = st.selectbox("Tipo de inmueble", TIPOS)
    superficie = st.number_input("Superficie (m2)", min_value=10, max_value=1500, value=80)
    habitaciones = st.number_input("Habitaciones", min_value=0, max_value=20, value=2)
    banos = st.number_input("Banos", min_value=0, max_value=10, value=1)

with col2:
    planta = st.number_input("Planta", min_value=-2, max_value=50, value=1)
    ascensor = st.selectbox("Ascensor", ["Desconocido", "Si", "No"])
    terraza = st.selectbox("Terraza", ["Desconocido", "Si", "No"])
    garaje = st.selectbox("Garaje", ["Desconocido", "Si", "No"])
    calefaccion = st.selectbox("Calefaccion", ["Desconocido", "Si", "No"])
    estado_reforma = st.selectbox("Reformado", ["Desconocido", "Si", "No"])


def mapear_binario(valor):
    """Convierte las opciones del desplegable a valores numericos."""
    if valor == "Si":
        return 1
    elif valor == "No":
        return 0
    else:
        return -1


# --- Seccion de fotografias ---
st.markdown("---")
st.subheader("Fotografias del interior")

if vision_disponible:
    st.markdown(
        "Sube una o varias fotografias del interior de la vivienda. "
        "El sistema analizara la calidad visual (acabados, iluminacion, estado) "
        "para ajustar la valoracion."
    )
    imagenes = st.file_uploader(
        "Selecciona las fotografias del interior",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
else:
    imagenes = []
    st.info(
        "El modulo de vision artificial no esta disponible en esta instancia. "
        "La valoracion se realizara usando el perfil visual medio del mercado "
        "madrileno como referencia (medianas PCA del dataset de entrenamiento)."
    )

# --- Boton de prediccion ---
st.markdown("---")
if st.button("Calcular valoracion", type="primary"):

    # Construyo el registro de entrada con las variables tabulares
    datos = {
        "superficie_m2": float(superficie),
        "habitaciones": float(habitaciones),
        "banos": float(banos),
        "planta": float(planta),
        "num_imagenes": float(len(imagenes)) if imagenes else 0.0,
        "codigo_postal": 0.0,
        "ascensor": mapear_binario(ascensor),
        "terraza": mapear_binario(terraza),
        "garaje": mapear_binario(garaje),
        "calefaccion": mapear_binario(calefaccion),
        "estado_reforma": mapear_binario(estado_reforma),
        "zona_scraping": zona,
        "tipo_inmueble": tipo
    }

    # Determino los valores PCA segun si hay fotos o no
    usa_fotos = False
    if imagenes and vision_disponible:
        with st.spinner("Analizando las fotografias con ResNet50..."):
            pca_resultado = procesar_imagenes(imagenes, pipeline_visual)

        if pca_resultado is not None:
            usa_fotos = True
            for col in cols_pca:
                datos[col] = pca_resultado.get(col, medianas_pca.get(col, 0.0))
        else:
            st.warning(
                "No se pudieron procesar las fotografias. "
                "Se usara el perfil visual medio como referencia."
            )
            for col in cols_pca:
                datos[col] = medianas_pca.get(col, 0.0)
    else:
        # Fallback a medianas PCA (representan la vivienda tipica de Madrid)
        for col in cols_pca:
            datos[col] = medianas_pca.get(col, 0.0)

    # Construyo el DataFrame y predigo
    df_input = pd.DataFrame([datos])
    df_input = df_input[cols_num + cols_cat + cols_pca]

    X_transformed = preprocessor.transform(df_input)
    pred_log = modelo.predict(X_transformed)[0]
    precio_estimado = float(np.expm1(pred_log))

    mae = metricas["MAE"]
    precio_bajo = max(0, precio_estimado - mae)
    precio_alto = precio_estimado + mae

    # --- Resultado ---
    st.markdown("---")
    st.subheader("Resultado de la valoracion")

    c1, c2, c3 = st.columns(3)
    c1.metric("Estimacion baja", f"{precio_bajo:,.0f} EUR")
    c2.metric("Valoracion central", f"{precio_estimado:,.0f} EUR")
    c3.metric("Estimacion alta", f"{precio_alto:,.0f} EUR")

    fuente_visual = "fotografias reales del inmueble" if usa_fotos else "medianas PCA del mercado madrileno"
    st.caption(
        f"Intervalo basado en el MAE historico del modelo ({mae:,.0f} EUR). "
        f"Fuente visual: {fuente_visual}."
    )

    # Muestro las fotos subidas como confirmacion
    if imagenes and usa_fotos:
        st.markdown("**Fotografias analizadas:**")
        cols_fotos = st.columns(min(len(imagenes), 4))
        for i, img in enumerate(imagenes[:4]):
            with cols_fotos[i]:
                st.image(img, use_container_width=True)

st.markdown("---")
st.caption("Valoralia Systems v1.0 | Maria Luisa Ros Bolea | TFM CEU San Pablo 2025-2026")
