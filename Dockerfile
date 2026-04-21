# Valoralia Systems. Contenedor productivo
FROM python:3.10-slim

LABEL maintainer="Maria Luisa Ros Bolea"
LABEL project="Valoralia Systems. TFM CEU San Pablo"
LABEL version="1.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Dependencias del sistema para XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY app_valoralia.py /app/app_valoralia.py
COPY preprocesador.pkl /app/preprocesador.pkl
COPY modelo_xgb.json /app/modelo_xgb.json
COPY medianas_pca.pkl /app/medianas_pca.pkl

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

EXPOSE 8501

CMD ["streamlit", "run", "app_valoralia.py", "--server.port=8501", "--server.address=0.0.0.0"]
