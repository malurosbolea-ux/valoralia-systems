FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app_valoralia.py .
COPY valoralia_production.pkl .
COPY valoralia_pca_transformer.pkl .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app_valoralia.py", "--server.port=8501", "--server.address=0.0.0.0"]
