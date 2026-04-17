#!/bin/bash
# deploy.sh -- Script de despliegue para Valoralia Systems
# Autora: Maria Luisa Ros Bolea
# Uso: bash deploy.sh

set -euo pipefail

echo "=== Valoralia Systems - Despliegue ==="

IMAGE_NAME="valoralia-systems"
IMAGE_TAG="v1.0"
CONTAINER_NAME="valoralia-app"
PORT=8501

echo "Construyendo imagen Docker..."
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .

docker stop ${CONTAINER_NAME} 2>/dev/null || true
docker rm ${CONTAINER_NAME} 2>/dev/null || true

echo "Lanzando contenedor en puerto ${PORT}..."
docker run -d \
    --name ${CONTAINER_NAME} \
    -p ${PORT}:8501 \
    --restart unless-stopped \
    ${IMAGE_NAME}:${IMAGE_TAG}

echo ""
echo "Valoralia Systems disponible en: http://localhost:${PORT}"
echo "Para ver logs: docker logs -f ${CONTAINER_NAME}"
