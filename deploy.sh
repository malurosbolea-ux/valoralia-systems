#!/bin/bash
# Valoralia Systems. Script de despliegue sobre AWS EC2
# Autora: Maria Luisa Ros Bolea

set -e

NOMBRE_IMAGEN="valoralia-saas"
NOMBRE_CONTENEDOR="valoralia-production"
PUERTO_HOST=80
PUERTO_CONTENEDOR=8501

echo "==> Actualizando paquetes del sistema"
sudo apt-get update -y

echo "==> Comprobando Docker"
if ! command -v docker &> /dev/null; then
    echo "==> Instalando Docker"
    sudo apt-get install -y docker.io
    sudo systemctl enable docker
    sudo systemctl start docker
    sudo usermod -aG docker $USER
fi

echo "==> Verificando artefactos"
for archivo in Dockerfile requirements.txt app_valoralia.py preprocesador.pkl modelo_xgb.json medianas_pca.pkl; do
    if [ ! -f "$archivo" ]; then
        echo "ERROR: falta $archivo en el directorio actual."
        exit 1
    fi
done

echo "==> Construyendo imagen $NOMBRE_IMAGEN"
sudo docker build -t $NOMBRE_IMAGEN:latest .

echo "==> Deteniendo contenedor previo si existe"
sudo docker stop $NOMBRE_CONTENEDOR 2>/dev/null || true
sudo docker rm $NOMBRE_CONTENEDOR 2>/dev/null || true

echo "==> Lanzando contenedor con reinicio automatico"
sudo docker run -d \
    --name $NOMBRE_CONTENEDOR \
    --restart always \
    -p $PUERTO_HOST:$PUERTO_CONTENEDOR \
    $NOMBRE_IMAGEN:latest

echo "==> Estado del contenedor"
sudo docker ps | grep $NOMBRE_CONTENEDOR || true

IP_PUBLICA=$(curl -s ifconfig.me || echo "desconocida")
echo ""
echo "============================================================"
echo " Despliegue completado. Accede desde tu navegador en:"
echo "   http://$IP_PUBLICA"
echo "============================================================"
