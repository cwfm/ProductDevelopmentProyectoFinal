# API Prediccion de Churn

API desarrollada con FastAPI para predecir la probabilidad de abandono de clientes
usando un modelo de machine learning.

## Instalacion

1. Crear entorno virtual
2. Instalar dependencias:
pip install -r requirements.txt

3. Colocar los archivos:
- modelo_churn.pkl
- preprocesador_churn.pkl

4. Ejecutar la API:
uvicorn main:app --host 0.0.0.0 --port 8000

## Endpoints

- GET /
- GET /health
- POST /predict

## Documentacion
http://localhost:8000/docs
