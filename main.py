import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="API Prediccion de Churn",
    description="API simple para predecir abandono de clientes",
    version="1.0.0"
)

# variables globales
modelo = None
preprocesador = None


# -------------------------
# carga de artefactos
# -------------------------
def cargar_modelo():
    global modelo, preprocesador

    if not os.path.exists("modelo_churn.pkl"):
        raise FileNotFoundError("No se encontro modelo_churn.pkl")

    if not os.path.exists("preprocesador_churn.pkl"):
        raise FileNotFoundError("No se encontro preprocesador_churn.pkl")

    modelo = joblib.load("modelo_churn.pkl")
    preprocesador = joblib.load("preprocesador_churn.pkl")

    print("modelo y preprocesador cargados correctamente")


@app.on_event("startup")
def startup_event():
    try:
        cargar_modelo()
    except Exception as e:
        print(f"error cargando el modelo: {e}")


# -------------------------
# esquema de entrada
# -------------------------
class ClienteEntrada(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    Contract: str
    PaymentMethod: str
    InternetService: str


# -------------------------
# endpoints
# -------------------------
@app.get("/")
def inicio():
    return {
        "mensaje": "API de prediccion de churn",
        "estado": "activa",
        "modelo_cargado": modelo is not None,
        "endpoints": {
            "health": "/health",
            "predict": "/predict"
        }
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "modelo_cargado": modelo is not None
    }


@app.post("/predict")
def predecir_churn(cliente: ClienteEntrada):

    if modelo is None or preprocesador is None:
        raise HTTPException(status_code=500, detail="modelo no cargado")

    try:
        # convertir entrada a dataframe
        datos = pd.DataFrame([cliente.dict()])

        # aplicar mismo preprocesamiento del entrenamiento
        datos_proc = preprocesador.transform(datos)

        # prediccion
        prob = modelo.predict_proba(datos_proc)[0][1]

        # banda de riesgo
        if prob >= 0.7:
            riesgo = "alto"
        elif prob >= 0.4:
            riesgo = "medio"
        else:
            riesgo = "bajo"

        return {
            "probabilidad_churn": round(float(prob), 4),
            "riesgo": riesgo
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)