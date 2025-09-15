"""
app.py - API de Model Serving para MLOps
Autor: Gabriel Demetrios Lafis
Certificação: IBM Machine Learning Engineering Professional Certificate
MLOps Capstone

API REST para servir modelos de ML em produção, com FastAPI.
Inclui logging, tratamento de erros, healthcheck e endpoint de predição.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import logging
import traceback

# Configuração Básica de Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ml-serving-api")

# Definição do app FastAPI
app = FastAPI(
    title="ML Model Serving API",
    description="API para serving de modelos do projeto MLOps Capstone",
    version="1.0.0"
)

# Exemplo: Permitir acessos CORS (para deploy local/testes)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, restringir domínios!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelo de entrada da predição (adapte conforme seu projeto)
class PredictionRequest(BaseModel):
    features: list = Field(..., example=[[0.5, 0.7, 1.2, 5.3]])

# Modelo de saída
class PredictionResponse(BaseModel):
    prediction: list
    confidence: float

# Carregar modelo pré-treinado
try:
    model = joblib.load("models/production_model.pkl")
    logger.info("Modelo carregado com sucesso.")
except Exception as e:
    logger.error(f"Erro ao carregar modelo: {e}")
    model = None

@app.get("/health", tags=["Monitoramento"])
def healthcheck():
    """
    Endpoint básico de health-check para Kubernetes e observabilidade.
    """
    if model is not None:
        return {"status": "ok", "model_loaded": True}
    else:
        return {"status": "erro", "model_loaded": False}

@app.post("/predict", response_model=PredictionResponse, tags=["Predição"])
async def predict(data: PredictionRequest, request: Request):
    """
    Recebe dados via POST e retorna predições do modelo carregado.
    """
    if model is None:
        logger.error("Modelo não carregado!")
        raise HTTPException(status_code=500, detail="Modelo não disponível")

    try:
        input_features = data.features
        prediction = model.predict(input_features)
        # Para modelos compatíveis com predict_proba
        try:
            confidence = float(model.predict_proba(input_features).max())
        except Exception:
            confidence = None

        # Logging da requisição
        logger.info(f"Predição realizada para input de shape: {len(input_features)} | IP: {request.client.host}")

        return PredictionResponse(
            prediction=prediction.tolist(),
            confidence=confidence if confidence is not None else -1.0
        )
    except Exception as err:
        logger.error(f"Erro durante a predição: {traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=f"Erro ao processar predição: {err}")

# Executar com: uvicorn src.serving.app:app --reload --host 0.0.0.0 --port 8080
