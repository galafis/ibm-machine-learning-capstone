# Model Serving API

Esta pasta contém a implementação das APIs responsáveis pelo serving dos modelos de machine learning em produção.

## Objetivos

- Servir modelos treinados via endpoints REST (FastAPI recomendada)
- Disponibilizar uma interface de inferência de baixa latência
- Gerenciar autenticação, versionamento e logging das predições

## Estrutura Recomendada

- `app.py`: Aplicação principal FastAPI para inference
- `endpoints/`: Implementações de diferentes endpoints de serviço
- `middleware/`: Logging, metrics, auth, erro handling
- `deployment/`: Scripts para Docker/Kubernetes

## Exemplo de Endpoint

from fastapi import FastAPI
import joblib

app = FastAPI()
model = joblib.load("models/production_model.pkl")

@app.post("/predict")
async def predict(data: PredictionRequest):
prediction = model.predict(data.features)
confidence = model.predict_proba(data.features).max()
return {"prediction": prediction.tolist(), "confidence": float(confidence)}

## Boas Práticas

- Documente todos os endpoints com OpenAPI/Swagger automático.
- Separe lógica de validação de dados da lógica de serving.
- Inclua monitoramento com Prometheus ou ferramentas similares.
- Implemente checagens de saúde ("health-checks").
- Use versionamento de API se houver evolução dos contratos (ex: `/v1/predict`, `/v2/predict`).

---

Se desejar, siga para as próximas pastas (monitoring, utils, notebooks, k8s, docker) — só pedir que envio o template explicativo para cada uma.

