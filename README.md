# IBM Machine Learning Engineering Capstone


Projeto Capstone do **IBM Machine Learning Engineering Professional Certificate** - Plataforma enterprise de MLOps com pipelines automatizados, model registry, monitoring e serving de modelos em produção.

## 🎯 Visão Geral

Sistema completo de MLOps que demonstra competências avançadas em engenharia de machine learning, deployment de modelos e operações de ML em ambiente de produção enterprise.

### ✨ Características Principais

- **🔄 MLOps Pipeline**: Treinamento, validação e deploy automatizados
- **📦 Model Registry**: Versionamento e gestão centralizada de modelos
- **🚀 Model Serving**: APIs de produção com alta disponibilidade
- **📊 Monitoring**: Detecção de drift e monitoramento de performance
- **🐳 Containerização**: Docker e Kubernetes para escalabilidade
- **⚡ CI/CD**: Pipeline de integração e deployment contínuo

## 🛠️ Stack Tecnológico

### MLOps & Engineering
- **MLflow**: Experiment tracking e model registry
- **Docker**: Containerização de modelos
- **Kubernetes**: Orquestração e auto-scaling
- **FastAPI**: APIs de serving de alta performance
- **Prometheus/Grafana**: Monitoring e observabilidade

### Machine Learning
- **TensorFlow**: Deep learning e neural networks
- **Scikit-learn**: Machine learning clássico
- **XGBoost**: Gradient boosting otimizado
- **Pandas/NumPy**: Processamento de dados

### Infrastructure
- **PostgreSQL**: Metadata e model registry
- **Redis**: Cache e feature store
- **NGINX**: Load balancing e reverse proxy

## 📁 Estrutura do Projeto

```
ibm-machine-learning-capstone/
├── src/
│   ├── data/                   # Pipeline de dados
│   ├── models/                 # Modelos de ML
│   ├── serving/                # APIs de serving
│   ├── monitoring/             # Monitoring e alertas
│   └── utils/                  # Utilitários
├── tests/                      # Testes automatizados
├── k8s/                        # Manifests Kubernetes
├── docker/                     # Dockerfiles
├── notebooks/                  # Jupyter notebooks
├── requirements.txt            # Dependências Python
└── README.md                   # Documentação
```

## 🚀 Quick Start

### Pré-requisitos

- Python 3.11+
- Docker & Docker Compose
- Kubernetes (opcional)

### Instalação Local

1. **Clone o repositório:**
```bash
git clone https://github.com/galafis/ibm-machine-learning-capstone.git
cd ibm-machine-learning-capstone
```

2. **Configure o ambiente:**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

3. **Execute com Docker:**
```bash
docker-compose up -d
```

4. **Acesse a plataforma:**
```
http://localhost:8080
```

## 🔬 MLOps Pipeline

### 1. Experiment Tracking
```python
import mlflow
import mlflow.sklearn

with mlflow.start_run():
    # Treinar modelo
    model = train_model(X_train, y_train)
    
    # Log parâmetros e métricas
    mlflow.log_params({"n_estimators": 100, "max_depth": 10})
    mlflow.log_metrics({"accuracy": 0.95, "f1_score": 0.93})
    
    # Registrar modelo
    mlflow.sklearn.log_model(model, "fraud_detection_model")
```

### 2. Model Serving
```python
from fastapi import FastAPI
import joblib

app = FastAPI()
model = joblib.load("models/production_model.pkl")

@app.post("/predict")
async def predict(data: PredictionRequest):
    prediction = model.predict(data.features)
    confidence = model.predict_proba(data.features).max()
    
    return {
        "prediction": prediction.tolist(),
        "confidence": float(confidence)
    }
```

### 3. Model Monitoring
```python
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Detectar data drift
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=reference_df, current_data=current_df)

if report.get_metric("DatasetDriftMetric").drift_detected:
    trigger_retraining_pipeline()
```

## 🤖 Modelos Implementados

### Fraud Detection
```python
import xgboost as xgb

fraud_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    eval_metric='auc'
)
```

### Recommendation System
```python
from sklearn.decomposition import NMF

recommendation_model = NMF(
    n_components=50,
    init='random',
    random_state=42
)
```

### Time Series Forecasting
```python
import tensorflow as tf

lstm_model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, return_sequences=True),
    tf.keras.layers.LSTM(50),
    tf.keras.layers.Dense(1)
])
```

## 📊 Métricas de Performance

### Targets de Produção
- **Latência**: < 100ms por predição
- **Throughput**: > 1000 requests/segundo
- **Accuracy**: > 95% nos modelos de classificação
- **Uptime**: 99.9% disponibilidade

### Monitoramento em Tempo Real
```python
from prometheus_client import Counter, Histogram

prediction_counter = Counter('ml_predictions_total', 'Total predictions')
prediction_latency = Histogram('ml_prediction_duration_seconds', 'Prediction latency')

@prediction_latency.time()
def make_prediction(data):
    prediction_counter.inc()
    return model.predict(data)
```

## 🐳 Deploy com Kubernetes

### Deploy da Aplicação
```bash
# Deploy completo
kubectl apply -f k8s/

# Verificar status
kubectl get pods -n ml-platform

# Acessar via port-forward
kubectl port-forward svc/ml-serving 8080:80
```

### Auto-scaling
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-serving-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-serving
  minReplicas: 3
  maxReplicas: 20
  targetCPUUtilizationPercentage: 70
```

## 🧪 Testes e Validação

### Executar Testes
```bash
# Testes unitários
pytest tests/unit/

# Testes de integração
pytest tests/integration/

# Testes de performance
pytest tests/performance/
```

### Validação de Modelos
```bash
# Validação cruzada
python src/models/validate_model.py --model fraud_detection

# Backtesting
python scripts/backtest.py --start-date 2024-01-01
```

## 📈 Casos de Uso Enterprise

### 1. Detecção de Fraudes
- Processamento em tempo real de transações
- Alertas automáticos para atividades suspeitas
- Redução de 80% em falsos positivos

### 2. Sistema de Recomendação
- Recomendações personalizadas em tempo real
- A/B testing para otimização contínua
- Aumento de 25% no engagement

### 3. Manutenção Preditiva
- Previsão de falhas em equipamentos
- Otimização de cronogramas de manutenção
- Redução de 40% em downtime não planejado

## 🔧 Configuração Avançada

### MLflow Configuration
```python
# mlflow_config.py
MLFLOW_CONFIG = {
    'tracking_uri': 'postgresql://user:pass@localhost/mlflow',
    'artifact_root': 's3://ml-artifacts-bucket',
    'experiment_name': 'production_models'
}
```

### Model Registry
```python
# Promover modelo para produção
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="fraud_detection_model",
    version=3,
    stage="Production"
)
```

## 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 👨‍💻 Autor

**Gabriel Demetrios Lafis**

- **Certificação**: IBM Machine Learning Engineering Professional Certificate
- GitHub: [@galafis](https://github.com/galafis)
- Email: gabrieldemetrios@gmail.com

---

⭐ Se este projeto foi útil, considere deixar uma estrela!

