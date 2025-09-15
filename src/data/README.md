# Data Pipeline

![Data Engineering](https://img.shields.io/badge/Data-Engineering-blue) ![MLOps](https://img.shields.io/badge/MLOps-Pipeline-green) ![Quality](https://img.shields.io/badge/Data%20Quality-Validated-success)

Este módulo contém o pipeline de dados da plataforma MLOps, responsável pela ingestão, processamento, validação e preparação de dados para treinamento e inferência de modelos de machine learning.

## 🎯 Objetivos

- **Ingestão Escalável**: Processamento de dados em lote e tempo real
- **Qualidade de Dados**: Validação e limpeza automatizada
- **Feature Engineering**: Transformações e criação de features
- **Versionamento**: Controle de versão de datasets e schemas
- **Monitoramento**: Observabilidade completa do pipeline

## 📊 Estrutura dos Dados

```
src/data/
├── ingestion/          # Módulos de ingestão de dados
│   ├── batch/          # Processamento em lote
│   ├── streaming/      # Processamento em tempo real
│   └── connectors/     # Conectores para fontes externas
├── processing/         # Pipeline de processamento
│   ├── validation/     # Validação de qualidade
│   ├── cleaning/       # Limpeza e tratamento
│   └── transformation/ # Transformações e features
├── storage/            # Camada de armazenamento
│   ├── raw/           # Dados brutos
│   ├── processed/     # Dados processados
│   └── features/      # Feature store
└── monitoring/         # Monitoramento de dados
    ├── quality/       # Métricas de qualidade
    ├── drift/         # Detecção de drift
    └── lineage/       # Linhagem de dados
```

## 🔄 Pipeline de Dados

### 1. Ingestão

```python
from src.data.ingestion import DataIngestion

# Ingestão em lote
batch_ingestion = DataIngestion(
    source="s3://raw-data-bucket/",
    format="parquet",
    schedule="@daily"
)

# Ingestão em tempo real
streaming_ingestion = DataIngestion(
    source="kafka://ml-events",
    format="json",
    window="5min"
)
```

### 2. Validação de Qualidade

```python
from src.data.processing.validation import DataValidator

validator = DataValidator()

# Definir regras de validação
validation_rules = {
    "completeness": {"threshold": 0.95},
    "uniqueness": {"columns": ["transaction_id"]},
    "range": {"amount": {"min": 0, "max": 100000}}
}

# Executar validação
results = validator.validate(data, validation_rules)
```

### 3. Feature Engineering

```python
from src.data.processing.transformation import FeatureEngineer

feature_engineer = FeatureEngineer()

# Transformações automáticas
features = feature_engineer.transform({
    "categorical_encoding": "target",
    "numerical_scaling": "robust",
    "feature_selection": "mutual_info",
    "temporal_features": ["hour", "day_of_week"]
})
```

## 📈 Fontes de Dados Suportadas

### Dados Estruturados
- **Transações Financeiras**: Detecção de fraudes
- **Comportamento de Usuários**: Sistema de recomendação
- **Sensores IoT**: Manutenção preditiva
- **Logs de Sistema**: Monitoramento operacional

### Formatos Suportados
- **Batch**: Parquet, CSV, JSON, Avro
- **Streaming**: Kafka, Kinesis, PubSub
- **APIs**: REST, GraphQL
- **Databases**: PostgreSQL, MongoDB, Cassandra

## 🎛️ Configuração

### Variáveis de Ambiente

```bash
# Storage
DATA_LAKE_BUCKET=s3://ml-data-lake
FEATURE_STORE_URI=postgresql://localhost:5432/features

# Streaming
KAFKA_BROKERS=localhost:9092
STREAM_PROCESSING_MODE=exactly_once

# Qualidade
DATA_QUALITY_THRESHOLD=0.95
DRIFT_DETECTION_WINDOW=7d
```

### Configuração do Pipeline

```yaml
# data_pipeline.yml
pipeline:
  ingestion:
    batch_size: 10000
    parallel_workers: 4
    retry_policy: exponential_backoff
  
  validation:
    enable_profiling: true
    fail_on_error: false
    quality_threshold: 0.95
  
  processing:
    cache_intermediate: true
    optimize_memory: true
    enable_monitoring: true
```

## 🔍 Monitoramento e Observabilidade

### Métricas Principais
- **Volume**: Registros processados por hora
- **Qualidade**: Taxa de aprovação nas validações
- **Latência**: Tempo de processamento end-to-end
- **Drift**: Desvio estatístico dos dados

### Alertas Automáticos
- **Falha na Ingestão**: > 5% de falha em 1h
- **Qualidade Baixa**: < 90% de aprovação
- **Drift Detectado**: Desvio > 2 desvios padrão
- **Latência Alta**: > 10min para processamento

## 🧪 Testes e Validação

### Executar Testes

```bash
# Testes unitários do pipeline
pytest tests/data/unit/

# Testes de integração
pytest tests/data/integration/

# Testes de qualidade de dados
pytest tests/data/quality/
```

### Validação de Schema

```python
from src.data.validation import SchemaValidator

# Definir schema esperado
schema = {
    "user_id": "int64",
    "transaction_amount": "float64",
    "timestamp": "datetime64[ns]",
    "merchant_category": "category"
}

# Validar dados contra schema
validator = SchemaValidator(schema)
results = validator.validate(df)
```

## 📝 Logs e Auditoria

### Estrutura de Logs

```json
{
  "timestamp": "2024-09-15T14:30:00Z",
  "level": "INFO",
  "component": "data.ingestion.batch",
  "message": "Processed 10000 records successfully",
  "metadata": {
    "source": "transactions_2024_09_15.parquet",
    "records_processed": 10000,
    "processing_time_ms": 1500,
    "data_quality_score": 0.97
  }
}
```

### Linhagem de Dados

- **Rastreabilidade**: Origem até modelo final
- **Transformações**: Log de todas as operações
- **Versionamento**: Controle de versão de datasets
- **Impacto**: Análise de downstream dependencies

## 🚀 Deploy e Produção

### Containerização

```dockerfile
# Dockerfile.data-pipeline
FROM python:3.11-slim

WORKDIR /app
COPY requirements-data.txt .
RUN pip install -r requirements-data.txt

COPY src/data/ ./src/data/
CMD ["python", "-m", "src.data.main"]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: data-pipeline
spec:
  replicas: 3
  selector:
    matchLabels:
      app: data-pipeline
  template:
    spec:
      containers:
      - name: data-pipeline
        image: ml-platform/data-pipeline:latest
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
```

## 📚 Documentação Adicional

- [Guia de Ingestão de Dados](./docs/ingestion-guide.md)
- [Padrões de Qualidade](./docs/quality-standards.md)
- [Feature Engineering](./docs/feature-engineering.md)
- [Troubleshooting](./docs/troubleshooting.md)

## 🤝 Contribuição

Para contribuir com melhorias no pipeline de dados:

1. Fork o repositório
2. Crie uma branch para sua feature: `git checkout -b feature/data-enhancement`
3. Implemente suas mudanças seguindo os padrões de qualidade
4. Execute todos os testes: `make test-data`
5. Submeta um Pull Request

---

**Mantido pela equipe de Data Engineering**  
📧 [data-engineering@company.com](mailto:data-engineering@company.com)  
📋 [Roadmap do Pipeline](https://github.com/galafis/ibm-machine-learning-capstone/projects/data-pipeline)
