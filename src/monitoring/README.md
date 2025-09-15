# Monitoring de Modelos ML

Esta pasta contém as implementações e configurações para monitoramento abrangente de modelos de machine learning em produção.

## Objetivos

- Monitorar performance e comportamento de modelos em produção
- Detectar data drift, concept drift e model degradation
- Coletar e visualizar métricas de negócio e técnicas
- Implementar alertas automáticos para anomalias
- Integrar com ferramentas como Prometheus, Grafana e Evidently

## Estrutura Recomendada

```
monitoring/
├── metrics/
│   ├── model_metrics.py       # Métricas de performance do modelo
│   ├── data_drift.py         # Detecção de drift nos dados
│   └── business_metrics.py   # Métricas de negócio
├── alerts/
│   ├── alert_manager.py      # Gerenciamento de alertas
│   └── thresholds.yaml       # Configuração de thresholds
├── dashboards/
│   ├── grafana_configs/      # Dashboards Grafana
│   └── evidently_reports/    # Reports Evidently AI
└── prometheus/
    └── prometheus.yml        # Configuração Prometheus
```

## Tipos de Monitoramento

### 1. Performance do Modelo

```python
# exemplo: model_metrics.py
import prometheus_client
from sklearn.metrics import accuracy_score, precision_score, recall_score

class ModelMetrics:
    def __init__(self):
        self.accuracy_gauge = prometheus_client.Gauge(
            'model_accuracy', 'Model accuracy score'
        )
        self.precision_gauge = prometheus_client.Gauge(
            'model_precision', 'Model precision score'
        )
        self.recall_gauge = prometheus_client.Gauge(
            'model_recall', 'Model recall score'
        )
    
    def update_metrics(self, y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        
        self.accuracy_gauge.set(accuracy)
        self.precision_gauge.set(precision)
        self.recall_gauge.set(recall)
```

### 2. Data Drift Detection

```python
# exemplo: data_drift.py
from evidently.metrics import DataDriftPreset
from evidently.report import Report
import pandas as pd

class DataDriftDetector:
    def __init__(self, reference_data):
        self.reference_data = reference_data
    
    def detect_drift(self, current_data):
        """Detecta drift usando Evidently AI"""
        report = Report(metrics=[DataDriftPreset()])
        report.run(
            reference_data=self.reference_data,
            current_data=current_data
        )
        
        result = report.as_dict()
        drift_detected = result['metrics'][0]['result']['dataset_drift']
        
        return {
            'drift_detected': drift_detected,
            'drift_score': result['metrics'][0]['result']['drift_score'],
            'report': result
        }
```

### 3. Business Metrics

```python
# exemplo: business_metrics.py
import prometheus_client

class BusinessMetrics:
    def __init__(self):
        self.prediction_counter = prometheus_client.Counter(
            'predictions_total', 'Total number of predictions'
        )
        self.response_time = prometheus_client.Histogram(
            'prediction_response_time_seconds', 'Response time for predictions'
        )
        self.error_rate = prometheus_client.Counter(
            'prediction_errors_total', 'Total number of prediction errors'
        )
    
    def record_prediction(self, response_time, error=False):
        self.prediction_counter.inc()
        self.response_time.observe(response_time)
        if error:
            self.error_rate.inc()
```

## Integração com Prometheus

### Configuração prometheus.yml

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'ml-model-api'
    static_configs:
      - targets: ['localhost:8000']
    scrape_interval: 10s
    metrics_path: /metrics

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### Regras de Alerta

```yaml
# alert_rules.yml
groups:
- name: ml_model_alerts
  rules:
  - alert: ModelAccuracyDrop
    expr: model_accuracy < 0.85
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Model accuracy dropped below threshold"
      description: "Model accuracy is {{ $value }}, below 0.85 threshold"
  
  - alert: DataDriftDetected
    expr: data_drift_detected == 1
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Data drift detected"
      description: "Significant data drift detected in production data"
```

## Integração com Evidently AI

### Report Automático

```python
# exemplo: evidently_monitoring.py
from evidently.report import Report
from evidently.metrics import DataDriftPreset, DataQualityPreset
from evidently.ui.workspace import Workspace

class EvidentlyMonitoring:
    def __init__(self, workspace_path="./evidently_workspace"):
        self.workspace = Workspace.create(workspace_path)
    
    def create_monitoring_report(self, reference_data, current_data):
        """Cria report completo de monitoramento"""
        report = Report(metrics=[
            DataDriftPreset(),
            DataQualityPreset()
        ])
        
        report.run(
            reference_data=reference_data,
            current_data=current_data
        )
        
        # Salva no workspace
        self.workspace.add_report(report)
        return report
    
    def schedule_monitoring(self):
        """Agenda monitoramento periódico"""
        # Implementar scheduling com APScheduler ou similar
        pass
```

## Dashboard Grafana

### Exemplo de Queries PromQL

```promql
# Accuracy do modelo ao longo do tempo
model_accuracy

# Taxa de erro de predições
rate(prediction_errors_total[5m])

# Tempo médio de resposta
histogram_quantile(0.95, rate(prediction_response_time_seconds_bucket[5m]))

# Volume de predições por minuto
rate(predictions_total[1m]) * 60
```

## Health Checks

```python
# exemplo: health_check.py
from fastapi import APIRouter
import prometheus_client

router = APIRouter()

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": check_model_health(),
        "drift_status": check_drift_status(),
        "last_training": get_last_training_date()
    }

@router.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return prometheus_client.generate_latest()
```

## Alertas e Notificações

### Sistema de Alertas

```python
# exemplo: alert_manager.py
import smtplib
from enum import Enum

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning" 
    CRITICAL = "critical"

class AlertManager:
    def __init__(self, smtp_config):
        self.smtp_config = smtp_config
    
    def send_alert(self, message, severity: AlertSeverity, recipients):
        """Envia alerta via email/Slack/Teams"""
        if severity == AlertSeverity.CRITICAL:
            # Notificação imediata
            self._send_email(message, recipients)
            self._send_slack(message)
        elif severity == AlertSeverity.WARNING:
            # Batch de alertas
            self._queue_alert(message)
```

## Boas Práticas

- **Métricas Balanceadas**: Colete métricas técnicas e de negócio
- **Thresholds Adaptativos**: Use thresholds que se ajustam ao comportamento do modelo
- **Monitoramento Contínuo**: Configure alertas em tempo real
- **Retenção de Dados**: Mantenha histórico para análise de tendências
- **Documentação**: Documente todos os dashboards e alertas
- **Testes de Alertas**: Teste regularmente o sistema de alertas
- **Escalabilidade**: Use ferramentas que escalam com o volume de dados

## Ferramentas Recomendadas

- **Prometheus + Grafana**: Métricas e dashboards
- **Evidently AI**: Detecção de drift e reports
- **MLflow**: Tracking de experimentos e modelos
- **Great Expectations**: Validação de qualidade dos dados
- **Slack/Teams**: Notificações de alertas
- **PagerDuty**: Gerenciamento de incidentes
