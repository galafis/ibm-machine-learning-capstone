# Kubernetes (k8s) — Orquestração MLOps

Pasta com os manifests, exemplos de deploy, políticas e instruções para rodar toda a stack do projeto em Kubernetes, seja em cloud, local ou plataforma híbrida.

## Estrutura Recomendada

- `deployment-api.yaml` — Deploy do serviço de serving/REST API com FastAPI
- `deployment-data-pipeline.yaml` — Deploy de workers/executores para ETL, ingestão e validação
- `deployment-mlflow.yaml` — MLflow tracking server como serviço separado
- `service-api.yaml` — Serviço para expor APIs internamente/externamente
- `service-mlflow.yaml` — Exposição do MLflow
- `configmap.yaml` — Variáveis de ambiente e configs customizadas
- `secrets.yaml` — Segredos sensíveis (tokens, senhas, URIs)
- `hpa-api.yaml` — HorizontalPodAutoscaler para autoscaling dinâmico
- `ingress.yaml` — Regras de roteamento HTTP/HTTPS
- `README.md` — Este arquivo explicativo

## Exemplo Simplificado de Deployment FastAPI

apiVersion: apps/v1
kind: Deployment
metadata:
name: ml-serving
spec:
replicas: 3
selector:
matchLabels:
app: ml-serving
template:
metadata:
labels:
app: ml-serving
spec:
containers:
- name: ml-api
image: mlops/ml-serving:latest
ports:
- containerPort: 8080
env:
- name: MODEL_PATH
value: "/models/production_model.pkl"
resources:
limits:
memory: "2Gi"
cpu: "2"
requests:
memory: "1Gi"
cpu: "1"

## Boas Práticas

- Use `resources.limits/requests` para evitar overprovisioning
- Separe *ConfigMap* de *Secret* (senhas, APIs, URIs sensíveis)
- Sempre especifique probes: `livenessProbe`/`readinessProbe` para APIs
- Manifests versionados! Use pastas para ambientes (dev, staging, prod)
- Use namespaces para isolar cada aplicativo/componente maior (ex: `ml-platform`, `monitoring`)

## Referências

- [Kubernetes Official Docs](https://kubernetes.io/docs/)
- [Kubernetes Patterns](https://www.oreilly.com/library/view/kubernetes-patterns/)
- [Kubeflow, Seldon, KFServing para MLOps](https://www.kubeflow.org/)

