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

