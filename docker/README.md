# Docker & Containerização MLOps

Pasta dedicada a Dockerfiles, builds multi-stage, scripts e configurações para deployment de pipelines, APIs e serviços do projeto em containers.

## Estrutura Recomendada

- `Dockerfile.api` — Container para serving FastAPI dos modelos
- `Dockerfile.data-pipeline` — Pipeline de ingestão, validação e feature store
- `Dockerfile.mlflow` — MLflow tracking e artifacts
- `docker-compose.yml` — Orquestração local de múltiplos serviços (MLflow, API, banco, worker etc)
- `docker-entrypoint.sh` — Entrypoint customizado para scripts de inicialização (opcional)

## Exemplo de Dockerfile para Serving com FastAPI
Dockerfile.api
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY ./src/ ./src/
COPY ./models/production_model.pkl ./models/

EXPOSE 8080

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8080"]

## Dicas & Boas Práticas

- Prefira imagens base leves (ex: `python:3.X-slim`)
- Sempre utilize `.dockerignore` para evitar build lento e vazamento de dados
- Para MLOps, segregue containers: serving, training, tracking e orquestração
- Use multi-stage builds para pipelines pesadas de dados ou ML
- Sempre documente variáveis de ambiente críticas no README

## Referências

- [Documentação Oficial Docker](https://docs.docker.com/)
- [Best Practices for Dockerfiles](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)
