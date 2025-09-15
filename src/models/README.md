# Modelos de Machine Learning

Esta pasta centraliza a implementação, o versionamento e a documentação dos modelos utilizados na plataforma de MLOps.

## Objetivos

- Organização dos arquivos de definição de modelos
- Scripts de treinamento e validação
- Versionamento e controle de experimentos (MLflow)
- Upload e download para o Model Registry

## Estrutura Recomendada

- `train_model.py`: Pipeline completo de treinamento
- `validate_model.py`: Validação e métricas customizadas
- `registry_manager.py`: Integração com MLflow ou outro registry
- Subpastas para cada use case (`fraud_detection/`, `recommender/`, `forecasting/`)

## Padrões

- Documente hiperparâmetros no início dos scripts
- Utilize docstrings detalhadas
- Cada subpasta deve ter seu próprio README.md descritivo

## Exemplo de comando para validação:

