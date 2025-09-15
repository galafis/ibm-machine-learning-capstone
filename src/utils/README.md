# Utilitários (Utils) para MLOps

Pasta destinada a funções auxiliares, classes utilitárias e scripts de apoio aos pipelines de dados, APIs de serving e automações de machine learning.

## Exemplos de Utilitários

- Manipulação, transformação e limpeza de dados (data cleaning, feature engineering)
- Funções genéricas reutilizáveis (logging customizado, serialização, conversão de formatos)
- Validações e asserts para integridade dos dados
- Utils para conexão com bancos, APIs externas, armazenamento (S3, Azure, GCP)
- Geração de hash/versionamento de arquivos e experimentos
- Scripts utilitários para orquestração de pipelines, scheduled jobs, etc.

## Estrutura Recomendada

- `data_cleaning.py` — Funções para limpeza, sanitização e padronização de datasets
- `feature_utils.py` — Funções para engenharia e seleção de features
- `api_helpers.py` — Validadores e estruturadores para APIs (entrada, saída, status)
- `storage.py` — Utils para ler/gravar arquivos em cloud/local
- `hashing.py` — Geração de hash e verificação de integridade
- `validation.py` — Classes para checagem de tipos e regras customizadas para cada pipeline
- `README.md` — (este arquivo), documentação dos utilitários disponíveis

## Boas Práticas

- Prefira funções puras e com docstrings detalhadas
- Garanta testes unitários para cada utilitário crítico
- Isolar dependências de terceiros ao máximo (caminhos no requirements separáveis)


