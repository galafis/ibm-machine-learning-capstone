# Notebooks Jupyter

Esta pasta armazena notebooks para análise exploratória, prototipação de modelos, validação de hipóteses e documentação visual do pipeline MLOps.

## Finalidade

- Experimentação interativa de dados, features e modelos
- Documentação visual de processos, gráficos e resultados
- Teste rápido de protótipos antes da implementação em produção
- Geração de artefatos de apresentação e suporte a auditorias técnicas

## Padrão de Organização

- 01_Exploratory_Data_Analysis.ipynb — EDA dos datasets principais
- 02_Feature_Engineering.ipynb — Transformações, seleção de variáveis
- 03_Model_Prototyping.ipynb — Treinamento/teste rápido de modelos
- 04_Metrics_Evaluation.ipynb — Análise quantitativa/visual de performance
- 05_Model_Interpretation.ipynb — SHAP, LIME, explicações de decisões
- 99_Reference_Playground.ipynb — Área livre para testes diversos

## Boas Práticas

- Sempre versionar notebooks relevantes (evitar dados brutos embarcados)
- Use células markdown explicativas e gráficos interativos
- Exportar resultados importantes para arquivos em /docs ou /reports
- Para notebooks de produção, documente requisitos e dependências extras

## Ferramentas Recomendadas

- JupyterLab/local ou cloud (Colab, Azure Notebooks)
- Papermill para parametrização/batch execution
- nbconvert para exportação em PDF/HTML/Markdown

---
