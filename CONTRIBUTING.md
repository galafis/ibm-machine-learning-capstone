# Guia de Contribuição - IBM Machine Learning Engineering Capstone

![IBM](https://img.shields.io/badge/IBM-052FAD?style=for-the-badge&logo=ibm&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![MLOps](https://img.shields.io/badge/MLOps-FF6B6B?style=for-the-badge&logo=mlflow&logoColor=white)

## 🎯 Bem-vindo(a)!

Obrigado pelo seu interesse em contribuir com o projeto IBM Machine Learning Engineering Capstone! Este guia estabelece as diretrizes para uma colaboração eficiente e mantém os padrões de qualidade enterprise do projeto.

## 📋 Índice

- [Como Contribuir](#-como-contribuir)
- [Configuração do Ambiente](#-configuração-do-ambiente)
- [Padrões de Código](#-padrões-de-código)
- [Fluxo de Trabalho](#-fluxo-de-trabalho)
- [Tipos de Contribuição](#-tipos-de-contribuição)
- [Revisão de Código](#-revisão-de-código)
- [Comunicação](#-comunicação)
- [Código de Conduta](#-código-de-conduta)

## 🚀 Como Contribuir

### Pré-requisitos

- **Conhecimento técnico**: Python 3.11+, MLOps, Docker, Kubernetes
- **Experiência**: Machine Learning, DevOps, ou desenvolvimento de software
- **Ferramentas**: Git, GitHub, IDE de sua preferência

### Primeiros Passos

1. **Fork** o repositório
2. **Clone** seu fork localmente
3. Configure o **ambiente de desenvolvimento**
4. Leia a **documentação completa**
5. Escolha uma **issue** para trabalhar

## 🛠️ Configuração do Ambiente

### Instalação Local

```bash
# Clone do repositório
git clone https://github.com/SEU_USUARIO/ibm-machine-learning-capstone.git
cd ibm-machine-learning-capstone

# Configuração do ambiente Python
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Instalação das dependências
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Dependências de desenvolvimento
```

### Configuração com Docker

```bash
# Build da imagem de desenvolvimento
docker build -f docker/Dockerfile.dev -t ml-capstone-dev .

# Execução do container
docker run -it -v $(pwd):/workspace ml-capstone-dev bash
```

### Verificação da Instalação

```bash
# Executar testes
pytest tests/

# Verificar linting
flake8 src/
black --check src/
isort --check-only src/

# Verificar type hints
mypy src/
```

## 📝 Padrões de Código

### Style Guide

- **Python**: PEP 8 + Black formatter
- **Imports**: isort para organização
- **Type Hints**: Obrigatório para funções públicas
- **Docstrings**: Google style para documentação
- **Testes**: pytest com cobertura > 90%

### Exemplo de Função Documentada

```python
from typing import Dict, List, Optional
import pandas as pd

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_params: Optional[Dict] = None,
    validation_split: float = 0.2
) -> Dict[str, float]:
    """Treina modelo de machine learning com validação.
    
    Args:
        X_train: Features de treinamento
        y_train: Target de treinamento  
        model_params: Parâmetros do modelo
        validation_split: Proporção para validação
        
    Returns:
        Métricas de performance do modelo
        
    Raises:
        ValueError: Se os dados estão malformados
        
    Examples:
        >>> metrics = train_model(X_train, y_train, {'n_estimators': 100})
        >>> print(f"Accuracy: {metrics['accuracy']:.3f}")
    """
    # Implementação aqui
    pass
```

### Estrutura de Arquivos

```
src/
├── data/              # Pipeline de dados
│   ├── __init__.py
│   ├── preprocessing.py
│   └── validation.py
├── models/            # Modelos ML
│   ├── __init__.py
│   ├── base_model.py
│   └── fraud_detection.py
├── serving/           # API serving
│   ├── __init__.py
│   ├── api.py
│   └── schemas.py
└── utils/             # Utilitários
    ├── __init__.py
    ├── config.py
    └── logger.py
```

## 🔄 Fluxo de Trabalho

### Branch Strategy

```bash
# Criar branch feature
git checkout -b feature/nome-da-funcionalidade

# Fazer commits atômicos
git commit -m "feat: adiciona validação de drift de dados"

# Push da branch
git push origin feature/nome-da-funcionalidade

# Abrir Pull Request
```

### Padrão de Commits (Conventional Commits)

- `feat:` Nova funcionalidade
- `fix:` Correção de bug
- `docs:` Documentação
- `style:` Formatação de código
- `refactor:` Refatoração
- `test:` Testes
- `chore:` Tarefas de manutenção

**Exemplos:**
```
feat: adiciona endpoint para predição em batch
fix: corrige vazamento de memória no model serving
docs: atualiza README com exemplos de uso
test: adiciona testes para validação de drift
```

### Pull Request Template

```markdown
## Descrição
Descrição clara das mudanças implementadas.

## Tipo de Mudança
- [ ] 🐛 Bug fix
- [ ] ✨ Nova funcionalidade  
- [ ] 💥 Breaking change
- [ ] 📚 Documentação
- [ ] 🧪 Testes

## Checklist
- [ ] Testes passando
- [ ] Cobertura > 90%
- [ ] Documentação atualizada
- [ ] Code review solicitado
- [ ] Performance verificada

## Screenshots/Logs
(Se aplicável)

## Issues Relacionadas
Fixes #123
```

## 🎯 Tipos de Contribuição

### 🔬 Machine Learning
- Novos algoritmos ou modelos
- Otimização de hyperparâmetros
- Feature engineering
- Validação cruzada
- Métricas customizadas

### 🚀 MLOps & Infrastructure
- Pipeline de CI/CD
- Containerização
- Kubernetes manifests
- Monitoring e observabilidade
- Auto-scaling

### 🔧 Engineering
- APIs de serving
- Otimização de performance
- Gerenciamento de configuração
- Logging estruturado
- Error handling

### 📊 Data Engineering
- Pipeline de dados
- Data validation
- Feature stores
- ETL processes
- Data quality checks

### 📚 Documentação
- README updates
- API documentation
- Tutorials e exemplos
- Architecture diagrams
- Best practices guides

### 🧪 Testes
- Unit tests
- Integration tests
- Performance tests
- End-to-end tests
- Load testing

## 👥 Revisão de Código

### Para Reviewers

#### Checklist de Review
- [ ] **Funcionalidade**: Código funciona corretamente
- [ ] **Legibilidade**: Código claro e bem documentado
- [ ] **Performance**: Sem gargalos desnecessários
- [ ] **Segurança**: Não introduz vulnerabilidades
- [ ] **Testes**: Cobertura adequada
- [ ] **Padrões**: Segue style guide do projeto

#### Tipos de Feedback
- **Must Fix**: Problemas que impedem merge
- **Should Fix**: Melhorias importantes
- **Consider**: Sugestões de melhoria
- **Praise**: Reconhecimento de bom trabalho

### Para Autores
- Responda a todos os comentários
- Faça commits de correção separados
- Teste todas as sugestões
- Atualize documentação se necessário
- Seja receptivo ao feedback

## 💬 Comunicação

### Canais Disponíveis
- **GitHub Issues**: Bugs, features, discussões técnicas
- **GitHub Discussions**: Perguntas gerais, ideias
- **Pull Requests**: Review de código, implementação

### Template de Issue

#### Bug Report
```markdown
## 🐛 Bug Report

**Descrição**
Descrição clara do problema.

**Reprodução**
1. Execute `python script.py`
2. Clique em '...'
3. Observe o erro

**Comportamento Esperado**
O que deveria acontecer.

**Ambiente**
- OS: [Ubuntu 22.04]
- Python: [3.11.0]
- Versão: [1.0.0]

**Logs**
```
Cole os logs aqui
```
```

#### Feature Request
```markdown
## ✨ Feature Request

**Problema**
Qual problema esta feature resolveria?

**Solução Proposta**
Descreva a solução ideal.

**Alternativas**
Descreva alternativas consideradas.

**Contexto Adicional**
Qualquer outra informação relevante.
```

### Boas Práticas de Comunicação
- Seja **claro** e **objetivo**
- Use **exemplos** quando necessário
- **Documente** decisões técnicas
- **Pergunte** antes de grandes mudanças
- Seja **respeitoso** e **construtivo**

## 📜 Código de Conduta

### Nossos Valores
- **Respeito**: Trate todos com dignidade
- **Inclusão**: Ambiente acolhedor para todos
- **Colaboração**: Trabalho em equipe efetivo
- **Excelência**: Padrões técnicos elevados
- **Aprendizado**: Crescimento mútuo constante

### Comportamentos Esperados
- ✅ Comunicação respeitosa e profissional
- ✅ Feedback construtivo e específico
- ✅ Reconhecimento de contribuições
- ✅ Paciência com diferentes níveis de experiência
- ✅ Foco na solução, não no problema

### Comportamentos Inaceitáveis
- ❌ Linguagem ofensiva ou discriminatória
- ❌ Ataques pessoais ou ad hominem
- ❌ Trolling ou provocações
- ❌ Assédio de qualquer forma
- ❌ Divulgação de informações privadas

### Aplicação
Violações podem resultar em:
1. **Aviso** informal
2. **Aviso** formal
3. **Suspensão** temporária
4. **Banimento** permanente

## 🏆 Reconhecimento

### Tipos de Contribuidores
- **Core Maintainers**: Revisão e merge de PRs
- **Regular Contributors**: Contribuições frequentes
- **Domain Experts**: Especialistas em áreas específicas
- **Community Helpers**: Suporte a novos contribuidores

### Sistema de Créditos
Contribuições são reconhecidas através de:
- **Contributors** no README
- **Changelog** entries
- **Release notes** highlights
- **Social media** shoutouts

## 🔗 Recursos Úteis

### Documentação Técnica
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Docker Best Practices](https://docs.docker.com/develop/best-practices/)

### MLOps Resources
- [Google MLOps Whitepaper](https://cloud.google.com/resources/mlops-whitepaper)
- [ML Engineering Best Practices](https://developers.google.com/machine-learning/guides/rules-of-ml)
- [Model Monitoring Guide](https://www.evidentlyai.com/blog/ml-monitoring-guide)

### Python Resources
- [PEP 8 Style Guide](https://pep8.org/)
- [Type Hints Guide](https://docs.python.org/3/library/typing.html)
- [pytest Documentation](https://pytest.org/)

## 📞 Contato

**Maintainer**: Gabriel Demetrios Lafis  
**Email**: gabrieldemetrios@gmail.com  
**GitHub**: [@galafis](https://github.com/galafis)  
**LinkedIn**: [Gabriel Lafis](https://linkedin.com/in/gabriel-lafis)

---

**⭐ Lembre-se**: Toda contribuição é valiosa! Seja um bug fix pequeno ou uma feature complexa, sua colaboração ajuda a construir uma plataforma de MLOps de classe mundial.

**🚀 Happy Coding!** 🎯
