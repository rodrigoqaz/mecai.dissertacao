# Dissertação - MECAI: Classificação de Algodão com Visão Computacional

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7.0-red.svg)
![License](https://img.shields.io/badge/license-Academic-green.svg)


## 📋 Visão Geral
Este projeto de dissertação de mestrado desenvolve um sistema de classificação automática de algodão utilizando técnicas de visão computacional e aprendizado profundo. O objetivo é criar modelos capazes de identificar e classificar diferentes tipos e qualidades de algodão através de análise de imagens. O repositório adota uma estrutura monorepo para facilitar a integração entre coleta de dados, processamento, experimentação e documentação acadêmica.

## 📒 Sumário
- [📋 Visão Geral](#visão-geral)
- [🎯 Objetivos](#-objetivos)
- [📁 Estrutura do Projeto](#-estrutura-do-projeto)
- [🚀 Principais Funcionalidades](#-principais-funcionalidades)
- [💾 Como Executar](#como-executar)
- [📊 Monitoramento e Logs](#-monitoramento-e-logs)
- [📈 Resultados](#-resultados)
- [🤝 Contribuição](#-contribuição)
- [📄 Licença](#-licença)
- [👨‍💻 Autor](#-autor)
- [🙏 Agradecimentos](#-agradecimentos)
- [☑️ Status do Projeto](#-status-do-projeto)


### 🎯 Objetivos

- Desenvolver modelos de deep learning para classificação de algodão
- Implementar pipeline completo de processamento de dados e treinamento
- Criar aplicação para coleta e anotação de dados
- Produzir documentação acadêmica completa em formato de dissertação



## 📁 Estrutura do Projeto

```
mecai.dissertacao/
├── data/                   # Camadas de dados (bronze, silver, gold)
│   ├── bronze/
│   ├── silver/
│   └── gold/
├── dissertacao/            # Arquivos LaTeX e recursos para a dissertação/documentação acadêmica
├── image_capture/          # Aplicação para captura de amostras de algodão
├── models/                 # Definição e arquivos de modelos treinados
├── notebooks/              # Jupyter Notebooks para experimentação
├── src/                    # Código-fonte principal
│   ├── config/             # Configurações do projeto
│   ├── data/               # Processamento e manipulação de dados
│   └── utils/              # Utilitários gerais
├── .gitignore
├── build_datasets.py       # Script para construir datasets
├── limpar.sh               # Script de limpeza de arquivos temporários
├── requirements.txt        # Dependências do projeto
└── train_model.py          # Script principal de treinamento
```

## 🚀 Principais Funcionalidades

- **Coleta de Dados:** Aplicação para captura de imagens de amostras de algodão.
- **Processamento de Dados:** Scripts para construção, manipulação e leitura de datasets em múltiplas camadas (bronze, silver, gold).
- **Modelagem:** Definição, treinamento e avaliação de modelos de classificação de imagens.
- **Documentação Acadêmica:** Estrutura completa para escrita e compilação da dissertação em LaTeX, conforme o padrão fornecido pela USP.
- **Notebooks:** Experimentação e análise exploratória de dados e resultados.

## 💾 Como Executar

1. **Pré-requisitos**

- Python 3.10
- MPS ou CUDA-compatible GPU (recomendado para treinamento)
- Git

2. **Clone o repositório:**

```bash
git clone https://github.com/rodrigoqaz/mecai.dissertacao
cd mecai.dissertacao
```

3. **Crie e ative o ambiente virtual:**

```bash
python -m venv .venv
source .venv/bin/activate # Linux/Mac
.venv\Scripts\activate # Windows
```

4. **Instale as dependências:**

```bash
pip install -r requirements.txt
```

5. **Construa os datasets (se necessário):**

adicionar aqui a etapa de download tbm
```bash
python build_datasets.py
```

5. **Treine o modelo:**

```bash
python train_model.py
```
6. **Limpe arquivos temporários (gerados pelo compilador Latex):**

```bash
.\limpar.sh
```

## 📊 Monitoramento e Logs

O projeto utiliza **MLflow** para rastreamento de experimentos:

- Métricas de treino e validação
- Hiperparâmetros
- Artefatos do modelo
- Comparação entre experimentos

## 📈 Resultados

### Modelos Treinados

- `best_model.pth`: Melhor modelo geral
- `best_model_restnet.pth`: Melhor modelo ResNet
- `best_densenet.pth`: Melhor modelo DenseNet

### Métricas (Exemplo)

```
Modelo: VGG16
Acurácia: XX.X%
F1-Score: XX.X%
Precisão: XX.X%
Recall: XX.X%
```

## 🤝 Contribuição

Contribuições são bem-vindas! Para sugerir melhorias ou reportar problemas, abra uma issue ou envie um pull request.

## 📄 Licença

Este projeto é de uso acadêmico. Consulte o arquivo LICENSE (se houver) para mais detalhes.

## 👨‍💻 Autor

**Rodrigo de Souza Oliveira** - Mestrando em [Matemática, Estatística e Computação aplicado à Indústria/ICMC-USP]

## 🙏 Agradecimentos

- Orientador(a): [Nome]
- Programa de Pós-graduação em [Área]
- [Instituição de Ensino]
- Colaboradores e colegas

---

## ☑️ Status do Projeto

![Status](https://img.shields.io/badge/status-Em%20Desenvolvimento-yellow)


---