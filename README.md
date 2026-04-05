# Classificação da Qualidade do Algodão via Deep Learning

Este projeto de dissertação de mestrado propõe um sistema de visão computacional para a classificação automática da pluma de algodão (tipagem), comparando arquiteturas de redes neurais sob diferentes condições de iluminação e profundidade espectral.

---

## 🚀 Guia de Execução da Pesquisa

O fluxo de trabalho é dividido em quatro estágios principais para garantir o rigor científico.

### 1. Preparação dos Dados
Converte as amostras brutas em conjuntos de dados processados (v1 a v12).
```bash
python build_datasets.py
```
*Consulte `data/gold/` para verificar as versões geradas.*

### 2. Otimização de Hiperparâmetros (Optuna)
Busca automática das melhores configurações para cada arquitetura e versão de dataset.
```bash
python optimize_model.py --model densenet --trials 30 --objective mcc_loss_composite
```
*As melhores configurações são salvas automaticamente em `src/config/models/optimized/`.*

### 3. Treinamento da Elite
Geração dos pesos finais (.pth) para os modelos campeões identificados na etapa anterior.
```bash
# Use o script de treinamento v2 com o YAML otimizado
python train_model_v2.py --config src/config/models/optimized/best_config_...yaml
```

### 4. Pipeline de Dissertação (Geração de Resultados)
Este pipeline transforma experimentos em evidência científica pronta para a tese.

#### A. Mineração (`select_best_models.py`)
Varre o MLflow e seleciona o melhor run de cada categoria baseado no MCC.
```bash
python select_best_models.py
```

#### B. Tribunal de Teste (`run_final_test_inference.py`)
Executa a inferência definitiva no conjunto de teste (`hold-out`) nunca visto pelos modelos.
- **Benchmarking:** Mede latência real (com sincronização de hardware) e complexidade computacional (**GFLOPs**).
```bash
python run_final_test_inference.py
```

#### C. Análise Estatística (`generate_hypothesis_results.py`)
Gera os artefatos finais para inclusão no documento LaTeX.
- **Teste de McNemar:** Valida a significância estatística entre as arquiteturas e os sensores.
- **Artefatos:** Tabelas LaTeX, Heatmaps de P-Value e Matrizes de Confusão.
```bash
python generate_hypothesis_results.py
```

---

## 📊 Matriz de Experimentos

| Versão | Luz | Canais | Foco Científico |
| :--- | :--- | :--- | :--- |
| **V1** | AB | 3 (RGB) | **Baseline:** Espectro pleno. |
| **V8** | AB | 15 | Impacto da **Expansão Espectral Digital**. |
| **V9** | AM | 3 (RGB) | Impacto do **Filtro Amarelo** (Cor). |
| **V10** | BR | 3 (RGB) | Impacto do **Filtro Branco** (Impurezas). |
| **V11** | AM | 15 | Sinergia Luz Amarela + 15 Canais. |
| **V12** | BR | 15 | Sinergia Luz Branca + 15 Canais. |

## 📚 Documentação Detalhada

Para informações aprofundadas sobre partes específicas do projeto, consulte:

*   **[Otimização de Hiperparâmetros](docs/optimization.md):** Detalhes sobre o espaço de busca do Optuna e checklist de experimentos.
*   **[Preparação de Dados](docs/data_preparation.md):** Pipeline de conversão Bronze -> Silver -> Gold.
*   **[Revisão de Arquiteturas](docs/architectures_review.md):** Análise técnica das redes neurais utilizadas (CNNs e Transformers).
*   **[Desenho Experimental](docs/experimental_plan.md):** Metodologia estatística e plano de testes de hipóteses.

---

## 🛠️ Tecnologias e Dependências
- **Deep Learning:** PyTorch, Torchvision, Torchinfo.
- **Eficiência:** fvcore (GFLOPs).
- **Estatística:** statsmodels (McNemar), scikit-learn.
- **Tracking:** MLflow, Optuna.
- **Documentação:** LaTeX (TeLive/MiKTeX).

---

## 📁 Estrutura de Resultados (`results/`)
- `dissertation_artifacts/`: Figuras e tabelas prontas para o PDF final.
- `predictions/`: Dados binários das predições de teste.
- `weights/`: Pesos oficiais da elite dos modelos.
