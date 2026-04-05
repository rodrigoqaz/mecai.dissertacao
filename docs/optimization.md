# Otimização de Hiperparâmetros com Optuna

Este documento explica como usar o sistema de otimização de hiperparâmetros integrado ao projeto.

## 🎯 Visão Geral

O sistema de otimização permite encontrar automaticamente os melhores hiperparâmetros para cada modelo usando o Optuna, mantendo total integração com MLflow para tracking e reprodutibilidade.

## 🚀 Uso Rápido

### Otimização Básica
```bash
# Otimizar ResNet com 50 trials
python optimize_model.py --model resnet --trials 50

# Otimizar Inception com objetivo balanceado
python optimize_model.py --model inception --trials 30 --objective balanced
```

### Treinamento Tradicional (mantido)
```bash
# Continua funcionando como antes
python train_model.py
```

## 📋 Parâmetros Disponíveis

### Modelos Suportados
- `resnet` - ResNet (18, 34, 50, 101)
- `inception` - Inception v3
- `efficientnet` - EfficientNet (B0-B3)
- `densenet` - DenseNet (121, 161, 169)
- `vgg` - VGG (11, 13, 16, 19)

### Objetivos de Otimização
- `accuracy` - Maximizar accuracy de validação (padrão)
- `balanced` - Combinar accuracy e loss (70% acc + 30% loss)
- `loss` - Minimizar loss de validação

### Parâmetros do CLI
```bash
python optimize_model.py [OPÇÕES]

--model, -m          Nome do modelo (obrigatório)
--trials, -t         Número de trials (padrão: 50)
--study-name, -s     Nome do estudo (padrão: {model}_optimization)
--timeout            Timeout em segundos (padrão: sem limite)
--objective          Tipo de objetivo (padrão: accuracy)
```

## 🔧 Hiperparâmetros Otimizados

### Parâmetros Globais
- **Batch size**: [16, 32, 64] (varia por modelo)
- **Learning rate**: 1e-5 a 1e-2 (log scale)
- **Weight decay**: 1e-6 a 1e-2 (log scale)

### Parâmetros do Modelo
- **Arquitetura**: Varia por tipo de modelo
- **Unfreeze layers**: 2-12 (número de camadas descongeladas)
- **Hidden units**: [256, 512, 1024, 2048]
- **Dropout**: 0.3-0.8

### Otimizadores
- **Tipo**: Adam, AdamW, SGD (varia por modelo)

### Schedulers
- **CosineAnnealingWarmRestarts**: T_0, T_mult
- **ReduceLROnPlateau**: patience, factor
- **StepLR**: step_size, gamma

### Loss Function
- **Label smoothing**: 0.0-0.3

## 📊 Monitoramento com MLflow

### Organização dos Experimentos
```
MLflow Experiments:
├── ResNet_Optimization/        # Trials de otimização
│   ├── Trial_001/
│   ├── Trial_002/
│   └── ...
├── Inception_Optimization/
└── ResNet/                     # Experimentos tradicionais
```

### Tags Automáticas
- `optuna_optimization: true` - Identifica trials de otimização
- `model_type: resnet` - Tipo do modelo
- `trial_number: 001` - Número do trial

### Acessar MLflow UI
```bash
mlflow ui --backend-store-uri file:///Users/rodrigoqaz/Documents/tmp/mlruns
```

## 📁 Estrutura de Arquivos

### Arquivos Principais
```
mecai.dissertacao/
├── optimize_model.py           # Script principal de otimização
├── train_model.py             # Treinamento (modificado para compatibilidade)
├── example_usage.py           # Exemplos de uso
└── src/optimization/
    ├── objective.py           # Função objetivo do Optuna
    ├── hyperparameter_config.py  # Espaços de busca
    └── dynamic_config.py      # Configurações dinâmicas
```

## 💡 Exemplos Práticos

### 1. Teste Rápido
```bash
# Teste com poucos trials para validar funcionamento
python optimize_model.py --model inception --trials 5
```

### 2. Otimização Completa
```bash
# Otimização robusta para produção
python optimize_model.py --model resnet --trials 100 --study-name resnet_production
```

### 3. Comparação de Modelos
```bash
# Compare diferentes modelos
python optimize_model.py --model resnet --trials 50 --study-name comparison_resnet
python optimize_model.py --model inception --trials 50 --study-name comparison_inception
python optimize_model.py --model efficientnet --trials 50 --study-name comparison_efficientnet
```

### 4. Otimização com Timeout
```bash
# Limite de 2 horas
python optimize_model.py --model densenet --trials 200 --timeout 7200
```

## 📈 Análise de Resultados

### Melhores Parâmetros
Os melhores parâmetros são exibidos no console e salvos automaticamente em arquivo YAML:
```
best_config_{model}_{study_name}.yaml
```

### Métricas no MLflow
- **Train/validation accuracy e loss por época**
- **Learning rate schedule**
- **Métricas por classe (precision, recall, F1)**
- **Modelo salvo automaticamente**

### Visualizações Recomendadas
1. **Parallel Coordinate Plot** - Relação entre parâmetros
2. **Optimization History** - Convergência do estudo
3. **Parameter Importance** - Importância de cada hiperparâmetro

## 🔄 Workflow Recomendado

### 1. Exploração Inicial (10-20 trials)
```bash
python optimize_model.py --model resnet --trials 15 --study-name resnet_exploration
```

### 2. Análise dos Resultados
- Verificar trends no MLflow
- Identificar hiperparâmetros mais importantes
- Ajustar espaços de busca se necessário

### 3. Otimização Focada (50-100 trials)
```bash
python optimize_model.py --model resnet --trials 80 --study-name resnet_focused
```

### 4. Validação Final
- Treinar modelo com melhores parâmetros
- Validar em conjunto de teste separado
- Usar configuração YAML gerada

## 🛠️ Personalização

### Modificar Espaços de Busca
Edite `src/optimization/hyperparameter_config.py` para:
- Adicionar novos hiperparâmetros
- Modificar ranges de valores
- Criar espaços condicionais

### Novos Objetivos
Adicione funções em `src/optimization/objective.py` para:
- Otimizar métricas customizadas
- Combinar múltiplos objetivos
- Incluir restrições específicas

## ⚠️ Considerações Importantes

### Performance
- Cada trial executa um treinamento completo
- Use GPUs para acelerar (MPS/CUDA)
- Considere early stopping agressivo para trials ruins

### Recursos
- Monitor uso de memória com muitos trials
- Considere executar overnight para estudos longos
- Salve estudos periodicamente

### Reprodutibilidade
- Seeds são fixas para reprodutibilidade
- Configurações são salvas automaticamente
- MLflow mantém histórico completo

## 🐛 Troubleshooting

### Erro: "Model not found"
- Verifique se o modelo está na lista suportada
- Confirme arquivos YAML de configuração existem

### Erro: "CUDA out of memory"
- Reduza batch size no espaço de busca
- Use modelos menores para testes
- Monitor GPU usage

### Erro: "MLflow tracking"
- Verifique caminho do MLflow URI
- Confirme permissões de escrita
- Teste MLflow UI standalone

### Trials falham consistentemente
- Verifique configurações base do modelo
- Teste treinamento tradicional primeiro
- Analise logs de erro detalhados

## 📚 Referências

- [Optuna Documentation](https://optuna.readthedocs.io/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [PyTorch Lightning + Optuna](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.loggers.MLFlowLogger.html)

## 🤝 Contribuição

Para adicionar novos modelos ou funcionalidades:
1. Adicione configuração em `hyperparameter_config.py`
2. Teste com poucos trials
3. Documente parâmetros específicos
4. Atualize este README



# OTIMIZAÇÕES

# Exemplo genérico
python optimize_model.py \
  --model vgg \
  --study-name vgg_v8_AB_opt_aug \
  --storage sqlite:///optuna_dissertacao.db \
  --objective mcc_loss_composite \
  --trials 30

## GRUPO A: TRANSFORMERS (Sensíveis, AdamW, Regularização Alta)
- Vit
    [OK]	v1	AB	vit_v1_AB_opt_aug			
    [  ]	v1	AM	vit_v1_AM_opt_aug			
    [  ]	v1	BR	vit_v1_BR_opt_aug			
    [OK]	v8	AB	vit_v8_AB_opt_aug			
    [  ]	v8	AM	vit_v8_AM_opt_aug			
    [  ]	v8	BR	vit_v8_BR_opt_aug
- Swin	
    [OK]	v1	AB	swin_v1_AB_opt_aug			
    [  ]	v1	AM	swin_v1_AM_opt_aug			
    [  ]	v1	BR	swin_v1_BR_opt_aug			
    [OK]	v8	AB	swin_v8_AB_opt_aug			
    [  ]	v8	AM	swin_v8_AM_opt_aug			
    [  ]	v8	BR	swin_v8_BR_opt_aug

## GRUPO B: CNNs MODERNAS (ConvNeXt, EfficientNet)
- ConvNeXt
    [OK]	v1	AB	convnext_v1_AB_opt_aug
    [  ]	v1	AM	convnext_v1_AM_opt_aug			
    [  ]	v1	BR	convnext_v1_BR_opt_aug			
    [OK]	v8	AB	convnext_v8_AB_opt_aug			
    [  ]	v8	AM	convnext_v8_AM_opt_aug			
    [  ]	v8	BR	convnext_v8_BR_opt_aug
- EfficientNet
    [OK]	v1	AB	efficientnet_v1_AB_opt_aug
    [  ]	v1	AM	efficientnet_v1_AM_opt_aug			
    [  ]	v1	BR	efficientnet_v1_BR_opt_aug			
    [OK]	v8	AB	efficientnet_v8_AB_opt_aug			
    [  ]	v8	AM	efficientnet_v8_AM_opt_aug			
    [  ]	v8	BR	efficientnet_v8_BR_opt_aug

## GRUPO C: CNNs ROBUSTAS (ResNet, DenseNet, Inception)
- ResNet
    [OK]	v1	AB	resnet_v1_AB_opt_aug
    [  ]	v1	AM	resnet_v1_AM_opt_aug			
    [  ]	v1	BR	resnet_v1_BR_opt_aug			
    [OK]	v8	AB	resnet_v8_AB_opt_aug			
    [  ]	v8	AM	resnet_v8_AM_opt_aug			
    [  ]	v8	BR	resnet_v8_BR_opt_aug
- DenseNet
    [OK]	v1	AB	densenet_v1_AB_opt_aug
    [  ]	v1	AM	densenet_v1_AM_opt_aug			
    [  ]	v1	BR	densenet_v1_BR_opt_aug			
    [OK]	v8	AB	densenet_v8_AB_opt_aug			
    [  ]	v8	AM	densenet_v8_AM_opt_aug			
    [  ]	v8	BR	densenet_v8_BR_opt_aug
- Inception
    [OK]	v1	AB	inception_v1_AB_opt_aug
    [  ]	v1	AM	inception_v1_AM_opt_aug			
    [  ]	v1	BR	inception_v1_BR_opt_aug			
    [OK]	v8	AB	inception_v8_AB_opt_aug			
    [  ]	v8	AM	inception_v8_AM_opt_aug			
    [  ]	v8	BR	inception_v8_BR_opt_aug

## GRUPO D: LEGACY (VGG)
- VGG
    [OK]	v1	AB	vgg_v1_AB_opt_aug
    [  ]	v1	AM	vgg_v1_AM_opt_aug			
    [  ]	v1	BR	vgg_v1_BR_opt_aug			
    [OK]	v8	AB	vgg_v8_AB_opt_aug			
    [  ]	v8	AM	vgg_v8_AM_opt_aug			
    [  ]	v8	BR	vgg_v8_BR_opt_aug


Faça uma pesquisa academica profunda da Vit. A ideia é entender desde a concepção até a evolução da arquitetura.

## GRUPO A: TRANSFORMERS (Sensíveis, AdamW, Regularização Alta)
- Vit
- Swin

## GRUPO B: CNNs MODERNAS (ConvNeXt, EfficientNet)
- ConvNeXt
- EfficientNet

## GRUPO C: CNNs ROBUSTAS (ResNet, DenseNet, Inception)
- ResNet
- DenseNet
- Inception

## GRUPO D: LEGACY (VGG)
- VGG