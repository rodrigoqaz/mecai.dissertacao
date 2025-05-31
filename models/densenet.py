import torch
import torch.nn as nn
from torchvision import models
from src.config.models.config import Config

def initialize_model(num_classes: int, **params) -> nn.Module:
    """
    Inicializa um modelo DenseNet pré-treinado, adaptando o head para fine-tuning.
    
    Args:
        num_classes (int): Número de classes de saída
        params: Parâmetros de configuração:
            - architecture: Arquitetura DenseNet (ex: densenet121)
            - pretrained: Usar pesos pré-treinados (default: True)
            - unfreeze_layers: Número de camadas finais para descongelar (default: 2)
            - hidden_units: Unidades na camada oculta (default: 256)
            - dropout: Taxa de dropout (default: 0.3)
            - weights: Versão dos pesos (ex: IMAGENET1K_V1)
    
    Returns:
        nn.Module: Modelo DenseNet configurado
    """
    # Obter parâmetros com valores padrão
    architecture = params.get('architecture', 'densenet121')
    pretrained = params.get('pretrained', True)
    unfreeze_layers = params.get('unfreeze_layers', 2)
    hidden_units = params.get('hidden_units', 256)
    dropout = params.get('dropout', 0.3)
    weights = params.get('weights', 'IMAGENET1K_V1')
    device = Config.device

    # Carregar modelo base
    try:
        weights_enum = models.DenseNet121_Weights.verify(weights) if pretrained else None
    except AttributeError:
        weights_enum = None
        
    base_model = getattr(models, architecture)(weights=weights_enum)

    # Congelar todas as camadas inicialmente
    for param in base_model.parameters():
        param.requires_grad = False

    # Descongelar as últimas camadas do encoder
    children = list(base_model.children())
    for layer in children[-unfreeze_layers:]:
        for param in layer.parameters():
            param.requires_grad = True

    # Substituir o classificador
    in_features = base_model.classifier.in_features
    base_model.classifier = nn.Sequential(
        nn.Dropout(dropout, inplace=False),
        nn.Linear(in_features, hidden_units),
        nn.ReLU(inplace=False),
        nn.Dropout(dropout * 0.8, inplace=False),
        nn.Linear(hidden_units, num_classes)
    ).to(device)

    return base_model.to(device)
