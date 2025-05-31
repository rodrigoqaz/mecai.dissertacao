import torch
import torch.nn as nn
from torchvision import models
from src.config.models.config import Config

def initialize_model(num_classes: int, **params) -> nn.Module:
    """
    Inicializa um modelo EfficientNet pré-treinado, adaptando o head para fine-tuning.
    """
    architecture = params.get('architecture', 'efficientnet_b0')
    pretrained = params.get('pretrained', True)
    unfreeze_layers = params.get('unfreeze_layers', 2)
    hidden_units = params.get('hidden_units', 256)
    dropout = params.get('dropout', 0.3)
    weights = params.get('weights', 'IMAGENET1K_V1')
    device = Config.device

    # Carregar modelo base
    base_model = getattr(models, architecture)(weights=weights if pretrained else None)

    # Congelar todas as camadas
    for param in base_model.parameters():
        param.requires_grad = False

    # Descongelar as últimas camadas do encoder
    children = list(base_model.children())
    for layer in children[-unfreeze_layers:]:
        for param in layer.parameters():
            param.requires_grad = True

    # Substituir o head
    in_features = base_model.classifier[1].in_features
    base_model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, hidden_units, device=device),
        nn.ReLU(),
        nn.Dropout(dropout*0.8),
        nn.Linear(hidden_units, num_classes, device=device)
    )

    return base_model.to(device)
