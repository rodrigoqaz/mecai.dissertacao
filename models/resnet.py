import torch.nn as nn
from torchvision import models
from src.config.models.config import Config

def initialize_model(device, num_classes: int, **params) -> nn.Module:
    """
    Inicializa um modelo ResNet pré-treinado, adaptando o head para fine-tuning.
    """
    architecture = params.get('architecture', 'resnet50')
    pretrained = params.get('pretrained', True)
    unfreeze_layers = params.get('unfreeze_layers', 2)
    hidden_units = params.get('hidden_units', 512)
    dropout = params.get('dropout', 0.3)
    weights = params.get('weights', 'IMAGENET1K_V1')
    input_channels = params.get('input_channels', 3)
    # device = Config.device

    # Carregar modelo base
    base_model = getattr(models, architecture)(weights=weights if pretrained else None)

    # Ajusta a primeira camada, caso o número de canais no input seja diferente de 3 (RGB)
    if input_channels != 3:
        old_conv1 = base_model.conv1
        base_model.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride,
            padding=old_conv1.padding,
            bias=old_conv1.bias
        )

    # Congelar todas as camadas
    for param in base_model.parameters():
        param.requires_grad = False

    # Descongelar as últimas camadas do encoder
    children = list(base_model.children())
    for layer in children[-unfreeze_layers:]:
        for param in layer.parameters():
            param.requires_grad = True

    # Substituir o head
    in_features = base_model.fc.in_features
    base_model.fc = nn.Sequential(
        nn.Linear(in_features, hidden_units, device=device),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_units, hidden_units // 2, device=device),
        nn.ReLU(),
        nn.Dropout(dropout*0.8),
        nn.Linear(hidden_units // 2, num_classes, device=device)
    )

    return base_model.to(device)


