import torch.nn as nn
import torch
from torchvision import models
from src.config.models.config import Config

def initialize_model(num_classes: int, **params) -> nn.Module:
    """
    Inicializa um modelo ResNet pré-treinado com 15 canais de entrada.
    """
    # Parâmetros do modelo
    architecture = params.get('architecture', 'resnet50')
    pretrained = params.get('pretrained', True)
    unfreeze_layers = params.get('unfreeze_layers', 4)  # Aumentado para 4 camadas
    hidden_units = params.get('hidden_units', 1024)
    dropout = params.get('dropout', 0.5)  # Aumentado o dropout
    weights = params.get('weights', 'IMAGENET1K_V1')
    device = Config.device

    # Carregar modelo base
    base_model = getattr(models, architecture)(weights=weights if pretrained else None)

    # Modificar a primeira camada convolucional para 15 canais
    original_conv1 = base_model.conv1
    new_conv1 = nn.Conv2d(
        in_channels=15,
        out_channels=original_conv1.out_channels,
        kernel_size=original_conv1.kernel_size,
        stride=original_conv1.stride,
        padding=original_conv1.padding,
        bias=False
    ).to(device)

    # Inicialização híbrida dos pesos
    if pretrained:
        with torch.no_grad():
            # Copiar pesos para os primeiros 3 canais
            new_conv1.weight[:, :3] = original_conv1.weight.clone()
            
            # Inicializar canais extras com média dos existentes + ruído
            mean_weights = original_conv1.weight.mean(dim=1, keepdim=True)
            new_conv1.weight[:, 3:] = mean_weights.repeat(1, 12, 1, 1) * 0.01

    base_model.conv1 = new_conv1

    # Congelar camadas
    for param in base_model.parameters():
        param.requires_grad = False

    # Descongelar camadas selecionadas
    children = list(base_model.children())
    for layer in children[-unfreeze_layers:]:
        for param in layer.parameters():
            param.requires_grad = True

    # Substituir head com maior capacidade
    in_features = base_model.fc.in_features
    base_model.fc = nn.Sequential(
        nn.Linear(in_features, hidden_units, device=device),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.BatchNorm1d(hidden_units),  # Adicionado BatchNorm
        nn.Linear(hidden_units, hidden_units // 2, device=device),
        nn.ReLU(),
        nn.Dropout(dropout // 1.5),
        nn.Linear(hidden_units // 2, num_classes, device=device)
    )

    return base_model.to(device)
