import torch
import torch.nn as nn
from torchvision import models
from src.config.models.config import Config

def initialize_model(num_classes: int, **params) -> nn.Module:
    """
    Inicializa o modelo VGG16 pré-treinado no ImageNet com ajustes para fine-tuning.

    - Congela as camadas convolucionais, exceto as 4 últimas.
    - Substitui o classificador final para o número de classes do problema.
    - Permite customização de dropout e hidden units.

    Args:
        num_classes (int): Número de classes de saída.
        **params: Parâmetros configuráveis:
            - dropout_rate_1 (float): Dropout após primeira camada do classificador.
            - dropout_rate_2 (float): Dropout após segunda camada do classificador.
            - hidden_units (int): Unidades da primeira camada do classificador.
            - unfreeze_layers: Número de camadas a descongelar no fim da rede (default: 4).
            - weights: Pesos pré-treinados a usar (default: 'IMAGENET1K_V1').
            - device: Dispositivo onde carregar o modelo (opcional).

    Returns:
        nn.Module: Modelo VGG16 ajustado.
    """

    # Carregar os parâmetros com valores padrão
    dropout_rate_1 = params.get('dropout_rate_1', 0.5)
    dropout_rate_2 = params.get('dropout_rate_2', 0.3)
    hidden_units = params.get('hidden_units', 256)
    unfreeze_layers = params.get('unfreeze_layers', 4)
    weights = params.get('weights', 'IMAGENET1K_V1')
    device = Config.device

    # Criar o modelo base
    base_model = models.vgg16(weights=weights)
    
    # Congelar camadas
    for param in base_model.features.parameters():
        param.requires_grad = False

    # Descongelar as últimas camadas conforme configuração
    for layer in base_model.features[-unfreeze_layers:]:
        for param in layer.parameters():
            param.requires_grad = True
    
    # Criar o classificador
    base_model.classifier = nn.Sequential(
        nn.Linear(25088, hidden_units, dtype=torch.float32, device=device),
        nn.BatchNorm1d(hidden_units, device=device),
        nn.ReLU(),
        nn.Dropout(dropout_rate_1),
        nn.Linear(hidden_units, hidden_units // 2, dtype=torch.float32, device=device),
        nn.BatchNorm1d(hidden_units // 2, device=device),
        nn.ReLU(),
        nn.Dropout(dropout_rate_2),
        nn.Linear(hidden_units // 2, num_classes, dtype=torch.float32, device=device)
    )
    
    # Converter o modelo para o device selecionado
    if device:
        base_model = base_model.to(device, dtype=torch.float32)

    return base_model
