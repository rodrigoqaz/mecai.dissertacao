import torch.nn as nn
from torchvision import models

def initialize_model(device, num_classes: int, input_channels: int = 3, **params) -> nn.Module:
    """
    Inicializa um modelo EfficientNet pré-treinado, adaptando o head para fine-tuning.
    """
    architecture = params.get('architecture', 'efficientnet_b0')
    pretrained = params.get('pretrained', True)
    unfreeze_layers = params.get('unfreeze_layers', 2)
    hidden_units = params.get('hidden_units', 256)
    dropout = params.get('dropout', 0.3)
    weights = params.get('weights', 'IMAGENET1K_V1')

    # Carregar modelo base
    base_model = getattr(models, architecture)(weights=weights if pretrained else None)

    # Ajustar a primeira camada convolucional se input_channels for diferente de 3
    if input_channels != 3:
        first_conv_layer = base_model.features[0]
        original_conv = first_conv_layer[0]
        new_conv = nn.Conv2d(
            input_channels,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=(original_conv.bias is not None)
        )
        first_conv_layer[0] = new_conv
        # base_model.features[0] = new_conv

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
