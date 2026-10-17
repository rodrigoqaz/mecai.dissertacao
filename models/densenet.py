import torch.nn as nn
from torchvision import models

def initialize_model(device, num_classes: int, input_channels: int = 3, **params) -> nn.Module:
    """
    Inicializa um modelo DenseNet pré-treinado, adaptando o head para fine-tuning.
    """
    # Obter parâmetros com valores padrão
    architecture = params.get('architecture', 'densenet121')
    pretrained = params.get('pretrained', True)
    unfreeze_layers = params.get('unfreeze_layers', 2)
    hidden_units = params.get('hidden_units', 256)
    dropout = params.get('dropout', 0.3)
    weights = params.get('weights', 'IMAGENET1K_V1')

    # Carregar modelo base
    try:
        weights_enum = models.DenseNet121_Weights.verify(weights) if pretrained else None
    except AttributeError:
        weights_enum = None
        
    base_model = getattr(models, architecture)(weights=weights_enum)

    # Ajustar a primeira camada convolucional se input_channels for diferente de 3
    if input_channels != 3:
        original_conv = base_model.features.conv0
        new_conv = nn.Conv2d(
            input_channels,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias
        )
        base_model.features.conv0 = new_conv

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
