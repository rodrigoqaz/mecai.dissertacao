import torch.nn as nn
from torchvision import models

def initialize_model(device, num_classes: int, input_channels: int = 3, **params) -> nn.Module:
    """
    Inicializa um modelo ConvNeXt pré-treinado, adaptando para fine-tuning.
    """
    architecture = params.get('architecture', 'convnext_tiny')
    pretrained = params.get('pretrained', True)
    unfreeze_layers = params.get('unfreeze_layers', 2)
    hidden_units = params.get('hidden_units', 512)
    dropout = params.get('dropout', 0.3)
    weights = params.get('weights', 'IMAGENET1K_V1')

    # Carregar modelo base
    base_model = getattr(models, architecture)(weights=weights if pretrained else None)

    # Ajustar a primeira camada convolucional se input_channels for diferente de 3
    if input_channels != 3:
        original_conv = base_model.features[0][0] # ConvNeXt's first convolutional layer
        new_conv = nn.Conv2d(
            input_channels,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=(original_conv.bias is not None)
        )
        base_model.features[0][0] = new_conv.to(device)

    # Congelar todas as camadas inicialmente
    for param in base_model.parameters():
        param.requires_grad = False

    # Descongelar as últimas camadas
    # ConvNeXt.features é um nn.Sequential com BasicStage, etc.
    # O classificador é base_model.classifier
    
    # Descongelar os últimos 'unfreeze_layers' módulos de 'features'
    if unfreeze_layers > 0:
        for i in range(1, unfreeze_layers + 1):
            if i <= len(base_model.features):
                for param in base_model.features[-i].parameters():
                    param.requires_grad = True

    # Sempre garantir que o classificador é descongelado
    if hasattr(base_model, 'classifier'):
        for param in base_model.classifier.parameters():
            param.requires_grad = True

    # Substituir o head de classificação
    in_features = base_model.classifier[-1].in_features
    base_model.classifier[-1] = nn.Sequential(
        nn.Linear(in_features, hidden_units),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_units, hidden_units // 2),
        nn.ReLU(),
        nn.Dropout(dropout*0.8),
        nn.Linear(hidden_units // 2, num_classes)
    ).to(device)

    return base_model.to(device)
