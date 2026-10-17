import torch.nn as nn
from torchvision import models

def initialize_model(device, num_classes: int, input_channels: int = 3, **params) -> nn.Module:
    """
    Inicializa um modelo Vision Transformer (ViT) pré-treinado, adaptando para fine-tuning.
    """
    architecture = params.get('architecture', 'vit_b_16')
    pretrained = params.get('pretrained', True)
    unfreeze_layers = params.get('unfreeze_layers', 2)
    hidden_units = params.get('hidden_units', 512)
    dropout = params.get('dropout', 0.3)
    weights = params.get('weights', 'IMAGENET1K_V1')

    # Carregar modelo base
    base_model = getattr(models, architecture)(weights=weights if pretrained else None)

    # Ajustar a primeira camada convolucional (patch embedding) se input_channels for diferente de 3
    if input_channels != 3:
        original_patch_embedding = base_model.conv_proj
        new_patch_embedding = nn.Conv2d(
            input_channels,
            original_patch_embedding.out_channels,
            kernel_size=original_patch_embedding.kernel_size,
            stride=original_patch_embedding.stride,
            padding=original_patch_embedding.padding,
            bias=(original_patch_embedding.bias is not None)
        )
        base_model.conv_proj = new_patch_embedding.to(device)

    # Congelar todas as camadas inicialmente
    for param in base_model.parameters():
        param.requires_grad = False

    # Descongelar as últimas camadas (parte do encoder e/ou head)
    children = list(base_model.children())
    # Note: ViT structure is slightly different, often has an `encoder` and `heads` module
    # We unfreeze parts of the encoder and then the classification head
    
    # Example for unfreezing the last 'unfreeze_layers' of the encoder
    # This might need adjustment based on the specific ViT architecture
    if hasattr(base_model, 'encoder'):
        encoder_blocks = list(base_model.encoder.layers)
        for layer in encoder_blocks[-unfreeze_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
    
    # Always ensure the classification head is unfrozen
    if hasattr(base_model, 'heads'):
        for param in base_model.heads.parameters():
            param.requires_grad = True

    # Substituir o head de classificação
    in_features = base_model.heads.head.in_features
    base_model.heads.head = nn.Sequential(
        nn.Linear(in_features, hidden_units),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_units, hidden_units // 2),
        nn.ReLU(),
        nn.Dropout(dropout*0.8),
        nn.Linear(hidden_units // 2, num_classes)
    ).to(device)

    return base_model.to(device)
