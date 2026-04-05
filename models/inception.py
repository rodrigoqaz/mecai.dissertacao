import torch.nn as nn
from torchvision import models

def initialize_model(device, num_classes: int, input_channels: int = 3, **params) -> nn.Module:
    architecture = params.get('architecture', 'inception_v3')
    pretrained = params.get('pretrained', True)
    unfreeze_layers = params.get('unfreeze_layers', 2)
    hidden_units = params.get('hidden_units', 1024)
    dropout = params.get('dropout', 0.7)
    weights = params.get('weights', 'IMAGENET1K_V1')

    # 1. Carrega com aux_logits=True para evitar o ValueError
    base_model = getattr(models, architecture)(
        weights=weights if pretrained else None,
        aux_logits=True,
        transform_input=False # IMPORTANTE
    )

    # 2. Ajusta entrada para 15 canais
    if input_channels != 3:
        old_layer = base_model.Conv2d_1a_3x3.conv
        
        # Criamos a camada
        new_conv = nn.Conv2d(
            input_channels,
            old_layer.out_channels,
            kernel_size=old_layer.kernel_size,
            stride=old_layer.stride,
            padding=old_layer.padding,
            bias=(old_layer.bias is not None)
        )
        
        # IMPORTANTE: Enviamos a camada específica para o device (MPS/CUDA)
        base_model.Conv2d_1a_3x3.conv = new_conv.to(device)

    # 3. IMPORTANTE: Desativa aux_logits para o forward retornar apenas 1 valor
    base_model.aux_logits = False
    # Opcional: deletar para economizar GPU
    if hasattr(base_model, 'AuxLogits'):
        base_model.AuxLogits = None 

    # 4. Congelar tudo
    for param in base_model.parameters():
        param.requires_grad = False

    # 5. Substituir o Head (fc)
    in_features = base_model.fc.in_features
    base_model.fc = nn.Sequential(
        nn.Linear(in_features, hidden_units),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_units, num_classes)
    ).to(device)

    # 6. Descongelar camadas (opcionalmente mais preciso)
    # Se quiser descongelar os últimos blocos, unfreeze_layers deve ser pequeno (ex: 1 ou 2)
    children = list(base_model.children())
    for layer in children[-unfreeze_layers:]:
        for param in layer.parameters():
            param.requires_grad = True

    return base_model.to(device)