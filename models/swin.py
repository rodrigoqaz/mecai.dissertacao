import torch.nn as nn
from torchvision import models

def initialize_model(device, num_classes: int, input_channels: int = 3, **params) -> nn.Module:
    """
    Inicializa um modelo Swin Transformer pré-treinado, adaptando para fine-tuning.
    """
    architecture = params.get('architecture', 'swin_t')
    pretrained = params.get('pretrained', True)
    unfreeze_layers = params.get('unfreeze_layers', 2)
    hidden_units = params.get('hidden_units', 512)
    dropout = params.get('dropout', 0.3)
    weights = params.get('weights', 'IMAGENET1K_V1')

    # Carregar modelo base
    base_model = getattr(models, architecture)(weights=weights if pretrained else None)

    # Ajustar a primeira camada convolucional (patch embedding) se input_channels for diferente de 3
    if input_channels != 3:
        original_patch_embedding = base_model.features[0][0] # Swin Transformer's patch embedding
        new_patch_embedding = nn.Conv2d(
            input_channels,
            original_patch_embedding.out_channels,
            kernel_size=original_patch_embedding.kernel_size,
            stride=original_patch_embedding.stride,
            padding=original_patch_embedding.padding,
            bias=(original_patch_embedding.bias is not None)
        )
        base_model.features[0][0] = new_patch_embedding.to(device)

    # Congelar todas as camadas inicialmente
    for param in base_model.parameters():
        param.requires_grad = False

    # Descongelar as últimas camadas do encoder
    # Swin Transformer tem uma estrutura hierárquica (features)
    # Precisamos descongelar blocos ou estágios completos
    
    # Exemplo: descongelar os últimos 'unfreeze_layers' blocos de um estágio final
    # A estrutura do Swin pode variar, este é um exemplo comum para swin_t
    if unfreeze_layers > 0:
        # Descongelar as camadas de Normalização antes do head
        for param in base_model.norm.parameters():
            param.requires_grad = True

        # Descongelar os últimos blocos do último estágio
        # Acessando o último estágio de camadas (ex: Stage 3 para swin_t)
        # base_model.features é um nn.Sequential com PatchEmbed, BasicLayer * 3, e o norm final
        # Os BasicLayer são os estágios principais
        
        # Swin_T tem 4 BasicLayer, cada um com seus próprios blocos
        # Vamos descongelar os últimos blocos dos últimos N estágios, ou os últimos N blocos globais
        # Para simplificar, descongelar o último BasicLayer completo
        if len(base_model.features) > 1: # Pelo menos um BasicLayer existe
            last_basic_layer = base_model.features[-2] # O penúltimo é o último BasicLayer antes do norm final
            for param in last_basic_layer.parameters():
                param.requires_grad = True

            # Se quisermos mais granularidade, podemos iterar dentro dos blocos do BasicLayer
            # for block in last_basic_layer.blocks[-unfreeze_layers:]:
            #     for param in block.parameters():
            #         param.requires_grad = True

    # Sempre garantir que o head de classificação é descongelado
    if hasattr(base_model, 'head'):
        for param in base_model.head.parameters():
            param.requires_grad = True

    # Substituir o head de classificação
    in_features = base_model.head.in_features
    base_model.head = nn.Sequential(
        nn.Linear(in_features, hidden_units),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_units, hidden_units // 2),
        nn.ReLU(),
        nn.Dropout(dropout*0.8),
        nn.Linear(hidden_units // 2, num_classes)
    ).to(device)

    return base_model.to(device)
