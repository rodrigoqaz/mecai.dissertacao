import numpy as np

def normalizar_imagem(image):
    """
    Normaliza uma imagem RGB para o intervalo [0, 1].
    
    Parâmetros:
    image: Array numpy de formato (altura, largura, 3)
    
    Retorna:
    imagem normalizada como um array numpy de floats no intervalo [0, 1]
    """
    # Obtêm o número de canais da imagem.
    if len(image.shape) == 2:
        NUM_CHANNELS = 1
    else:
        NUM_CHANNELS = image.shape[2]
    
    # Converte para float para evitar overflow
    image_float = image.astype(float)
    
    # Normaliza cada canal separadamente
    for channel in range(NUM_CHANNELS):
        channel_min = np.min(image_float[:,:,channel])
        channel_max = np.max(image_float[:,:,channel])
        
        # Evita divisão por zero
        if channel_min != channel_max:
            image_float[:,:,channel] = (image_float[:,:,channel] - channel_min) / (channel_max - channel_min)
        else:
            image_float[:,:,channel] = image_float[:,:,channel] - channel_min
    
    return image_float
