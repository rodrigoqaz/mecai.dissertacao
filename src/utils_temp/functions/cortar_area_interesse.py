import os
import numpy as np
import cv2
import largestinteriorrectangle as lir
from rembg import remove, new_session


def cortar_area_interesse(caminho_origem: str, caminho_destino: str) -> dict:
    """
    Corta a área de interesse (AOI) da imagem original, selecionando apenas áreas com algodão.

    Args:
        caminho_origem (str): Path do arquivo da imagem original.
        caminho_destino (str): Patha para a pasta que vai salvar.

    Returns: 
        Salva o arquivo na pasta de destino e retorna um dict com os atributos codigo, caminho_aoi, pt1, pt2, shape_aoi.
    """
    codigo = os.path.basename(caminho_origem)[:-4]
    img = cv2.imread(caminho_origem)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Remove o BG
    model_name = 'birefnet-dis'
    session = new_session(model_name)
    img_bg_removed = remove(img, session=session)
    
    # Criar as máscaras

    # Máscara com base no canal H do HSV:
    img_hsv = cv2.cvtColor(img_bg_removed, cv2.COLOR_BGR2HSV)
    h_channel, _, _ = cv2.split(img_hsv) # Seleciona apenas o canal Hue do HSV

    media = np.mean(h_channel) # Valor médio do Hue
    _, h_mask = cv2.threshold(h_channel,media,255,cv2.THRESH_BINARY) # Máscara Binária
    

    # Filtro de amarelo
    tolerancia = 0
    lower_yellow = np.array([120, 100, 55]) - tolerancia
    upper_yellow = np.array([150, 150, 110]) + tolerancia
    y_mask = cv2.inRange(img, lower_yellow, upper_yellow)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # Retangular 5x5
    y_mask_closed = cv2.morphologyEx(y_mask, cv2.MORPH_CLOSE, kernel, iterations=5) # Faz um fechamento em 5 iteração com a máscara de amarelo

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(y_mask_closed, connectivity=8) # Pega os componentes conectados
    filtered_mask = np.zeros_like(y_mask_closed, dtype=np.uint8)
    
    min_size = 500 # filtra somente os componentes de tamanho > 500
    for i in range(1, num_labels): 
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_size:
            filtered_mask[labels == i] = 255

    y_mask_final = cv2.dilate(filtered_mask, kernel*10, iterations=10) # Faz 10 iterações de dilação

    # Subtrair as máscaras
    subtract_mask = cv2.subtract(h_mask, y_mask_final)

    # Identificar os componentes conectados
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(subtract_mask, connectivity=8)

    # Remove o ruído
    removed_noise = np.zeros_like(subtract_mask, dtype=np.uint8)
    min_size = 500
    for i in range(1, num_labels): 
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_size:
            removed_noise[labels == i] = 255

    # Erosao
    eroded = cv2.erode(removed_noise, kernel, iterations=10)

    # Dilação
    dilated = cv2.dilate(eroded, kernel, iterations=15)

    final_mask = dilated.copy()
    cv2.rectangle(final_mask, (0, 0), (final_mask.shape[1] - 1, final_mask.shape[0] - 1), 0, thickness=30)

    # Criar o retangulo
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    contour = np.array([largest_contour[:, 0, :]])
    inner_bb = lir.lir(contour)

    desloc = 10 # DEslocar 30 pixels para dentro do retangulo
    pt1 = tuple([x+desloc for x in lir.pt1(inner_bb)])
    pt2 = tuple([x-desloc for x in lir.pt2(inner_bb)])
    
    # Cortar
    img_crop = img[pt1[1]:pt2[1], pt1[0]:pt2[0]]
    img_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)

    # Salvar
    cv2.imwrite(f"{caminho_destino}/{codigo}.png", img_crop)

    return {
        'codigo': codigo[3:],
        'atributos': {
            'caminho_aoi': f"{caminho_destino}/{codigo}.png", 
            'pt1': pt1, 
            'pt2': pt2, 
            'shape_aoi': img_crop.shape
        }
    }
