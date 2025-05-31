import cv2
import numpy as np
import json

def recortar_imagem(imagem_path):
    imagem = cv2.imread(imagem_path)
    imagem_original = imagem.copy()

    x_start, y_start, x_end, y_end = -1, -1, -1, -1
    cropping = False

    def mouse_crop(event, x, y, flags, param):
        nonlocal x_start, y_start, x_end, y_end, cropping, imagem

        if event == cv2.EVENT_LBUTTONDOWN:
            x_start, y_start, x_end, y_end = x, y, x, y
            cropping = True

        elif event == cv2.EVENT_MOUSEMOVE:
            if cropping:
                x_end, y_end = x, y

        elif event == cv2.EVENT_LBUTTONUP:
            x_end, y_end = x, y
            cropping = False
            
            cv2.rectangle(imagem, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
            cv2.imshow("Imagem", imagem)

    cv2.namedWindow("Imagem")
    cv2.setMouseCallback("Imagem", mouse_crop)

    while True:
        if not cropping:
            cv2.imshow("Imagem", imagem)
        elif cropping:
            imagem_temp = imagem.copy()
            cv2.rectangle(imagem_temp, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
            cv2.imshow("Imagem", imagem_temp)

        key = cv2.waitKey(1) & 0xFF
        
        # Pressione 'r' para resetar a seleção
        if key == ord('r'):
            imagem = imagem_original.copy()
            x_start, y_start, x_end, y_end = -1, -1, -1, -1
        
        # Pressione 'c' para confirmar o recorte
        elif key == ord('c'):
            if x_start != -1 and y_start != -1 and x_end != -1 and y_end != -1:
                break
        
        # Pressione 'q' para sair sem recortar
        elif key == ord('q'):
            return None

    # Garantir que x_start < x_end e y_start < y_end
    if x_start > x_end:
        x_start, x_end = x_end, x_start
    if y_start > y_end:
        y_start, y_end = y_end, y_start

    # Recortar a imagem
    imagem_recortada = imagem_original[y_start:y_end, x_start:x_end]
    list(imagem_recortada.shape)

    cv2.destroyAllWindows()
    return imagem_recortada, {'pt1': [x_start, y_start], 'pt2': [x_end, y_end], 'shape_aoi': list(imagem_recortada.shape)}


codigo = '00078986195518316924'
imagem_path = f"data/bronze/amostras/AB_{codigo}.png"
imagem_recortada, atributos = recortar_imagem(imagem_path)

if imagem_recortada is not None:
    cv2.imshow("Imagem Recortada", imagem_recortada)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite(f"data/silver/amostras_recortadas/AB_{codigo}.png", imagem_recortada)

    with open('data/silver/amostras_recortadas.json', 'r') as f:
        amostras_recortadas = json.load(f)

    for item in amostras_recortadas:
        if item['codigo'] == codigo:
            print(f"Atributos do fardinho {codigo} alterados de:\n")
            print(item['atributos'])
            item['atributos'] = atributos
            print("Para:\n")
            print(atributos)

    with open('data/silver/amostras_recortadas.json', 'w') as f:
        json.dump(amostras_recortadas, f, ensure_ascii=False, indent=4)
        
else:
    print("Recorte cancelado.")
