import json
import cv2
import os
from functions.normalizar_imagem import normalizar_imagem


print('Abrindo o arquivo')
with open('data/silver/amostras_recortadas.json', 'r') as f:
    amostras_recortadas = json.load(f)


# Cria as features:
print('Criando as features')
path_origem = 'data/silver/amostras_recortadas_quadrado/'

path_destino = 'data/gold/dataset_224_v1/'

try: 
    os.mkdir(path_destino)
except FileExistsError:
    pass

for amostra in amostras_recortadas:
    codigo = amostra['codigo']
    classe = amostra['classe']
    qtd_images = amostra['atributos']['quadrados']['qtd']
    for i in range(1, qtd_images):
        img = cv2.imread(f"{path_origem}AB_{codigo}_{i}.png")
        img = cv2.resize(img, (224,244))
        img_norm = normalizar_imagem(img)
        try: 
            os.mkdir(f"{path_destino}/{classe}")
        except FileExistsError:
            pass
        
        cv2.imwrite(f"{path_destino}/{classe}/AB_{codigo}_{i}.png", img)
        print(f"Imagem {classe}/AB_{codigo}_{i} Salva com Sucesso!!")
