from functions.cortar_aoi_bg_removido import cortar_aoi_bg_removido
import os
import json
import logging
from numpyencoder import NumpyEncoder

logging.basicConfig(level=logging.INFO)

def main():
    
    logging.info(f"Selecionando as amostras")
    codigos = []
    for root, dirs, files in os.walk('data/bronze/amostras/'):
        for file in files:
            if file[:2] == 'AB':
                codigos.append(f'{file[3:-4]}')

    logging.info(f"Verificando se alguma amostra foi processada")
    try:
        with open('data/silver/amostras_recortadas.json') as f:
            d = json.load(f)
            amostras_processadas = [amostra.get('codigo') for amostra in d]
            amostras_para_processar = [i for i in codigos if i not in amostras_processadas]
    except Exception as e:
        logging.info(f"Nenhuma amostra processada")
        amostras_para_processar = codigos

    output_dir = 'data/silver/amostras_recortadas'
    output_file = "data/silver/amostras_recortadas.json"
    amostras_recortadas = []
    i=1
    for amostra in amostras_para_processar:
        amostra_caminho = f'data/bronze/amostras/AB_{amostra}.png'
        logging.info(f"Processando amostra {amostra} ({i}/{len(amostras_para_processar)}) - {i/len(amostras_para_processar)*100:.2f}% concluído")
        try:
            amostra_recortada = cortar_aoi_bg_removido(amostra_caminho, output_dir)
            if amostra_recortada:
                amostras_recortadas.append(amostra_recortada)
        except Exception as e:
            logging.error(f"Erro ao processar a amostra {amostra}")
        i+=1
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(amostras_recortadas, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)
    

if __name__ == '__main__':
    main()
    
