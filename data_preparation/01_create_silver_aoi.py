from lib.cortar_aoi_bg_removido import cortar_aoi_bg_removido
import os
import sys
import json
import logging
import cv2
from numpyencoder import NumpyEncoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # --- Definição de Caminhos e Parâmetros ---
    BRONZE_DATA_DIR = 'data/bronze/amostras/'
    SILVER_CROPS_DIR = 'data/silver/amostras_recortadas'
    SILVER_JSON_FILE = 'data/silver/amostras_recortadas.json'
    PREFIXES = ["AB", "AM", "BR"]
    MAIN_PREFIX = "AB"

    # --- Verificação de Pré-requisitos (Camada Bronze) ---
    logging.info("Verificando a existência da camada 'bronze'...")
    if not os.path.isdir(BRONZE_DATA_DIR):
        logging.error(f"ERRO: Diretório da camada 'bronze' não encontrado em '{BRONZE_DATA_DIR}'")
        logging.error("Execute o script '00_download_bronze.py' para baixar os dados antes de continuar.")
        sys.exit(1)
    logging.info("Camada 'bronze' encontrada.")

    # --- Garantir a Existência dos Diretórios de Saída (Camada Silver) ---
    os.makedirs(SILVER_CROPS_DIR, exist_ok=True)
    logging.info(f"Diretório de saída '{SILVER_CROPS_DIR}' garantido.")

    # --- Carregamento de Amostras e Progresso ---
    logging.info("Selecionando amostras da camada bronze com base no PREFIXO_PRINCIPAL '{}'...".format(MAIN_PREFIX))
    codigos = []
    for file in os.listdir(BRONZE_DATA_DIR):
        if file.startswith(f'{MAIN_PREFIX}_') and file.endswith('.png'):
            codigos.append(file[len(MAIN_PREFIX)+1:-4])

    logging.info(f"Verificando progresso anterior em '{SILVER_JSON_FILE}'...")
    amostras_recortadas = []
    amostras_ja_processadas = set()
    try:
        with open(SILVER_JSON_FILE, 'r', encoding='utf-8') as f:
            dados_salvos = json.load(f)
            if isinstance(dados_salvos, list):
                amostras_ja_processadas = {item.get('codigo') for item in dados_salvos if isinstance(item, dict) and 'codigo' in item}
                amostras_recortadas = dados_salvos  # Continua a partir dos dados existentes
                logging.info(f"Encontrado progresso de {len(amostras_ja_processadas)} amostras.")
            else:
                logging.warning("Arquivo de progresso com formato inesperado. Começando do zero.")
    except FileNotFoundError:
        logging.info("Arquivo de progresso não encontrado. Começando do zero.")
    except json.JSONDecodeError:
        logging.warning("Arquivo de progresso está vazio ou corrompido. Começando do zero.")

    amostras_para_processar = [codigo for codigo in codigos if codigo not in amostras_ja_processadas]

    if not amostras_para_processar:
        logging.info("Todas as amostras já foram processadas. Nenhuma ação necessária.")
        return

    # --- Loop de Processamento ---
    total_a_processar = len(amostras_para_processar)
    logging.info(f"Iniciando processamento de {total_a_processar} novas amostras.")
    
    for i, amostra_codigo in enumerate(amostras_para_processar, 1):
        logging.info(f"Processando amostra {amostra_codigo} ({i}/{total_a_processar}) - {i/total_a_processar*100:.2f}% concluído")
        
        main_image_path = os.path.join(BRONZE_DATA_DIR, f'{MAIN_PREFIX}_{amostra_codigo}.png')
        
        if not os.path.exists(main_image_path):
            logging.warning(f"Imagem de referência principal não encontrada para o código {amostra_codigo} em '{main_image_path}'. Pulando.")
            continue

        try:
            # Processa a imagem principal para obter as coordenadas de recorte
            main_crop_info = cortar_aoi_bg_removido(main_image_path, SILVER_CROPS_DIR)
            
            if not main_crop_info:
                logging.error(f"Falha ao processar a imagem principal para a amostra {amostra_codigo}.")
                continue

            atributos = main_crop_info['atributos']
            pt1 = atributos['pt1']
            pt2 = atributos['pt2']
            
            amostras_aoi_list = []

            # Processa todos os prefixos usando as mesmas coordenadas
            for prefix in PREFIXES:
                prefix_lower = prefix.lower()
                bronze_path = os.path.join(BRONZE_DATA_DIR, f'{prefix}_{amostra_codigo}.png')
                silver_path = os.path.join(SILVER_CROPS_DIR, f'{prefix}_{amostra_codigo}.png')

                if prefix == MAIN_PREFIX:
                    # Este já foi criado por cortar_aoi_bg_removido
                    amostras_aoi_list.append({prefix_lower: silver_path})
                elif os.path.exists(bronze_path):
                    try:
                        img = cv2.imread(bronze_path)
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        if img is None:
                            raise IOError(f"A imagem em {bronze_path} não pôde ser lida.")
                        
                        # Recorta usando as coordenadas da imagem principal
                        img_crop = img[pt1[1]:pt2[1], pt1[0]:pt2[0]]
                        img_crop_bgr = cv2.cvtColor(img_crop, cv2.COLOR_RGB2BGR)
                        
                        cv2.imwrite(silver_path, img_crop_bgr)
                        amostras_aoi_list.append({prefix_lower: silver_path})
                    except Exception as e:
                        logging.error(f"Erro ao recortar a imagem {bronze_path} para a amostra {amostra_codigo}: {e}", exc_info=True)
                else:
                    logging.warning(f"Imagem não encontrada para o prefixo {prefix} e código {amostra_codigo}. Pulando este prefixo.")

            # Monta o objeto JSON final para a amostra
            final_sample_info = {
                'codigo': amostra_codigo,
                'atributos': atributos,
                'amostras_aoi': amostras_aoi_list
            }
            amostras_recortadas.append(final_sample_info)

        except Exception as e:
            logging.error(f"Ocorreu um erro inesperado ao processar a amostra {amostra_codigo}: {e}", exc_info=True)
        
        # Salva o progresso a cada iteração para evitar perda de trabalho
        with open(SILVER_JSON_FILE, 'w', encoding='utf-8') as f:
            json.dump(amostras_recortadas, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)

    logging.info("Processamento da camada 'silver' concluído.")

if __name__ == '__main__':
    main()
