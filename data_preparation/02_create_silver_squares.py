import os
import json
import cv2
import math
import logging
import sys
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # --- Definição de Caminhos e Parâmetros ---
    SILVER_JSON_FILE = 'data/silver/amostras_recortadas.json'
    PARQUET_FILE = 'data/bronze/dados_fardinhos.parquet'
    OUTPUT_DIR = 'data/silver/amostras_recortadas_quadrado'
    TAMANHO_QUADRADO = 256

    # --- Verificação de Pré-requisitos ---
    logging.info(f"Verificando a existência do arquivo JSON '{SILVER_JSON_FILE}'...")
    if not os.path.exists(SILVER_JSON_FILE):
        logging.error(f"ERRO: Arquivo JSON não encontrado em '{SILVER_JSON_FILE}'")
        logging.error("Execute o script '01_create_silver_aoi_v2.py' para gerar as amostras recortadas primeiro.")
        sys.exit(1)
    logging.info("Arquivo JSON encontrado.")

    logging.info(f"Verificando a existência do arquivo Parquet '{PARQUET_FILE}'...")
    if not os.path.exists(PARQUET_FILE):
        logging.error(f"ERRO: Arquivo Parquet não encontrado em '{PARQUET_FILE}'")
        sys.exit(1)
    logging.info("Arquivo Parquet encontrado.")

    # --- Garantir a Existência do Diretório de Saída ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logging.info(f"Diretório de saída '{OUTPUT_DIR}' garantido.")

    # --- Carregar Dados ---
    with open(SILVER_JSON_FILE, 'r', encoding='utf-8') as f:
        amostras_recortadas = json.load(f)
    
    logging.info("Carregando dados de classe do arquivo Parquet...")
    df_classes = pd.read_parquet(PARQUET_FILE)
    # Assumindo que a coluna de código no parquet é 'codigosai' e a de classe é 'classificacao'
    # E que o 'codigo' no JSON corresponde ao 'codigosai'
    df_classes.set_index('codigosai', inplace=True)
    classes_map = df_classes['classificacao'].to_dict()
    logging.info("Dados de classe carregados.")


    # --- 1. Atualizar metadados no JSON ---
    logging.info("Atualizando metadados no JSON (quadrados e classes)...")
    for amostra in amostras_recortadas:
        try:
            codigo = amostra.get('codigo')
            if not codigo:
                logging.warning("Amostra sem 'codigo' encontrada. Pulando.")
                continue

            # Adicionar metadados dos quadrados se não existirem
            if 'quadrados' not in amostra.get('atributos', {}):
                altura = amostra['atributos']['shape_aoi'][0]
                largura = amostra['atributos']['shape_aoi'][1]

                qtd_altura = math.ceil(altura / TAMANHO_QUADRADO)
                qtd_largura = math.ceil(largura / TAMANHO_QUADRADO)

                quadrados = []
                for y_idx in range(qtd_altura):
                    for x_idx in range(qtd_largura):
                        x_ini = x_idx * TAMANHO_QUADRADO
                        y_ini = y_idx * TAMANHO_QUADRADO
                        if (x_ini + TAMANHO_QUADRADO) > largura:
                            x_ini = largura - TAMANHO_QUADRADO
                        if (y_ini + TAMANHO_QUADRADO) > altura:
                            y_ini = altura - TAMANHO_QUADRADO
                        
                        x_end = x_ini + TAMANHO_QUADRADO
                        y_end = y_ini + TAMANHO_QUADRADO
                        quadrados.append([[x_ini, y_ini], [x_end, y_end]])
                
                amostra['atributos']['quadrados'] = {
                    'qtd': len(quadrados),
                    'pontos': quadrados
                }
            
            # Adicionar metadados da classe se não existirem
            if 'classe' not in amostra:
                if codigo in classes_map:
                    amostra['classe'] = classes_map[codigo]
                else:
                    logging.warning(f"Classe para o código {codigo} não encontrada no arquivo Parquet. Atributo 'classe' não será adicionado.")

        except KeyError as e:
            logging.error(f"Atributo ausente {e} para a amostra {amostra.get('codigo', 'N/A')}. Pulando.")
            continue

    # Salvar arquivo JSON atualizado
    with open(SILVER_JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump(amostras_recortadas, f, ensure_ascii=False, indent=4)
    logging.info("Atualização dos metadados no JSON concluída.")

    # --- 2. Criar recortes quadrados das imagens ---
    total_amostras = len(amostras_recortadas)
    logging.info(f"Iniciando processamento de {total_amostras} amostras para criar os recortes quadrados.")

    for i, amostra in enumerate(amostras_recortadas, 1):
        codigo = amostra.get('codigo')
        if not codigo:
            continue # Já foi avisado anteriormente
            
        logging.info(f"Processando amostra {codigo} ({i}/{total_amostras}) - {i/total_amostras*100:.2f}% concluído")

        try:
            quadrados_pontos = amostra['atributos']['quadrados']['pontos']
            
            for aoi_info in amostra['amostras_aoi']:
                prefixo = list(aoi_info.keys())[0]
                caminho_aoi = aoi_info[prefixo]

                if not os.path.exists(caminho_aoi):
                    logging.warning(f"Imagem AOI não encontrada em '{caminho_aoi}' para a amostra {codigo}. Pulando este prefixo.")
                    continue

                img_aoi = cv2.imread(caminho_aoi)
                if img_aoi is None:
                    logging.warning(f"Não foi possível ler a imagem em '{caminho_aoi}' para a amostra {codigo}. Pulando.")
                    continue

                for j, pontos in enumerate(quadrados_pontos, 1):
                    pt1 = pontos[0]
                    pt2 = pontos[1]
                    
                    img_quadrado = img_aoi[pt1[1]:pt2[1], pt1[0]:pt2[0]]
                    
                    output_filename = f"{prefixo.upper()}_{codigo}_{j}.png"
                    output_path = os.path.join(OUTPUT_DIR, output_filename)
                    cv2.imwrite(output_path, img_quadrado)

        except KeyError as e:
            logging.error(f"Atributo ausente {e} durante o recorte da imagem para a amostra {codigo}. Pulando.")
        except Exception as e:
            logging.error(f"Ocorreu um erro inesperado ao processar a amostra {codigo}: {e}", exc_info=True)

    logging.info("Processamento de recortes quadrados concluído.")

if __name__ == '__main__':
    main()
