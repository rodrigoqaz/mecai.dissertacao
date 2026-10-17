from src.data.dataset_builder import DatasetBuilder
import logging

# Para o rebuild do p3.1 (re-split por fardo), use
# data_preparation/03_build_gold_datasets.py com VERSIONS = ['v1','v9','v10','v8','v11','v12'],
# nessa ordem (ver correcoes_banca/p3.1_resplit_por_fardo.md).
VERSIONS = [
    # 'v1',
    'v2',
    # 'v3',
    # 'v4',
    # 'v5',
    # 'v6'
]

if __name__ == "__main__":
    for version in VERSIONS:
        builder = DatasetBuilder(f"src/config/data/versions/{version}.yaml")
        builder.build()
        logging.info(f"Versão {version} concluída!\n")
