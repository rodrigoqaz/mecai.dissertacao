from src.data.dataset_builder import DatasetBuilder
import logging

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
