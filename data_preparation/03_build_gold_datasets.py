import sys
import os
import logging

# Adiciona a raiz do projeto ao sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.dataset_builder import DatasetBuilder

VERSIONS = [
    # 'v1',
    # 'v2',
    # 'v3',
    # 'v4',
    # 'v5',
    # 'v6',
    # 'v7',
    # 'v8',
    # 'v9', 
    # 'v10',
    'v11',
    'v12'
]

if __name__ == "__main__":
    for version in VERSIONS:
        builder = DatasetBuilder(f"src/config/data/versions/{version}.yaml")
        builder.build()
        logging.info(f"Versão {version} concluída!\n")
