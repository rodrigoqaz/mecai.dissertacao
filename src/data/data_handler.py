import logging
import json
import cv2
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
from src.utils.error_logger import ErrorLogger
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

class DataLoader:
    """
    Responsável por carregar imagens e dados associados.
    """
    def __init__(self, config_manager):
        self.config_manager = config_manager
        
    def load_dataset(self) -> Dict[str, List[Tuple[np.ndarray, str]]]:
        if self.config_manager.is_initial_dataset():
            logging.info(f"\nCarregando dataset inicial {self.config_manager.get_version_name()}")
            return self._load_raw_from_json()
        else:
            logging.info(f"\nCarregando dataset {self.config_manager.get_version_name()} a partir de {self.config_manager.get_input_version()}")
            return self._load_processed_images()
    
    def _load_raw_from_json(self):
        dataset = defaultdict(list)
        json_path = self.config_manager.get_json_mapping_path()
        input_dir = self.config_manager.get_input_path()
        light_source = self.config_manager.get_light_source()
        
        with open(json_path, 'r') as f:
            class_mapping = json.load(f)
        
        code_to_class = {item['codigo']: item['classe'] for item in class_mapping}
        
        for img_path in input_dir.glob(f"{light_source}_*.png"):
            try:
                parts = img_path.stem.split('_')
                if len(parts) < 2:
                    continue
                    
                codigo = parts[1]
                classe = code_to_class.get(codigo)
                if not classe:
                    continue
                    
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                    
                dataset[classe].append((img, img_path.name))
            except Exception as e:
                ErrorLogger.log_error(self.config_manager.get_output_path(), img_path, f"Erro ao carregar imagem: {e}")
                
        total_classes = len(dataset)
        total_images = sum(len(imgs) for imgs in dataset.values())
        logging.info(f"Total de classes carregadas: {total_classes}")
        logging.info(f"Total de imagens carregadas: {total_images}")
        
        return dataset
        
    def _load_processed_images(self) -> Dict[str, Dict[str, List[Tuple[np.ndarray, str]]]]:
        """
        Carrega imagens processadas organizadas por split (ex: train_val, test).
        Retorna: { 'split_name': { 'class_name': [(img, filename), ...] } }
        """
        dataset_splits = defaultdict(lambda: defaultdict(list))
        input_dir = self.config_manager.get_input_path()

        try:
            if not input_dir.exists():
                ErrorLogger.log_error(
                    self.config_manager.get_output_path(),
                    input_dir,
                    f"Diretório não encontrado: {input_dir}"
                )
                logging.error(f"Diretório de entrada não encontrado: {input_dir}")
                return dataset_splits
            
            # Busca todos os arquivos .npy recursivamente
            npy_files = list(input_dir.rglob('*.npy'))
            
            if not npy_files:
                logging.warning(f"Nenhum arquivo .npy encontrado em {input_dir}")
                return dataset_splits

            for img_path in npy_files:
                try:
                    # Estrutura esperada: input_dir / split / class / file.npy
                    # Ou: input_dir / class / file.npy
                    class_name = img_path.parent.name
                    parent_of_class = img_path.parent.parent
                    
                    split_name = 'all'
                    if parent_of_class.name in ['train_val', 'test']:
                        split_name = parent_of_class.name
                    elif class_name in ['train_val', 'test']:
                        # Caso o arquivo esteja direto no split (improvável mas possível)
                        continue
                        
                    img = np.load(str(img_path))
                    dataset_splits[split_name][class_name].append((img, img_path.name))
                except Exception as e:
                    ErrorLogger.log_error(
                        self.config_manager.get_output_path(),
                        img_path,
                        f"Erro ao carregar arquivo NPY: {e}"
                    )
        except Exception as e:
            ErrorLogger.log_error(
                self.config_manager.get_output_path(),
                "load_processed_images",
                f"Erro ao processar diretório de entrada: {e}"
            )
        
        for split, classes in dataset_splits.items():
            total_imgs = sum(len(imgs) for imgs in classes.values())
            logging.info(f"Split '{split}': {total_imgs} imagens em {len(classes)} classes.")
                    
        return dataset_splits

class DatasetExporter:
    """
    Responsável pela exportação e salvamento do dataset processado.
    """
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.output_dir = config_manager.get_output_path()
        self._create_dirs()
        
    def _create_dirs(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def save_dataset(self, dataset: Dict, split_name: str):
        """Salva um conjunto de dados em um subdiretório específico."""
        split_dir = self.output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for class_name, images in dataset.items():
                class_dir = split_dir / class_name
                class_dir.mkdir(parents=True, exist_ok=True)
                
                for img, filename in images:
                    npy_filename = Path(filename).stem + ".npy"
                    out_path = class_dir / npy_filename
                    futures.append(
                        executor.submit(
                            np.save,
                            str(out_path),
                            img,
                            allow_pickle=False
                        )
                    )
            
            desc = f"Salvando imagens em '{split_name}'"
            for _ in tqdm(executor.map(lambda f: f.result(), futures), total=len(futures), desc=desc):
                pass
                
    def save_metadata(self, dataset=None):
        class_dist = None
        if dataset:
            class_dist = {class_name: len(images) for class_name, images in dataset.items()}
        
        metadata = {
            'version': self.config_manager.get_version_name(),
            'creation_date': datetime.now().isoformat(),
            'processing_steps': self.config_manager.get_processing_steps()
        }
        
        if self.config_manager.is_initial_dataset():
            metadata.update({
                'light_source': self.config_manager.get_light_source(),
                'source': 'raw',
                'json_mapping': str(self.config_manager.get_json_mapping_path()),
                'classes': self._get_class_distribution() if not class_dist else class_dist
            })
        else:
            metadata.update({
                'source': self.config_manager.get_input_version(),
                'class_distribution': class_dist
            })
            
        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
            
    def _get_class_distribution(self):
        return {
            folder.name: len(list(folder.glob('*.png')))
            for folder in self.output_dir.iterdir()
            if folder.is_dir()
        }
