from src.config.data.config import ConfigurationManager
from src.data.data_handler import DataLoader, DatasetExporter
from src.utils.image_utils import ImageProcessor
import logging
from sklearn.model_selection import train_test_split
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class DatasetBuilder:
    """
    Classe principal que coordena o processo de construção do dataset.
    """
    def __init__(self, version_config):
        self.config_manager = ConfigurationManager(version_config)
        self.data_loader = DataLoader(self.config_manager)
        self.image_processor = ImageProcessor(self.config_manager)
        self.dataset_exporter = DatasetExporter(self.config_manager)
        
    def build(self):
        # Carrega o dataset (agora pode vir com splits: {'train_val': ..., 'test': ...} ou {'all': ...})
        dataset_splits = self.data_loader.load_dataset()
        
        # Lê a configuração de split do YAML
        split_config = self.config_manager.get_split_config()
        
        if not split_config and ('train_val' in dataset_splits or 'test' in dataset_splits):
            logging.info("Configuração de split não encontrada. Mantendo os splits originais.")
            
            all_processed_data = defaultdict(list)
            
            for split_name, dataset in dataset_splits.items():
                if not dataset:
                    continue
                
                logging.info(f"\nProcessando split: {split_name}")
                processed_split = self.image_processor.process_dataset(dataset)
                
                # Salva o split processado
                self.dataset_exporter.save_dataset(processed_split, split_name)
                
                # Acumula para o metadado global
                for class_name, items in processed_split.items():
                    all_processed_data[class_name].extend(items)
            
            # Salva metadados globais
            self.dataset_exporter.save_metadata(all_processed_data)
            
        else:
            # Comportamento antigo ou forçado por configuração: junta tudo e faz novo split
            logging.info("Realizando novo split de dados (estratificado).")
            
            # Achata todos os splits em um único dataset
            full_dataset = defaultdict(list)
            for split_dataset in dataset_splits.values():
                for class_name, items in split_dataset.items():
                    full_dataset[class_name].extend(items)
            
            # Processa o dataset completo
            processed_dataset = self.image_processor.process_dataset(full_dataset)
            
            # Prepara para o split
            test_size = split_config.get('test_size', 0.2)
            random_state = split_config.get('random_state', 42)
            
            all_items = []
            for class_name, images in processed_dataset.items():
                for img, filename in images:
                    all_items.append({'image': img, 'label': class_name, 'filename': filename})
            
            if not all_items:
                raise ValueError("Nenhuma imagem encontrada para processar.")

            labels = [item['label'] for item in all_items]
            
            # Divide os dados de forma estratificada
            train_val_items, test_items = train_test_split(
                all_items,
                test_size=test_size,
                random_state=random_state,
                stratify=labels
            )

            # Reconstrói o formato de dicionário para cada conjunto
            train_val_dataset = defaultdict(list)
            for item in train_val_items:
                train_val_dataset[item['label']].append((item['image'], item['filename']))
                
            test_dataset = defaultdict(list)
            for item in test_items:
                test_dataset[item['label']].append((item['image'], item['filename']))

            # Salva os dois conjuntos de dados
            self.dataset_exporter.save_dataset(train_val_dataset, "train_val")
            self.dataset_exporter.save_dataset(test_dataset, "test")
            
            # Salva o metadado com a distribuição completa
            self.dataset_exporter.save_metadata(processed_dataset)
