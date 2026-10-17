from src.config.data.config import ConfigurationManager
from src.data.data_handler import DataLoader, DatasetExporter
from src.utils.image_utils import ImageProcessor
from src.utils.bale_id import extract_bale_id
import logging
from sklearn.model_selection import train_test_split, StratifiedGroupKFold, GroupShuffleSplit
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
            logging.info("Realizando novo split de dados.")

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
            strategy = split_config.get('strategy', 'random')

            all_items = []
            for class_name, images in processed_dataset.items():
                for img, filename in images:
                    all_items.append({'image': img, 'label': class_name, 'filename': filename})

            if not all_items:
                raise ValueError("Nenhuma imagem encontrada para processar.")

            labels = [item['label'] for item in all_items]

            if strategy == 'group_by_bale':
                # Split agrupado por fardo físico: elimina o vazamento de dados a
                # nível de patch (ver correcoes_banca/p3.1_resplit_por_fardo.md).
                bale_ids = [extract_bale_id(item['filename']) for item in all_items]

                train_idx, test_idx, split_method = self._split_group_by_bale(
                    labels, bale_ids, test_size, random_state
                )
                train_val_items = [all_items[i] for i in train_idx]
                test_items = [all_items[i] for i in test_idx]

                split_info = {
                    'strategy': split_method,
                    'seed': random_state,
                    'test_size': test_size,
                    'n_bales_train': len({bale_ids[i] for i in train_idx}),
                    'n_bales_test': len({bale_ids[i] for i in test_idx}),
                }
            else:
                # Split legado a nível de patch (mantido para versões não cobertas
                # pelo p3.1). Produz vazamento de dados: patches do mesmo fardo
                # podem cair em treino e teste simultaneamente.
                logging.warning(
                    "Estratégia de split 'random' (nível de patch, legado): "
                    "patches do mesmo fardo podem vazar entre treino e teste."
                )
                train_val_items, test_items = train_test_split(
                    all_items,
                    test_size=test_size,
                    random_state=random_state,
                    stratify=labels
                )
                split_info = {
                    'strategy': 'random (patch-level, legado)',
                    'seed': random_state,
                    'test_size': test_size,
                }

            # Reconstrói o formato de dicionário para cada conjunto
            train_val_dataset = defaultdict(list)
            for item in train_val_items:
                train_val_dataset[item['label']].append((item['image'], item['filename']))

            test_dataset = defaultdict(list)
            for item in test_items:
                test_dataset[item['label']].append((item['image'], item['filename']))

            split_info['class_distribution_train_val'] = {
                class_name: len(images) for class_name, images in train_val_dataset.items()
            }
            split_info['class_distribution_test'] = {
                class_name: len(images) for class_name, images in test_dataset.items()
            }

            # Salva os dois conjuntos de dados
            self.dataset_exporter.save_dataset(train_val_dataset, "train_val")
            self.dataset_exporter.save_dataset(test_dataset, "test")

            # Salva o metadado com a distribuição completa e a estratégia de split usada
            self.dataset_exporter.save_metadata(processed_dataset, split_info=split_info)

    @staticmethod
    def _split_group_by_bale(labels, groups, test_size, random_state):
        """
        Particiona treino/teste agrupando por fardo físico (groups), para que
        nenhum patch de um fardo do teste apareça no treino.

        Tenta primeiro StratifiedGroupKFold (preserva a proporção de classe e
        isola os fardos). Se alguma classe tiver menos fardos distintos do que o
        número de dobras necessário para atingir `test_size` (comum em classes
        minoritárias), cai para GroupShuffleSplit — ainda isola os fardos, mas
        sem garantir a proporção de classe — e registra o desvio.
        """
        dummy_X = [0] * len(labels)
        n_splits = max(2, round(1 / test_size))

        try:
            splitter = StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=random_state
            )
            train_idx, test_idx = next(splitter.split(dummy_X, labels, groups))
            logging.info(
                f"Split agrupado por fardo via StratifiedGroupKFold "
                f"(n_splits={n_splits}, 1ª dobra usada como teste)."
            )
            return train_idx, test_idx, 'group_by_bale (StratifiedGroupKFold)'
        except ValueError as e:
            logging.warning(
                f"StratifiedGroupKFold falhou ({e}). Caindo para GroupShuffleSplit "
                "sem estratificação de classe (a proporção por classe pode desviar "
                "do test_size configurado para classes minoritárias)."
            )
            splitter = GroupShuffleSplit(
                n_splits=1, test_size=test_size, random_state=random_state
            )
            train_idx, test_idx = next(splitter.split(dummy_X, labels, groups))
            return train_idx, test_idx, 'group_by_bale (GroupShuffleSplit, fallback)'
