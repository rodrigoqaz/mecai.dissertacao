from src.config.data.config import ConfigurationManager
from src.data.data_handler import DataLoader, DatasetExporter
from src.utils.image_utils import ImageProcessor
import logging

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
        dataset = self.data_loader.load_dataset()
        dataset = self.image_processor.process_dataset(dataset)
        self.dataset_exporter.save_dataset(dataset)
        self.dataset_exporter.save_metadata(dataset)
