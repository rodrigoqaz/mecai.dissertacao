from pathlib import Path
import yaml

class ConfigurationManager:
    """
    Gerencia as configurações do processamento do dataset.
    Responsável por carregar arquivos de configuração YAML e 
    fornecer acesso estruturado às configurações.
    """
    def __init__(self, version_config_path):
        self.config = self._load_config(version_config_path)
        self.paths = self._load_paths()
        
    def _load_config(self, config_path):
        with open(config_path) as f:
            return yaml.safe_load(f)
            
    def _load_paths(self):
        with open('src/config/data/paths.yaml') as f:
            return yaml.safe_load(f)
            
    def get_input_path(self):
        if self.config.get('type') == 'initial':
            return Path(self.config['raw_source'])
        else:
            return Path(self.paths['gold']) / self.config['input_version']
            
    def get_output_path(self):
        return Path(self.paths['gold']) / self.config['version_name']
        
    def get_json_mapping_path(self):
        if self.config.get('type') == 'initial':
            return Path(self.config['json_mapping'])
        return None
        
    def is_initial_dataset(self):
        return self.config.get('type') == 'initial'
        
    def get_processing_steps(self):
        return self.config.get('processing_steps', [])
        
    def get_version_name(self):
        return self.config['version_name']
        
    def get_light_source(self):
        return self.config.get('light_source')
        
    def get_input_version(self):
        return self.config.get('input_version')
