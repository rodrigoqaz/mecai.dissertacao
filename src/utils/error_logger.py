from datetime import datetime
import json

class ErrorLogger:
    """
    Classe utilitária para registro de erros de processamento.
    """
    @staticmethod
    def log_error(output_dir, img_path, error_msg):
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'file': str(img_path),
            'error': error_msg
        }
        
        output_dir.mkdir(parents=True, exist_ok=True)
        error_log = output_dir / 'processing_errors.json'
        
        if error_log.exists():
            with open(error_log, 'r') as f:
                existing_errors = json.load(f)
        else:
            existing_errors = []
            
        existing_errors.append(error_entry)
        
        with open(error_log, 'w') as f:
            json.dump(existing_errors, f, indent=2)
