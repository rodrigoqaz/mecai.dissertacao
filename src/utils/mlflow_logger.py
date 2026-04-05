import mlflow
import os

class MLflowLogger:
    _active_run = None
    _tracking_uri = None

    @classmethod
    def set_tracking_uri(cls, uri):
        """Define o tracking URI do MLflow"""
        # Se for um caminho relativo, converte para absoluto relativo ao projeto
        if uri.startswith("file:///mlruns"):
            # Converte para caminho relativo ao diretório do projeto
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            mlruns_path = os.path.join(project_root, "mlruns")
            uri = f"file://{mlruns_path}"
        
        cls._tracking_uri = uri
        mlflow.set_tracking_uri(uri)

    @classmethod
    def start_run(cls, experiment_name, tags=None):
        # Se não foi definido um tracking URI, usa o padrão relativo
        if cls._tracking_uri is None:
            cls.set_tracking_uri("file:///mlruns")
        
        mlflow.set_experiment(experiment_name)
        cls._active_run = mlflow.start_run(tags=tags)

    @classmethod
    def log_params(cls, params):
        mlflow.log_params(params)

    @classmethod
    def log_metrics(cls, metrics, step=None):
        mlflow.log_metrics(metrics, step=step)

    @classmethod
    def log_model(cls, model, artifact_path, input_schema):
        mlflow.pytorch.log_model(model, artifact_path, signature=input_schema)

    @classmethod
    def log_text(cls, text, filename="model_summary.txt"):
        mlflow.log_text(text=text, artifact_file=filename)

    @classmethod
    def log_artifact(cls, local_path, artifact_path=None):
        mlflow.log_artifact(local_path=local_path, artifact_path=artifact_path)

    @classmethod
    def get_active_run(cls):
        """Retorna o objeto do run ativo"""
        return cls._active_run

    @classmethod
    def set_tag(cls, key, value):
        """Adiciona uma tag ao run ativo"""
        mlflow.set_tag(key, value)

    @classmethod
    def end_run(cls):
        mlflow.end_run()
