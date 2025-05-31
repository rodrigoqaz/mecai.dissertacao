import mlflow

class MLflowLogger:
    _active_run = None

    @classmethod
    def start_run(cls, experiment_name):
        mlflow.set_tracking_uri("file:///Users/rodrigoqaz/Documents/tmp/mlruns")
        mlflow.set_experiment(experiment_name)
        cls._active_run = mlflow.start_run()

    @classmethod
    def log_params(cls, params):
        mlflow.log_params(params)

    @classmethod
    def log_metrics(cls, metrics, step=None):
        mlflow.log_metrics(metrics, step=step)

    @classmethod
    def log_model(cls, model, artifact_path, input_schema):
        mlflow.pytorch.log_model(model, artifact_path, signature=input_schema)  # type: ignore

    @classmethod
    def log_text(cls, text):
        mlflow.log_text(text=text, artifact_file="model_summary.txt")

    @classmethod
    def log_artifact(cls, local_path, artifact_path):
        mlflow.log_artifact(local_path=local_path, artifact_path=artifact_path)

    @classmethod
    def end_run(cls):
        mlflow.end_run()
