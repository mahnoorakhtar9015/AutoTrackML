import mlflow
import mlflow.sklearn

def init_mlflow(uri, experiment_name):
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)

def log_run(model_name, params, metrics, model):
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")
