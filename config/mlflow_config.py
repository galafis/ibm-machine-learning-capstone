"""
Configuração central de tracking MLflow para experimentos e produção.
"""
MLFLOW_CONFIG = {
    'tracking_uri': 'postgresql://user:pass@localhost/mlflow',
    'artifact_root': 's3://ml-artifacts-bucket',
    'experiment_name': 'production_models'
}

def get_config():
    return MLFLOW_CONFIG
