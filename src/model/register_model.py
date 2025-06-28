import numpy as np
import pandas as pd
import json
import logging
import mlflow
import mlflow.sklearn
import dagshub


mlflow.set_tracking_uri(
    "https://dagshub.com/Saiful0044/Analyzing-Emotional-Sentiment-in-Tweets-Using-Machine-Learning.mlflow"
)
dagshub.init(
    repo_owner="Saiful0044",
    repo_name="Analyzing-Emotional-Sentiment-in-Tweets-Using-Machine-Learning",
    mlflow=True,
)

# logging configuration
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_evaluation_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model_info(file_path: str)->dict:
    """Load the model info from a json file"""
    try:
        with open(file_path,'r') as file:
            model_info = json.load(file)
        logger.info(f"Model info loaded from {file_path}")
        return model_info
    except FileNotFoundError:
        logger.error(f"File not found {file_path}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error occurred while loading the model info: {e}")
        raise

def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry."""
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"

        # Register the model
        model_version = mlflow.register_model(model_uri,model_name)

        # Transition the model to "Staging" stage
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )
        logger.info(f'Model {model_name} version {model_version.version} registered and transitioned to Staging.')
    except Exception as e:
        logger.error('Error during model registration: %s', e)
        raise

def main():
    try:
        model_info_path = 'reports/experiment_info.json'
        model_info = load_model_info(model_info_path)

        model_name = 'my_model'
        register_model(model_name=model_name, model_info=model_info)
    except Exception as e:
        logger.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
