import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml
import logging

# logging configuration
logger = logging.getLogger('data_ingestion')
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("ERROR")

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# data load 
def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a csv file"""
    try:
        df = pd.read_csv(data_url)
        logger.info("Data loaded from %s", data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the csv file: %s", e)
    except Exception as e:
        logger.error("Unexpected error occurred while loading the data: %s", e)
        raise
# data preprocessing
def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """preprocess the data"""
    try:
        data.drop(columns=['tweet_id'], inplace=True)
        data = data[data['sentiment'].isin(['happiness', 'sadness'])]
        data['sentiment'].replace({'happiness': 1, 'sadness':0}, inplace=True)
        logger.info("Data preprocessing completed")
        return data
    except KeyError as e:
        logger.error('Missing column in the dataframe: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error during preprocessing: %s', e)
        raise


# Load parameters from a yaml file
def load_params(params_path: str) -> dict:
    """Load parameters from a Yaml file"""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.info('Parameters retrieved from %s',params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error("YAML error: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        raise

# save data
def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets"""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, 'train.csv'), index=False)
        test_data.to_csv(os.path.join(raw_data_path, 'test.csv'), index=False)
        logger.info('Train and test data saved to %s', raw_data_path)
    except Exception as e:
        logger.error("Unexpected error occurred while saving the data: %s", e)
        raise


def main():
    try:
        df = load_data(data_url="https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv")

        df = preprocess_data(data=df)
        # load yaml file
        params = load_params(params_path='params.yaml')
        test_size = params['data_ingestion']['test_size']

        # train test 
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)
        
        # save data
        save_data(train_data=train_data, test_data=test_data, data_path='./data')
    
    except Exception as e:
        logger.error('failed to complete the data ingestion process: %s', e)

if __name__=='__main__':
    main()

