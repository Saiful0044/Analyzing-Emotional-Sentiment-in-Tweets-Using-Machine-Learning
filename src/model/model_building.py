import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
import yaml
import logging

# logging configuration
logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_building_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# data load 
def load_data(file_path: str)->pd.DataFrame:
    """Load data from a csv file"""
    try:
        data = pd.read_csv(file_path)
        logger.info(f'Data loaded from {file_path}')
        return data
    except pd.errors.ParserError as e:
        logger.error(f"Failed to parse the csv file: {e}")
    except Exception as e:
        logger.error(f"Unexpected error occurred while loading the data: {e}")
        raise

def train_model(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """Train the logistic Regression model"""
    try:
        clf = LogisticRegression(C=1, solver='liblinear', penalty='l2')
        clf.fit(X_train,y_train)
        logger.info('Model training completed')
        return clf
    except Exception as e:
        logger.error(f"Error during model training {e}")
        raise

# model save
def save_model(model, file_path: str)-> None:
    """Save the trained model to a file"""
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.info(f"Model saved to {file_path}")
    except Exception as e:
        logger.error(f"Error occurred while saving the model {e}")
        raise

def main():
    try:
        train_data = load_data('./data/processed/train_bow.csv')
        X_train = train_data.iloc[:,:-1].values
        y_train = train_data.iloc[:,-1].values

        clf = train_model(X_train=X_train, y_train=y_train)

        save_model(clf, 'models/model.pkl')
    
    except Exception as e:
        logger.error(f"Failed to complete the model building process: {e}")
        raise

if __name__=="__main__":
    main()