import numpy as np
import pandas as pd
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging

# logging configuration
logger = logging.getLogger('data_transformation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('transformation_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

nltk.download('wordnet')
nltk.download('stopwords')


# difine text preprocessing functions
def lemmatization(text):
    """Lemmatize the text"""
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return ' '.join(text)

def remove_stop_words(text):
    """remove stop words from the text"""
    stop_words = set(stopwords.words('english'))
    text = [word for word in text.split() if word not in stop_words]
    return ' '.join(text)

def removing_numbers(text):
    """remove numbers from the text"""
    text = ''.join([char for char in text if not char.isdigit()])
    return text

def lower_case(text):
    """Convert text to lower case"""
    text = text.split()
    text = [word.lower() for word in text]
    return ' '.join(text)

def removing_punctuations(text):
    """remove punctuations from the text"""
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace(':', '')
    text = re.sub('\s+', ' ', text).strip()
    return text

def removes_urls(text):
    """Remove urls from the text"""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def normalize_text(df):
    """Normalize the text data"""
    try:
        df['content'] = df['content'].apply(lower_case)
        df['content'] = df['content'].apply(remove_stop_words)
        df['content'] = df['content'].apply(removing_numbers)
        df['content'] = df['content'].apply(removing_punctuations)
        df['content'] = df['content'].apply(removes_urls)
        df['content'] = df['content'].apply(lemmatization)
        return df
    except Exception as e:
        print(f"Error during text normalization: {e}")
        raise

def main():
    try:

        # fetch the data from data/raw
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.info('Data loaded properly')

        # transform the data
        train_processed_data = normalize_text(train_data)
        test_processed_data = normalize_text(test_data)

        # store the data inside data/processed
        data_path = os.path.join('./data', 'interim')
        os.makedirs(data_path, exist_ok=True)

        train_processed_data.to_csv(os.path.join(data_path, 'train_processed.csv'), index=False)
        test_processed_data.to_csv(os.path.join(data_path,'test_processed.csv'), index=False)
        logger.info("Processed data saved to %s", data_path)
    except Exception as e:
        logger.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")

if __name__=='__main__':
    main()
