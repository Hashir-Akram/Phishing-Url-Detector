"""
Data preprocessing module for phishing detection
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os
import sys
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.utils import extract_url_features, normalize_url


class URLPreprocessor:
    """
    Preprocessor for URL data
    """
    
    def __init__(self, max_length=200, max_features=10000):
        self.max_length = max_length
        self.max_features = max_features
        self.tokenizer = Tokenizer(num_words=max_features, char_level=True)
        self.scaler = StandardScaler()
        
    def fit(self, urls):
        """
        Fit tokenizer on URLs
        """
        normalized_urls = [normalize_url(url) for url in urls]
        self.tokenizer.fit_on_texts(normalized_urls)
        
    def transform_urls(self, urls):
        """
        Transform URLs to sequences
        """
        normalized_urls = [normalize_url(url) for url in urls]
        sequences = self.tokenizer.texts_to_sequences(normalized_urls)
        padded = pad_sequences(sequences, maxlen=self.max_length, padding='post', truncating='post')
        return padded
    
    def extract_features(self, urls):
        """
        Extract numerical features from URLs
        """
        features_list = []
        for url in urls:
            features = extract_url_features(normalize_url(url))
            features_list.append(list(features.values()))
        return np.array(features_list)
    
    def fit_scaler(self, features):
        """
        Fit scaler on features
        """
        self.scaler.fit(features)
        
    def transform_features(self, features):
        """
        Scale features
        """
        return self.scaler.transform(features)
    
    def save(self, tokenizer_path, scaler_path):
        """
        Save tokenizer and scaler
        """
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
    
    def load(self, tokenizer_path, scaler_path):
        """
        Load tokenizer and scaler
        """
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)


def load_and_preprocess_data(dataset_path):
    """
    Load and preprocess the phishing dataset
    """
    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv(dataset_path)
    
    # PhiUSIIL dataset format: 'URL' and 'label' columns
    if 'URL' in df.columns and 'label' in df.columns:
        print("Detected PhiUSIIL dataset format")
        urls = df['URL'].values
        labels = df['label'].values
    else:
        print("ERROR: Dataset must have 'URL' and 'label' columns")
        return None
    
    # Convert labels to binary (0 = legitimate, 1 = phishing)
    labels = labels.astype(int)
    
    print(f"Loaded {len(urls)} URLs")
    print(f"Phishing URLs: {sum(labels)}, Legitimate URLs: {len(labels) - sum(labels)}")
    
    # Create preprocessor
    preprocessor = URLPreprocessor(max_length=config.MAX_URL_LENGTH)
    
    # Fit and transform
    print("Preprocessing URLs...")
    preprocessor.fit(urls)
    X_sequences = preprocessor.transform_urls(urls)
    
    print("Extracting features...")
    X_features = preprocessor.extract_features(urls)
    preprocessor.fit_scaler(X_features)
    X_features = preprocessor.transform_features(X_features)
    
    # Split data
    X_seq_train, X_seq_test, X_feat_train, X_feat_test, y_train, y_test = train_test_split(
        X_sequences, X_features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    return (X_seq_train, X_seq_test, X_feat_train, X_feat_test, y_train, y_test), preprocessor
