"""
Configuration file for Phishing Detector
"""
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Model Parameters
MAX_URL_LENGTH = 200
EMBEDDING_DIM = 128
LSTM_UNITS = 128
ATTENTION_UNITS = 64
DROPOUT_RATE = 0.3
BATCH_SIZE = 64
EPOCHS = 20
VALIDATION_SPLIT = 0.2

# Dataset
DATASET_PATH = os.path.join(BASE_DIR, 'PhiUSIIL_Phishing_URL_Dataset.csv')

# Model Save Path
MODEL_PATH = os.path.join(MODEL_DIR, 'phishing_detector_lstm_attention.h5')
TOKENIZER_PATH = os.path.join(MODEL_DIR, 'tokenizer.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')

# Flask Configuration
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000
FLASK_DEBUG = False

# Streamlit Configuration
STREAMLIT_PORT = 8501
