"""
Debug script to test model predictions
"""
import numpy as np
from tensorflow import keras
from src.model import AttentionLayer
from src.preprocessing import URLPreprocessor
from src.utils import normalize_url
import config

print("Loading model and preprocessor...")
model = keras.models.load_model(
    config.MODEL_PATH,
    custom_objects={'AttentionLayer': AttentionLayer}
)

preprocessor = URLPreprocessor()
preprocessor.load(config.TOKENIZER_PATH, config.SCALER_PATH)

# Test URLs
test_urls = [
    "https://www.google.com",
    "https://www.facebook.com",
    "http://phishing-test-site.com",
    "http://192.168.1.1/login",
    "http://secure-paypal-verify.tk/verify",
    "https://www.amazon.com"
]

print("\nTesting predictions:")
print("=" * 80)

for url in test_urls:
    normalized_url = normalize_url(url)
    
    # Preprocessing
    X_seq = preprocessor.transform_urls([normalized_url])
    X_feat = preprocessor.extract_features([normalized_url])
    X_feat_scaled = preprocessor.transform_features(X_feat)
    
    # Prediction
    prediction_proba = model.predict([X_seq, X_feat_scaled], verbose=0)[0][0]
    prediction = int(prediction_proba > 0.5)
    
    print(f"\nURL: {url}")
    print(f"  Normalized: {normalized_url}")
    print(f"  Sequence shape: {X_seq.shape}")
    print(f"  Features shape: {X_feat_scaled.shape}")
    print(f"  Raw prediction probability: {prediction_proba}")
    print(f"  Prediction: {'Phishing' if prediction == 1 else 'Legitimate'}")
    print(f"  Confidence: {prediction_proba if prediction == 1 else (1 - prediction_proba):.4f}")
    
    # Show some feature values
    print(f"  Sample features: {X_feat_scaled[0][:5]}")
    print(f"  Sample sequence: {X_seq[0][:20]}")

print("\n" + "=" * 80)
