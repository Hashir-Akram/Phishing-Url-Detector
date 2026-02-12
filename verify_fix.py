"""
Verify the prediction fix
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

# Test URLs - known legitimate and suspicious
test_cases = [
    {"url": "https://www.google.com", "expected": "Legitimate"},
    {"url": "https://www.facebook.com", "expected": "Legitimate"},
    {"url": "https://www.amazon.com", "expected": "Legitimate"},
    {"url": "https://www.microsoft.com", "expected": "Legitimate"},
    {"url": "http://phishing-test-site.com", "expected": "Phishing"},
    {"url": "http://192.168.1.1/login", "expected": "Phishing"},
    {"url": "http://secure-paypal-verify.tk/verify", "expected": "Phishing"},
]

print("\nTesting with FIXED predictions:")
print("=" * 80)

correct = 0
total = len(test_cases)

for test in test_cases:
    url = test["url"]
    expected = test["expected"]
    
    normalized_url = normalize_url(url)
    
    # Preprocessing
    X_seq = preprocessor.transform_urls([normalized_url])
    X_feat = preprocessor.extract_features([normalized_url])
    X_feat_scaled = preprocessor.transform_features(X_feat)
    
    # Prediction (with fix)
    raw_prediction_proba = model.predict([X_seq, X_feat_scaled], verbose=0)[0][0]
    prediction_proba = 1 - raw_prediction_proba  # INVERT THE PREDICTION
    prediction = int(prediction_proba > 0.5)
    
    result = "Phishing" if prediction == 1 else "Legitimate"
    is_correct = result == expected
    if is_correct:
        correct += 1
    
    print(f"\nURL: {url}")
    print(f"  Raw model output: {raw_prediction_proba:.4f}")
    print(f"  Inverted probability: {prediction_proba:.4f}")
    print(f"  Prediction: {result}")
    print(f"  Expected: {expected}")
    print(f"  Status: {'✓ CORRECT' if is_correct else '✗ WRONG'}")

print("\n" + "=" * 80)
print(f"Accuracy: {correct}/{total} ({100*correct/total:.1f}%)")
print("=" * 80)
