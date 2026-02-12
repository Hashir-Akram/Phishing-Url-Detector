"""
Test script to verify model and preprocessor loading
"""
import os
import sys

try:
    print("Testing model loading...")
    print("=" * 70)
    
    # Import required modules
    print("1. Importing TensorFlow...")
    from tensorflow import keras
    print("   ✓ TensorFlow imported")
    
    print("2. Importing custom modules...")
    from src.model import AttentionLayer
    from src.preprocessing import URLPreprocessor
    import config
    print("   ✓ Custom modules imported")
    
    # Check if files exist
    print("\n3. Checking if files exist...")
    print(f"   Model path: {config.MODEL_PATH}")
    print(f"   Model exists: {os.path.exists(config.MODEL_PATH)}")
    print(f"   Tokenizer path: {config.TOKENIZER_PATH}")
    print(f"   Tokenizer exists: {os.path.exists(config.TOKENIZER_PATH)}")
    print(f"   Scaler path: {config.SCALER_PATH}")
    print(f"   Scaler exists: {os.path.exists(config.SCALER_PATH)}")
    
    # Load model
    print("\n4. Loading model...")
    model = keras.models.load_model(
        config.MODEL_PATH,
        custom_objects={'AttentionLayer': AttentionLayer}
    )
    print("   ✓ Model loaded successfully")
    
    # Load preprocessor
    print("\n5. Loading preprocessor...")
    preprocessor = URLPreprocessor()
    preprocessor.load(config.TOKENIZER_PATH, config.SCALER_PATH)
    print("   ✓ Preprocessor loaded successfully")
    
    # Test prediction
    print("\n6. Testing prediction on sample URL...")
    test_url = "https://www.google.com"
    from src.utils import normalize_url
    normalized_url = normalize_url(test_url)
    
    X_seq = preprocessor.transform_urls([normalized_url])
    X_feat = preprocessor.extract_features([normalized_url])
    X_feat = preprocessor.transform_features(X_feat)
    
    prediction_proba = model.predict([X_seq, X_feat], verbose=0)[0][0]
    print(f"   Test URL: {test_url}")
    print(f"   Phishing probability: {prediction_proba:.4f}")
    print(f"   Prediction: {'Phishing' if prediction_proba > 0.5 else 'Legitimate'}")
    
    print("\n" + "=" * 70)
    print("✓ ALL TESTS PASSED!")
    print("=" * 70)
    
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
