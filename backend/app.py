"""
Flask Backend API for Phishing Detection
"""
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import numpy as np
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from tensorflow import keras
    TF_AVAILABLE = True
except:
    TF_AVAILABLE = False

import config
from src.preprocessing import URLPreprocessor
from src.utils import normalize_url, is_valid_url
from src.model import AttentionLayer

app = Flask(__name__)
CORS(app)

# Global variables
model = None
preprocessor = None


def load_model_and_preprocessor():
    """Load the trained model and preprocessor"""
    global model, preprocessor
    
    if not TF_AVAILABLE:
        print("TensorFlow not available")
        return False
    
    try:
        if os.path.exists(config.MODEL_PATH):
            print(f"Loading model from {config.MODEL_PATH}", flush=True)
            model = keras.models.load_model(
                config.MODEL_PATH, 
                custom_objects={'AttentionLayer': AttentionLayer}
            )
            print("✓ Model loaded successfully", flush=True)
        else:
            print(f"Model not found at {config.MODEL_PATH}", flush=True)
            return False
        
        if os.path.exists(config.TOKENIZER_PATH) and os.path.exists(config.SCALER_PATH):
            print(f"Loading preprocessor...", flush=True)
            preprocessor = URLPreprocessor()
            preprocessor.load(config.TOKENIZER_PATH, config.SCALER_PATH)
            print("✓ Preprocessor loaded successfully", flush=True)
        else:
            print(f"Preprocessor files not found:", flush=True)
            print(f"  Tokenizer: {os.path.exists(config.TOKENIZER_PATH)}", flush=True)
            print(f"  Scaler: {os.path.exists(config.SCALER_PATH)}", flush=True)
            return False
        
        return True
    except Exception as e:
        print(f"✗ Error loading model/preprocessor: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return False


def predict_url(url):
    """Predict if a URL is phishing or legitimate"""
    try:
        normalized_url = normalize_url(url)
        
        X_seq = preprocessor.transform_urls([normalized_url])
        X_feat = preprocessor.extract_features([normalized_url])
        X_feat = preprocessor.transform_features(X_feat)
        
        # Get raw prediction from model
        raw_prediction_proba = model.predict([X_seq, X_feat], verbose=0)[0][0]
        
        # IMPORTANT: The model predictions are inverted, so we need to flip them
        # The model outputs high probability for legitimate sites, so we invert it
        prediction_proba = 1 - raw_prediction_proba
        prediction = int(prediction_proba > 0.5)
        
        confidence = prediction_proba if prediction == 1 else (1 - prediction_proba)
        
        if prediction_proba >= 0.8:
            risk_level = "Very High"
        elif prediction_proba >= 0.6:
            risk_level = "High"
        elif prediction_proba >= 0.4:
            risk_level = "Medium"
        elif prediction_proba >= 0.2:
            risk_level = "Low"
        else:
            risk_level = "Very Low"
        
        return {
            'success': True,
            'url': url,
            'is_phishing': bool(prediction),
            'phishing_probability': float(prediction_proba),
            'legitimate_probability': float(1 - prediction_proba),
            'confidence': float(confidence),
            'risk_level': risk_level,
            'prediction': 'Phishing' if prediction == 1 else 'Legitimate'
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


@app.route('/')
def home():
    """Serve the HTML interface"""
    try:
        frontend_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'frontend')
        return send_from_directory(frontend_path, 'index.html')
    except Exception as e:
        return jsonify({
            'message': 'Phishing Detector API',
            'version': '1.0',
            'error': str(e),
            'endpoints': {
                '/': 'Web interface (HTML)',
                '/api': 'API information',
                '/predict': 'POST - Predict if a URL is phishing',
                '/health': 'GET - Check API health'
            }
        })


@app.route('/api')
def api_info():
    """API information endpoint"""
    return jsonify({
        'message': 'Phishing Detector API',
        'version': '1.0',
        'endpoints': {
            '/': 'Web interface (HTML)',
            '/api': 'This API information',
            '/predict': 'POST - Predict if a URL is phishing',
            '/health': 'GET - Check API health'
        }
    })


@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'preprocessor_loaded': preprocessor is not None
    })


@app.route('/predict', methods=['POST'])
def predict():
    if model is None or preprocessor is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded. Please train the model first.'
        }), 500
    
    try:
        data = request.get_json()
        
        if not data or 'url' not in data:
            return jsonify({
                'success': False,
                'error': 'No URL provided. Please send JSON with "url" field.'
            }), 400
        
        url = data['url']
        
        if not is_valid_url(url):
            return jsonify({
                'success': False,
                'error': 'Invalid URL format'
            }), 400
        
        result = predict_url(url)
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    print("=" * 70, flush=True)
    print("PHISHING DETECTOR - FLASK BACKEND", flush=True)
    print("=" * 70, flush=True)
    
    if load_model_and_preprocessor():
        print(f"\n✓ Server running on http://{config.FLASK_HOST}:{config.FLASK_PORT}", flush=True)
        print("=" * 70 + "\n", flush=True)
        app.run(host=config.FLASK_HOST, port=config.FLASK_PORT, debug=config.FLASK_DEBUG)
    else:
        print("\n✗ Failed to load model. Please train first: python src/train.py", flush=True)
        print("=" * 70 + "\n", flush=True)
