"""
Utility functions for URL feature extraction and preprocessing
"""
import re
import tldextract
from urllib.parse import urlparse
import numpy as np

def extract_url_features(url):
    """
    Extract features from a URL for phishing detection
    Returns a dictionary of features
    """
    features = {}
    
    try:
        # Basic URL parsing
        parsed_url = urlparse(url)
        ext = tldextract.extract(url)
        
        # Length features
        features['url_length'] = len(url)
        features['hostname_length'] = len(parsed_url.netloc)
        features['path_length'] = len(parsed_url.path)
        features['query_length'] = len(parsed_url.query)
        
        # Character counts
        features['dot_count'] = url.count('.')
        features['dash_count'] = url.count('-')
        features['underscore_count'] = url.count('_')
        features['slash_count'] = url.count('/')
        features['question_count'] = url.count('?')
        features['equal_count'] = url.count('=')
        features['at_count'] = url.count('@')
        features['ampersand_count'] = url.count('&')
        features['exclamation_count'] = url.count('!')
        features['space_count'] = url.count(' ')
        features['tilde_count'] = url.count('~')
        features['comma_count'] = url.count(',')
        features['plus_count'] = url.count('+')
        features['asterisk_count'] = url.count('*')
        features['hash_count'] = url.count('#')
        features['dollar_count'] = url.count('$')
        features['percent_count'] = url.count('%')
        
        # Digit and letter counts
        features['digit_count'] = sum(c.isdigit() for c in url)
        features['letter_count'] = sum(c.isalpha() for c in url)
        
        # Protocol features
        features['https'] = 1 if parsed_url.scheme == 'https' else 0
        features['http'] = 1 if parsed_url.scheme == 'http' else 0
        
        # Domain features
        features['has_ip'] = 1 if re.match(r'\d+\.\d+\.\d+\.\d+', parsed_url.netloc) else 0
        features['has_port'] = 1 if ':' in parsed_url.netloc else 0
        
        # Subdomain count
        features['subdomain_count'] = len(ext.subdomain.split('.')) if ext.subdomain else 0
        
        # Suspicious patterns
        features['has_double_slash'] = 1 if '//' in parsed_url.path else 0
        features['has_at_symbol'] = 1 if '@' in url else 0
        
        # TLD length
        features['tld_length'] = len(ext.suffix) if ext.suffix else 0
        
    except Exception as e:
        print(f"Error extracting features from {url}: {e}")
        # Return default features if parsing fails
        for key in features.keys():
            features[key] = 0
    
    return features


def normalize_url(url):
    """
    Normalize URL for consistent processing
    """
    url = url.strip()
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url
    return url


def is_valid_url(url):
    """
    Check if URL is valid
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False
