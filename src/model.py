"""
LSTM + Attention Model Architecture for Phishing Detection
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


class AttentionLayer(layers.Layer):
    """
    Custom Attention Layer for focusing on important parts of the URL
    """
    
    def __init__(self, units, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.units = units
        
    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        self.u = self.add_weight(
            name='attention_context',
            shape=(self.units,),
            initializer='glorot_uniform',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)
        
    def call(self, x):
        # x shape: (batch_size, time_steps, features)
        
        # Calculate attention scores
        uit = tf.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        ait = tf.tensordot(uit, self.u, axes=1)
        
        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(ait, axis=1)
        attention_weights = tf.expand_dims(attention_weights, axis=-1)
        
        # Apply attention weights to input
        weighted_input = x * attention_weights
        output = tf.reduce_sum(weighted_input, axis=1)
        
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


def build_lstm_attention_model(
    max_length=200,
    vocab_size=10000,
    embedding_dim=128,
    lstm_units=128,
    attention_units=64,
    num_features=27,
    dropout_rate=0.3
):
    """
    Build LSTM model with Attention mechanism for phishing detection
    """
    
    # Input for URL sequences
    sequence_input = layers.Input(shape=(max_length,), name='sequence_input')
    
    # Embedding layer
    embedding = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=max_length,
        name='embedding'
    )(sequence_input)
    
    # Bidirectional LSTM layers
    lstm_out = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=True, dropout=dropout_rate),
        name='bidirectional_lstm_1'
    )(embedding)
    
    lstm_out = layers.Bidirectional(
        layers.LSTM(lstm_units // 2, return_sequences=True, dropout=dropout_rate),
        name='bidirectional_lstm_2'
    )(lstm_out)
    
    # Attention layer
    attention_out = AttentionLayer(attention_units, name='attention')(lstm_out)
    
    # Dense layers for LSTM branch
    lstm_dense = layers.Dense(64, activation='relu', name='lstm_dense_1')(attention_out)
    lstm_dense = layers.Dropout(dropout_rate)(lstm_dense)
    lstm_dense = layers.Dense(32, activation='relu', name='lstm_dense_2')(lstm_dense)
    
    # Input for numerical features
    feature_input = layers.Input(shape=(num_features,), name='feature_input')
    
    # Dense layers for feature branch
    feature_dense = layers.Dense(64, activation='relu', name='feature_dense_1')(feature_input)
    feature_dense = layers.Dropout(dropout_rate)(feature_dense)
    feature_dense = layers.Dense(32, activation='relu', name='feature_dense_2')(feature_dense)
    
    # Concatenate both branches
    concatenated = layers.Concatenate(name='concatenate')([lstm_dense, feature_dense])
    
    # Final dense layers
    dense = layers.Dense(64, activation='relu', name='final_dense_1')(concatenated)
    dense = layers.Dropout(dropout_rate)(dense)
    dense = layers.Dense(32, activation='relu', name='final_dense_2')(dense)
    dense = layers.Dropout(dropout_rate)(dense)
    
    # Output layer
    output = layers.Dense(1, activation='sigmoid', name='output')(dense)
    
    # Create model
    model = Model(
        inputs=[sequence_input, feature_input],
        outputs=output,
        name='phishing_detector_lstm_attention'
    )
    
    return model


def compile_model(model, learning_rate=0.001):
    """
    Compile the model with optimizer and metrics
    """
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    
    return model


def create_phishing_detector():
    """
    Create and compile the complete phishing detector model
    """
    model = build_lstm_attention_model(
        max_length=config.MAX_URL_LENGTH,
        vocab_size=10000,
        embedding_dim=config.EMBEDDING_DIM,
        lstm_units=config.LSTM_UNITS,
        attention_units=config.ATTENTION_UNITS,
        num_features=31,  # Updated to match actual feature count
        dropout_rate=config.DROPOUT_RATE
    )
    
    model = compile_model(model)
    
    return model
