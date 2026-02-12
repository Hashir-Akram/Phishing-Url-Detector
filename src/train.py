"""
Training script for phishing detector model
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import os
import sys
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.model import create_phishing_detector
from src.preprocessing import load_and_preprocess_data


def plot_training_history(history, save_path='models/training_history.png'):
    """
    Plot training history
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0, 0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train Loss')
    axes[0, 1].plot(history.history['val_loss'], label='Val Loss')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Precision
    axes[1, 0].plot(history.history['precision'], label='Train Precision')
    axes[1, 0].plot(history.history['val_precision'], label='Val Precision')
    axes[1, 0].set_title('Model Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Recall
    axes[1, 1].plot(history.history['recall'], label='Train Recall')
    axes[1, 1].plot(history.history['val_recall'], label='Val Recall')
    axes[1, 1].set_title('Model Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to {save_path}")
    plt.close()


def evaluate_model(model, X_seq_test, X_feat_test, y_test, save_path='models/evaluation.png'):
    """
    Evaluate model and create visualizations
    """
    # Predictions
    y_pred_proba = model.predict([X_seq_test, X_feat_test])
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Legitimate', 'Phishing'],
                yticklabels=['Legitimate', 'Phishing'])
    axes[0].set_title('Confusion Matrix')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    
    # ROC Curve
    axes[1].plot(fpr, tpr, color='darkorange', lw=2, 
                 label=f'ROC curve (AUC = {roc_auc:.2f})')
    axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('Receiver Operating Characteristic (ROC) Curve')
    axes[1].legend(loc="lower right")
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Evaluation plot saved to {save_path}")
    plt.close()
    
    return {
        'accuracy': np.mean(y_pred == y_test),
        'auc': roc_auc,
        'confusion_matrix': cm
    }


def train_model():
    """
    Main training function
    """
    print("=" * 70)
    print("PHISHING DETECTOR - LSTM + ATTENTION MODEL TRAINING")
    print("=" * 70)
    
    # Check if dataset exists
    if not os.path.exists(config.DATASET_PATH):
        print(f"\nDataset not found at {config.DATASET_PATH}")
        return
    
    # Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    result = load_and_preprocess_data(config.DATASET_PATH)
    if result is None:
        return
    
    (X_seq_train, X_seq_test, X_feat_train, X_feat_test, y_train, y_test), preprocessor = result
    
    print(f"\nTraining set: {len(X_seq_train)} samples")
    print(f"Test set: {len(X_seq_test)} samples")
    print(f"Sequence shape: {X_seq_train.shape}")
    print(f"Features shape: {X_feat_train.shape}")
    
    # Create model
    print("\n2. Creating model...")
    model = create_phishing_detector()
    model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            config.MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            verbose=1,
            min_lr=1e-7
        )
    ]
    
    # Train model
    print("\n3. Training model...")
    history = model.fit(
        [X_seq_train, X_feat_train],
        y_train,
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS,
        validation_split=config.VALIDATION_SPLIT,
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    print("\n4. Plotting training history...")
    plot_training_history(history)
    
    # Evaluate model
    print("\n5. Evaluating model...")
    metrics = evaluate_model(model, X_seq_test, X_feat_test, y_test)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test AUC: {metrics['auc']:.4f}")
    print(f"Model saved to: {config.MODEL_PATH}")
    
    # Save preprocessor
    print("\n6. Saving preprocessor...")
    preprocessor.save(config.TOKENIZER_PATH, config.SCALER_PATH)
    print(f"Tokenizer saved to: {config.TOKENIZER_PATH}")
    print(f"Scaler saved to: {config.SCALER_PATH}")
    
    print("\n" + "=" * 70)
    print("All done! You can now use the model for predictions.")
    print("=" * 70)


if __name__ == "__main__":
    train_model()
