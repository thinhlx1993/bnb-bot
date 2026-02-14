"""
Deep Learning Model for Trading Signal Prediction
This script trains a neural network to predict profitable trades based on technical indicators.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, f1_score
from imblearn.over_sampling import SMOTE, RandomOverSampler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
import logging
import warnings
warnings.filterwarnings('ignore')

# Configure logging
Path("logs").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/train_model.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
TRAINING_DATA_DIR = Path("results/training_data")
MODEL_SAVE_DIR = Path("models")
MODEL_SAVE_DIR.mkdir(exist_ok=True)

# Model hyperparameters
BATCH_SIZE = 32
EPOCHS = 1000
VALIDATION_SPLIT = 0.3
TEST_SPLIT = 0.1
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 100
REDUCE_LR_PATIENCE = 50


def load_training_data(data_dir=TRAINING_DATA_DIR, strategy_filter=None, ticker_filter=None):
    """
    Load all training data CSV files from the training data directory.
    
    Args:
        data_dir: Directory containing training data CSV files
        strategy_filter: Optional filter for specific strategy (e.g., 'Combined', 'MACD_Trend_Reversal')
        ticker_filter: Optional filter for specific ticker (e.g., 'BTCUSDT', 'ETHUSDT')
    
    Returns:
        combined_df: Combined DataFrame with all training data
        file_info: Dictionary mapping file names to their data
    """
    logger.info(f"Loading training data from {data_dir}...")
    
    if not data_dir.exists():
        logger.error(f"Training data directory not found: {data_dir}")
        return None, {}
    
    csv_files = list(data_dir.glob("*.csv"))
    
    if len(csv_files) == 0:
        logger.error(f"No CSV files found in {data_dir}")
        return None, {}
    
    logger.info(f"Found {len(csv_files)} training data files")
    
    all_dataframes = []
    file_info = {}
    
    for csv_file in csv_files:
        # Parse filename: {ticker}_{strategy}_training_data.csv
        filename = csv_file.stem
        parts = filename.replace('_training_data', '').split('_')
        
        if len(parts) < 2:
            logger.warning(f"Could not parse filename: {filename}, skipping...")
            continue
        
        ticker = parts[0]
        strategy = '_'.join(parts[1:])
        
        # Apply filters
        if strategy_filter and strategy_filter not in strategy:
            continue
        if ticker_filter and ticker_filter not in ticker:
            continue
        
        try:
            df = pd.read_csv(csv_file)
            
            if len(df) == 0:
                logger.warning(f"Empty file: {csv_file}, skipping...")
                continue
            
            # Add metadata columns
            df['source_ticker'] = ticker
            df['source_strategy'] = strategy
            df['source_file'] = filename
            
            all_dataframes.append(df)
            file_info[filename] = {
                'ticker': ticker,
                'strategy': strategy,
                'samples': len(df),
                'positive_labels': df['label'].sum() if 'label' in df.columns else 0
            }
            
            logger.info(f"  Loaded {filename}: {len(df)} samples ({file_info[filename]['positive_labels']} positive)")
            
        except Exception as e:
            logger.error(f"Error loading {csv_file}: {e}")
            continue
    
    if len(all_dataframes) == 0:
        logger.error("No valid training data loaded!")
        return None, {}
    
    # Combine all dataframes
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    logger.info(f"\nTotal combined dataset: {len(combined_df)} samples")
    profit_count = (combined_df['label'] == 1).sum()
    loss_count = (combined_df['label'] == -1).sum()
    logger.info(f"Positive labels (profit=1): {profit_count} ({profit_count/len(combined_df)*100:.1f}%)")
    logger.info(f"Negative labels (loss=-1): {loss_count} ({loss_count/len(combined_df)*100:.1f}%)")
    
    return combined_df, file_info


def preprocess_data(df, test_size=TEST_SPLIT, validation_size=VALIDATION_SPLIT):
    """
    Preprocess the training data: separate features and labels, split data, normalize.
    
    Uses MinMaxScaler to normalize features to [0, 1] range instead of standardization.
    
    Args:
        df: DataFrame with features and labels
        test_size: Proportion of data for testing
        validation_size: Proportion of training data for validation
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names
    """
    logger.info("Preprocessing data...")
    
    # Separate features and labels
    # Exclude metadata columns and label
    exclude_cols = ['label', 'source_ticker', 'source_strategy', 'source_file']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].copy()
    y = df['label'].copy()
    
    # Convert labels from -1/1 to 0/1 for binary classification
    # -1 (loss) -> 0, 1 (profit) -> 1
    y = y.replace({-1: 0, 1: 1})
    
    logger.info(f"Features: {len(feature_cols)}")
    logger.info(f"Feature names: {', '.join(feature_cols[:10])}..." if len(feature_cols) > 10 else f"Feature names: {', '.join(feature_cols)}")
    logger.info(f"Labels converted: -1 (loss) -> 0, 1 (profit) -> 1 for binary classification")
    
    # Handle missing values
    if X.isnull().sum().sum() > 0:
        logger.warning(f"Found {X.isnull().sum().sum()} missing values, filling with median...")
        X = X.fillna(X.median())
    
    # Check for infinite values
    if np.isinf(X.select_dtypes(include=[np.number])).sum().sum() > 0:
        logger.warning("Found infinite values, replacing with finite max...")
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
    
    # Split data: train -> (train + validation) and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Split train_val into train and validation
    val_size_adjusted = validation_size / (1 - test_size)  # Adjust for already split data
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size_adjusted, random_state=42, stratify=y_train_val
    )
    
    logger.info(f"Data split (before upsampling):")
    logger.info(f"  Training: {len(X_train)} samples")
    logger.info(f"    - Loss (0, originally -1): {(y_train == 0).sum()} ({(y_train == 0).mean()*100:.1f}%)")
    logger.info(f"    - Profit (1): {(y_train == 1).sum()} ({(y_train == 1).mean()*100:.1f}%)")
    logger.info(f"  Validation: {len(X_val)} samples ({y_val.mean()*100:.1f}% positive)")
    logger.info(f"  Testing: {len(X_test)} samples ({y_test.mean()*100:.1f}% positive)")
    
    # Upsample the minority class (Loss) in training data
    logger.info("\nUpsampling minority class (Loss) in training data...")
    
    # First normalize features before SMOTE (SMOTE works better with normalized data)
    # Using MinMaxScaler to normalize features to [0, 1] range
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply SMOTE to balance the training data
    # Try SMOTE first, fallback to RandomOverSampler if SMOTE fails
    try:
        smote = SMOTE(random_state=42, k_neighbors=min(5, (y_train == 0).sum() - 1))
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train.values)
        logger.info(f"  Using SMOTE for upsampling")
    except Exception as e:
        logger.warning(f"  SMOTE failed ({e}), using RandomOverSampler instead...")
        ros = RandomOverSampler(random_state=42)
        X_train_balanced, y_train_balanced = ros.fit_resample(X_train_scaled, y_train.values)
        logger.info(f"  Using RandomOverSampler for upsampling")
    
    logger.info(f"\nData after upsampling:")
    logger.info(f"  Training: {len(X_train_balanced)} samples")
    logger.info(f"    - Loss (0, originally -1): {(y_train_balanced == 0).sum()} ({(y_train_balanced == 0).mean()*100:.1f}%)")
    logger.info(f"    - Profit (1): {(y_train_balanced == 1).sum()} ({(y_train_balanced == 1).mean()*100:.1f}%)")
    logger.info(f"  Upsampled {len(X_train_balanced) - len(X_train)} additional Loss samples")
    
    logger.info("Data preprocessing complete!")
    
    return (X_train_balanced, X_val_scaled, X_test_scaled, 
            y_train_balanced, y_val.values, y_test.values, 
            scaler, feature_cols)


def create_model(input_dim, learning_rate=LEARNING_RATE):
    """
    Create a deep neural network model for binary classification.
    
    Args:
        input_dim: Number of input features
        learning_rate: Learning rate for optimizer
    
    Returns:
        model: Compiled Keras model
    """
    logger.info(f"Creating model with {input_dim} input features...")
    
    model = models.Sequential([
        # Input layer
        layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Hidden layers
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.2),
        
        # Output layer (binary classification)
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    logger.info("Model architecture:")
    model.summary(print_fn=lambda x: logger.info(x))
    
    return model


def train_model(model, X_train, y_train, X_val, y_val, epochs=EPOCHS, batch_size=BATCH_SIZE):
    """
    Train the model with callbacks for early stopping and learning rate reduction.
    
    Args:
        model: Keras model
        X_train, y_train: Training data
        X_val, y_val: Validation data
        epochs: Maximum number of epochs
        batch_size: Batch size for training
    
    Returns:
        history: Training history
    """
    logger.info("Starting model training...")
    
    # Note: Training data is already balanced via upsampling, but we can still use class weights
    # for additional emphasis if needed. Since data is balanced, weights will be close to 1:1
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    logger.info(f"Class distribution in training set (after upsampling):")
    logger.info(f"  Loss (0, originally -1): {(y_train == 0).sum()} samples ({(y_train == 0).mean()*100:.1f}%)")
    logger.info(f"  Profit (1): {(y_train == 1).sum()} samples ({(y_train == 1).mean()*100:.1f}%)")
    logger.info(f"Class weights: {class_weight_dict}")
    
    # Define callbacks
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=REDUCE_LR_PATIENCE,
            min_lr=1e-7,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=MODEL_SAVE_DIR / 'best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train model with class weights
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks_list,
        class_weight=class_weight_dict,  # Add class weights to handle imbalance
        verbose=1
    )
    
    logger.info("Training complete!")
    
    return history


def evaluate_model(model, X_test, y_test, scaler=None, feature_names=None):
    """
    Evaluate the model on test data and generate comprehensive metrics.
    
    Args:
        model: Trained Keras model
        X_test: Test features
        y_test: Test labels
        scaler: Optional scaler (for future predictions)
        feature_names: Optional feature names (for future predictions)
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    logger.info("Evaluating model on test data...")
    
    # Predictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Calculate metrics
    test_loss, test_accuracy, test_precision, test_recall = model.evaluate(
        X_test, y_test, verbose=0
    )
    
    # Additional metrics
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    # Try to find optimal threshold using F1 score
    thresholds = np.arange(0.1, 0.9, 0.05)
    f1_scores = []
    for threshold in thresholds:
        y_pred_thresh = (y_pred_proba > threshold).astype(int).flatten()
        f1 = f1_score(y_test, y_pred_thresh, average='weighted')
        f1_scores.append(f1)
    
    optimal_threshold_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_threshold_idx]
    optimal_f1 = f1_scores[optimal_threshold_idx]
    
    logger.info(f"\nOptimal threshold analysis:")
    logger.info(f"  Default threshold (0.5): F1 = {f1_score(y_test, y_pred, average='weighted'):.4f}")
    logger.info(f"  Optimal threshold ({optimal_threshold:.2f}): F1 = {optimal_f1:.4f}")
    
    # Use optimal threshold for predictions
    y_pred_optimal = (y_pred_proba > optimal_threshold).astype(int).flatten()
    
    # Classification report with default threshold
    logger.info("\nClassification Report (threshold=0.5):")
    logger.info("\n"+classification_report(y_test, y_pred, target_names=['Loss', 'Profit']))
    
    # Classification report with optimal threshold
    logger.info(f"\nClassification Report (optimal threshold={optimal_threshold:.2f}):")
    logger.info("\n"+classification_report(y_test, y_pred_optimal, target_names=['Loss', 'Profit']))
    
    # Confusion matrix with default threshold
    cm = confusion_matrix(y_test, y_pred)
    logger.info("\nConfusion Matrix (threshold=0.5):")
    logger.info(f"True Negatives (Loss predicted as Loss): {cm[0][0]}")
    logger.info(f"False Positives (Loss predicted as Profit): {cm[0][1]}")
    logger.info(f"False Negatives (Profit predicted as Loss): {cm[1][0]}")
    logger.info(f"True Positives (Profit predicted as Profit): {cm[1][1]}")
    
    # Confusion matrix with optimal threshold
    cm_optimal = confusion_matrix(y_test, y_pred_optimal)
    logger.info(f"\nConfusion Matrix (optimal threshold={optimal_threshold:.2f}):")
    logger.info(f"True Negatives (Loss predicted as Loss): {cm_optimal[0][0]}")
    logger.info(f"False Positives (Loss predicted as Profit): {cm_optimal[0][1]}")
    logger.info(f"False Negatives (Profit predicted as Loss): {cm_optimal[1][0]}")
    logger.info(f"True Positives (Profit predicted as Profit): {cm_optimal[1][1]}")
    
    metrics = {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_auc': auc_score,
        'confusion_matrix': cm_optimal,  # Use optimal threshold confusion matrix
        'optimal_threshold': float(optimal_threshold),
        'optimal_f1': float(optimal_f1)
    }
    
    logger.info(f"\nTest Metrics (threshold=0.5):")
    logger.info(f"  Loss: {test_loss:.4f}")
    logger.info(f"  Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    logger.info(f"  Precision: {test_precision:.4f} ({test_precision*100:.2f}%)")
    logger.info(f"  Recall: {test_recall:.4f} ({test_recall*100:.2f}%)")
    logger.info(f"  AUC-ROC: {auc_score:.4f}")
    
    # Calculate metrics with optimal threshold
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    test_accuracy_optimal = accuracy_score(y_test, y_pred_optimal)
    test_precision_optimal = precision_score(y_test, y_pred_optimal, zero_division=0)
    test_recall_optimal = recall_score(y_test, y_pred_optimal, zero_division=0)
    
    logger.info(f"\nTest Metrics (optimal threshold={optimal_threshold:.2f}):")
    logger.info(f"  Accuracy: {test_accuracy_optimal:.4f} ({test_accuracy_optimal*100:.2f}%)")
    logger.info(f"  Precision: {test_precision_optimal:.4f} ({test_precision_optimal*100:.2f}%)")
    logger.info(f"  Recall: {test_recall_optimal:.4f} ({test_recall_optimal*100:.2f}%)")
    logger.info(f"  F1-Score: {optimal_f1:.4f}")
    
    return metrics, y_pred_proba, y_pred_optimal


def plot_training_history(history, save_path=None):
    """Plot training history (loss and accuracy)."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history.history['loss'], label='Training Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy
    axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0, 1].set_title('Model Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Precision
    if 'precision' in history.history:
        axes[1, 0].plot(history.history['precision'], label='Training Precision')
        axes[1, 0].plot(history.history['val_precision'], label='Validation Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Recall
    if 'recall' in history.history:
        axes[1, 1].plot(history.history['recall'], label='Training Recall')
        axes[1, 1].plot(history.history['val_recall'], label='Validation Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Training history plot saved: {save_path}")
    else:
        plt.savefig(MODEL_SAVE_DIR / 'training_history.png', dpi=150, bbox_inches='tight')
        logger.info(f"Training history plot saved: {MODEL_SAVE_DIR / 'training_history.png'}")
    
    plt.close()


def plot_confusion_matrix(cm, save_path=None):
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Loss', 'Profit'],
                yticklabels=['Loss', 'Profit'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.savefig(MODEL_SAVE_DIR / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    
    plt.close()


def plot_roc_curve(y_test, y_pred_proba, save_path=None):
    """Plot ROC curve."""
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.savefig(MODEL_SAVE_DIR / 'roc_curve.png', dpi=150, bbox_inches='tight')
    
    plt.close()


def save_model_artifacts(model, scaler, feature_names, metrics, file_info):
    """Save model, scaler, and metadata for future use."""
    # Save model
    model_path = MODEL_SAVE_DIR / 'trading_predictor_model.h5'
    model.save(model_path)
    logger.info(f"Model saved: {model_path}")
    
    # Save scaler (using pickle)
    import pickle
    scaler_path = MODEL_SAVE_DIR / 'scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"Scaler saved: {scaler_path}")
    
    # Save feature names
    feature_names_path = MODEL_SAVE_DIR / 'feature_names.txt'
    with open(feature_names_path, 'w') as f:
        for name in feature_names:
            f.write(f"{name}\n")
    logger.info(f"Feature names saved: {feature_names_path}")
    
    # Save metadata
    metadata = {
        'metrics': metrics,
        'file_info': file_info,
        'feature_count': len(feature_names),
        'feature_names': feature_names
    }
    
    import json
    metadata_path = MODEL_SAVE_DIR / 'model_metadata.json'
    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    with open(metadata_path, 'w') as f:
        json.dump(convert_to_serializable(metadata), f, indent=2)
    logger.info(f"Metadata saved: {metadata_path}")


def main():
    """Main execution function."""
    logger.info("="*60)
    logger.info("Deep Learning Model Training for Trading Signal Prediction")
    logger.info("="*60)
    
    # Load training data
    df, file_info = load_training_data()
    
    if df is None:
        logger.error("Failed to load training data. Exiting...")
        return
    
    # Preprocess data
    (X_train, X_val, X_test, 
     y_train, y_val, y_test, 
     scaler, feature_names) = preprocess_data(df)
    
    # Create model
    model = create_model(input_dim=X_train.shape[1])
    
    # Train model
    history = train_model(model, X_train, y_train, X_val, y_val)
    
    # Evaluate model
    metrics, y_pred_proba, y_pred = evaluate_model(model, X_test, y_test, scaler, feature_names)
    
    # Plot results
    plot_training_history(history)
    plot_confusion_matrix(metrics['confusion_matrix'])
    plot_roc_curve(y_test, y_pred_proba)
    
    # Save model artifacts
    save_model_artifacts(model, scaler, feature_names, metrics, file_info)
    
    logger.info("="*60)
    logger.info("Training pipeline complete!")
    logger.info(f"Model and artifacts saved to: {MODEL_SAVE_DIR}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
