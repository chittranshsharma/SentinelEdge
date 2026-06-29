#!/usr/bin/env python3
import numpy as np
import json
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import tensorflow as tf
from tensorflow import keras

ML_DIR = Path(__file__).parent
DATA_DIR = ML_DIR / "data"
MODELS_DIR = ML_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

FEATURE_COUNT = 42
NUM_CLASSES = 4
LABELS = {0: "stationary", 1: "movement", 2: "rotation", 3: "shake"}

def build_nn(input_dim=FEATURE_COUNT, num_classes=NUM_CLASSES):
    inputs = keras.Input(shape=(input_dim,))
    x = keras.layers.Dense(32, activation='relu')(inputs)
    x = keras.layers.Dense(16, activation='relu')(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    return keras.Model(inputs, outputs, name='SentinelEdge_RealClassifier')

def main():
    train_features_path = DATA_DIR / "real_features_train.npz"
    test_features_path = DATA_DIR / "real_features_test.npz"
    
    if not train_features_path.exists() or not test_features_path.exists():
        print(f"Error: missing train or test features.")
        return

    train_data = np.load(train_features_path)
    test_data = np.load(test_features_path)
    
    X_train, y_train = train_data['X'], train_data['y']
    X_test, y_test = test_data['X'], test_data['y']
    
    print(f"Loaded {len(X_train)} Train windows, {len(X_test)} Test windows.")
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler params for C++ quantization
    scaler_params = {
        'mean': scaler.mean_.tolist(),
        'scale': scaler.scale_.tolist()
    }
    with open(MODELS_DIR / "real_scaler_params.json", 'w') as f:
        json.dump(scaler_params, f)
        
    print("\n--- Training Random Forest Baseline ---")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    rf_pred = rf.predict(X_test_scaled)
    rf_acc = accuracy_score(y_test, rf_pred)
    print(f"Random Forest Accuracy: {rf_acc*100:.2f}%")
    
    print("\n--- Training Dense Neural Network ---")
    model = build_nn()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Train
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    history = model.fit(X_train_scaled, y_train, epochs=200, validation_split=0.2, 
                        callbacks=[early_stop], verbose=1)
    
    # Evaluate
    print("\nEvaluating Dense NN...")
    train_loss, train_acc = model.evaluate(X_train_scaled, y_train, verbose=0)
    test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Dense NN Train Accuracy: {train_acc*100:.2f}%")
    print(f"Dense NN Test Accuracy:  {test_acc*100:.2f}%")
    
    y_pred_prob = model.predict(X_test_scaled, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, average=None)
    
    print("\nClass-wise metrics:")
    for i in range(NUM_CLASSES):
        class_name = LABELS.get(i, "Unknown")
        # Ensure we have metrics for this class
        if i < len(p):
            print(f"  {class_name:12s} - Precision: {p[i]:.2f}, Recall: {r[i]:.2f}, F1: {f[i]:.2f}")
        
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save Keras Model
    keras_path = MODELS_DIR / "real_fault_classifier.keras"
    model.save(keras_path)
    print(f"\nSaved best model to {keras_path}")
    
if __name__ == '__main__':
    main()
