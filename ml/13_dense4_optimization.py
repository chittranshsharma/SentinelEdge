#!/usr/bin/env python3
import numpy as np
import warnings
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow import keras

warnings.filterwarnings('ignore')

ML_DIR = Path(__file__).parent
DATA_DIR = ML_DIR / "data"

LABELS = {0: "stationary", 1: "movement", 2: "rotation", 3: "shake"}
NUM_CLASSES = 4

def build_model(input_dim):
    inputs = keras.Input(shape=(input_dim,))
    outputs = keras.layers.Dense(NUM_CLASSES, activation='softmax')(inputs)
    return keras.Model(inputs, outputs)

def main():
    train_data = np.load(DATA_DIR / "real_features_train.npz")
    test_data = np.load(DATA_DIR / "real_features_test.npz")
    
    X_train, y_train = train_data['X'], train_data['y']
    X_test, y_test = test_data['X'], test_data['y']
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    configs = [
        {"name": "Adam, lr=0.01", "opt": keras.optimizers.Adam(learning_rate=0.01)},
        {"name": "Adam, lr=0.001", "opt": keras.optimizers.Adam(learning_rate=0.001)},
        {"name": "SGD, lr=0.1, mom=0.9", "opt": keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)},
        {"name": "SGD, lr=0.01, mom=0.9", "opt": keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)},
    ]
    
    epochs = 1000
    batch_size = 32
    
    print(f"Running optimization investigation for Dense(4) model over {epochs} epochs (no early stopping)...")
    
    best_acc = 0
    best_name = ""
    
    for config in configs:
        print(f"\n================================================")
        print(f"Config: {config['name']}")
        
        # We use a fixed seed for reproducibility across trials
        tf.random.set_seed(42)
        np.random.seed(42)
        
        model = build_model(X_train_scaled.shape[1])
        model.compile(optimizer=config['opt'], loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        # Train without early stopping
        model.fit(X_train_scaled, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        
        train_pred = np.argmax(model.predict(X_train_scaled, verbose=0), axis=1)
        test_pred = np.argmax(model.predict(X_test_scaled, verbose=0), axis=1)
        
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        print(f"Train Accuracy: {train_acc*100:.2f}%")
        print(f"Test Accuracy : {test_acc*100:.2f}%")
        
        cm = confusion_matrix(y_test, test_pred)
        header = "  " + " " * 18 + "".join([f"{LABELS[i][:10]:>11}" for i in range(4)])
        print(header)
        for i, row in enumerate(cm):
            print(f"  {LABELS[i]:<18}" + "".join([f"{v:>11}" for v in row]))
            
        if test_acc > best_acc:
            best_acc = test_acc
            best_name = config['name']
            
    print(f"\nBest Config: {best_name} with {best_acc*100:.2f}% Test Accuracy")

if __name__ == "__main__":
    main()
