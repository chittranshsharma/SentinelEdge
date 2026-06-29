#!/usr/bin/env python3
import numpy as np
import warnings
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import tensorflow as tf
from tensorflow import keras

warnings.filterwarnings('ignore')

ML_DIR = Path(__file__).parent
DATA_DIR = ML_DIR / "data"

LABELS = {0: "stationary", 1: "movement", 2: "rotation", 3: "shake"}
NUM_CLASSES = 4

def build_nn_current(input_dim):
    inputs = keras.Input(shape=(input_dim,))
    x = keras.layers.Dense(32, activation='relu')(inputs)
    x = keras.layers.Dense(16, activation='relu')(x)
    outputs = keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_nn_small_16(input_dim):
    inputs = keras.Input(shape=(input_dim,))
    x = keras.layers.Dense(16, activation='relu')(inputs)
    outputs = keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_nn_small_8(input_dim):
    inputs = keras.Input(shape=(input_dim,))
    x = keras.layers.Dense(8, activation='relu')(inputs)
    outputs = keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def evaluate_model(name, y_train, train_pred, y_test, test_pred):
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    p, r, f, _ = precision_recall_fscore_support(y_test, test_pred, average=None, zero_division=0)
    cm = confusion_matrix(y_test, test_pred)
    
    print(f"\n{'='*60}")
    print(f"Model: {name}")
    print(f"Train Accuracy : {train_acc*100:.2f}%")
    print(f"Test Accuracy  : {test_acc*100:.2f}%")
    
    print("\nClass-wise metrics (Test):")
    for i in range(NUM_CLASSES):
        class_name = LABELS.get(i, "Unknown")
        if i < len(p):
            print(f"  {class_name:12s} - Precision: {p[i]:.2f}, Recall: {r[i]:.2f}, F1: {f[i]:.2f}")
            
    print("\nConfusion Matrix (Test):")
    header = "  " + " " * 18 + "".join([f"{LABELS[i][:10]:>11}" for i in range(4)])
    print(header)
    for i, row in enumerate(cm):
        print(f"  {LABELS[i]:<18}" + "".join([f"{v:>11}" for v in row]))

def main():
    train_data = np.load(DATA_DIR / "real_features_train.npz")
    test_data = np.load(DATA_DIR / "real_features_test.npz")
    
    X_train, y_train = train_data['X'], train_data['y']
    X_test, y_test = test_data['X'], test_data['y']
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Running evaluation on {len(X_train)} train windows, {len(X_test)} test windows...")
    
    # 1. Logistic Regression
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    evaluate_model("1. Logistic Regression", y_train, lr.predict(X_train_scaled), y_test, lr.predict(X_test_scaled))
    
    # 2. SVM (RBF)
    svm = SVC(kernel='rbf', random_state=42)
    svm.fit(X_train_scaled, y_train)
    evaluate_model("2. SVM (RBF)", y_train, svm.predict(X_train_scaled), y_test, svm.predict(X_test_scaled))
    
    # 3. Gradient Boosting
    gb = GradientBoostingClassifier(random_state=42)
    gb.fit(X_train_scaled, y_train)
    evaluate_model("3. Gradient Boosting", y_train, gb.predict(X_train_scaled), y_test, gb.predict(X_test_scaled))
    
    # 4. Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    evaluate_model("4. Random Forest", y_train, rf.predict(X_train_scaled), y_test, rf.predict(X_test_scaled))
    
    # NN params
    epochs = 400
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
    
    # 5. Current Dense NN (32 -> 16 -> 4)
    nn_curr = build_nn_current(X_train_scaled.shape[1])
    nn_curr.fit(X_train_scaled, y_train, epochs=epochs, validation_split=0.2, callbacks=[early_stop], verbose=0)
    train_pred_5 = np.argmax(nn_curr.predict(X_train_scaled, verbose=0), axis=1)
    test_pred_5 = np.argmax(nn_curr.predict(X_test_scaled, verbose=0), axis=1)
    evaluate_model("5. Current Dense NN (32 -> 16 -> 4)", y_train, train_pred_5, y_test, test_pred_5)
    
    # 6. Smaller Dense NN (16 -> 4)
    nn_16 = build_nn_small_16(X_train_scaled.shape[1])
    nn_16.fit(X_train_scaled, y_train, epochs=epochs, validation_split=0.2, callbacks=[early_stop], verbose=0)
    train_pred_6 = np.argmax(nn_16.predict(X_train_scaled, verbose=0), axis=1)
    test_pred_6 = np.argmax(nn_16.predict(X_test_scaled, verbose=0), axis=1)
    evaluate_model("6. Smaller Dense NN (16 -> 4)", y_train, train_pred_6, y_test, test_pred_6)
    
    # 7. Smaller Dense NN (8 -> 4)
    nn_8 = build_nn_small_8(X_train_scaled.shape[1])
    nn_8.fit(X_train_scaled, y_train, epochs=epochs, validation_split=0.2, callbacks=[early_stop], verbose=0)
    train_pred_7 = np.argmax(nn_8.predict(X_train_scaled, verbose=0), axis=1)
    test_pred_7 = np.argmax(nn_8.predict(X_test_scaled, verbose=0), axis=1)
    evaluate_model("7. Smaller Dense NN (8 -> 4)", y_train, train_pred_7, y_test, test_pred_7)

if __name__ == "__main__":
    main()
