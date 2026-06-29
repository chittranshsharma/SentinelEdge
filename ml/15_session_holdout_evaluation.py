#!/usr/bin/env python3
import numpy as np
import warnings
from pathlib import Path
import json

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
REPORTS_DIR = ML_DIR.parent / "reports"
MODELS_DIR = ML_DIR / "models"

LABELS = {0: "stationary", 1: "movement", 2: "rotation", 3: "shake"}
NUM_CLASSES = 4

def build_dense4(input_dim):
    inputs = keras.Input(shape=(input_dim,))
    outputs = keras.layers.Dense(NUM_CLASSES, activation='softmax')(inputs)
    model = keras.Model(inputs, outputs, name='SentinelEdge_RealClassifier')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def format_cm(cm):
    header = "  " + " " * 18 + "".join([f"{LABELS[i][:10]:>11}" for i in range(4)])
    out = header + "\n"
    for i, row in enumerate(cm):
        out += f"  {LABELS[i]:<18}" + "".join([f"{v:>11}" for v in row]) + "\n"
    return out

def evaluate_model(name, y_train, train_pred, y_test, test_pred, f):
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    p, r, f1, _ = precision_recall_fscore_support(y_test, test_pred, average=None, zero_division=0)
    cm = confusion_matrix(y_test, test_pred)
    
    out = f"### {name}\n"
    out += f"- **Train Accuracy (Session 1):** {train_acc*100:.2f}%\n"
    out += f"- **Test Accuracy (Session 2):**  {test_acc*100:.2f}%\n\n"
    
    out += "**Class-wise Metrics (Test):**\n"
    for i in range(NUM_CLASSES):
        class_name = LABELS.get(i, "Unknown")
        if i < len(p):
            out += f"- `{class_name:10s}` Precision: {p[i]:.2f}, Recall: {r[i]:.2f}, F1: {f1[i]:.2f}\n"
            
    out += "\n**Confusion Matrix (Test):**\n```text\n"
    out += format_cm(cm)
    out += "```\n\n"
    
    print(out)
    f.write(out)
    return test_acc

def main():
    s1_path = DATA_DIR / "session1_features.npz"
    s2_path = DATA_DIR / "session2_features.npz"
    
    if not s1_path.exists() or not s2_path.exists():
        print("Error: Missing session1 or session2 feature files.")
        return
        
    train_data = np.load(s1_path)
    test_data = np.load(s2_path)
    
    X_train, y_train = train_data['X'], train_data['y']
    X_test, y_test = test_data['X'], test_data['y']
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    report_path = REPORTS_DIR / "session_holdout_report.md"
    
    print(f"Evaluating {len(X_train)} Train windows (S1), {len(X_test)} Test windows (S2)...\n")
    
    with open(report_path, "w") as f:
        f.write("# Session Holdout Evaluation\n\n")
        f.write(f"- **Train**: Session 1 ({len(X_train)} windows)\n")
        f.write(f"- **Test**: Session 2 ({len(X_test)} windows)\n\n")
        
        models_results = {}
        
        # 1. Logistic Regression
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(X_train_scaled, y_train)
        acc = evaluate_model("Logistic Regression", y_train, lr.predict(X_train_scaled), y_test, lr.predict(X_test_scaled), f)
        models_results["Logistic Regression"] = acc
        
        # 2. Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train_scaled, y_train)
        acc = evaluate_model("Random Forest", y_train, rf.predict(X_train_scaled), y_test, rf.predict(X_test_scaled), f)
        models_results["Random Forest"] = acc
        
        # 3. Gradient Boosting
        gb = GradientBoostingClassifier(random_state=42)
        gb.fit(X_train_scaled, y_train)
        acc = evaluate_model("Gradient Boosting", y_train, gb.predict(X_train_scaled), y_test, gb.predict(X_test_scaled), f)
        models_results["Gradient Boosting"] = acc
        
        # 4. SVM (RBF)
        svm = SVC(kernel='rbf', random_state=42)
        svm.fit(X_train_scaled, y_train)
        acc = evaluate_model("SVM (RBF)", y_train, svm.predict(X_train_scaled), y_test, svm.predict(X_test_scaled), f)
        models_results["SVM (RBF)"] = acc
        
        # 5. Dense(4) Softmax
        dense4 = build_dense4(X_train_scaled.shape[1])
        # Use 1000 epochs to ensure full convergence like LR
        tf.random.set_seed(42)
        np.random.seed(42)
        dense4.fit(X_train_scaled, y_train, epochs=1000, batch_size=32, verbose=0)
        
        d4_train_pred = np.argmax(dense4.predict(X_train_scaled, verbose=0), axis=1)
        d4_test_pred = np.argmax(dense4.predict(X_test_scaled, verbose=0), axis=1)
        d4_acc = evaluate_model("Dense(4)", y_train, d4_train_pred, y_test, d4_test_pred, f)
        models_results["Dense(4)"] = d4_acc
        
        # Check deployment
        best_classical = max([v for k,v in models_results.items() if k != "Dense(4)"])
        
        f.write("## Verdict\n")
        print("\n==============================")
        f.write(f"- Best Classical Model Accuracy: {best_classical*100:.2f}%\n")
        f.write(f"- Dense(4) Accuracy: {d4_acc*100:.2f}%\n\n")
        print(f"Best Classical Model Accuracy: {best_classical*100:.2f}%")
        print(f"Dense(4) Accuracy: {d4_acc*100:.2f}%")
        
        if d4_acc >= (best_classical - 0.02) and d4_acc >= 0.90:
            msg = "[SUCCESS] Dense(4) is within 2% of the best classical model and above 90%. Saving as the final deployment candidate."
            print(msg)
            f.write(msg + "\n")
            
            # Save the new model
            keras_path = MODELS_DIR / "real_fault_classifier.keras"
            dense4.save(keras_path)
            
            # Save scaler params
            scaler_params = {
                'mean': scaler.mean_.tolist(),
                'scale': scaler.scale_.tolist()
            }
            with open(MODELS_DIR / "real_scaler_params.json", 'w') as sf:
                json.dump(scaler_params, sf)
                
            print("Successfully updated real_fault_classifier.keras and scaler params.")
            print("Next: Run `python 09_convert_real_model.py` to regenerate firmware headers.")
        else:
            msg = "[FAIL] Dense(4) did not meet the deployment threshold (either < 90% or > 2% worse than classical)."
            print(msg)
            f.write(msg + "\n")

if __name__ == "__main__":
    main()
