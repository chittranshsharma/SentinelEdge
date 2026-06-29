#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

ML_DIR = Path(__file__).parent
DATA_DIR = ML_DIR / "data"
REPORTS_DIR = ML_DIR.parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

LABELS = {0: "stationary", 1: "movement", 2: "rotation", 3: "shake"}
AXES = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
FEATURE_NAMES_PER_AXIS = [
    'mean', 'std', 'variance', 'rms', 'peak_to_peak', 'dominant_freq_bin', 'spectral_energy'
]
ALL_FEATURE_NAMES = [f"{ax}_{f}" for ax in AXES for f in FEATURE_NAMES_PER_AXIS]

def main():
    print("Loading datasets...")
    train_data = np.load(DATA_DIR / "real_features_train.npz")
    test_data = np.load(DATA_DIR / "real_features_test.npz")
    
    X_train, y_train = train_data['X'], train_data['y']
    X_test, y_test = test_data['X'], test_data['y']
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n--- 1. Random Forest Accuracy & Confusion Matrix ---")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    
    rf_train_pred = rf.predict(X_train_scaled)
    rf_test_pred = rf.predict(X_test_scaled)
    
    rf_train_acc = accuracy_score(y_train, rf_train_pred)
    rf_test_acc = accuracy_score(y_test, rf_test_pred)
    
    print(f"RF Train Accuracy: {rf_train_acc*100:.2f}%")
    print(f"RF Test Accuracy:  {rf_test_acc*100:.2f}%")
    print("\nRF Test Confusion Matrix:")
    
    cm = confusion_matrix(y_test, rf_test_pred)
    header = "  " + " " * 18 + "".join([f"{LABELS[i][:10]:>11}" for i in range(4)])
    print(header)
    for i, row in enumerate(cm):
        print(f"  {LABELS[i]:<18}" + "".join([f"{v:>11}" for v in row]))
        
    print("\n--- 2. Feature Importance ---")
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    top_n = 15
    plt.figure(figsize=(10, 6))
    plt.title("Top 15 Feature Importances (Random Forest)")
    plt.bar(range(top_n), importances[indices][:top_n], align="center")
    plt.xticks(range(top_n), [ALL_FEATURE_NAMES[i] for i in indices[:top_n]], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "feature_importance.png", dpi=300)
    print("Saved reports/feature_importance.png")
    
    print("\n--- 3. PCA & t-SNE Visualization ---")
    # Combine for visualization
    X_all = np.vstack((X_train_scaled, X_test_scaled))
    y_all = np.concatenate((y_train, y_test))
    
    # Only pick shake and rotation for a clear comparison? No, let's plot all.
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_all)
    
    # tsne takes a bit longer but with 400 samples it's instant
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_all)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = ['blue', 'green', 'orange', 'red']
    for i in range(4):
        mask = (y_all == i)
        ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], c=colors[i], label=LABELS[i], alpha=0.7, edgecolors='k')
        ax2.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=colors[i], label=LABELS[i], alpha=0.7, edgecolors='k')
        
    ax1.set_title("PCA Visualization")
    ax1.set_xlabel("Principal Component 1")
    ax1.set_ylabel("Principal Component 2")
    ax1.legend()
    
    ax2.set_title("t-SNE Visualization")
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "tsne_pca.png", dpi=300)
    print("Saved reports/tsne_pca.png")
    
if __name__ == "__main__":
    main()
