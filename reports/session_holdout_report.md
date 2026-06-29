# Session Holdout Evaluation

- **Train**: Session 1 (413 windows)
- **Test**: Session 2 (339 windows)

### Logistic Regression
- **Train Accuracy (Session 1):** 100.00%
- **Test Accuracy (Session 2):**  98.82%

**Class-wise Metrics (Test):**
- `stationary` Precision: 0.99, Recall: 1.00, F1: 0.99
- `movement  ` Precision: 0.96, Recall: 1.00, F1: 0.98
- `rotation  ` Precision: 1.00, Recall: 0.96, F1: 0.98
- `shake     ` Precision: 1.00, Recall: 1.00, F1: 1.00

**Confusion Matrix (Test):**
```text
                     stationary   movement   rotation      shake
  stationary                 85          0          0          0
  movement                    0         75          0          0
  rotation                    1          3         96          0
  shake                       0          0          0         79
```

### Random Forest
- **Train Accuracy (Session 1):** 100.00%
- **Test Accuracy (Session 2):**  98.82%

**Class-wise Metrics (Test):**
- `stationary` Precision: 0.99, Recall: 1.00, F1: 0.99
- `movement  ` Precision: 0.96, Recall: 1.00, F1: 0.98
- `rotation  ` Precision: 1.00, Recall: 0.96, F1: 0.98
- `shake     ` Precision: 1.00, Recall: 1.00, F1: 1.00

**Confusion Matrix (Test):**
```text
                     stationary   movement   rotation      shake
  stationary                 85          0          0          0
  movement                    0         75          0          0
  rotation                    1          3         96          0
  shake                       0          0          0         79
```

### Gradient Boosting
- **Train Accuracy (Session 1):** 100.00%
- **Test Accuracy (Session 2):**  97.94%

**Class-wise Metrics (Test):**
- `stationary` Precision: 0.98, Recall: 1.00, F1: 0.99
- `movement  ` Precision: 0.96, Recall: 1.00, F1: 0.98
- `rotation  ` Precision: 0.98, Recall: 0.95, F1: 0.96
- `shake     ` Precision: 1.00, Recall: 0.97, F1: 0.99

**Confusion Matrix (Test):**
```text
                     stationary   movement   rotation      shake
  stationary                 85          0          0          0
  movement                    0         75          0          0
  rotation                    2          3         95          0
  shake                       0          0          2         77
```

### SVM (RBF)
- **Train Accuracy (Session 1):** 99.76%
- **Test Accuracy (Session 2):**  94.10%

**Class-wise Metrics (Test):**
- `stationary` Precision: 0.99, Recall: 1.00, F1: 0.99
- `movement  ` Precision: 0.96, Recall: 1.00, F1: 0.98
- `rotation  ` Precision: 0.86, Recall: 0.96, F1: 0.91
- `shake     ` Precision: 1.00, Recall: 0.80, F1: 0.89

**Confusion Matrix (Test):**
```text
                     stationary   movement   rotation      shake
  stationary                 85          0          0          0
  movement                    0         75          0          0
  rotation                    1          3         96          0
  shake                       0          0         16         63
```

### Dense(4)
- **Train Accuracy (Session 1):** 100.00%
- **Test Accuracy (Session 2):**  98.23%

**Class-wise Metrics (Test):**
- `stationary` Precision: 0.99, Recall: 1.00, F1: 0.99
- `movement  ` Precision: 0.94, Recall: 1.00, F1: 0.97
- `rotation  ` Precision: 1.00, Recall: 0.94, F1: 0.97
- `shake     ` Precision: 1.00, Recall: 1.00, F1: 1.00

**Confusion Matrix (Test):**
```text
                     stationary   movement   rotation      shake
  stationary                 85          0          0          0
  movement                    0         75          0          0
  rotation                    1          5         94          0
  shake                       0          0          0         79
```

## Verdict
- Best Classical Model Accuracy: 98.82%
- Dense(4) Accuracy: 98.23%

[SUCCESS] Dense(4) is within 2% of the best classical model and above 90%. Saving as the final deployment candidate.
