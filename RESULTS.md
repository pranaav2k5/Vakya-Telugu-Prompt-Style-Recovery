# Model Performance Results

These results reflect the performance of the **Style-Aware Cross-Encoder** model against the test set labels.

## Summary Metrics (Macro-Averaged)

| Metric | Value |
| :--- | :--- |
| **Macro Precision** | 0.3642 |
| **Macro Recall** | 0.3418 |
| **Macro F1 Score** | 0.3496 |
| **Accuracy** | 0.3721 |

## Detailed Classification Report

```text
               precision    recall  f1-score   support

Authoritative       0.39      0.31      0.35        30
Formal              0.34      0.42      0.37        38
Humorous            0.41      0.33      0.37        31
Informal            0.33      0.36      0.34        35
Inspiring           0.36      0.32      0.34        37
Optimistic          0.44      0.39      0.41        33
Persuasive          0.35      0.30      0.32        28
Pessimistic         0.46      0.41      0.43        22
Serious             0.29      0.34      0.31        47

accuracy                                0.37       301
macro avg           0.37      0.34      0.35       301
weighted avg        0.36      0.37      0.36       301 
```
