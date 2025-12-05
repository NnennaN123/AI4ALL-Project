import json
import matplotlib.pyplot as plt
import numpy as np

# Load metrics from all three models
with open('logistic_regression/model_metrics.json', 'r') as f:
    lr_metrics = json.load(f)

with open('xgboost/xgboost_model_metrics.json', 'r') as f:
    xgb_metrics = json.load(f)

with open('random_forest/random_forest_model_metrics.json', 'r') as f:
    rf_metrics = json.load(f)

# Extract key metrics
models = ['Logistic Regression', 'XGBoost', 'Random Forest']
metrics_data = [lr_metrics, xgb_metrics, rf_metrics]

# Extract metrics
roc_auc = [m['roc_auc'] for m in metrics_data]
accuracy = [m['classification_report']['accuracy'] for m in metrics_data]
weighted_precision = [m['classification_report']['weighted avg']['precision'] for m in metrics_data]
weighted_recall = [m['classification_report']['weighted avg']['recall'] for m in metrics_data]
weighted_f1 = [m['classification_report']['weighted avg']['f1-score'] for m in metrics_data]

# Create figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

# Define colors for each model
colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']

# Plot ROC AUC
axes[0, 0].bar(models, roc_auc, color=colors)
axes[0, 0].set_title('ROC AUC Score', fontweight='bold')
axes[0, 0].set_ylabel('Score')
axes[0, 0].set_ylim([0, 1])
axes[0, 0].grid(axis='y', alpha=0.3)
for i, v in enumerate(roc_auc):
    axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

# Plot Accuracy
axes[0, 1].bar(models, accuracy, color=colors)
axes[0, 1].set_title('Accuracy', fontweight='bold')
axes[0, 1].set_ylabel('Score')
axes[0, 1].set_ylim([0, 1])
axes[0, 1].grid(axis='y', alpha=0.3)
for i, v in enumerate(accuracy):
    axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

# Plot Weighted Precision
axes[0, 2].bar(models, weighted_precision, color=colors)
axes[0, 2].set_title('Weighted Precision', fontweight='bold')
axes[0, 2].set_ylabel('Score')
axes[0, 2].set_ylim([0, 1])
axes[0, 2].grid(axis='y', alpha=0.3)
for i, v in enumerate(weighted_precision):
    axes[0, 2].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

# Plot Weighted Recall
axes[1, 0].bar(models, weighted_recall, color=colors)
axes[1, 0].set_title('Weighted Recall', fontweight='bold')
axes[1, 0].set_ylabel('Score')
axes[1, 0].set_ylim([0, 1])
axes[1, 0].grid(axis='y', alpha=0.3)
for i, v in enumerate(weighted_recall):
    axes[1, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

# Plot Weighted F1-Score
axes[1, 1].bar(models, weighted_f1, color=colors)
axes[1, 1].set_title('Weighted F1-Score', fontweight='bold')
axes[1, 1].set_ylabel('Score')
axes[1, 1].set_ylim([0, 1])
axes[1, 1].grid(axis='y', alpha=0.3)
for i, v in enumerate(weighted_f1):
    axes[1, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

# Combined comparison bar chart
x = np.arange(len(models))
width = 0.15
axes[1, 2].bar(x - 2*width, roc_auc, width, label='ROC AUC', color=colors[0], alpha=0.8)
axes[1, 2].bar(x - width, accuracy, width, label='Accuracy', color=colors[1], alpha=0.8)
axes[1, 2].bar(x, weighted_precision, width, label='Precision', color=colors[2], alpha=0.8)
axes[1, 2].bar(x + width, weighted_recall, width, label='Recall', color='#FFA07A', alpha=0.8)
axes[1, 2].bar(x + 2*width, weighted_f1, width, label='F1-Score', color='#98D8C8', alpha=0.8)
axes[1, 2].set_title('All Metrics Comparison', fontweight='bold')
axes[1, 2].set_ylabel('Score')
axes[1, 2].set_xticks(x)
axes[1, 2].set_xticklabels(models)
axes[1, 2].set_ylim([0, 1])
axes[1, 2].legend(loc='upper left', fontsize=8)
axes[1, 2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("Comparison chart saved as 'model_comparison.png'")
plt.show()

