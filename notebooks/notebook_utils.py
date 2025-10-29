import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import numpy as np


def setup_plotting():
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10


def plot_metric_comparison(metrics_dict: Dict[str, float], title: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    models = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    ax.bar(models, values)
    ax.set_title(title)
    ax.set_ylabel('Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig


def plot_feature_distributions(df: pd.DataFrame, features: List[str]):
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()
    
    for idx, feature in enumerate(features):
        axes[idx].hist(df[feature], bins=50, edgecolor='black')
        axes[idx].set_title(f'{feature} Distribution')
        axes[idx].set_xlabel(feature)
        axes[idx].set_ylabel('Frequency')
    
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig


def plot_correlation_matrix(df: pd.DataFrame, figsize=(12, 10)):
    corr = df.corr()
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, annot=False, cmap='coolwarm', center=0,
                square=True, ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_title('Feature Correlation Matrix')
    plt.tight_layout()
    return fig


def plot_roc_curve(y_true, y_pred_proba, model_name='Model'):
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig


def plot_pr_curve(y_true, y_pred_proba, model_name='Model'):
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    ap = average_precision_score(y_true, y_pred_proba)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, label=f'{model_name} (AP = {ap:.3f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig


def plot_confusion_matrix(y_true, y_pred, labels=None):
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    if labels:
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
    plt.tight_layout()
    return fig