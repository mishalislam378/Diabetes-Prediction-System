#visualization.py
"""

Visualization utilities for the Diabetes Prediction System
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from constants import COLORS, TARGET_COL


def plot_confusion_matrix(cm):
    """Plot confusion matrix heatmap"""
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'])
    ax.set_title('Confusion Matrix', fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.tight_layout()
    return fig


def plot_feature_importance(feat_df):  #Positive coefficient increases diabetes probability
    """Plot feature importance bar chart"""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [COLORS['red'] if v < 0 else COLORS['green'] for v in feat_df['Coefficient']]
    ax.barh(feat_df['Feature'], feat_df['Coefficient'], color=colors)
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_xlabel('Coefficient Value')
    ax.set_title('Feature Importance (LR Coefficients)', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    return fig


def plot_class_distribution(df):
    """Plot pie chart of class distribution"""
    counts = df[TARGET_COL].value_counts()
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(counts, labels=['No Diabetes', 'Diabetes'], autopct='%1.1f%%',
           colors=[COLORS['green'], COLORS['red']], startangle=90)
    ax.set_title('Class Distribution', fontweight='bold')
    plt.tight_layout()
    return fig


def plot_bmi_distribution(df):
    """Plot histogram of BMI distribution by outcome"""
    fig, ax = plt.subplots(figsize=(5, 3))
    for label, color, name in [(0, COLORS['green'], 'No Diabetes'),
                                (1, COLORS['red'], 'Diabetes')]:
        ax.hist(df[df[TARGET_COL] == label]['bmi'], bins=40,
                alpha=0.7, color=color, label=name, edgecolor='white')
    ax.set_xlabel('BMI')
    ax.set_ylabel('Count')
    ax.set_title('BMI Distribution by Outcome')
    ax.legend()
    plt.tight_layout()
    return fig


def plot_hba1c_distribution(df):
    """Plot histogram of HbA1c distribution by outcome"""
    fig, ax = plt.subplots(figsize=(8, 3))
    for label, color, name in [(0, COLORS['green'], 'No Diabetes'),
                                (1, COLORS['red'], 'Diabetes')]:
        ax.hist(df[df[TARGET_COL] == label]['HbA1c_level'], bins=30,
                alpha=0.7, color=color, label=name, edgecolor='white')
    ax.set_xlabel('HbA1c Level')
    ax.set_ylabel('Count')
    ax.legend()
    plt.tight_layout()
    return fig


def plot_runs_bargraph(metrics_runs, n_runs=5):
    """Plot grouped bar chart for multiple evaluation runs"""
    metrics_keys = ['accuracy', 'precision', 'recall', 'f1']
    metrics_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    bar_colors = [COLORS['blue'], COLORS['green'], COLORS['orange'], COLORS['purple']]
    
    x = np.arange(n_runs)
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for i, (key, label, color) in enumerate(zip(metrics_keys, metrics_labels, bar_colors)):
        values = metrics_runs[key]
        bars = ax.bar(x + i * width, values, width, label=label, color=color, alpha=0.85)
        
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f'{val*100:.1f}%',
                    ha='center', va='bottom', fontsize=7, fontweight='bold')
    
    ax.set_xlabel('Run Number')
    ax.set_ylabel('Score (%)')
    ax.set_title('Metric Scores Across 5 Runs', fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f'Run {i+1}' for i in range(n_runs)])
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f'{val*100:.0f}%'))
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    return fig


def plot_prediction_probability(probabilities):
    """Plot prediction probability bar chart"""
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(['No Diabetes', 'Diabetes'], probabilities,
                  color=[COLORS['green'], COLORS['red']], alpha=0.85)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Probability')
    ax.set_title('Prediction Probability', fontweight='bold')
    
    for bar, p in zip(bars, probabilities):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f'{p*100:.1f}%', ha='center', fontweight='bold')
    
    plt.tight_layout()
    return fig