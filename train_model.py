"""
Model Training & Evaluation Module for Student Placement Prediction System
Trains 5 classification algorithms, evaluates performance across multiple metrics,
generates professional visualizations, and saves models using joblib.
"""

import os
import json
import shutil
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Headless backend for web/server environments
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)
from sklearn.inspection import permutation_importance
import joblib

from preprocessing import prepare_dataset

# Configuration Paths
MODELS_DIR = 'models'
OUTPUTS_DIR = 'outputs'
CONF_MATRIX_DIR = os.path.join(OUTPUTS_DIR, 'confusion_matrices')
STATIC_IMAGES_DIR = os.path.join('static', 'images')

for d in [MODELS_DIR, OUTPUTS_DIR, CONF_MATRIX_DIR, STATIC_IMAGES_DIR]:
    os.makedirs(d, exist_ok=True)

# Visual styling setup
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 300


def get_models():
    """Returns a dictionary of initialized candidate models."""
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, C=1.0),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_split=5, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5, weights='distance'),
        'Support Vector Machine': SVC(probability=True, kernel='rbf', C=1.0, random_state=42)
    }


def train_and_evaluate_all():
    """
    Main training and evaluation pipeline.
    """
    print("[1/5] Loading and preprocessing dataset...")
    data = prepare_dataset()
    
    df = data['df']
    target_col = data['target_col']
    X_train = data['X_train_scaled']
    X_test = data['X_test_scaled']
    y_train = data['y_train']
    y_test = data['y_test']
    preprocessor = data['preprocessor']
    feature_names = data['feature_names']
    metadata = data['metadata']
    inverse_mapping = data['inverse_mapping']

    # Save Preprocessing Pipeline and Metadata
    joblib.dump(preprocessor, os.path.join(MODELS_DIR, 'preprocessing_pipeline.pkl'))
    with open(os.path.join(MODELS_DIR, 'metadata.json'), 'w') as f:
        json.dump({
            'feature_names': feature_names,
            'features': metadata,
            'target_col': target_col,
            'classes': inverse_mapping,
            'total_samples': len(df),
            'placed_count': int((df[target_col] == 'Placed').sum()),
            'not_placed_count': int((df[target_col] == 'Not Placed').sum())
        }, f, indent=4)

    models = get_models()
    evaluation_results = []
    confusion_matrices = {}
    trained_models = {}
    model_predictions = {}
    model_probabilities = {}

    print("[2/5] Training 5 classification algorithms with cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        print(f"  -> Training {name}...")
        
        # 5-fold Cross-Validation on Training Data
        cv_scores = cross_validate(
            model, X_train, y_train, cv=cv,
            scoring=['accuracy', 'precision', 'recall', 'f1']
        )
        
        # Fit on full training set
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Save individual model file
        file_slug = name.lower().replace(' ', '_').replace('-', '_')
        joblib.dump(model, os.path.join(MODELS_DIR, f"{file_slug}.pkl"))
        if name == 'K-Nearest Neighbors':
            joblib.dump(model, os.path.join(MODELS_DIR, "knn.pkl"))
        elif name == 'Support Vector Machine':
            joblib.dump(model, os.path.join(MODELS_DIR, "svm.pkl"))
        
        # Predictions on Test Set
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        model_predictions[name] = y_pred
        model_probabilities[name] = y_prob
        
        # Calculate Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0.0
        
        # Confusion matrix breakdown
        tn, fp, fn, tp = cm.ravel()
        confusion_matrices[name] = {
            'matrix': cm.tolist(),
            'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp)
        }
        
        evaluation_results.append({
            'Algorithm': name,
            'Accuracy': round(float(acc), 4),
            'Precision': round(float(prec), 4),
            'Recall': round(float(rec), 4),
            'F1 Score': round(float(f1), 4),
            'ROC AUC': round(float(roc_auc), 4),
            'CV Accuracy Mean': round(float(cv_scores['test_accuracy'].mean()), 4),
            'CV Accuracy Std': round(float(cv_scores['test_accuracy'].std()), 4),
            'CV F1 Mean': round(float(cv_scores['test_f1'].mean()), 4),
            'True Negatives': int(tn),
            'False Positives': int(fp),
            'False Negatives': int(fn),
            'True Positives': int(tp),
            'File Slug': file_slug
        })

    # Convert to DataFrame
    comparison_df = pd.DataFrame(evaluation_results)
    
    # Sort and Identify Best Model based on F1 Score then Accuracy
    comparison_df = comparison_df.sort_values(by=['F1 Score', 'Accuracy'], ascending=False).reset_index(drop=True)
    best_algorithm_name = comparison_df.iloc[0]['Algorithm']
    best_model = trained_models[best_algorithm_name]
    
    print(f"\n[INFO] Selected Best Model based on F1 Score: {best_algorithm_name} (F1: {comparison_df.iloc[0]['F1 Score']:.4f}, Accuracy: {comparison_df.iloc[0]['Accuracy']:.4f})")
    joblib.dump(best_model, os.path.join(MODELS_DIR, 'best_model.pkl'))
    
    # Save comparison files
    comparison_df.to_csv(os.path.join(OUTPUTS_DIR, 'model_comparison.csv'), index=False)
    
    summary_json = {
        'best_model': best_algorithm_name,
        'criteria': 'F1 Score and Accuracy',
        'comparison': evaluation_results,
        'confusion_matrices': confusion_matrices
    }
    with open(os.path.join(OUTPUTS_DIR, 'evaluation_metrics.json'), 'w') as f:
        json.dump(summary_json, f, indent=4)

    print("[3/5] Generating static EDA & Evaluation Visualizations...")
    generate_all_visualizations(df, target_col, comparison_df, confusion_matrices, trained_models, X_test, y_test, feature_names)

    print("[4/5] Syncing visualizations to static assets directory...")
    sync_outputs_to_static()

    print("[5/5] All models trained, evaluated, and saved successfully!")
    return comparison_df, best_algorithm_name


def generate_all_visualizations(df, target_col, comparison_df, confusion_matrices, trained_models, X_test, y_test, feature_names):
    """Generates all 10+ professional charts with high visual quality."""
    
    # Palette
    custom_palette = ['#4361EE', '#3A0CA3', '#7209B7', '#F72585', '#4CC9F0']
    status_palette = {'Placed': '#10B981', 'Not Placed': '#EF4444'}

    # 1. Placement Status Distribution
    plt.figure(figsize=(8, 5))
    counts = df[target_col].value_counts()
    ax = sns.barplot(x=counts.index, y=counts.values, palette=status_palette)
    plt.title('Overall Placement Status Distribution', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Placement Outcome', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Students', fontsize=12, fontweight='bold')
    for p in ax.patches:
        height = p.get_height()
        pct = (height / len(df)) * 100
        ax.annotate(f'{int(height)} ({pct:.1f}%)',
                    (p.get_x() + p.get_width() / 2., height / 2),
                    ha='center', va='center', fontsize=11, color='white', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, 'placement_distribution.png'))
    plt.close()

    # 2. CGPA Distribution by Placement Status
    if 'CGPA' in df.columns:
        plt.figure(figsize=(9, 5))
        sns.histplot(data=df, x='CGPA', hue=target_col, kde=True, bins=25, palette=status_palette, alpha=0.6)
        plt.title('CGPA Distribution by Placement Status', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Cumulative Grade Point Average (CGPA)', fontsize=12, fontweight='bold')
        plt.ylabel('Student Count', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUTS_DIR, 'cgpa_distribution.png'))
        plt.close()

    # 3. Feature Correlation Heatmap
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    if target_col in df.columns:
        # Create temporary numeric target for correlation
        numeric_df['Placement_Encoded'] = (df[target_col] == 'Placed').astype(int)
    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1,
                linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, 'correlation_heatmap.png'))
    plt.close()

    # 4. Model Accuracy Comparison
    plot_single_metric_bar(comparison_df, 'Accuracy', 'Model Accuracy Comparison', 'accuracy_comparison.png', '#4361EE')

    # 5. Model Precision Comparison
    plot_single_metric_bar(comparison_df, 'Precision', 'Model Precision Comparison', 'precision_comparison.png', '#3A0CA3')

    # 6. Model Recall Comparison
    plot_single_metric_bar(comparison_df, 'Recall', 'Model Recall Comparison', 'recall_comparison.png', '#7209B7')

    # 7. Model F1 Score Comparison
    plot_single_metric_bar(comparison_df, 'F1 Score', 'Model F1 Score Comparison (Selection Metric)', 'f1_comparison.png', '#F72585')

    # 8. Confusion Matrices for Each Algorithm
    for name, data in confusion_matrices.items():
        cm = np.array(data['matrix'])
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Not Placed', 'Placed'],
                    yticklabels=['Not Placed', 'Placed'],
                    annot_kws={"size": 14, "weight": "bold"})
        plt.title(f'Confusion Matrix: {name}', fontsize=13, fontweight='bold', pad=12)
        plt.xlabel('Predicted Label', fontsize=11, fontweight='bold')
        plt.ylabel('Actual Label', fontsize=11, fontweight='bold')
        slug = name.lower().replace(' ', '_').replace('-', '_')
        plt.tight_layout()
        plt.savefig(os.path.join(CONF_MATRIX_DIR, f'cm_{slug}.png'))
        plt.close()

    # Combined Confusion Matrix Grid
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    for idx, (name, data) in enumerate(confusion_matrices.items()):
        cm = np.array(data['matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Not Placed', 'Placed'],
                    yticklabels=['Not Placed', 'Placed'],
                    ax=axes[idx], annot_kws={"size": 12, "weight": "bold"})
        axes[idx].set_title(name, fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Predicted Label', fontsize=10)
        axes[idx].set_ylabel('Actual Label', fontsize=10)
    # Hide empty 6th subplot
    fig.delaxes(axes[5])
    plt.suptitle('Confusion Matrix Benchmark across All Algorithms', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, 'all_confusion_matrices.png'))
    plt.close()

    # 9. Feature Importance Chart
    plt.figure(figsize=(10, 6))
    if 'Random Forest' in trained_models:
        rf = trained_models['Random Forest']
        importances = rf.feature_importances_
        feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feat_df = feat_df.sort_values(by='Importance', ascending=True)
        ax = sns.barplot(x='Importance', y='Feature', data=feat_df, palette='viridis')
        plt.title('Feature Importance (Random Forest Gini Impurity)', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Relative Feature Importance Score', fontsize=12, fontweight='bold')
        plt.ylabel('Feature', fontsize=12, fontweight='bold')
        for p in ax.patches:
            val = p.get_width()
            ax.annotate(f'{val:.3f}', (val + 0.005, p.get_y() + p.get_height() / 2),
                        ha='left', va='center', fontsize=10, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUTS_DIR, 'feature_importance.png'))
        plt.close()

    # 10. Actual vs Predicted Distribution (Best Model)
    best_model_name = comparison_df.iloc[0]['Algorithm']
    best_pred = trained_models[best_model_name].predict(X_test)
    plt.figure(figsize=(8, 5))
    comp_df = pd.DataFrame({
        'Status': ['Not Placed', 'Placed', 'Not Placed', 'Placed'],
        'Count': [
            int((y_test == 0).sum()), int((y_test == 1).sum()),
            int((best_pred == 0).sum()), int((best_pred == 1).sum())
        ],
        'Type': ['Actual', 'Actual', f'Predicted ({best_model_name})', f'Predicted ({best_model_name})']
    })
    sns.barplot(data=comp_df, x='Status', y='Count', hue='Type', palette=['#3B82F6', '#10B981'])
    plt.title(f'Actual vs Predicted Distribution on Test Set ({best_model_name})', fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('Placement Outcome', fontsize=11, fontweight='bold')
    plt.ylabel('Student Count', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, 'actual_vs_predicted.png'))
    plt.close()


def plot_single_metric_bar(df, metric, title, filename, color):
    """Helper to plot an aesthetic horizontal bar chart for a single metric."""
    plt.figure(figsize=(9, 5))
    sorted_df = df.sort_values(by=metric, ascending=True)
    ax = sns.barplot(x=metric, y='Algorithm', data=sorted_df, color=color)
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.xlabel(metric, fontsize=12, fontweight='bold')
    plt.ylabel('Algorithm', fontsize=12, fontweight='bold')
    plt.xlim(0, 1.08)
    for p in ax.patches:
        val = p.get_width()
        ax.annotate(f'{val:.4f} ({val*100:.1f}%)',
                    (val + 0.01, p.get_y() + p.get_height() / 2),
                    ha='left', va='center', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, filename))
    plt.close()


def sync_outputs_to_static():
    """Copies output PNG charts to static/images so Flask frontend can serve them directly."""
    for root, dirs, files in os.walk(OUTPUTS_DIR):
        for f in files:
            if f.endswith('.png'):
                src = os.path.join(root, f)
                rel = os.path.relpath(src, OUTPUTS_DIR)
                dest = os.path.join(STATIC_IMAGES_DIR, rel)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copy(src, dest)


if __name__ == '__main__':
    print("==========================================================")
    print(" STUDENT PLACEMENT PREDICTION - MODEL TRAINING PIPELINE   ")
    print("==========================================================")
    train_and_evaluate_all()
