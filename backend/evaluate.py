"""
evaluate.py
Evaluate and compare performance:
Original Data vs Baseline (Latent Only) vs Your Encoding (Ψ)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import warnings
import os
warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════
#  TRAIN AND EVALUATE ONE ML MODEL
# ══════════════════════════════════════════════════════
def train_evaluate_model(
        X_train, X_test,
        y_train, y_test,
        model_name='KNN'
):
    """
    Train one ML model with randomized grid search.

    Returns:
        f1 score (macro)
    """
    models_config = {
        'KNN': {
            'model': KNeighborsClassifier(),
            'params': {
                'n_neighbors': [3, 5, 7, 10, 15],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        },
        'SVM': {
            'model': SVC(probability=True),
            'params': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            }
        },
        'RF': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 50],
                'min_samples_split': [2, 5, 10]
            }
        },
        'MLP': {
            'model': MLPClassifier(max_iter=500, random_state=42),
            'params': {
                'hidden_layer_sizes': [(100,), (200, 100), (300, 200, 100)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
                'solver': ['adam', 'sgd']
            }
        }
    }

    config = models_config[model_name]

    search = RandomizedSearchCV(
        config['model'],
        config['params'],
        n_iter=10,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        scoring='f1_macro',
        random_state=42,
        n_jobs=-1
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    y_pred = best_model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='macro')

    return f1, best_model, y_pred


# ══════════════════════════════════════════════════════
#  COMPARE ALL THREE VERSIONS
# ══════════════════════════════════════════════════════
def compare_all(
        X_org_train, X_org_test,
        X_base_train, X_base_test,
        X_enc_train, X_enc_test,
        y_train, y_test,
        dataset_name='Dataset'
):
    """
    Compare performance:
    Original (Org) vs Baseline (Base) vs Your Encoding (Enc)

    Args:
        X_org_*  : original data
        X_base_* : latent space only (baseline)
        X_enc_*  : your concatenated encoding Ψ
        y_*      : labels

    Returns:
        results DataFrame
    """
    model_names = ['KNN', 'SVM', 'RF', 'MLP']
    results = []

    print(f"\n{'='*65}")
    print(f"  PERFORMANCE COMPARISON: {dataset_name}")
    print(f"{'='*65}")
    print(f"{'Model':<8} {'Original':>12} {'Baseline':>12} {'Encoding':>12} {'Improvement':>12}")
    print(f"{'-'*65}")

    for model_name in model_names:
        print(f"Training {model_name}...", end=' ', flush=True)

        # Train on original data
        f1_org, _, _ = train_evaluate_model(
            X_org_train, X_org_test, y_train, y_test, model_name
        )

        # Train on baseline (latent space only)
        f1_base, _, _ = train_evaluate_model(
            X_base_train, X_base_test, y_train, y_test, model_name
        )

        # Train on your encoding
        f1_enc, _, y_pred = train_evaluate_model(
            X_enc_train, X_enc_test, y_train, y_test, model_name
        )

        improvement = f1_enc - f1_base

        results.append({
            'Model': model_name,
            'Original': f1_org * 100,
            'Baseline': f1_base * 100,
            'Encoding': f1_enc * 100,
            'Improvement': improvement * 100
        })

        print(
            f"\r{model_name:<8} {f1_org*100:>11.2f}% "
            f"{f1_base*100:>11.2f}% "
            f"{f1_enc*100:>11.2f}% "
            f"{improvement*100:>+11.2f}%"
        )

    print(f"{'='*65}")

    df = pd.DataFrame(results)
    return df


# ══════════════════════════════════════════════════════
#  SILHOUETTE SCORE (Center Loss Evaluation)
# ══════════════════════════════════════════════════════
def compute_silhouette(encoding, labels, name=''):
    """
    Compute silhouette score of encoding.
    Higher = better separated clusters.
    """
    # Reduce to 2D with t-SNE first
    print(f"Computing t-SNE for {name}...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    enc_2d = tsne.fit_transform(encoding)

    score = silhouette_score(enc_2d, labels)
    print(f"Silhouette Score ({name}): {score:.4f}")
    return score, enc_2d


# ══════════════════════════════════════════════════════
#  PLOTS
# ══════════════════════════════════════════════════════
def plot_comparison_bar(results_df, dataset_name, save_path=None):
    """Bar chart comparing Org vs Baseline vs Encoding."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(results_df))
    width = 0.25

    bars1 = ax.bar(x - width, results_df['Original'],
                   width, label='Original', color='#4CAF50', alpha=0.8)
    bars2 = ax.bar(x, results_df['Baseline'],
                   width, label='Baseline (Latent)', color='#2196F3', alpha=0.8)
    bars3 = ax.bar(x + width, results_df['Encoding'],
                   width, label='Our Encoding Ψ', color='#FF5722', alpha=0.8)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Macro F1 Score (%)', fontsize=12)
    ax.set_title(
        f'Performance Comparison - {dataset_name}\n'
        f'Original vs Baseline vs Our Encoding',
        fontsize=13
    )
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['Model'])
    ax.legend()
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Plot saved: {save_path}")
    plt.savefig('results/plots/comparison.png', dpi=100, bbox_inches='tight')
    plt.close()


def plot_tsne_clusters(enc_2d, labels, title, save_path=None):
    """t-SNE visualization of encoding clusters."""
    fig, ax = plt.subplots(figsize=(8, 6))

    scatter = ax.scatter(
        enc_2d[:, 0], enc_2d[:, 1],
        c=labels, cmap='tab10',
        alpha=0.6, s=20
    )
    plt.colorbar(scatter, ax=ax, label='Class')
    ax.set_title(f't-SNE Visualization\n{title}', fontsize=12)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"t-SNE plot saved: {save_path}")
    plt.savefig('results/plots/tsne.png', dpi=100, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path=None):
    """Confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm, annot=True, fmt='d',
        xticklabels=class_names,
        yticklabels=class_names,
        cmap='Blues', ax=ax
    )
    ax.set_title(f'Confusion Matrix\n{title}', fontsize=12)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.savefig('results/plots/confusion_matrix.png', dpi=100, bbox_inches='tight')
    plt.close()


# ══════════════════════════════════════════════════════
#  FULL EVALUATION PIPELINE
# ══════════════════════════════════════════════════════
def full_evaluation(
        X_org_train, X_org_test,
        psi_train, psi_test,
        y_train, y_test,
        dataset_name='Leukemia',
        class_names=None
):
    """
    Run complete evaluation.

    For baseline, use only the latent space (last 96 dims of 480).
    For encoding, use full 480-dim Ψ.
    """
    # Baseline = last 96 dimensions (latent space only)
    X_base_train = psi_train[:, 384:]  # last 96 dims
    X_base_test = psi_test[:, 384:]

    # Your encoding = full 480 dims
    X_enc_train = psi_train
    X_enc_test = psi_test

    # Compare performance
    results_df = compare_all(
        X_org_train, X_org_test,
        X_base_train, X_base_test,
        X_enc_train, X_enc_test,
        y_train, y_test,
        dataset_name
    )

    # Plot comparison
    plot_comparison_bar(
        results_df, dataset_name,
        f'results/plots/{dataset_name}_comparison.png'
    )

    # Silhouette scores
    print("\nComputing Silhouette Scores...")
    score_base, tsne_base = compute_silhouette(
        X_base_test, y_test, 'Baseline'
    )
    score_enc, tsne_enc = compute_silhouette(
        X_enc_test, y_test, 'Our Encoding'
    )

    print(f"\nSilhouette Score Comparison:")
    print(f"  Baseline:     {score_base:.4f}")
    print(f"  Our Encoding: {score_enc:.4f}")
    improvement_pct = ((score_enc - score_base) / abs(score_base)) * 100
    print(f"  Improvement:  {improvement_pct:+.1f}%")

    # t-SNE plots
    plot_tsne_clusters(
        tsne_base, y_test,
        f'Baseline - {dataset_name}',
        f'results/plots/{dataset_name}_tsne_baseline.png'
    )
    plot_tsne_clusters(
        tsne_enc, y_test,
        f'Our Encoding - {dataset_name}',
        f'results/plots/{dataset_name}_tsne_encoding.png'
    )

    # Save results
    os.makedirs('results/metrics', exist_ok=True)
    results_df.to_csv(
        f'results/metrics/{dataset_name}_results.csv',
        index=False
    )
    print(f"\nResults saved to results/metrics/{dataset_name}_results.csv")

    return results_df, score_base, score_enc


if __name__ == '__main__':
    print("Import and call full_evaluation() with your data.")
    print("See README for usage examples.")
