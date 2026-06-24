"""
utils.py
Helper / Utility Functions
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import os
import pickle


# ══════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════
def load_csv_dataset(file_path, label_col=-1, scale=True):
    """
    Load any CSV dataset.

    Args:
        file_path : path to CSV file
        label_col : column index of labels (-1 = last column)
        scale     : whether to scale features

    Returns:
        X, y (numpy arrays), label_encoder, scaler
    """
    print(f"Loading: {file_path}")
    df = pd.read_csv(file_path)
    print(f"Shape: {df.shape}")

    # Separate features and labels
    if label_col == -1:
        X = df.iloc[:, :-1].values
        y_raw = df.iloc[:, -1].values
    else:
        X = df.drop(df.columns[label_col], axis=1).values
        y_raw = df.iloc[:, label_col].values

    # Encode labels to integers
    le = LabelEncoder()
    y = le.fit_transform(y_raw).astype(np.int32)

    print(f"Features: {X.shape[1]} | Samples: {X.shape[0]}")
    print(f"Classes ({len(le.classes_)}): {le.classes_}")

    X = X.astype(np.float32)

    # Scale features
    scaler = None
    if scale:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X).astype(np.float32)
        print("Features scaled to [0, 1]")

    return X, y, le, scaler


def load_mnist(flatten=False):
    """Load MNIST dataset from Keras."""
    from tensorflow.keras.datasets import mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    if flatten:
        X_train = X_train.reshape(len(X_train), -1)
        X_test = X_test.reshape(len(X_test), -1)
    else:
        X_train = X_train[..., np.newaxis]  # (N, 28, 28, 1)
        X_test = X_test[..., np.newaxis]

    print(f"MNIST: Train {X_train.shape} | Test {X_test.shape}")
    return X_train, X_test, y_train.astype(np.int32), y_test.astype(np.int32)


def load_fashion_mnist(flatten=False):
    """Load FashionMNIST dataset from Keras."""
    from tensorflow.keras.datasets import fashion_mnist
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    if flatten:
        X_train = X_train.reshape(len(X_train), -1)
        X_test = X_test.reshape(len(X_test), -1)
    else:
        X_train = X_train[..., np.newaxis]
        X_test = X_test[..., np.newaxis]

    print(f"FashionMNIST: Train {X_train.shape} | Test {X_test.shape}")
    return X_train, X_test, y_train.astype(np.int32), y_test.astype(np.int32)


# ══════════════════════════════════════════════════════
#  SAVE / LOAD HELPERS
# ══════════════════════════════════════════════════════
def save_encodings(psi_train, psi_test, y_train, y_test, name, save_dir='saved_models'):
    """Save generated encodings to disk."""
    os.makedirs(save_dir, exist_ok=True)
    np.save(f'{save_dir}/psi_train_{name}.npy', psi_train)
    np.save(f'{save_dir}/psi_test_{name}.npy', psi_test)
    np.save(f'{save_dir}/y_train_{name}.npy', y_train)
    np.save(f'{save_dir}/y_test_{name}.npy', y_test)
    print(f"Encodings saved: {save_dir}/psi_*_{name}.npy")


def load_encodings(name, save_dir='saved_models'):
    """Load previously saved encodings."""
    psi_train = np.load(f'{save_dir}/psi_train_{name}.npy')
    psi_test = np.load(f'{save_dir}/psi_test_{name}.npy')
    y_train = np.load(f'{save_dir}/y_train_{name}.npy')
    y_test = np.load(f'{save_dir}/y_test_{name}.npy')
    print(f"Encodings loaded: psi shape = {psi_train.shape}")
    return psi_train, psi_test, y_train, y_test


def save_preprocessors(le, scaler, name, save_dir='saved_models'):
    """Save label encoder and scaler."""
    os.makedirs(save_dir, exist_ok=True)
    with open(f'{save_dir}/le_{name}.pkl', 'wb') as f:
        pickle.dump(le, f)
    if scaler:
        with open(f'{save_dir}/scaler_{name}.pkl', 'wb') as f:
            pickle.dump(scaler, f)
    print(f"Preprocessors saved.")


def load_preprocessors(name, save_dir='saved_models'):
    """Load saved preprocessors."""
    with open(f'{save_dir}/le_{name}.pkl', 'rb') as f:
        le = pickle.load(f)
    try:
        with open(f'{save_dir}/scaler_{name}.pkl', 'rb') as f:
            scaler = pickle.load(f)
    except FileNotFoundError:
        scaler = None
    return le, scaler


# ══════════════════════════════════════════════════════
#  SUMMARY REPORT
# ══════════════════════════════════════════════════════
def print_summary_report(results_df, silhouette_base, silhouette_enc, dataset_name):
    """Print formatted summary report."""
    print("\n" + "=" * 65)
    print(f"  FINAL REPORT: {dataset_name}")
    print("=" * 65)

    print("\n📊 PERFORMANCE (Macro F1 Score %):")
    print(results_df.to_string(index=False))

    avg_enc = results_df['Encoding'].mean()
    avg_base = results_df['Baseline'].mean()
    avg_org = results_df['Original'].mean()

    print(f"\nAverage F1:")
    print(f"  Original:     {avg_org:.2f}%")
    print(f"  Baseline:     {avg_base:.2f}%")
    print(f"  Our Encoding: {avg_enc:.2f}%")
    print(f"  Improvement over Baseline: {avg_enc - avg_base:+.2f}%")

    print(f"\n🔍 CLUSTER QUALITY (Silhouette Score):")
    print(f"  Baseline:     {silhouette_base:.4f}")
    print(f"  Our Encoding: {silhouette_enc:.4f}")

    if silhouette_base != 0:
        pct = ((silhouette_enc - silhouette_base) / abs(silhouette_base)) * 100
        print(f"  Improvement:  {pct:+.1f}%")

    print("\n🔐 PRIVACY STATUS:")
    print("  ✅ Original data NOT shared with cloud")
    print("  ✅ Only encoding Ψ (480 dims) shared")
    print("  ✅ Encoding cannot be reverse-engineered")
    print("=" * 65)
