"""
train.py
Training Pipeline for SRAE
Run this file in Google Colab for GPU acceleration
"""

import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import os
import time

from model import build_srae_tabular, build_srae_image
from losses import CenterLoss, compute_total_loss


# ══════════════════════════════════════════════════════
#  TRAINING FUNCTION
# ══════════════════════════════════════════════════════
def train_srae(
        X_train, y_train,
        X_val, y_val,
        num_classes,
        model_type='tabular',
        input_shape=None,
        epochs=100,
        batch_size=32,
        save_path='saved_models/srae_model.h5',
        lambda1=1.0, lambda2=1.0,
        lambda3=0.5, lambda4=0.5
):
    """
    Train the Supervised Residual Autoencoder.

    Args:
        X_train, y_train : training data and labels
        X_val, y_val     : validation data and labels
        num_classes      : number of output classes
        model_type       : 'tabular' or 'image'
        input_shape      : required if model_type='image'
        epochs           : max training epochs
        batch_size       : samples per batch
        save_path        : where to save best model
        lambda1-4        : loss weights

    Returns:
        trained model, training history dict
    """
    print("=" * 60)
    print("  PRIVACY-PRESERVING ML - SRAE TRAINING")
    print("=" * 60)

    # ── BUILD MODEL ──────────────────────────────────
    if model_type == 'tabular':
        input_dim = X_train.shape[1]
        model = build_srae_tabular(input_dim, num_classes)
        print(f"Model: SRAE Tabular | Input: {input_dim} | Classes: {num_classes}")
    else:
        model = build_srae_image(input_shape, num_classes)
        print(f"Model: C-SRAE Image | Input: {input_shape} | Classes: {num_classes}")

    # ── OPTIMIZER ────────────────────────────────────
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # ── CENTER LOSS INIT ─────────────────────────────
    center_loss_fn = CenterLoss(
        num_classes=num_classes,
        feature_dim=480,
        alpha=0.5
    )

    # ── PRE-COMPUTE PCA ───────────────────────────────
    print("\nComputing PCA of training data...")
    if model_type == 'tabular':
        pca = PCA(n_components=2)
        x_pca_train = pca.fit_transform(X_train).astype(np.float32)
    else:
        X_flat = X_train.reshape(len(X_train), -1)
        pca = PCA(n_components=2)
        x_pca_train = pca.fit_transform(X_flat).astype(np.float32)
    print("PCA computed!")

    # ── TRAINING HISTORY ──────────────────────────────
    history = {
        'total_loss': [], 'val_loss': [],
        'L1_recon': [], 'L2_class': [],
        'L3_center': [], 'L4_pca': []
    }

    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"\nTraining for {epochs} epochs (patience={patience})...")
    print("-" * 60)

    start_time = time.time()

    for epoch in range(epochs):

        # Shuffle training data
        idx = np.random.permutation(len(X_train))
        X_shuffled = X_train[idx]
        y_shuffled = y_train[idx]
        pca_shuffled = x_pca_train[idx]

        epoch_losses = {'total': 0, 'L1': 0, 'L2': 0, 'L3': 0, 'L4': 0}
        num_batches = max(1, len(X_train) // batch_size)

        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = start + batch_size

            x_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end].astype(np.int32)
            pca_batch = pca_shuffled[start:end]

            # ── FORWARD + BACKWARD PASS ───────────────
            with tf.GradientTape() as tape:
                outputs = model(x_batch, training=True)

                reconstruction = outputs[0]
                clf_1 = outputs[1]
                clf_2 = outputs[2]
                clf_3 = outputs[3]
                clf_final = outputs[4]
                psi_concat = outputs[5]
                pca_proj = outputs[6]

                total, L1, L2, L3, L4 = compute_total_loss(
                    x_batch, reconstruction,
                    y_batch,
                    [clf_1, clf_2, clf_3, clf_final],
                    psi_concat, pca_proj, pca_batch,
                    center_loss_fn,
                    lambda1, lambda2, lambda3, lambda4
                )

            # Update weights
            grads = tape.gradient(total, model.trainable_variables)
            optimizer.apply_gradients(
                zip(grads, model.trainable_variables)
            )

            epoch_losses['total'] += total.numpy()
            epoch_losses['L1'] += L1.numpy()
            epoch_losses['L2'] += L2.numpy()
            epoch_losses['L3'] += L3.numpy()
            epoch_losses['L4'] += L4.numpy()

        # Average losses
        for k in epoch_losses:
            epoch_losses[k] /= num_batches

        # ── VALIDATION ───────────────────────────────
        val_outputs = model(X_val, training=False)
        val_loss = tf.reduce_mean(
            tf.square(X_val - val_outputs[0])
        ).numpy()

        # Save history
        history['total_loss'].append(epoch_losses['total'])
        history['val_loss'].append(val_loss)
        history['L1_recon'].append(epoch_losses['L1'])
        history['L2_class'].append(epoch_losses['L2'])
        history['L3_center'].append(epoch_losses['L3'])
        history['L4_pca'].append(epoch_losses['L4'])

        # ── PRINT PROGRESS ───────────────────────────
        if (epoch + 1) % 5 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch+1:3d}/{epochs} | "
                f"Loss: {epoch_losses['total']:.4f} | "
                f"ValLoss: {val_loss:.4f} | "
                f"L1:{epoch_losses['L1']:.3f} "
                f"L2:{epoch_losses['L2']:.3f} "
                f"L3:{epoch_losses['L3']:.3f} "
                f"L4:{epoch_losses['L4']:.3f} | "
                f"Time: {elapsed:.1f}s"
            )

        # ── EARLY STOPPING ───────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save(save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                print(f"Best val loss: {best_val_loss:.4f}")
                break

    total_time = time.time() - start_time
    print(f"\nTraining complete! Total time: {total_time:.1f}s")
    print(f"Best model saved to: {save_path}")

    return model, history


# ══════════════════════════════════════════════════════
#  GENERATE ENCODING
# ══════════════════════════════════════════════════════
def generate_encoding(model, X_data, batch_size=256):
    """
    Generate concatenated encoding Ψ for all data.

    Args:
        model      : trained SRAE model
        X_data     : input data
        batch_size : processing batch size

    Returns:
        numpy array of encodings shape (N, 480)
    """
    encodings = []
    for i in range(0, len(X_data), batch_size):
        batch = X_data[i:i + batch_size]
        outputs = model(batch, training=False)
        psi = outputs[5].numpy()  # psi_concat
        encodings.append(psi)
    return np.vstack(encodings)


# ══════════════════════════════════════════════════════
#  PLOT TRAINING HISTORY
# ══════════════════════════════════════════════════════
def plot_training_history(history, save_path='results/plots/training.png'):
    """Plot and save training curves."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('SRAE Training History', fontsize=14)

    plots = [
        ('total_loss', 'Total Loss', 'blue'),
        ('val_loss', 'Validation Loss', 'red'),
        ('L1_recon', 'Reconstruction Loss (L1)', 'green'),
        ('L2_class', 'Classification Loss (L2)', 'orange'),
        ('L3_center', 'Center Loss (L3)', 'purple'),
        ('L4_pca', 'PCA Cosine Loss (L4)', 'brown'),
    ]

    for ax, (key, title, color) in zip(axes.flat, plots):
        ax.plot(history[key], color=color)
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()
    print(f"Training plot saved: {save_path}")


# ══════════════════════════════════════════════════════
#  FULL PIPELINE: LOAD DATA → TRAIN → ENCODE
# ══════════════════════════════════════════════════════
def full_pipeline_leukemia(data_path, save_dir='saved_models'):
    """
    Complete pipeline for Leukemia dataset.
    Run this in Google Colab.

    Args:
        data_path : path to leukemia CSV file
        save_dir  : directory to save model and encodings
    """
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

    print("Loading Leukemia dataset...")
    df = pd.read_csv(data_path)

    X = df.iloc[:, :-1].values.astype(np.float32)
    y_raw = df.iloc[:, -1].values

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw).astype(np.int32)
    num_classes = len(le.classes_)

    print(f"Dataset: {X.shape} | Classes: {num_classes} | {le.classes_}")

    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42
    )

    print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

    # Train SRAE
    model, history = train_srae(
        X_train, y_train,
        X_val, y_val,
        num_classes=num_classes,
        model_type='tabular',
        epochs=100,
        batch_size=16,
        save_path=f'{save_dir}/srae_leukemia.h5'
    )

    # Generate encodings
    print("\nGenerating encodings...")
    psi_train = generate_encoding(model, X_train)
    psi_test = generate_encoding(model, X_test)

    print(f"Encoding shape: {psi_train.shape}")  # Should be (N, 480)

    # Save encodings
    os.makedirs(save_dir, exist_ok=True)
    np.save(f'{save_dir}/psi_train_leukemia.npy', psi_train)
    np.save(f'{save_dir}/psi_test_leukemia.npy', psi_test)
    np.save(f'{save_dir}/y_train_leukemia.npy', y_train)
    np.save(f'{save_dir}/y_test_leukemia.npy', y_test)

    print("Encodings saved!")

    # Plot training
    plot_training_history(history)

    return model, psi_train, psi_test, y_train, y_test, le


if __name__ == '__main__':
    print("Run full_pipeline_leukemia() with your dataset path.")
    print("Example:")
    print("  model, psi_train, psi_test, y_tr, y_te, le =")
    print("  full_pipeline_leukemia('data/leukemia/leukemia.csv')")
