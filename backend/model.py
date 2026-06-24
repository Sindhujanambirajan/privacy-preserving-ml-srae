"""
model.py
Supervised Residual Autoencoder (SRAE)
Privacy-Preserving Machine Learning Project
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, BatchNormalization, Activation,
    Add, Concatenate, Conv2D, MaxPool2D, Flatten,
    Reshape, Conv2DTranspose, UpSampling2D, Dropout
)
from tensorflow.keras.models import Model
import tensorflow.keras.regularizers as reg


# ══════════════════════════════════════════════════════
#  RESIDUAL BLOCK FOR TABULAR DATA
# ══════════════════════════════════════════════════════
def residual_block_dense(x, units, name=None):
    """
    One Residual Block for tabular (non-image) data
    Formula: Output = F(x) + x  (skip connection)
    """
    shortcut = x

    # Main path
    out = Dense(
        units,
        kernel_regularizer=reg.l1_l2(l1=1e-5, l2=1e-5)
    )(x)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    out = Dense(
        units,
        kernel_regularizer=reg.l1_l2(l1=1e-5, l2=1e-5)
    )(out)
    out = BatchNormalization()(out)

    # Match dimensions if needed
    if int(shortcut.shape[-1]) != units:
        shortcut = Dense(units)(shortcut)
        shortcut = BatchNormalization()(shortcut)

    # Skip connection: F(x) + x
    out = Add()([out, shortcut])
    out = Activation('relu', name=name)(out)
    return out


# ══════════════════════════════════════════════════════
#  RESIDUAL BLOCK FOR IMAGE DATA
# ══════════════════════════════════════════════════════
def residual_block_conv(x, filters, downsample=False):
    """
    One Residual Block for image data (Convolutional)
    """
    shortcut = x
    stride = 2 if downsample else 1

    out = Conv2D(
        filters, (3, 3),
        strides=stride,
        padding='same',
        kernel_regularizer=reg.l1_l2(l1=1e-5, l2=1e-5)
    )(x)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    out = Conv2D(
        filters, (3, 3),
        padding='same',
        kernel_regularizer=reg.l1_l2(l1=1e-5, l2=1e-5)
    )(out)
    out = BatchNormalization()(out)

    # Match dimensions
    if downsample or int(shortcut.shape[-1]) != filters:
        shortcut = Conv2D(filters, (1, 1), strides=stride, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    out = Add()([out, shortcut])
    out = Activation('relu')(out)
    return out


# ══════════════════════════════════════════════════════
#  SRAE FOR TABULAR DATA (Leukemia, TCGA)
# ══════════════════════════════════════════════════════
def build_srae_tabular(input_dim, num_classes, latent_dim=64):
    """
    Supervised Residual Autoencoder for tabular data.

    Args:
        input_dim  : number of input features
        num_classes: number of output classes
        latent_dim : size of latent space

    Returns:
        Keras Model with outputs:
        [reconstruction, clf1, clf2, clf3,
         clf_final, psi_concat, pca_proj]
    """
    inputs = Input(shape=(input_dim,), name='input')

    # ── ENCODER ──────────────────────────────────────
    psi_1 = residual_block_dense(inputs, 256)
    psi_1 = Dense(
        256,
        activity_regularizer=reg.l1(1e-5),
        name='psi_1'
    )(psi_1)

    psi_2 = residual_block_dense(psi_1, 128)
    psi_2 = Dense(
        128,
        activity_regularizer=reg.l1(1e-5),
        name='psi_2'
    )(psi_2)

    psi_3 = residual_block_dense(psi_2, 96)
    psi_3 = Dense(
        96,
        activity_regularizer=reg.l1(1e-5),
        name='psi_3'
    )(psi_3)  # Latent space

    # ── CLASSIFIERS ON EACH LAYER ─────────────────────
    clf_1 = Dense(
        num_classes, activation='softmax', name='clf_1'
    )(psi_1)

    clf_2 = Dense(
        num_classes, activation='softmax', name='clf_2'
    )(psi_2)

    clf_3 = Dense(
        num_classes, activation='softmax', name='clf_3'
    )(psi_3)

    # ── CONCATENATED ENCODING Ψ ───────────────────────
    # 256 + 128 + 96 = 480 dimensions
    psi_concat = Concatenate(name='encoding')([psi_1, psi_2, psi_3])

    # Final classifier on concatenated encoding
    clf_final = Dense(
        num_classes, activation='softmax', name='clf_final'
    )(psi_concat)

    # PCA projection layer (for PCA cosine loss)
    pca_proj = Dense(2, name='pca_projection')(psi_concat)

    # ── DECODER ──────────────────────────────────────
    dec = residual_block_dense(psi_3, 128)
    dec = residual_block_dense(dec, 256)
    reconstruction = Dense(
        input_dim, activation='sigmoid', name='reconstruction'
    )(dec)

    # ── BUILD MODEL ───────────────────────────────────
    model = Model(
        inputs=inputs,
        outputs=[
            reconstruction,  # index 0
            clf_1,           # index 1
            clf_2,           # index 2
            clf_3,           # index 3
            clf_final,       # index 4
            psi_concat,      # index 5
            pca_proj         # index 6
        ],
        name='SRAE_Tabular'
    )

    return model


# ══════════════════════════════════════════════════════
#  C-SRAE FOR IMAGE DATA (MNIST, FashionMNIST, OCT)
# ══════════════════════════════════════════════════════
def build_srae_image(input_shape, num_classes):
    """
    Convolutional Supervised Residual Autoencoder for images.

    Args:
        input_shape : (H, W, C) e.g. (28, 28, 1) for MNIST
        num_classes : number of output classes

    Returns:
        Keras Model
    """
    inputs = Input(shape=input_shape, name='input')

    # ── ENCODER ──────────────────────────────────────
    x = residual_block_conv(inputs, 32)
    x = MaxPool2D((2, 2))(x)
    psi_1_raw = x

    x = residual_block_conv(x, 64)
    x = MaxPool2D((2, 2))(x)
    psi_2_raw = x

    x = residual_block_conv(x, 128)
    psi_3_raw = x

    # Flatten
    psi_1_flat = Flatten()(psi_1_raw)
    psi_2_flat = Flatten()(psi_2_raw)
    psi_3_flat = Flatten()(psi_3_raw)

    # Project to fixed sizes
    psi_1 = Dense(
        256, activity_regularizer=reg.l1(1e-5), name='psi_1'
    )(psi_1_flat)
    psi_2 = Dense(
        128, activity_regularizer=reg.l1(1e-5), name='psi_2'
    )(psi_2_flat)
    psi_3 = Dense(
        96, activity_regularizer=reg.l1(1e-5), name='psi_3'
    )(psi_3_flat)

    # ── CLASSIFIERS ───────────────────────────────────
    clf_1 = Dense(num_classes, activation='softmax', name='clf_1')(psi_1)
    clf_2 = Dense(num_classes, activation='softmax', name='clf_2')(psi_2)
    clf_3 = Dense(num_classes, activation='softmax', name='clf_3')(psi_3)

    # ── CONCATENATED ENCODING ─────────────────────────
    psi_concat = Concatenate(name='encoding')([psi_1, psi_2, psi_3])
    clf_final = Dense(
        num_classes, activation='softmax', name='clf_final'
    )(psi_concat)
    pca_proj = Dense(2, name='pca_projection')(psi_concat)

    # ── DECODER ──────────────────────────────────────
    h = input_shape[0] // 4
    w = input_shape[1] // 4

    dec = Dense(h * w * 64)(psi_3)
    dec = Reshape((h, w, 64))(dec)
    dec = Conv2DTranspose(64, (3, 3), padding='same', activation='relu')(dec)
    dec = UpSampling2D((2, 2))(dec)
    dec = Conv2DTranspose(32, (3, 3), padding='same', activation='relu')(dec)
    dec = UpSampling2D((2, 2))(dec)
    reconstruction = Conv2DTranspose(
        input_shape[-1], (3, 3),
        padding='same',
        activation='sigmoid',
        name='reconstruction'
    )(dec)

    model = Model(
        inputs=inputs,
        outputs=[
            reconstruction,
            clf_1, clf_2, clf_3,
            clf_final,
            psi_concat,
            pca_proj
        ],
        name='CSRAE_Image'
    )

    return model


# ══════════════════════════════════════════════════════
#  QUICK TEST
# ══════════════════════════════════════════════════════
if __name__ == '__main__':
    print("Testing SRAE Tabular...")
    model_tab = build_srae_tabular(
        input_dim=100,
        num_classes=7
    )
    model_tab.summary()

    print("\nTesting SRAE Image...")
    model_img = build_srae_image(
        input_shape=(28, 28, 1),
        num_classes=10
    )
    model_img.summary()

    print("\n✅ Models built successfully!")
