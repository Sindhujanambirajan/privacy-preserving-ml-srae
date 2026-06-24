"""
losses.py
Four Loss Functions for Multi-Objective Training
1. Reconstruction Loss (MSE)
2. Classification Loss (Cross Entropy)
3. Center Loss
4. PCA Cosine Similarity Loss
"""

import tensorflow as tf
import numpy as np


# ══════════════════════════════════════════════════════
#  LOSS 1: RECONSTRUCTION LOSS
# ══════════════════════════════════════════════════════
def reconstruction_loss(x_true, x_pred):
    """
    MSE between original and reconstructed data.
    Ensures encoding retains all information.
    """
    return tf.reduce_mean(tf.square(x_true - x_pred))


# ══════════════════════════════════════════════════════
#  LOSS 2: CLASSIFICATION LOSS
# ══════════════════════════════════════════════════════
def classification_loss(y_true, clf_outputs):
    """
    Average cross-entropy across all classifiers.
    Ensures encoding is discriminative.

    Args:
        y_true      : true labels (batch_size,)
        clf_outputs : list of classifier softmax outputs
    """
    total = 0.0
    for clf_out in clf_outputs:
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            y_true, clf_out
        )
        total += tf.reduce_mean(loss)
    return total / len(clf_outputs)


# ══════════════════════════════════════════════════════
#  LOSS 3: CENTER LOSS
# ══════════════════════════════════════════════════════
class CenterLoss:
    """
    Center Loss: keeps same-class samples close together.

    Formula: Lc = sum_i || xi - c_yi ||^2
    where c_yi = center of class yi

    Effect: Creates well-separated clusters in encoding space.
    """

    def __init__(self, num_classes, feature_dim, alpha=0.5):
        """
        Args:
            num_classes : number of output classes
            feature_dim : dimension of encoding Ψ (e.g. 480)
            alpha       : center update rate (0 to 1)
        """
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.alpha = alpha

        # Initialize class centers at zero
        self.centers = tf.Variable(
            tf.zeros([num_classes, feature_dim]),
            trainable=False,
            dtype=tf.float32,
            name='class_centers'
        )

    def __call__(self, features, labels):
        """
        Compute center loss and update centers.

        Args:
            features : encoded data Ψ  shape (batch, feature_dim)
            labels   : true class labels shape (batch,)

        Returns:
            center loss scalar
        """
        features = tf.cast(features, tf.float32)
        labels = tf.cast(labels, tf.int32)

        # Get center for each sample's class
        centers_batch = tf.gather(self.centers, labels)

        # Distance from each sample to its class center
        diff = features - centers_batch
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(diff), axis=1))

        # Update centers
        self._update_centers(features, labels)

        return loss

    def _update_centers(self, features, labels):
        """
        Move centers toward current batch samples.
        New center = Old center - alpha * (Old center - batch mean)
        """
        for class_id in range(self.num_classes):
            # Find samples of this class
            mask = tf.equal(labels, class_id)
            class_features = tf.boolean_mask(features, mask)

            if tf.shape(class_features)[0] > 0:
                # Batch mean for this class
                class_mean = tf.reduce_mean(class_features, axis=0)

                # Move center toward mean
                diff = self.centers[class_id] - class_mean
                new_center = self.centers[class_id] - self.alpha * diff

                # Assign updated center
                indices = tf.constant([[class_id]])
                update = tf.expand_dims(new_center, 0)
                self.centers.assign(
                    tf.tensor_scatter_nd_update(self.centers, indices, update)
                )

    def reset_centers(self):
        """Reset all centers to zero."""
        self.centers.assign(
            tf.zeros([self.num_classes, self.feature_dim])
        )


# ══════════════════════════════════════════════════════
#  LOSS 4: PCA COSINE SIMILARITY LOSS
# ══════════════════════════════════════════════════════
def pca_cosine_loss(pca_projection, x_pca_batch):
    """
    PCA Alignment Loss: forces encoding to align with PCA structure.

    Formula: Lpca = sum_i [ 1 - cosine_similarity(f2(ψi), x_pca_i) ]

    Args:
        pca_projection : model's 2D projection of Ψ  shape (batch, 2)
        x_pca_batch    : 2D PCA of original data      shape (batch, 2)

    Returns:
        PCA cosine loss scalar
    """
    pca_projection = tf.cast(pca_projection, tf.float32)
    x_pca_batch = tf.cast(x_pca_batch, tf.float32)

    # Normalize both to unit vectors
    f2_norm = tf.nn.l2_normalize(pca_projection, axis=1)
    pca_norm = tf.nn.l2_normalize(x_pca_batch, axis=1)

    # Cosine similarity = dot product of normalized vectors
    cosine_sim = tf.reduce_sum(f2_norm * pca_norm, axis=1)

    # Loss = 1 - similarity (0 = perfect, 2 = worst)
    loss = tf.reduce_mean(1.0 - cosine_sim)
    return loss


# ══════════════════════════════════════════════════════
#  COMBINED TOTAL LOSS
# ══════════════════════════════════════════════════════
def compute_total_loss(
        x_true, x_pred,
        y_true, clf_outputs,
        psi_concat, pca_proj, x_pca_batch,
        center_loss_fn,
        lambda1=1.0, lambda2=1.0,
        lambda3=0.5, lambda4=0.5
):
    """
    Combine all 4 losses into one training objective.

    Args:
        x_true          : original input data
        x_pred          : reconstructed data
        y_true          : true class labels
        clf_outputs     : list of classifier outputs
        psi_concat      : concatenated encoding Ψ
        pca_proj        : model's 2D projection of Ψ
        x_pca_batch     : 2D PCA of original data
        center_loss_fn  : CenterLoss instance
        lambda1-4       : loss weights

    Returns:
        total_loss, L1, L2, L3, L4
    """
    # Loss 1: Reconstruction
    L1 = reconstruction_loss(x_true, x_pred)

    # Loss 2: Classification
    L2 = classification_loss(y_true, clf_outputs)

    # Loss 3: Center Loss
    L3 = center_loss_fn(psi_concat, y_true)

    # Loss 4: PCA Cosine Loss
    L4 = pca_cosine_loss(pca_proj, x_pca_batch)

    # Weighted sum
    total = lambda1 * L1 + lambda2 * L2 + lambda3 * L3 + lambda4 * L4

    return total, L1, L2, L3, L4


# ══════════════════════════════════════════════════════
#  QUICK TEST
# ══════════════════════════════════════════════════════
if __name__ == '__main__':
    import numpy as np

    print("Testing Loss Functions...")

    batch = 16
    feat_dim = 480
    num_cls = 7

    # Dummy data
    x_true = tf.random.normal([batch, 100])
    x_pred = tf.random.normal([batch, 100])
    y_true = tf.random.uniform([batch], 0, num_cls, dtype=tf.int32)
    features = tf.random.normal([batch, feat_dim])
    pca_proj = tf.random.normal([batch, 2])
    x_pca = tf.random.normal([batch, 2])
    clf_outs = [
        tf.nn.softmax(tf.random.normal([batch, num_cls]))
        for _ in range(4)
    ]

    # Test center loss
    cl = CenterLoss(num_cls, feat_dim)
    l3 = cl(features, y_true)
    print(f"Center Loss: {l3:.4f}")

    # Test PCA loss
    l4 = pca_cosine_loss(pca_proj, x_pca)
    print(f"PCA Loss: {l4:.4f}")

    # Test total loss
    total, L1, L2, L3, L4 = compute_total_loss(
        x_true, x_pred,
        y_true, clf_outs,
        features, pca_proj, x_pca,
        cl
    )
    print(f"Total Loss: {total:.4f}")
    print(f"  L1(Recon): {L1:.4f}")
    print(f"  L2(Class): {L2:.4f}")
    print(f"  L3(Center): {L3:.4f}")
    print(f"  L4(PCA): {L4:.4f}")
    print("\n✅ All losses working!")
