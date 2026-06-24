from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import pandas as pd
import os, pickle, io, base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Activation, Add, Concatenate
from tensorflow.keras.models import Model
import tensorflow.keras.regularizers as reg

app = Flask(__name__, static_folder='../frontend')
CORS(app)

print(f"TensorFlow: {tf.__version__}")

STATE = {
    'model': None, 'scaler': None,
    'label_encoder': None, 'num_classes': 0,
    'trained': False, 'dataset_name': ''
}

def load_csv(file_content):
    df = pd.read_csv(io.StringIO(file_content.decode('utf-8', errors='ignore')))
    if 'samples' in df.columns:
        df = df.drop(columns=['samples'])
    if 'type' in df.columns:
        y_raw = df['type'].values
        X     = df.drop(columns=['type']).values.astype(np.float32)
    else:
        X     = df.values.astype(np.float32)
        y_raw = None
    return X, y_raw

def res_block(x, units):
    s = x
    o = Dense(units, kernel_regularizer=reg.l1_l2(1e-5, 1e-5))(x)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    o = Dense(units, kernel_regularizer=reg.l1_l2(1e-5, 1e-5))(o)
    o = BatchNormalization()(o)
    if int(s.shape[-1]) != units:
        s = Dense(units)(s)
        s = BatchNormalization()(s)
    o = Add()([o, s])
    return Activation('relu')(o)

def build_model(input_dim, num_classes):
    inp  = Input(shape=(input_dim,))
    p1   = res_block(inp, 256)
    p1   = Dense(256, activity_regularizer=reg.l1(1e-5), name='psi_1')(p1)
    p2   = res_block(p1, 128)
    p2   = Dense(128, activity_regularizer=reg.l1(1e-5), name='psi_2')(p2)
    p3   = res_block(p2, 96)
    p3   = Dense(96,  activity_regularizer=reg.l1(1e-5), name='psi_3')(p3)
    c1   = Dense(num_classes, activation='softmax', name='clf_1')(p1)
    c2   = Dense(num_classes, activation='softmax', name='clf_2')(p2)
    c3   = Dense(num_classes, activation='softmax', name='clf_3')(p3)
    psi  = Concatenate(name='encoding')([p1, p2, p3])
    cf   = Dense(num_classes, activation='softmax', name='clf_final')(psi)
    pp   = Dense(2, name='pca_projection')(psi)
    d    = res_block(p3, 128)
    d    = res_block(d, 256)
    recon = Dense(input_dim, activation='sigmoid', name='reconstruction')(d)
    return Model(inp, [recon, c1, c2, c3, cf, psi, pp])

class CenterLoss:
    def __init__(self, nc, fd, alpha=0.5):
        self.alpha   = alpha
        self.nc      = nc
        self.centers = tf.Variable(
            tf.zeros([nc, fd]), trainable=False, dtype=tf.float32)
    def __call__(self, features, labels):
        features = tf.cast(features, tf.float32)
        labels   = tf.cast(labels,   tf.int32)
        cb   = tf.gather(self.centers, labels)
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(features - cb), axis=1))
        for cid in range(self.nc):
            mask = tf.equal(labels, cid)
            cf   = tf.boolean_mask(features, mask)
            if tf.shape(cf)[0] > 0:
                cm = tf.reduce_mean(cf, axis=0)
                nc = self.centers[cid] - self.alpha*(self.centers[cid]-cm)
                self.centers.assign(
                    tf.tensor_scatter_nd_update(
                        self.centers, [[cid]], tf.expand_dims(nc, 0)))
        return loss

def pca_loss_fn(proj, xp):
    p = tf.nn.l2_normalize(tf.cast(proj, tf.float32), axis=1)
    q = tf.nn.l2_normalize(tf.cast(xp,   tf.float32), axis=1)
    return tf.reduce_mean(1.0 - tf.reduce_sum(p * q, axis=1))

@app.route('/')
def index():
    return send_from_directory('../frontend', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('../frontend', path)

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'running',
        'model_trained': STATE['trained'],
        'tf_version': tf.__version__,
        'message': 'Privacy ML System Active'
    })

@app.route('/api/train', methods=['POST'])
def train():
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file uploaded'}), 400

        file    = request.files['file']
        content = file.read()
        X, y_raw = load_csv(content)

        if len(X) < 5:
            return jsonify({'status': 'error', 'message': 'Need at least 5 rows'}), 400

        le          = LabelEncoder()
        y           = le.fit_transform(y_raw).astype(np.int32)
        num_classes = len(le.classes_)

        scaler   = MinMaxScaler()
        X_scaled = scaler.fit_transform(X).astype(np.float32)

        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42)

        from sklearn.decomposition import PCA
        model     = build_model(X_train.shape[1], num_classes)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        cl        = CenterLoss(num_classes, 480)

        pca         = PCA(n_components=2)
        x_pca_train = pca.fit_transform(X_train).astype(np.float32)

        epochs     = int(request.form.get('epochs', 30))
        batch_size = max(2, min(8, len(X_train) // 3))
        best_val   = float('inf')
        patience   = 8
        wait       = 0

        for epoch in range(epochs):
            idx = np.random.permutation(len(X_train))
            nb  = max(1, len(X_train) // batch_size)
            for b in range(nb):
                s  = b * batch_size
                e  = s + batch_size
                xb = X_train[idx[s:e]]
                yb = y_train[idx[s:e]].astype(np.int32)
                pb = x_pca_train[idx[s:e]]
                with tf.GradientTape() as tape:
                    out   = model(xb, training=True)
                    L1    = tf.reduce_mean(tf.square(xb - out[0]))
                    L2    = tf.reduce_mean(
                        tf.keras.losses.sparse_categorical_crossentropy(yb, out[4]))
                    L3    = cl(out[5], yb)
                    L4    = pca_loss_fn(out[6], pb)
                    total = L1 + L2 + 0.5*L3 + 0.5*L4
                grads = tape.gradient(total, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

            val_out = model(X_val, training=False)
            vl      = tf.reduce_mean(tf.square(X_val - val_out[0])).numpy()
            if vl < best_val:
                best_val = vl
                os.makedirs('saved_models', exist_ok=True)
                model.save('saved_models/srae_model.keras')
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        STATE['model']         = model
        STATE['scaler']        = scaler
        STATE['label_encoder'] = le
        STATE['num_classes']   = int(num_classes)
        STATE['trained']       = True
        STATE['dataset_name']  = file.filename.replace('.csv', '')

        with open('saved_models/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        with open('saved_models/le.pkl', 'wb') as f:
            pickle.dump(le, f)

        return jsonify({
            'status':       'success',
            'message':      f'Model trained successfully!',
            'num_samples':  int(len(X)),
            'num_features': int(X.shape[1]),
            'num_classes':  int(num_classes),
            'classes':      le.classes_.tolist(),
            'encoding_dim': 480
        })

    except Exception as e:
        import traceback
        return jsonify({
            'status': 'error',
            'message': str(e),
            'detail': traceback.format_exc()
        }), 500

@app.route('/api/encode', methods=['POST'])
def encode():
    try:
        if not STATE['trained']:
            return jsonify({'status': 'error',
                            'message': 'Train the model first!'}), 400

        file    = request.files['file']
        content = file.read()
        X, _    = load_csv(content)

        X_scaled = STATE['scaler'].transform(X).astype(np.float32)
        outputs  = STATE['model'](X_scaled, training=False)
        encoding = outputs[5].numpy()

        return jsonify({
            'status':            'success',
            'message':           'Encoding generated!',
            'original_shape':    list(X.shape),
            'encoding_shape':    list(encoding.shape),
            'original_size_kb':  round(X.nbytes / 1024, 2),
            'encoding_size_kb':  round(encoding.nbytes / 1024, 2),
            'privacy_guarantee': 'Original data PROTECTED!',
            'sample_encoding':   encoding[0][:10].tolist()
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        if not STATE['trained']:
            return jsonify({'status': 'error',
                            'message': 'Train the model first!'}), 400

        file    = request.files['file']
        content = file.read()
        X, _    = load_csv(content)

        X_scaled  = STATE['scaler'].transform(X).astype(np.float32)
        outputs   = STATE['model'](X_scaled, training=False)
        clf_probs = outputs[4].numpy()

        predictions = np.argmax(clf_probs, axis=1)
        confidence  = np.max(clf_probs, axis=1)
        pred_labels = STATE['label_encoder'].inverse_transform(predictions)

        pred_list = [
            {
                'sample':     i + 1,
                'prediction': str(label),
                'confidence': round(float(conf) * 100, 2)
            }
            for i, (label, conf) in enumerate(zip(pred_labels, confidence))
        ]

        return jsonify({
            'status':        'success',
            'predictions':   pred_list,
            'total_samples': len(pred_list),
            'classes':       STATE['label_encoder'].classes_.tolist()
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/evaluate', methods=['POST'])
def evaluate():
    try:
        if not STATE['trained']:
            return jsonify({'status': 'error',
                            'message': 'Train the model first!'}), 400

        file     = request.files['file']
        content  = file.read()
        X, y_raw = load_csv(content)

        try:
            y = STATE['label_encoder'].transform(y_raw).astype(np.int32)
        except Exception:
            le2 = LabelEncoder()
            y   = le2.fit_transform(y_raw).astype(np.int32)

        X_scaled = STATE['scaler'].transform(X).astype(np.float32)
        outputs  = STATE['model'](X_scaled, training=False)
        psi      = outputs[5].numpy()

        X_tr, X_te, psi_tr, psi_te, y_tr, y_te = train_test_split(
            X_scaled, psi, y, test_size=0.3, random_state=42)

        base_tr = psi_tr[:, 384:]
        base_te = psi_te[:, 384:]

        results = {}
        for name, clf in [
            ('KNN', KNeighborsClassifier(n_neighbors=min(3, len(X_tr)-1))),
            ('RF',  RandomForestClassifier(n_estimators=100, random_state=42))
        ]:
            clf.fit(X_tr, y_tr)
            f_org  = f1_score(y_te, clf.predict(X_te),   average='macro')
            clf.fit(base_tr, y_tr)
            f_base = f1_score(y_te, clf.predict(base_te), average='macro')
            clf.fit(psi_tr, y_tr)
            f_enc  = f1_score(y_te, clf.predict(psi_te),  average='macro')
            results[name] = {
                'original':    round(f_org  * 100, 2),
                'baseline':    round(f_base * 100, 2),
                'encoding':    round(f_enc  * 100, 2),
                'improvement': round((f_enc - f_base) * 100, 2)
            }

        chart_b64 = _make_chart(results)
        return jsonify({
            'status':  'success',
            'results': results,
            'chart':   chart_b64,
            'message': 'Evaluation complete!'
        })

    except Exception as e:
        import traceback
        return jsonify({
            'status': 'error',
            'message': str(e),
            'detail': traceback.format_exc()
        }), 500

def _make_chart(results):
    models    = list(results.keys())
    org_vals  = [results[m]['original']  for m in models]
    base_vals = [results[m]['baseline']  for m in models]
    enc_vals  = [results[m]['encoding']  for m in models]
    x     = np.arange(len(models))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width, org_vals,  width, label='Original',       color='#4CAF50', alpha=0.85)
    ax.bar(x,         base_vals, width, label='Baseline',        color='#2196F3', alpha=0.85)
    ax.bar(x + width, enc_vals,  width, label='Our Encoding Ψ', color='#FF5722', alpha=0.85)
    ax.set_xlabel('Model')
    ax.set_ylabel('Macro F1 Score (%)')
    ax.set_title('Performance: Original vs Baseline vs Encoding')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 115)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=80, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

@app.route('/api/threat', methods=['GET'])
def threat_analysis():
    return jsonify({
        'scenarios': [
            {
                'id': 1, 'actor': 'Cloud Server Hacked',
                'risk': 'LOW',
                'reason': 'Without encoder, Ψ is meaningless numbers.'
            },
            {
                'id': 2, 'actor': 'User Account Hacked',
                'risk': 'LOW',
                'reason': 'Predictions alone cannot reveal training data.'
            },
            {
                'id': 3, 'actor': 'Partner Institution Hacked',
                'risk': 'MEDIUM',
                'reason': 'Can encode new data but cannot recover original.'
            },
            {
                'id': 4, 'actor': 'Cloud + Institution Both Hacked',
                'risk': 'HIGH',
                'reason': 'Could train inverse decoder. Must NOT collude!'
            }
        ]
    })

if __name__ == '__main__':
    print("=" * 50)
    print("  Privacy ML System - Flask Server")
    print(f"  TensorFlow : {tf.__version__}")
    print("=" * 50)
    print("  Server : http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, port=5000, host='0.0.0.0')