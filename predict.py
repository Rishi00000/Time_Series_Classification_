import numpy as np
import pandas as pd
import pickle

from features import extract_features

CLASS_MAP = {0: 'Normal', 1: 'Early Fault', 2: 'Fault'}


def load_model(model_path='model.pkl'):
    with open(model_path, 'rb') as f:
        bundle = pickle.load(f)
    return bundle['gbm'], bundle['encoder'], bundle['lr']


def predict(X_raw, model_path='model.pkl'):
    """
    Full stacked inference pipeline.

    Layer 1 — GBM maps each signal's features to leaf node indices.
    Layer 2 — Logistic Regression converts those indices to smooth
              class probabilities via softmax.

    Input:  X_raw of shape (N, 1024)
    Output: predicted labels, confidence scores, full probability matrix
    """
    gbm, enc, lr = load_model(model_path)

    # Feature extraction
    X_feat = extract_features(X_raw).values

    # GBM leaf embeddings → one-hot encoding
    leaves = gbm.apply(X_feat)[:, :, 0]
    X_enc  = enc.transform(leaves)

    # Logistic Regression softmax probabilities
    proba      = lr.predict_proba(X_enc)
    pred_class = np.argmax(proba, axis=1)
    confidence = np.max(proba, axis=1)

    return pred_class, confidence, proba


if __name__ == '__main__':
    data = np.load('data/val.npz')
    X_raw, y_true = data['X'], data['y']

    pred_class, confidence, proba = predict(X_raw)

    results = pd.DataFrame({
        'sample_id'       : np.arange(len(pred_class)),
        'predicted_label' : [CLASS_MAP[p] for p in pred_class],
        'confidence_score': np.round(confidence, 4),
        'p_normal'        : np.round(proba[:, 0], 4),
        'p_early_fault'   : np.round(proba[:, 1], 4),
        'p_fault'         : np.round(proba[:, 2], 4),
    })

    results.to_csv('predictions.csv', index=False)
    print(results.to_string(index=False))
    print(f"\nSaved to predictions.csv")