import numpy as np
import pickle
import warnings
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
warnings.filterwarnings('ignore')

from features import extract_features
from augment import augment_dataset

CLASS_MAP     = {0: 'Normal', 1: 'Early Fault', 2: 'Fault'}
CLASS_WEIGHTS = {0: 1, 1: 5, 2: 5}


def train(train_path='data/train.npz', val_path='data/val.npz', save_path='model.pkl'):

    # ── LOAD DATA ─────────────────────────────────────────────────────
    train_data = np.load(train_path)
    val_data   = np.load(val_path)
    X_train_raw, y_train = train_data['X'], train_data['y']
    X_val_raw,   y_val   = val_data['X'],   val_data['y']

    # ── AUGMENT (train only) ───────────────────────────────────────────
    print("Augmenting training data...")
    X_train_raw, y_train = augment_dataset(X_train_raw, y_train, multiplier=4)
    print(f"  Training samples: {len(y_train) // 4} → {len(y_train)}")

    # ── EXTRACT FEATURES ──────────────────────────────────────────────
    print("Extracting features...")
    X_train = extract_features(X_train_raw).values
    X_val   = extract_features(X_val_raw).values

    # ── LAYER 1: GRADIENT BOOSTING ────────────────────────────────────
    print("Training GradientBoosting layer...")
    gbm = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=8,
        random_state=42
    )
    gbm.fit(X_train, y_train)

    # ── LEAF EMBEDDINGS ───────────────────────────────────────────────
    X_train_leaves = gbm.apply(X_train)[:, :, 0]
    X_val_leaves   = gbm.apply(X_val)[:, :, 0]

    # ── ONE-HOT ENCODE LEAF INDICES ───────────────────────────────────
    enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_train_enc = enc.fit_transform(X_train_leaves)
    X_val_enc   = enc.transform(X_val_leaves)

    # ── LAYER 2: LOGISTIC REGRESSION ──────────────────────────────────
    print("Training Logistic Regression layer...")
    lr = LogisticRegression(
        C=1.0,
        class_weight=CLASS_WEIGHTS,
        max_iter=1000,
        random_state=42
    )
    lr.fit(X_train_enc, y_train)

    # ── VALIDATE ──────────────────────────────────────────────────────
    print("Validating...")
    val_proba = lr.predict_proba(X_val_enc)
    val_pred  = np.argmax(val_proba, axis=1)
    val_acc   = (val_pred == y_val).mean()
    print(f"  Validation Accuracy : {val_acc * 100:.2f}%")
    print(f"  Confidence range    : {val_proba.max(axis=1).min():.4f} → {val_proba.max(axis=1).max():.4f}")

    # ── SAVE ──────────────────────────────────────────────────────────
    with open(save_path, 'wb') as f:
        pickle.dump({'gbm': gbm, 'encoder': enc, 'lr': lr}, f)

    print(f"Model saved to {save_path}")


if __name__ == '__main__':
    train()