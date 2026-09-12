import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from predict import predict

CLASS_NAMES = ['Normal', 'Early Fault', 'Fault']


def evaluate(data_path='data/val.npz', model_path='model.pkl'):

    data  = np.load(data_path)
    X_raw = data['X']
    y_true = data['y']

    pred_class, confidence, proba = predict(X_raw, model_path)

    acc = (pred_class == y_true).mean()
    print(f"\nAccuracy : {acc*100:.2f}%")
    print(f"Errors   : {(pred_class != y_true).sum()} / {len(y_true)}\n")
    print(classification_report(y_true, pred_class, target_names=CLASS_NAMES))

    # Confusion matrix plot
    cm = confusion_matrix(y_true, pred_class)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    print("Confusion matrix saved to confusion_matrix.png")


if __name__ == '__main__':
    evaluate()