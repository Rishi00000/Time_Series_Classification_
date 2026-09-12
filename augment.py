import numpy as np


def augment_dataset(X, y, multiplier=4, random_state=42):
    """
    Expand training data by creating augmented copies of each signal.

    Three augmentations applied per copy:
      - Additive Gaussian noise  : simulates sensor measurement noise
      - Amplitude scaling        : simulates different operating loads
      - Time shift               : simulates different capture moments

    Args:
        X          : raw signal array of shape (N, 1024)
        y          : labels of shape (N,)
        multiplier : how many total copies to create (4 = 4x the data)
        random_state: for reproducibility

    Returns:
        X_aug, y_aug : augmented arrays
    """
    rng = np.random.RandomState(random_state)
    X_aug, y_aug = [X], [y]   # start with original data

    for _ in range(multiplier - 1):
        X_new = X.copy()

        # 1. Additive Gaussian noise — 2% of each signal's std
        noise = rng.normal(0, 0.02 * X.std(axis=1, keepdims=True), X.shape)
        X_new = X_new + noise

        # 2. Amplitude scaling — random scale between 0.9x and 1.1x
        scale = rng.uniform(0.9, 1.1, size=(X.shape[0], 1))
        X_new = X_new * scale

        # 3. Time shift — shift signal up to 50 samples left or right
        shift = rng.randint(-50, 50)
        X_new = np.roll(X_new, shift, axis=1)

        X_aug.append(X_new)
        y_aug.append(y)

    return np.vstack(X_aug), np.concatenate(y_aug)