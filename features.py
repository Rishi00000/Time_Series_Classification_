import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import fft


def extract_features(X):
    """
    Convert raw signal matrix to feature matrix.
    Input:  X of shape (N, 1024)
    Output: DataFrame of shape (N, 22)
    """
    features = []
    n = X.shape[1]

    for sig in X:
        feat = {}

        # Time domain
        feat['mean']             = np.mean(sig)
        feat['std']              = np.std(sig)
        feat['rms']              = np.sqrt(np.mean(sig**2))
        feat['peak']             = np.max(np.abs(sig))
        feat['peak_to_peak']     = np.max(sig) - np.min(sig)
        feat['skewness']         = stats.skew(sig)
        feat['kurtosis']         = stats.kurtosis(sig)
        feat['crest_factor']     = feat['peak'] / (feat['rms'] + 1e-10)
        feat['shape_factor']     = feat['rms']  / (np.mean(np.abs(sig)) + 1e-10)
        feat['impulse_factor']   = feat['peak'] / (np.mean(np.abs(sig)) + 1e-10)
        feat['clearance_factor'] = feat['peak'] / (np.mean(np.sqrt(np.abs(sig)))**2 + 1e-10)
        feat['variance']         = np.var(sig)
        feat['zcr']              = np.sum(np.diff(np.sign(sig)) != 0) / n

        # Frequency domain
        fft_vals    = np.abs(fft(sig))[:n//2]
        freqs       = np.arange(n//2)
        fft_power   = fft_vals**2
        total_power = np.sum(fft_power) + 1e-10

        feat['spectral_centroid'] = np.sum(freqs * fft_power) / total_power
        feat['spectral_spread']   = np.sqrt(np.sum(((freqs - feat['spectral_centroid'])**2) * fft_power) / total_power)
        feat['spectral_energy']   = np.sum(fft_power)
        feat['spectral_entropy']  = -np.sum((fft_power / total_power) * np.log(fft_power / total_power + 1e-10))
        feat['spectral_kurtosis'] = stats.kurtosis(fft_power)
        feat['band_low']          = np.sum(fft_power[:n//6])        / total_power
        feat['band_mid']          = np.sum(fft_power[n//6 : n//3])  / total_power
        feat['band_high']         = np.sum(fft_power[n//3:])        / total_power
        feat['dom_freq_mag']      = np.max(fft_vals)

        features.append(feat)

    return pd.DataFrame(features)