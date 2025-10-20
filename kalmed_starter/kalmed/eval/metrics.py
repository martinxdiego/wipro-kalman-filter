import numpy as np

def rmse(truth, estimate, mask=None):
    truth = np.asarray(truth).ravel()
    estimate = np.asarray(estimate).ravel()
    if mask is not None:
        truth = truth[mask]
        estimate = estimate[mask]
    return float(np.sqrt(np.mean((truth - estimate)**2)))

def snr_gain(truth, noisy, denoised):
    """SNR gain in dB between noisy and denoised vs truth."""
    truth = np.asarray(truth).ravel()
    noisy = np.asarray(noisy).ravel()
    den = np.asarray(denoised).ravel()
    # align lengths
    n = min(truth.size, noisy.size, den.size)
    truth, noisy, den = truth[:n], noisy[:n], den[:n]
    def snr_db(signal, ref):
        num = np.sum((ref)**2)
        deno = np.sum((ref - signal)**2) + 1e-12
        return 10.0*np.log10(num/deno)
    return float(snr_db(noisy, truth) - snr_db(den, truth))
