"""Blind channel magnitude response estimation, inversion, and residual mismatch.

The novelty core. Estimates the recording device's magnitude response from the
non-cough (silence / breath / ambient) frames of a clip, with no reference
signal and no device metadata, then inverts it toward a canonical channel.

Only the *shape* of the response is blindly identifiable — absolute gain is
confounded with source loudness, so every response here is mean-removed and
lives in dB. Phase is neither estimable this way nor relevant to mel features.
"""
import numpy as np
import librosa

import config


def band_centers(sr=config.SR, n_bands=config.N_BANDS):
    """Geometric centres of log-spaced analysis bands.

    Log spacing, not mel: mel filterbanks apply bandwidth normalization that
    imposes a spectral tilt of their own, which would be indistinguishable from
    the device tilt we are trying to measure.
    """
    edges = np.geomspace(config.FMIN, sr / 2, n_bands + 1)
    return np.sqrt(edges[:-1] * edges[1:])


def _band_average(spec_db, sr, n_fft, n_bands):
    """Average a dB spectrum within log-spaced bands."""
    bin_f = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    edges = np.geomspace(config.FMIN, sr / 2, n_bands + 1)
    idx = np.digitize(bin_f, edges) - 1
    out = np.empty(n_bands)
    for b in range(n_bands):
        sel = idx == b
        # Empty low bands can occur at coarse FFT resolution; fall back to the
        # nearest bin so the response stays defined across the whole range.
        out[b] = spec_db[sel].mean() if sel.any() else spec_db[np.argmin(np.abs(bin_f - edges[b]))]
    return out


def noncough_mask(y, sr=config.SR):
    """Frames that carry channel information rather than cough energy.

    Cough is loud and spectrally structured; the channel is best observed in
    quiet, noise-like frames where the source is broadband and featureless.
    Selects low-energy frames, then keeps the spectrally flattest of those.
    """
    S = np.abs(librosa.stft(y, n_fft=config.N_FFT, hop_length=config.HOP))
    energy = S.sum(axis=0)
    flatness = librosa.feature.spectral_flatness(S=S)[0]

    quiet = energy <= np.percentile(energy, config.ENERGY_PERCENTILE)
    if quiet.sum() == 0:
        return quiet
    # Among quiet frames only, keep the flatter (more noise-like) half.
    flat_thresh = np.percentile(flatness[quiet], config.FLATNESS_PERCENTILE)
    return quiet & (flatness >= flat_thresh)


def estimate_response(y, sr=config.SR, mask=None, n_bands=config.N_BANDS):
    """Blind estimate of the channel magnitude response, in dB, mean-removed.

    Returns None when too few channel-carrying frames exist to estimate from —
    the caller must treat that as "cannot equalize", not as a flat response.
    """
    if mask is None:
        mask = noncough_mask(y, sr)
    if mask.sum() < config.MIN_NONCOUGH_FRAMES:
        return None

    S = np.abs(librosa.stft(y, n_fft=config.N_FFT, hop_length=config.HOP))

    # A channel is multiplicative in magnitude, hence ADDITIVE in log. Averaging
    # log-spectra across frames is therefore the correct estimator; averaging
    # power first would let the loudest frames dominate the channel estimate.
    log_spec = 20 * np.log10(S[:, mask] + 1e-10)
    mean_log = log_spec.mean(axis=1)

    resp_db = _band_average(mean_log, sr, config.N_FFT, n_bands)

    if config.SMOOTH_BANDS > 1:
        # Edge-replicate before smoothing. A zero-padded "same" convolution
        # drags the first and last bands toward zero, which reads as a fake
        # roll-off at exactly the band edges where device responses differ most.
        k = np.ones(config.SMOOTH_BANDS) / config.SMOOTH_BANDS
        pad = config.SMOOTH_BANDS // 2
        resp_db = np.convolve(np.pad(resp_db, pad, mode="edge"), k, mode="valid")

    return resp_db - resp_db.mean()


def _bands_to_bins(resp_db, sr, n_fft, n_bands):
    """Interpolate a band-spaced response up to FFT-bin resolution."""
    bin_f = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    return np.interp(bin_f, band_centers(sr, n_bands), resp_db)


def invert(y, sr=config.SR, response=None):
    """Apply the inverse of the estimated response — channel equalization.

    Zero-phase: only magnitudes are scaled, the original phase is kept, so the
    result stays a valid waveform without introducing group-delay artifacts.
    """
    if response is None:
        response = estimate_response(y, sr)
    if response is None:
        return y  # cannot estimate -> leave audio untouched, abstention catches it

    D = librosa.stft(y, n_fft=config.N_FFT, hop_length=config.HOP)
    gain_db = -_bands_to_bins(response, sr, config.N_FFT, len(response))
    # Clamp the boost. Where a device's resampling or codec has zeroed a band,
    # the estimated response dives toward -inf and an unclamped inverse applies
    # enormous gain to what is now only noise -- which measurably degrades the
    # features rather than restoring them. Destroyed bandwidth cannot be
    # equalized back; it is detected instead, and handled by abstention.
    gain_db = np.clip(gain_db, -config.MAX_CUT_DB, config.MAX_BOOST_DB)
    # Leave destroyed bands strictly alone. Boosting a band that holds no signal
    # only raises its noise floor, which is worse than the untouched band: there
    # is nothing there to restore. These bands are reported via
    # dead_band_fraction and handled by abstention instead.
    resp_bins = _bands_to_bins(response, sr, config.N_FFT, len(response))
    gain_db[resp_bins < -config.DEAD_BAND_DB] = 0.0
    gain = (10 ** (gain_db / 20))[:, None]
    return librosa.istft(D * gain, hop_length=config.HOP, length=len(y))


def calibrate_bound(mismatches, target_abstain=config.TARGET_ABSTAIN_RATE):
    """Pick the abstention bound from data rather than guessing it.

    The bound is the quantile of in-domain mismatch that leaves
    `target_abstain` of matched-condition clips rejected. A hardcoded dB value
    is meaningless across corpora: absolute mismatch depends on how quiet the
    recordings are and how much non-cough audio each clip contains.
    """
    m = np.asarray(mismatches, dtype=float)
    m = m[np.isfinite(m)]
    if len(m) == 0:
        return config.MISMATCH_BOUND
    return float(np.quantile(m, 1.0 - target_abstain))


def dead_band_fraction(y, sr=config.SR):
    """Fraction of the spectrum a device destroyed outright (codec / resampling).

    Distinct from channel colouring: colouring is invertible, destroyed
    bandwidth is not. This is the part of the mismatch that no equalizer can fix.
    """
    resp = estimate_response(y, sr)
    if resp is None:
        return 1.0
    return float((resp < -config.DEAD_BAND_DB).mean())


def residual_mismatch(y, sr=config.SR, reference=None):
    """RMS dB deviation of a clip's channel from the canonical reference.

    This is the abstention signal: how far the clip's channel still sits from
    the training distribution's centre after equalization. Large value means the
    equalization did not bring this device into the domain the model was trained
    on, so the prediction should not be trusted.
    """
    resp = estimate_response(y, sr)
    if resp is None:
        return np.inf  # unestimable channel is maximally mismatched -> abstain
    if reference is None:
        reference = np.zeros_like(resp)
    return float(np.sqrt(np.mean((resp - reference) ** 2)))


# ---------------------------------------------------------------- A0 gate ----
def _demo():
    """A0 gate: does blind estimation recover a KNOWN filter from non-cough frames?

    The load-bearing assumption of the whole project. Synthesize a clip with a
    cough-like burst plus quiet noise, push it through a known colouring filter,
    and check the blind estimate recovers that filter's shape — using only the
    non-cough frames, never the ground-truth filter.
    """
    rng = np.random.default_rng(config.SEED)
    sr = config.SR
    n = int(sr * 4.0)

    # Clip: quiet broadband noise floor + two loud cough-like bursts.
    y = rng.normal(0, 0.005, n)
    for start in (int(sr * 1.0), int(sr * 2.5)):
        d = int(sr * 0.3)
        burst = rng.normal(0, 0.3, d) * np.exp(-np.linspace(0, 8, d))
        y[start:start + d] += burst

    # Known device colouring: a tilt plus a resonant bump, applied in frequency.
    bin_f = librosa.fft_frequencies(sr=sr, n_fft=config.N_FFT)
    true_db = 6 * np.log10(bin_f / 1000 + 0.1) + 5 * np.exp(-((bin_f - 3000) / 800) ** 2)
    D = librosa.stft(y, n_fft=config.N_FFT, hop_length=config.HOP)
    y_col = librosa.istft(D * (10 ** (true_db / 20))[:, None],
                          hop_length=config.HOP, length=n)

    est = estimate_response(y_col, sr)
    assert est is not None, "A0 FAILED: no usable non-cough frames found"

    # Compare shapes on a common band grid, both mean-removed.
    true_bands = np.interp(band_centers(sr), bin_f, true_db)
    true_bands -= true_bands.mean()

    corr = float(np.corrcoef(est, true_bands)[0, 1])
    err = float(np.sqrt(np.mean((est - true_bands) ** 2)))

    # Baseline: how wrong would we be assuming a flat response (doing nothing)?
    flat_err = float(np.sqrt(np.mean(true_bands ** 2)))

    print(f"  correlation with true response : {corr:.3f}")
    print(f"  RMS error (estimated)          : {err:.2f} dB")
    print(f"  RMS error (assume flat/no-op)  : {flat_err:.2f} dB")
    print(f"  error reduction                : {100 * (1 - err / flat_err):.0f}%")

    assert corr > 0.9, f"A0 FAILED: correlation {corr:.3f} too low"
    assert err < flat_err / 2, f"A0 FAILED: {err:.2f} dB not < half of {flat_err:.2f} dB"

    # Inversion must actually flatten the channel, not just estimate it.
    y_eq = invert(y_col, sr)
    resid = estimate_response(y_eq, sr)
    resid_err = float(np.sqrt(np.mean((resid - true_bands * 0) ** 2)))
    before = float(np.sqrt(np.mean(estimate_response(y_col, sr) ** 2)))
    print(f"  channel spread before invert   : {before:.2f} dB")
    print(f"  channel spread after  invert   : {resid_err:.2f} dB")
    assert resid_err < before / 2, "A0 FAILED: inversion did not flatten the channel"

    # Unestimable clip (pure loud cough, no quiet frames) must abstain, not guess.
    assert residual_mismatch(np.zeros(100)) == np.inf, "A0 FAILED: short clip must abstain"

    print("\nA0 GATE PASSED - blind estimation recovers a known channel.")


if __name__ == "__main__":
    _demo()
