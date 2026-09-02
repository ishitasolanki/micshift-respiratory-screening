"""Device-variation simulation: real codecs, real resampling, synthetic mic responses.

No public respiratory corpus is recorded across multiple device types, so the
device mismatch this project defends against has to be synthesized. Fidelity
matters: these use actual ffmpeg encoders rather than spectral approximations,
because a phone's damage comes from a specific encoder's psychoacoustic model,
which is not reproducible by hand-waving a lowpass filter over the signal.
"""
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import imageio_ffmpeg

import config

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

# (codec, ffmpeg args, container) — the pipelines real phones and apps actually use.
CODECS = {
    "none":     (None, None),
    "amr_nb":   (["-ar", "8000", "-ab", "12.2k", "-c:a", "libopencore_amrnb"], ".amr"),
    "opus_24k": (["-c:a", "libopus", "-b:a", "24k"], ".ogg"),
    "opus_64k": (["-c:a", "libopus", "-b:a", "64k"], ".ogg"),
    "mp3_64k":  (["-c:a", "libmp3lame", "-b:a", "64k"], ".mp3"),
    "mp3_128k": (["-c:a", "libmp3lame", "-b:a", "128k"], ".mp3"),
    "aac_64k":  (["-c:a", "aac", "-b:a", "64k"], ".m4a"),
}

SAMPLE_RATES = [8000, 16000, 22050, 44100]


def available_codecs():
    """Codecs this ffmpeg build can actually encode.

    Builds differ in which encoders they ship (libopencore_amrnb especially).
    Probing once beats a training run that dies on batch 400.
    """
    out = subprocess.run([FFMPEG, "-encoders"], capture_output=True, text=True).stdout
    ok = ["none"]
    for name, (args, _) in CODECS.items():
        if args is None:
            continue
        enc = args[args.index("-c:a") + 1]
        if enc in out:
            ok.append(name)
    return ok


def apply_codec(y, sr, codec):
    """Round-trip audio through a real encoder and back to a float array."""
    if codec == "none" or CODECS[codec][0] is None:
        return y
    args, ext = CODECS[codec]
    with tempfile.TemporaryDirectory() as td:
        src, enc = Path(td) / "in.wav", Path(td) / f"enc{ext}"
        dec = Path(td) / "dec.wav"
        sf.write(src, y, sr)
        r = subprocess.run([FFMPEG, "-y", "-i", str(src), *args, str(enc)],
                           capture_output=True)
        if r.returncode != 0 or not enc.exists():
            return y  # encoder unavailable -> pass through rather than kill training
        # Decode with ffmpeg as well: libsndfile cannot read AMR or M4A, and
        # guessing which container it supports is how this breaks in the field.
        r = subprocess.run([FFMPEG, "-y", "-i", str(enc), "-ar", str(sr),
                            "-ac", "1", str(dec)], capture_output=True)
        if r.returncode != 0 or not dec.exists():
            return y
        out, _ = sf.read(dec, dtype="float32")
    # Encoders pad to their frame size (AMR especially). Restore the original
    # length so a codec round-trip never changes a clip's duration.
    if len(out) < len(y):
        out = np.pad(out, (0, len(y) - len(out)))
    return out[:len(y)]


def resample_roundtrip(y, sr, target_sr):
    """Downsample to a device's rate and back — models the bandwidth loss.

    The information destroyed above the target Nyquist does not come back; the
    return trip only restores the array shape the model expects.
    """
    if target_sr == sr:
        return y
    return librosa.resample(librosa.resample(y, orig_sr=sr, target_sr=target_sr),
                            orig_sr=target_sr, target_sr=sr)


def random_mic_response(rng, sr=config.SR, n_bands=config.N_BANDS, strength=6.0):
    """A plausible random microphone magnitude response, in dB, mean-removed.

    Built as a low-frequency tilt plus one or two resonant bumps — the shape
    real handset mics actually exhibit, not white noise over frequency.
    """
    from micshift.channel import band_centers
    f = band_centers(sr, n_bands)
    tilt = rng.uniform(-1, 1) * strength * np.log10(f / 1000 + 0.1)
    resp = tilt
    for _ in range(rng.integers(1, 3)):
        fc = rng.uniform(500, min(6000, sr / 2 * 0.8))
        resp = resp + rng.uniform(-1, 1) * strength * np.exp(-((f - fc) / rng.uniform(400, 1500)) ** 2)
    # Handset mics roll off hard below ~150 Hz.
    resp = resp - strength * 1.5 * np.exp(-(f / 150) ** 2)
    return resp - resp.mean()


def apply_response(y, response, sr=config.SR):
    """Colour audio with a given magnitude response (the inverse of channel.invert)."""
    from micshift.channel import _bands_to_bins
    D = librosa.stft(y, n_fft=config.N_FFT, hop_length=config.HOP)
    gain = (10 ** (_bands_to_bins(response, sr, config.N_FFT, len(response)) / 20))[:, None]
    return librosa.istft(D * gain, hop_length=config.HOP, length=len(y))


def simulate_device(y, sr, rng, codecs=None, strength=6.0):
    """Full device pipeline: mic response, then rate limit, then codec.

    Ordered to match physical reality — the microphone colours the sound before
    the phone's audio stack ever resamples or encodes it.
    """
    codecs = codecs or available_codecs()
    resp = random_mic_response(rng, sr, strength=strength)
    out = apply_response(y, resp, sr)
    out = resample_roundtrip(out, sr, rng.choice(SAMPLE_RATES))
    out = apply_codec(out, sr, str(rng.choice(codecs)))
    return out, resp


def _demo():
    rng = np.random.default_rng(config.SEED)
    sr, n = config.SR, config.SR * 3
    y = rng.normal(0, 0.05, n)
    t = np.arange(n) / sr
    y += 0.3 * np.sin(2 * np.pi * 440 * t)   # tone gives codecs something real to mangle

    ok = available_codecs()
    print("  available codecs:", ", ".join(ok))
    assert len(ok) >= 3, f"need real codecs, only found {ok}"

    for c in ok:
        out = apply_codec(y, sr, c)
        assert len(out) > 0 and np.isfinite(out).all(), f"{c} produced invalid audio"
        delta = np.abs(librosa.stft(out[:n], n_fft=512)).mean() - np.abs(librosa.stft(y, n_fft=512)).mean()
        print(f"    {c:9s} len={len(out):6d}  mean-spectrum delta={delta:+.4f}")
        if c != "none":
            assert not np.allclose(out[:len(y)], y, atol=1e-4), f"{c} was a no-op"

    # Resampling to 8 kHz must destroy energy above 4 kHz.
    lo = resample_roundtrip(y, sr, 8000)
    hi_before = np.abs(librosa.stft(y, n_fft=512))[160:].mean()
    hi_after = np.abs(librosa.stft(lo, n_fft=512))[160:].mean()
    print(f"  >5kHz energy  {hi_before:.4f} -> {hi_after:.4f} after 8k round-trip")
    assert hi_after < hi_before * 0.2, "8 kHz round-trip did not remove high band"

    # A simulated response must be recoverable by the estimator -- this is the
    # closed loop the whole project rests on: augment.py colours it, channel.py
    # must be able to take that colour back out.
    from micshift import channel
    quiet = rng.normal(0, 0.005, sr * 4)
    resp = random_mic_response(rng)
    col = apply_response(quiet, resp, sr)
    est = channel.estimate_response(col, sr)
    corr = float(np.corrcoef(est, resp)[0, 1])
    print(f"  simulated-response recovery correlation: {corr:.3f}")
    assert corr > 0.9, f"estimator cannot recover simulated device response ({corr:.3f})"

    print("\naugment OK - real codecs applied, responses recoverable.")


if __name__ == "__main__":
    _demo()
