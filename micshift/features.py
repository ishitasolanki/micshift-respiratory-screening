"""Mel-spectrogram features, and dataset construction with device simulation.

Two design decisions here decide whether any downstream number is meaningful:

1. **Split by participant, never by clip.** Coswara gives each participant two
   cough recordings. Splitting by clip puts the same person's throat in both
   train and test, and the model scores well by recognising the person rather
   than the condition.

2. **Train and test device pools are disjoint.** The claim is cross-device
   retention on *unseen* devices. Test devices are drawn from a separate RNG
   stream and a held-out codec set, so a test device is never one the model was
   augmented with during training.
"""
import numpy as np
import librosa

import config
from micshift import channel, augment

# Codecs disjoint between the pools: a test device's encoder was never trained on.
TRAIN_CODECS = ["none", "mp3_128k", "opus_64k", "aac_64k"]
TEST_CODECS = ["amr_nb", "opus_24k", "mp3_64k"]


def melspec(y, sr=config.SR):
    """Fixed-shape log-mel spectrogram, per-clip normalized."""
    m = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=config.N_FFT, hop_length=config.HOP,
        n_mels=config.N_MELS, fmin=config.FMIN, fmax=sr / 2)
    m = librosa.power_to_db(m, ref=np.max)
    # Per-clip standardization removes overall level, which carries no class
    # information and varies with how close the phone was held.
    return ((m - m.mean()) / (m.std() + 1e-6)).astype(np.float32)


def apply_device(y, rng, codecs, strength=6.0):
    """Push a clip through one simulated device (mic response + rate + codec)."""
    resp = augment.random_mic_response(rng, strength=strength)
    out = augment.apply_response(y, resp, config.SR)
    out = augment.resample_roundtrip(out, config.SR, int(rng.choice(augment.SAMPLE_RATES)))
    out = augment.apply_codec(out, config.SR, str(rng.choice(codecs)))
    if len(out) < len(y):
        out = np.pad(out, (0, len(y) - len(out)))
    return out[:len(y)]


def featurize(y, invert_channel):
    """Clip -> (features, residual mismatch).

    Mismatch is measured on the *equalized* audio: it reports how far this clip's
    channel still sits from canonical after MicShift did its work, which is
    exactly the quantity the abstention rule needs.
    """
    resp = channel.estimate_response(y)
    if resp is None:
        return melspec(y), np.inf

    y_eq = channel.invert(y, config.SR, resp) if invert_channel else y

    resid = channel.estimate_response(y_eq)
    shape_mismatch = float(np.sqrt(np.mean(resid ** 2))) if resid is not None else np.inf

    # Residual channel shape is what equalization can still be blamed for;
    # destroyed bandwidth is what it can never fix. The abstention signal has to
    # carry both, or a clip that has been lowpassed to 4 kHz looks perfectly
    # well-matched right up until the classifier gets it wrong.
    dead = channel.dead_band_fraction(y)
    return melspec(y_eq), shape_mismatch + config.DEAD_BAND_WEIGHT * dead


def split_participants(clips, seed=config.SEED, fracs=(0.7, 0.15)):
    """Partition PARTICIPANTS (not clips) into train/val/test."""
    ids = sorted({pid for _, _, pid in clips})
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_tr = int(len(ids) * fracs[0])
    n_va = int(len(ids) * fracs[1])
    return set(ids[:n_tr]), set(ids[n_tr:n_tr + n_va]), set(ids[n_tr + n_va:])


def _demo():
    from micshift import data
    rng = np.random.default_rng(config.SEED)

    # Local data only: a self-check must never trigger a multi-GB download.
    clips = data.collect_clips(dates=data.local_dates(), download=False)
    assert clips, "no clips on disk - run `python -m micshift.data` first"
    print(f"  {len(clips)} clips from {len({c[2] for c in clips})} participants")

    tr, va, te = split_participants(clips)
    assert not (tr & te) and not (tr & va), "participant leakage between splits"
    print(f"  split: {len(tr)} train / {len(va)} val / {len(te)} test participants (disjoint)")

    y = data.load_audio(clips[0][0])
    f = melspec(y)
    assert f.shape[0] == config.N_MELS and np.isfinite(f).all()
    print(f"  feature shape {f.shape}")

    ys = [a for c in clips[:12] if (a := data.load_audio(c[0])) is not None]

    def mean_shift(make_device):
        raw, inv, mism = [], [], []
        for i, a in enumerate(ys):
            r = np.random.default_rng(100 + i)
            d = make_device(a, r)
            raw.append(np.abs(melspec(a) - melspec(d)).mean())
            fo, _ = featurize(a, True)
            fd, m = featurize(d, True)
            inv.append(np.abs(fo - fd).mean())
            mism.append(m)
        return np.mean(raw), np.mean(inv), np.mean(mism)

    # 1. Channel COLOURING is invertible -- this is what MicShift actually fixes.
    colour = lambda a, r: augment.apply_response(a, augment.random_mic_response(r), config.SR)
    r0, i0, _ = mean_shift(colour)
    print(f"  mic colouring   : raw={r0:.3f}  inverted={i0:.3f}  ({100*(1-i0/r0):.0f}% reduction)")
    assert i0 < r0, f"inversion failed on invertible channel colouring ({i0:.3f} vs {r0:.3f})"

    # 2. Destroyed BANDWIDTH is not invertible -- no equalizer restores a band
    #    that resampling zeroed. The requirement is that MicShift DETECTS it, so
    #    the reject option fires instead of the model guessing. This asymmetry is
    #    the reason the abstention rule exists at all.
    clean_mm = np.mean([featurize(a, True)[1] for a in ys])
    _, _, band_mm = mean_shift(lambda a, r: augment.resample_roundtrip(a, config.SR, 8000))
    print(f"  mismatch score  : clean={clean_mm:.1f}   bandwidth-destroyed={band_mm:.1f}")
    assert band_mm > clean_mm * 3, "mismatch score fails to flag destroyed bandwidth"
    assert clean_mm < config.MISMATCH_BOUND, "clean clips would be wrongly abstained on"

    print("\nfeatures OK - splits clean; colouring inverted, destroyed bandwidth detected.")


if __name__ == "__main__":
    _demo()
