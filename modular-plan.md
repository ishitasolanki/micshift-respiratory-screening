# MicShift — Modular Plan

## Build order (dependency-ordered, gate first)

| # | Module | Depends on | Gate |
|---|---|---|---|
| 0 | `micshift/channel.py` | numpy, scipy, librosa | **A0 — recover a known synthetic filter.** Project stops if this fails. |
| 1 | `micshift/augment.py` | ffmpeg, channel | Round-trip through real codecs changes the spectrum measurably and reversibly. |
| 2 | `micshift/data.py` | channel, augment | Coswara/ICBHI load, segment, cache without manual steps. |
| 3 | `micshift/features.py` | librosa | Deterministic mel features, fixed shape. |
| 4 | `micshift/model.py` | torch | Small CNN, forward pass shape-correct. |
| 5 | `micshift/train.py` | 2,3,4 | Trains, beats chance in-domain. |
| 6 | `micshift/evaluate.py` | 5 | Ablation table across all configs. |
| 7 | `app.py` | 0,3,4 | Upload clip, get verdict + plots. |

## File structure

```
config.py                 all tunables in one place, no YAML layer
micshift/
  __init__.py
  channel.py              blind estimation, inversion, residual mismatch
  augment.py              codec + sample-rate simulation via real ffmpeg
  data.py                 download, segment (cough vs non-cough), cache
  features.py             mel-spectrogram extraction
  model.py                CNN classifier
  train.py                training loop
  evaluate.py             ablation harness + metrics
app.py                    Streamlit demo
tests/test_channel.py     the A0 gate, runnable standalone
data/                     gitignored — datasets + feature cache
```

## Module responsibilities & interfaces

### `channel.py` — the novelty core
```python
estimate_response(y, sr, noncough_mask) -> np.ndarray   # log-magnitude response, n_bands
invert(y, sr, response) -> np.ndarray                    # channel-equalized audio
residual_mismatch(response, reference) -> float          # scalar, feeds abstention
```
Estimation: average log-magnitude spectrum over non-cough frames, smoothed into
mel-spaced bands, mean-removed (only the *shape* of the response is identifiable
blindly — absolute gain is not, and does not matter). Inversion: apply the
negated response as a zero-phase filter. Magnitude-only by design; phase response
is neither estimable blindly nor relevant to mel features.

### `augment.py`
Simulate device variation with **real encoders**, not approximations: AMR-NB,
Opus, MP3, AAC at assorted bitrates × sample rates {8k, 16k, 22.05k, 44.1k},
plus synthetic IIR device-response filters. Two roles: training augmentation, and
constructing the held-out *simulated unseen device* test conditions.

### `data.py`
Cough/non-cough segmentation by short-time energy + spectral flatness — non-cough
frames are the low-energy, high-flatness ones. Caches features to `.npy` so
retraining never re-runs DSP (NFR-5).

### `model.py`
Small CNN on mel-spectrograms, ~1–2M params. Sized to the dataset (~10k clips),
not to fashion. CPU-inferable (NFR-1).

### `evaluate.py`
Ablations: `{inversion on/off} × {augmentation on/off} × {abstention on/off}`,
plus the reviewer-demanded baselines (SpecAugment, PCEN, heavy-augmentation-only).
Primary output: cross-device retention table + accuracy-vs-abstention curve.

## Testing requirement per module

Each module leaves one runnable `assert`-based check. No framework, no fixtures.
The A0 gate in `tests/test_channel.py` is the only one that can kill the project.

## Integration points

- `data.py` → `channel.py`: segmentation mask feeds estimation.
- `channel.py` → `features.py`: inverted audio feeds mel extraction.
- `augment.py` → `train.py`: augmentation applied per-batch at train time.
- `channel.py` → `app.py`: mismatch score drives the abstain decision in the UI.
