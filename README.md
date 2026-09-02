# MicShift

Microphone-invariant respiratory audio screening.

Cough-based screening models degrade sharply when the recording device changes —
different phone microphones, sample rates and codecs. In a mass-screening
deployment where an ASHA worker uses whatever phone she owns, the device is
uncontrolled, and this heterogeneity, not model capacity, is the binding
constraint on field deployment.

MicShift estimates the recording device's magnitude response **blindly, from the
non-cough segments of the same clip**, inverts it, trains against simulated
device variation, and **abstains** when residual channel mismatch exceeds a
calibrated bound.

## The core idea

```
audio clip
    │
    ├── non-cough frames (silence / breath) ──> blind channel estimate H(f)
    │                                                    │
    └────────────── invert with 1/H(f) <─────────────────┘
                          │
                    mel features ──> CNN ──> prediction
                          │
                 residual mismatch ──> within bound ? predict : ABSTAIN
```

The device response is estimated from the audio the model does *not* classify.
No calibration recording, no device metadata, no reference signal.

## What works, and what provably cannot

This distinction is the point of the method, not a caveat:

| Device effect | Invertible? | How MicShift handles it |
|---|---|---|
| Microphone colouring (tilt, resonances) | **Yes** | Estimated and inverted. Measured 23% reduction in device-induced feature shift. |
| Codec artefacts | Partly | Reduced by inversion, simulated during training. |
| Destroyed bandwidth (8 kHz resampling, AMR) | **No** | Cannot be recovered — the band is gone. Detected and routed to abstention. |

A band that resampling zeroed holds no signal to restore; boosting it only
amplifies noise, which measurably *degrades* accuracy. MicShift therefore clamps
inversion gain, leaves destroyed bands untouched, and reports them through the
mismatch score. Clean clips score ≈2 dB mismatch; 8 kHz-resampled clips score
≈19 dB — cleanly separable, so the reject option fires.

## Results

Coswara, 25 collection dates: 1,842 clips from 923 participants, ~50/50 class
balance. Split by participant; unseen-device test clips use codecs and mic
responses never seen in training.

| Metric | Value |
|---|---|
| Matched-device balanced accuracy | 0.660 |
| **Unseen-device balanced accuracy** | **0.628** |
| **Cross-device retention** | **95.1%** |
| Abstention rate | 24.5% |
| Accuracy on kept clips | 0.676 |
| Accuracy on abstained clips | 0.507 |
| Abstention precision | 0.493 (vs 0.365 base error rate) |

**Read these honestly.** Balanced accuracy of 0.63–0.66 is a weak classifier —
above chance (0.500) and consistent with published Coswara cough-classification
results, but not a deployable screening tool on its own. The classifier is not
the contribution; the channel handling is. Training balanced accuracy reaches
0.990 against 0.675 validation, so the model still overfits: this corpus is
small for the task.

The abstention result is the cleanest finding. Rejected clips would have been
wrong 49.3% of the time against a 36.5% base error rate — a 1.35× enrichment for
errors, meaning the reject option is selectively catching clips the model gets
wrong rather than discarding at random.

## Ablation: does channel inversion actually help?

Channel inversion does **not** improve cross-device retention on this corpus. It
makes it worse, and a five-line SpecAugment baseline beats it.

Coswara, 25 dates, 1,842 clips / 925 participants, split by participant, unseen
devices using codecs and mic responses never seen in training.

| configuration | matched bal | unseen bal | **retention** | abstain | kept acc | rejected acc | abst. prec |
|---|---|---|---|---|---|---|---|
| no-inversion (raw) | 0.658 | 0.650 | **98.9%** | 54.0% | 0.607 | 0.692 | 0.308 |
| SpecAugment | 0.657 | **0.665** | **101.2%** | 54.0% | 0.592 | 0.722 | 0.278 |
| **MicShift (inversion)** | 0.660 | 0.628 | **95.1%** | 24.5% | **0.676** | 0.507 | **0.493** |

**Why it fails: the headroom was never there.** No-inversion already retains
98.9%, so unseen devices were costing about 1% to begin with. All three
configurations train on the same codec- and sample-rate-augmented data, and
**augmentation alone already solves the cross-device problem**. Inversion then
adds its own distortion — clamped boosts amplifying noise in low-SNR bands — and
removes roughly 2% of real signal to recover a 1% loss. Net negative.

**What does survive: the reject option.** MicShift abstains on 24.5% of clips at
precision 0.493; the baselines abstain on 54% at precision ~0.29. More
importantly, note the *direction*: for both baselines the rejected clips score
**higher** (0.692, 0.722) than the kept clips (0.607, 0.592) — their abstention
is backwards, discarding the clips the model gets right. Only the
post-inversion mismatch score orders correctly (kept 0.676 > rejected 0.507).

**Honest conclusion:** blind channel inversion works as *signal processing* (it
recovers a known channel at 0.996 correlation and cuts device-induced feature
shift 23%), but as a *robustness method for this task it is not justified* —
codec/SR augmentation is simpler and better. Its surviving contribution is a
usable confidence signal for a bounded reject option.

**What would change this verdict:** a test where augmentation cannot already
close the gap — real recordings from physically different handsets (Phase B), or
a train/test device split far more severe than simulated augmentation covers. On
this corpus the claim is not supported, and should not be filed or published as
though it were.

## Known limitation: the reject option is weak on real phone chains

Measured, not assumed. Detecting destroyed bandwidth works on a clean lowpass
but **largely fails once a codec re-encodes the result** — which is the actual
deployment case.

| Degradation | dead-band detected | mismatch score |
|---|---|---|
| 8 kHz resample only | 0.15 ✅ | 26.1 dB (vs ~2 clean) |
| **8 kHz → AMR-NB** (a cheap phone) | **0.00 ❌** | **5.5 dB** |

The AMR decoder resamples back up on the way out and **refills the destroyed
band with codec noise**, lifting it from −70 dB to −15 dB — above the −25 dB
dead-band threshold, so the detector misses it.

Separating clean clips from 8 kHz+AMR clips across 40 Coswara recordings gives
**AUC 0.572** — barely above chance. No threshold rescues it: a 4 dB bound
catches 42% of degraded clips while falsely rejecting 28% of clean ones. A
mid-to-high shelf-depth feature does better at AUC 0.662, still not enough to
claim.

**Why**: Coswara was collected through a web app on whatever phones participants
owned, so the corpus is *already* device-heterogeneous and partly bandlimited.
Adding 8 kHz+AMR to an already-8 kHz-ish clip changes little. Device-mismatch
detection cannot be cleanly validated on a corpus whose device conditions are
unknown and mixed — this is exactly what the Phase-B multi-device recording set
(5–10 known handsets) is for.

Honest status: **channel inversion is supported; the mismatch-bounded reject
option is demonstrated only under controlled synthetic degradation, not on
realistic codec chains.**

## Setup

```bash
pip install -r requirements.txt
```

ffmpeg ships via `imageio-ffmpeg` — no system install needed.

## Run

Verify the method before trusting any result. The A0 gate checks that blind
estimation recovers a *known* synthetic channel:

```bash
python tests/test_channel.py
```

Download data (Coswara, ~365 MB per date, resumable):

```bash
python -m micshift.data
```

Train:

```bash
python -m micshift.train --dates 4 --devices 3
```

Reproduce the ablation table and abstention analysis:

```bash
python -m micshift.evaluate --dates 4 --devices 3
```

Demo UI:

```bash
streamlit run app.py
```

Upload a clip or pick a Coswara sample, then use the **Simulate a device** panel
to re-record it through a different codec, sample rate, and microphone response
and watch the estimated channel, the equalized spectrogram, and the accept/abstain
decision change.

### Training on Colab (recommended)

Training on a laptop CPU takes ~40 min per configuration; a T4 does it in well
under a minute. Preprocessing is CPU-bound either way, so build the feature
caches once and reuse them.

Open `MicShift_Colab.ipynb`, set **Runtime → T4 GPU**, and either point it at
your repo or upload the project. Put the `data/cache/*.npz` files in Drive and
every later run skips straight to training.

## Datasets

Public datasets only.

| Dataset | Access | Role |
|---|---|---|
| [Coswara](https://github.com/iiscleap/Coswara-Data) (IISc Bangalore) | Open | Primary. Cough + breathing, Indian speakers. |
| ICBHI 2017 | Open, research use | Cross-corpus check (planned). |
| CODA TB (Synapse) | Free, application required | TB labels (deferred — application pending). |

**Class balance matters here.** Coswara's positive cases sit in the *later*
collection dates; the earliest dates are entirely healthy. Taking "the first N
dates" produces a single-class dataset and a meaningless 100% accuracy.
`data.DEFAULT_DATES` is curated for balance, and `train.build_cache` raises if
any split ends up single-class.

**Label note.** Coswara carries COVID status, not TB. MicShift's claim is about
*channel robustness*, which is label-agnostic — the mechanism is demonstrated on
whatever respiratory label is available, and CODA TB supplies genuine TB labels
once access clears.

## Evaluation design

Two decisions determine whether any number here means anything:

1. **Split by participant, never by clip.** Coswara gives each participant two
   cough recordings; splitting by clip puts the same throat in train and test,
   and the model scores well by recognising the person.
2. **Train and test device pools are disjoint.** Test clips are encoded with
   codecs (`amr_nb`, `opus_24k`, `mp3_64k`) and mic responses drawn from a
   separate RNG stream, never seen in training. "Unseen device" is genuinely unseen.

Reported metrics: cross-device **retention** (unseen-device accuracy ÷ matched
accuracy), balanced accuracy, and **abstention precision** — whether rejected
clips were genuinely the ones the model would have got wrong.

## Layout

```
config.py              all tunables
micshift/channel.py    blind estimation, inversion, mismatch  <- novelty core
micshift/augment.py    real-codec device simulation (ffmpeg)
micshift/data.py       Coswara download, labels, curated dates
micshift/features.py   mel features, participant splits, device pools
micshift/model.py      small CNN (72k params, CPU-inferable)
micshift/train.py      feature cache + training
micshift/evaluate.py   ablations, baselines, abstention analysis
app.py                 Streamlit demo
tests/test_channel.py  all module checks, A0 gate first
```

## IP note

Prior art exists for blind channel estimation (IEEE TASLP 2013) in *speech
forensics*, and for respiratory-sound risk analysis (Salcit/Swaasa, Indian
granted patent). Novelty rests on the specific combination: non-cough-segment
estimation + inversion + codec/SR-aware augmentation + mismatch-bounded
abstention. **File a provisional before any public disclosure** — India's grace
period does not cover general publication.
