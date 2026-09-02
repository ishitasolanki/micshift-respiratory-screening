# MicShift — Microphone-Invariant Respiratory Audio Screening

## Problem statement

Cough-based respiratory/TB screening models degrade sharply when the recording
device changes. A model trained on clips from one phone loses accuracy on clips
from another, because each microphone imposes its own magnitude response, and
each phone/app pipeline applies its own sample rate and lossy codec. In a mass
screening deployment — an ASHA worker using whatever phone she owns — the device
is uncontrolled. This device heterogeneity, not model capacity, is the binding
constraint on field deployment.

## Objective

Retain screening accuracy across unseen recording devices, on unmodified consumer
hardware, without requiring device calibration, a reference recording, or a
device-specific model. Where retention cannot be guaranteed, abstain rather than
emit an unreliable prediction.

## Proposed solution

Four coupled components:

1. **Blind channel magnitude response estimation.** Estimate the recording
   device's magnitude response from the *non-cough* segments (silence, ambient,
   breath) of the same clip that contains the cough. No reference signal, no
   device metadata, no separate calibration recording.
2. **Response inversion.** Invert the estimated response and apply it to the clip,
   mapping the audio toward a canonical channel before feature extraction.
3. **Codec- and sample-rate-aware augmentation.** During training, simulate the
   device/codec/sample-rate variation the model will meet in the field, so the
   classifier is trained on the distribution of channels it will actually see.
4. **Mismatch-bounded abstention.** Measure residual channel mismatch after
   inversion. When it exceeds a calibrated bound, the system abstains ("retake
   recording" / "refer to clinician") instead of predicting.

## Novelty core

Transfer of blind-channel-estimation from audio forensics (where it is used to
identify recording devices) into health acoustics (where it is used to *remove*
device identity), coupled to a mismatch-bounded reject option. The technical
effect is cross-device accuracy retention on unmodified consumer hardware.

## Prior art (checked)

| Source | Relation |
|---|---|
| IEEE TASLP 2013, "Blind Channel Magnitude Response Estimation in Speech Using Spectrum Classification" | Estimation technique exists — in speech forensics, not health acoustics. |
| Microphone classification via blind channel analysis (2016); arXiv 2204.02841 | Uses channel response to *identify* device; we use it to *cancel* device. |
| Salcit/Swaasa, Indian granted patent, "A Method and System for Analyzing Risk Associated with Respiratory Sounds" | Same application domain, same jurisdiction. Nearest neighbour. Does not claim blind channel estimation or a mismatch-bounded reject option. |

**IP verdict: MEDIUM risk.** Technique-exists-elsewhere invites an obviousness
attack. Patentable only on the specific combination (non-cough-segment
estimation + inversion + codec/SR-aware augmentation + mismatch-bounded
abstention). File provisional before any public disclosure — India's grace period
does not cover general publication.

## Target users

- **Primary:** ASHA workers / community health workers screening in the field on
  personally-owned Android phones of unknown make and quality.
- **Secondary:** Researchers evaluating cross-device robustness of respiratory
  audio models.
- **Tertiary:** Clinicians receiving referred cases, who need to know when a
  prediction was made confidently vs. abstained on.

## Scope

**Phase A (this build): proof-of-concept.** Prove the core mechanism works —
channel estimation, inversion, augmentation, abstention — on open datasets, with
a demo UI. Structured so it extends without rewrite.

**Phase B (later): full system.** Full dataset, complete ablation set, real
multi-device held-out test set, paper submission.

## Datasets

Public datasets only. No private or unknown-provenance data.

| Dataset | Source | License / access | Role |
|---|---|---|---|
| **Coswara** | IISc Bangalore — https://github.com/iiscleap/Coswara-Data | Open (CC BY 4.0) | Primary. Cough + breathing from Indian speakers. Breathing/ambient segments feed channel estimation. |
| **ICBHI 2017** | ICBHI respiratory sound database | Open, research use | Secondary. Cross-corpus generalization check — different recording conditions entirely. |
| **CODA TB** | Synapse (free, application required) | Requires approved application | Deferred. TB-labeled cough. Application submitted in parallel; project does not block on it. |

**Critical dataset limitation, stated explicitly:** none of these three are
natively recorded across multiple device types. They are largely
single-recording-condition corpora. This is precisely why augmentation-based
channel simulation is necessary — the device mismatch must be *synthesized*
because no public corpus measures it directly. Validating that synthetic
mismatch transfers to real mismatch requires a small self-recorded multi-device
held-out set (Phase B, 5–10 phones).

## Functional requirements

| ID | Requirement |
|---|---|
| FR-1 | Load and preprocess audio clips from Coswara and ICBHI into a common format (mono, resampled, normalized). |
| FR-2 | Segment each clip into cough/event regions vs. non-cough regions (silence, breath, ambient). |
| FR-3 | Estimate the recording device's magnitude response blindly from non-cough regions only. |
| FR-4 | Invert the estimated response and apply it to the full clip. |
| FR-5 | Apply codec- and sample-rate-aware augmentation during training (simulating device variation). |
| FR-6 | Train a classifier on inverted + augmented features. |
| FR-7 | Compute residual channel mismatch at inference and abstain when it exceeds a calibrated bound. |
| FR-8 | Report accuracy, and accuracy-vs-abstention-rate, on held-out simulated device conditions. |
| FR-9 | Demo UI: upload a clip, show estimated device response, inverted spectrum, prediction, and accept/abstain decision with the mismatch value. |
| FR-10 | Ablation harness: run with/without inversion, with/without augmentation, with/without abstention. |

## Non-functional requirements

| ID | Requirement |
|---|---|
| NFR-1 | Preprocessing and inference run on CPU (field devices and demo have no GPU). |
| NFR-2 | Training feasible on a free-tier Colab GPU (≤ 20 min per run). |
| NFR-3 | Fully reproducible: fixed seeds, pinned deps, documented dataset version. |
| NFR-4 | No patient-identifying data stored or transmitted by the demo. |
| NFR-5 | Preprocessed features cached to disk — re-running training must not re-do DSP. |

## Inputs / outputs

**Input:** a single audio clip (WAV/MP3/M4A/OGG) containing at least one cough
and some non-cough audio, from an arbitrary consumer device.

**Output:**
- Screening prediction (class + probability), **or** an abstention.
- Estimated device magnitude response (for inspection/plotting).
- Residual channel mismatch score and the bound it was compared against.

## Evaluation criteria

| Metric | Purpose |
|---|---|
| In-domain accuracy / AUC | Baseline sanity — model learned the task at all. |
| **Cross-device accuracy retention** | Primary metric. Accuracy on held-out simulated device conditions ÷ in-domain accuracy. This is the headline claim. |
| Accuracy vs. abstention-rate curve | Shows the reject option buys real reliability, not just discarded hard cases. |
| Abstention precision | Are abstained clips genuinely the ones the model would have gotten wrong? |
| Ablation deltas | Isolates the contribution of inversion, augmentation, and abstention separately. |

**Required baselines** (reviewers will demand these — "why not just augment
harder?"): no-inversion + heavy augmentation; SpecAugment; PCEN. If MicShift does
not beat these, the channel-inversion step is not earning its place.

## Constraints & assumptions

**Constraints**
- Public datasets only.
- Free-tier Colab GPU budget.
- CPU-only inference.
- No public multi-device corpus exists → device mismatch must be simulated.

**Assumptions**
- Non-cough segments in a clip carry enough channel information to estimate the
  device magnitude response. *This is the load-bearing assumption of the entire
  project and is validated first, before anything is built on top of it.*
- Device response is approximately time-invariant within a single clip.
- Codec and resampling effects can be simulated faithfully enough with standard
  encoders to transfer to real devices.

## Tech stack

| Layer | Choice | Reason |
|---|---|---|
| Language | Python 3.11 | Ecosystem. |
| Audio DSP | `librosa`, `soundfile`, `scipy` | Standard, no custom DSP needed. |
| Codec simulation | `ffmpeg` via `subprocess` | Real encoders, not approximations — the point is fidelity to actual phone pipelines. |
| ML | PyTorch | Small CNN on mel-spectrograms. |
| Demo UI | Streamlit | Upload + plots + verdict in one file. No frontend build, no API layer. |
| Config | single `config.py` | No YAML/Hydra layer for a proof-of-concept. |

## Architecture

```
audio clip
    ↓
[segment]  cough regions ─────────────────┐
           non-cough regions              │
    ↓                                     │
[blind channel estimation]                │
    → estimated magnitude response H(f)   │
    ↓                                     │
[invert]  apply 1/H(f) ───────────────────┤
    ↓                                     │
[mel features] ←──────────────────────────┘
    ↓
  train:  + codec/SR augmentation → CNN
  infer:  CNN → prediction
    ↓
[residual mismatch check]
    ↓
mismatch ≤ bound → emit prediction
mismatch >  bound → abstain
```

## Findings during implementation

Recorded because they changed the design, and because two of them would have
produced published numbers that were quietly meaningless.

| # | Finding | Consequence |
|---|---|---|
| 1 | A channel is multiplicative in magnitude, therefore **additive in log**. Averaging power spectra across frames lets the loudest frames dominate. | Estimator corrected to average log-spectra. Correlation 0.66 → 0.87. |
| 2 | `np.convolve(..., mode="same")` zero-pads at the edges, dragging the first and last bands toward zero — a fake roll-off exactly where device responses differ most. | Edge-replicated smoothing. Correlation 0.87 → **0.996**. |
| 3 | **Destroyed bandwidth is not invertible.** Inverting a band that 8 kHz resampling zeroed applies enormous gain to pure noise and *degrades* accuracy. | Inversion gain clamped; dead bands left untouched and routed to abstention. This is the reason the reject option exists. |
| 4 | **Coswara's early dates are 100% healthy.** "First N dates" yields a single-class dataset reporting 100% accuracy having learned nothing. | Curated `DEFAULT_DATES`; `build_cache` raises on any single-class split. |
| 5 | Retention measured as test-split ÷ val-split confounds device shift with participant difficulty. | Added `test_matched`: the same participants and audio through the training device pool. Only the device varies. |
| 6 | Class-skewed test sets make plain accuracy look stable while the model collapses to the majority class. | Retention reported on **balanced** accuracy. |
| 7 | **Model selection on plain validation accuracy rewards majority collapse.** On a 68%-healthy val split, "always healthy" scores 0.68 and wins checkpoint selection. The first full ablation returned balanced accuracy of exactly 0.500 and accuracy of exactly 0.750 — the majority fraction — across every configuration. | Checkpoint selection switched to balanced accuracy. Model then predicted both classes and rose above chance. |
| 8 | With 5 dates (129 participants) the model overfit hard: train balanced accuracy 0.958 vs test 0.561. | Corpus expanded to 25 dates; weight decay added. Small-data overfitting, not a method defect. |
| 9 | Evaluating a whole split in one forward pass allocated ~800 MB of conv activations and the process was **killed silently, with no traceback** — twice. | Inference batched in `model.predict` and in the validation loop. |
| 10 | The abstention bound was a hardcoded 3.0 dB that `project.md` *claimed* was calibrated. It sat near the validation p75 and would have wrongly rejected ~25% of good clips. | `channel.calibrate_bound` sets it from the validation mismatch quantile (9.57 dB → 10% in-domain abstention), persisted to `bound.json`. |
| 11 | **The reject option largely fails on realistic phone chains.** An AMR decoder resamples on output and refills the destroyed band with codec noise (−70 dB → −15 dB), defeating the dead-band threshold. Clean vs 8 kHz+AMR separates at only **AUC 0.572**. | Documented as a known limitation, not patched over. Requires the Phase-B multi-device set to fix properly. |
| 12 | `results.json` was written only after all three ablation configs finished, so an interrupt discarded ~40 min of completed training. | Each config's row is written as it completes, and reused on restart. |

### Final verdict (ablation, 25 dates, 925 participants)

| configuration | unseen balanced acc | retention | abstention precision |
|---|---|---|---|
| no-inversion (raw) | 0.650 | 98.9% | 0.308 |
| SpecAugment | **0.665** | **101.2%** | 0.278 |
| MicShift (inversion) | 0.628 | 95.1% | **0.493** |

**The novelty claim is not supported on this corpus.** Channel inversion has the
worst retention and worst unseen accuracy of the three. Codec/sample-rate
augmentation alone already retains 98.9%, leaving ~1% for inversion to recover
while inversion itself costs ~2% in added distortion.

**Implication for the patent.** The stated technical effect — "cross-device
accuracy retention on unmodified consumer hardware" — is the effect that failed
to materialise. Filing on that effect, as measured here, would claim something
the evidence contradicts. What remains defensible is narrower: the
mismatch-bounded reject option, which is the one component that outperformed
both baselines, and where the baselines' abstention was actively inverted.

**Before any filing or submission**, this needs the Phase-B multi-device test
(real handsets, not simulated devices). If real device shift is larger than
augmentation can cover, inversion may yet earn its place; on simulated shift it
does not.

**Scientific position this produces:** MicShift fixes what is physically fixable
(microphone colouring, measured 23% reduction in device-induced feature shift)
and *detects* what is not (destroyed bandwidth: ~2 dB mismatch on clean clips vs
~19 dB on 8 kHz-resampled ones). Claiming inversion solves all device mismatch
would be false; the honest asymmetry is a stronger result, because it is what
justifies a bounded reject option rather than an unconditional prediction.

## Acceptance criteria / Definition of Done

- [ ] **A0 (gate):** Blind channel estimation demonstrably recovers a *known*
      synthetic filter from non-cough segments. If this fails, the premise is
      wrong and the project stops here rather than building on sand.
- [ ] Coswara + ICBHI download, verify, preprocess, and cache without manual steps.
- [ ] Classifier trains and beats chance by a clear margin in-domain.
- [ ] Cross-device retention with MicShift measurably exceeds the no-inversion
      baseline, and exceeds the augmentation-only baselines.
- [ ] Abstention rule demonstrably removes more errors than correct predictions
      (abstention precision above the base error rate).
- [ ] Ablation table produced for all configurations.
- [ ] Demo UI accepts an arbitrary uploaded clip and returns prediction or
      abstention with the response plot and mismatch value.
- [ ] Fresh-environment run reproduces the reported numbers from README alone.
- [ ] Dataset licenses documented; no private data anywhere in the repo.
