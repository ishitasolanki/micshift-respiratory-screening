"""MicShift demo — upload a cough clip, see the device channel and the verdict.

Shows what the method actually does, rather than only a score: the estimated
device response, the equalized spectrum, the residual mismatch, and whether that
mismatch put the clip inside or outside the model's trustworthy operating range.
"""
import io
import json

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import streamlit as st
import torch

import config
from micshift import channel, features as F, model as M, augment

st.set_page_config(page_title="MicShift", layout="wide")


@st.cache_resource
def load_model():
    path = config.DATA_DIR / "model.pt"
    if not path.exists():
        return None
    net = M.CoughCNN()
    net.load_state_dict(torch.load(path, map_location="cpu"))
    net.eval()
    return net


@st.cache_data
def calibrated_bound():
    """Bound calibrated on the validation split, not a hardcoded dB value."""
    p = config.DATA_DIR / "bound.json"
    if p.exists():
        return float(json.loads(p.read_text())["bound"]), True
    return float(config.MISMATCH_BOUND), False


@st.cache_data
def sample_clips(n=6):
    """A few real Coswara clips, so the demo is usable without hunting for a file."""
    try:
        from micshift import data
        clips = data.collect_clips(dates=data.local_dates()[:2], download=False)
        return [(f"{'positive' if lab else 'healthy'} — {pid[:8]}", str(w))
                for w, lab, pid in clips[:n]]
    except Exception:
        return []


st.title("MicShift")
st.caption("Microphone-invariant respiratory audio screening — "
           "blind channel estimation, inversion, and a mismatch-bounded reject option.")

net = load_model()
if net is None:
    st.warning("No trained model found. Run `python -m micshift.train` first.")

default_bound, was_calibrated = calibrated_bound()

with st.sidebar:
    st.header("Abstention")
    bound = st.slider("Mismatch bound (dB)", 0.5, 25.0, default_bound, 0.5)
    if was_calibrated:
        st.caption(f"Default {default_bound:.2f} dB calibrated on the validation "
                   f"split to abstain on ~{config.TARGET_ABSTAIN_RATE:.0%} of "
                   "in-domain clips.")
    else:
        st.caption("No calibration file found — using the fallback default. "
                   "Run training to calibrate.")
    apply_inv = st.checkbox("Apply channel inversion", value=True)

    st.header("Simulate a device")
    st.caption("Re-record the clip through a different phone to watch the "
               "channel shift, and the reject option respond.")
    sim = st.selectbox("Codec", ["none"] + [c for c in augment.CODECS if c != "none"])
    sim_sr = st.selectbox("Sample rate", ["original", 8000, 16000, 22050, 44100])
    sim_mic = st.checkbox("Random microphone response", value=False)

tab_up, tab_sample = st.tabs(["Upload a clip", "Use a sample clip"])
with tab_up:
    up = st.file_uploader("Upload a cough recording",
                          type=["wav", "mp3", "m4a", "ogg", "flac", "amr"])
with tab_sample:
    samples = sample_clips()
    if samples:
        pick = st.selectbox("Coswara sample", [s[0] for s in samples])
        if st.button("Analyze this sample"):
            st.session_state["sample_path"] = dict(samples)[pick]
    else:
        st.caption("No local Coswara data found. Run `python -m micshift.data`.")

src = None
if up is not None:
    src = io.BytesIO(up.read())
elif st.session_state.get("sample_path"):
    src = st.session_state["sample_path"]

if src is not None:
    y, _ = librosa.load(src, sr=config.SR, mono=True)
    if len(y) < config.SR // 2:
        st.error("Clip too short — need at least half a second.")
        st.stop()

    # Optional device simulation, applied in physical order: the microphone
    # colours the sound before the phone's stack resamples or encodes it.
    if sim_mic:
        y = augment.apply_response(
            y, augment.random_mic_response(np.random.default_rng()), config.SR)
    if sim_sr != "original":
        y = augment.resample_roundtrip(y, config.SR, int(sim_sr))
    if sim != "none":
        y = augment.apply_codec(y, config.SR, sim)

    st.audio(y, sample_rate=config.SR)

    resp = channel.estimate_response(y)
    if resp is None:
        st.error("**Abstained.** Not enough non-cough audio to estimate the "
                 "recording channel. Ask for a recording with a moment of "
                 "silence before or after the cough.")
        st.stop()

    y_eq = channel.invert(y, config.SR, resp) if apply_inv else y
    feat, mismatch = F.featurize(y, invert_channel=apply_inv)
    dead = channel.dead_band_fraction(y)

    c1, c2, c3 = st.columns(3)
    c1.metric("Residual mismatch", f"{mismatch:.2f} dB", help="Distance from the canonical channel.")
    c2.metric("Destroyed bandwidth", f"{dead:.0%}", help="Spectrum the device's codec/resampling erased. Not recoverable.")
    c3.metric("Channel spread", f"{np.sqrt(np.mean(resp**2)):.2f} dB")

    # Verdict
    if mismatch > bound:
        st.error(f"**ABSTAIN** — residual mismatch {mismatch:.2f} dB exceeds the "
                 f"{bound:.1f} dB bound. This recording's channel is outside the "
                 "range the model was validated on, so no screening result is issued.")
        if dead > config.DEAD_BAND_BOUND:
            st.info(f"Cause: {dead:.0%} of the spectrum was destroyed by the "
                    "device's codec or sample rate. Equalization cannot recover it. "
                    "Re-record at a higher quality setting if possible.")
    elif net is None:
        st.info("Channel is in range, but no trained model is available to score it.")
    else:
        prob = M.predict(net, feat)[0]
        label = ["Healthy", "Respiratory condition"][int(prob.argmax())]
        st.success(f"**{label}** — confidence {prob.max():.1%}  "
                   f"(mismatch {mismatch:.2f} dB, within bound)")
        st.progress(float(prob[1]), text=f"P(respiratory condition) = {prob[1]:.1%}")

    # Plots
    st.subheader("What MicShift measured")
    fig, ax = plt.subplots(1, 3, figsize=(16, 3.6))

    ax[0].semilogx(channel.band_centers(), resp, lw=2)
    ax[0].axhline(0, color="grey", ls=":")
    ax[0].axhline(-config.DEAD_BAND_DB, color="crimson", ls="--", lw=1, label="dead-band threshold")
    ax[0].set(title="Estimated device response", xlabel="Hz", ylabel="dB")
    ax[0].legend(fontsize=7)

    librosa.display.specshow(librosa.power_to_db(librosa.feature.melspectrogram(
        y=y, sr=config.SR, n_mels=config.N_MELS), ref=np.max),
        sr=config.SR, x_axis="time", y_axis="mel", ax=ax[1])
    ax[1].set(title="Original")

    librosa.display.specshow(librosa.power_to_db(librosa.feature.melspectrogram(
        y=y_eq, sr=config.SR, n_mels=config.N_MELS), ref=np.max),
        sr=config.SR, x_axis="time", y_axis="mel", ax=ax[2])
    ax[2].set(title="Channel-equalized" if apply_inv else "Unequalized")

    fig.tight_layout()
    st.pyplot(fig)

    with st.expander("Non-cough frames used for estimation"):
        mask = channel.noncough_mask(y)
        st.write(f"{int(mask.sum())} of {len(mask)} frames "
                 f"({mask.mean():.0%}) were quiet and noise-like enough to carry "
                 "channel information. The device response is estimated from "
                 "these frames only — the cough itself is never used.")
