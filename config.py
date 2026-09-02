"""All tunables in one place."""
from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"

SEED = 1337

# Audio
SR = 16000
CLIP_SECONDS = 4.0
N_FFT = 512
HOP = 160          # 10 ms at 16 kHz
N_MELS = 64

# Channel estimation
N_BANDS = 40       # log-spaced bands for the estimated response
FMIN = 50.0        # Hz; below this, phone mics carry no reliable information
SMOOTH_BANDS = 3   # moving-average width over bands

# Non-cough (channel-carrying) frame selection
ENERGY_PERCENTILE = 40    # frames below this energy percentile are candidates
FLATNESS_PERCENTILE = 50  # of those, keep the flatter (noise-like) half
MIN_NONCOUGH_FRAMES = 20  # below this, estimation is unreliable

# Inversion limits — a device's destroyed bandwidth cannot be equalized back,
# and trying only amplifies noise. Boost is clamped harder than cut.
MAX_BOOST_DB = 12.0
MAX_CUT_DB = 20.0
DEAD_BAND_DB = 25.0    # bands this far down are considered destroyed, not coloured

# Abstention
# Fallback only. The real bound is calibrated from the validation split by
# channel.calibrate_bound and stored in data/bound.json -- absolute mismatch in
# dB is not comparable across corpora, so a fixed value here would be arbitrary.
MISMATCH_BOUND = 3.0
TARGET_ABSTAIN_RATE = 0.10  # fraction of in-domain clips allowed to abstain
DEAD_BAND_BOUND = 0.25  # abstain when this fraction of the spectrum is destroyed
DEAD_BAND_WEIGHT = 20.0  # dB-equivalent penalty per unit dead-band fraction

# Training
BATCH_SIZE = 64
EPOCHS = 50
LR = 3e-4
WEIGHT_DECAY = 1e-3
