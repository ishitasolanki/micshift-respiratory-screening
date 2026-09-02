"""Coswara download, extraction, labelling, and feature caching.

Coswara ships as ~365 MB-per-date tarballs split into 100 MB parts (GitHub's
file cap). Full corpus is ~15 GB across 45 dates; a few dates is plenty for the
proof-of-concept, so the number of dates is a parameter rather than all-or-nothing.

Label note: Coswara carries COVID status, not TB. MicShift's claim is about
*channel robustness*, which is label-agnostic — the cross-device retention
mechanism is proven on whatever respiratory label is available, and CODA TB
supplies genuine TB labels once its access application clears.
"""
import csv
import json
import io
import os
import shutil
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf

import config

REPO = "iiscleap/Coswara-Data"
API = f"https://api.github.com/repos/{REPO}/contents"
RAW = f"https://raw.githubusercontent.com/{REPO}/master"

COSWARA_DIR = config.DATA_DIR / "coswara"
EXTRACT_DIR = COSWARA_DIR / "extracted"

# Cough recordings are the signal; breathing gives extra channel-carrying audio.
COUGH_FILES = ["cough-heavy.wav", "cough-shallow.wav"]

# Coswara covid_status -> binary label. Grouped by whether the participant had a
# confirmed respiratory infection, not by symptom self-report.
POSITIVE = {"positive_mild", "positive_moderate", "positive_asymp"}
NEGATIVE = {"healthy"}


def _get(url, retries=4):
    req = urllib.request.Request(url, headers={"User-Agent": "micshift"})
    for attempt in range(retries):
        try:
            return urllib.request.urlopen(req, timeout=60).read()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def _download_file(url, dest, retries=5):
    """Stream a large file to disk, resuming byte ranges after a dropped connection.

    GitHub regularly truncates these 100 MB part downloads, so a plain read()
    into memory fails most of the time. Partial files are kept and resumed via
    Range requests rather than restarting the whole part.
    """
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries):
        have = tmp.stat().st_size if tmp.exists() else 0
        headers = {"User-Agent": "micshift"}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as r:
                total = have + int(r.headers.get("Content-Length", 0))
                with open(tmp, "ab" if have else "wb") as f:
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
                        have += len(chunk)
                        print(f"\r      {have/1e6:6.1f} / {total/1e6:6.1f} MB", end="", flush=True)
            print()
            # os.replace, not Path.rename: on Windows rename raises when the
            # destination already exists, which turns a harmless re-download
            # into a hard failure.
            os.replace(tmp, dest)
            return dest
        except Exception as e:
            print(f"\n      retry {attempt+1}: {type(e).__name__}")
            time.sleep(2 ** attempt)
    raise RuntimeError(f"failed to download {url}")


def list_dates():
    entries = json.loads(_get(API))
    return sorted(e["name"] for e in entries
                  if e["type"] == "dir" and e["name"][:2] == "20")


def download_date(date, dest=COSWARA_DIR):
    """Download one date's split tarball, reassemble, and extract it."""
    out_dir = EXTRACT_DIR / date
    if out_dir.exists():
        return out_dir

    entries = json.loads(_get(f"{API}/{date}"))
    parts = sorted(e["name"] for e in entries if ".tar.gz." in e["name"])
    if not parts:
        return None

    parts_dir = dest / "parts" / date
    parts_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for p in parts:
        print(f"    {date}/{p}", flush=True)
        paths.append(_download_file(f"{RAW}/{date}/{p}", parts_dir / p))

    tgz = parts_dir / f"{date}.tar.gz"
    with open(tgz, "wb") as out:
        for p in paths:
            with open(p, "rb") as f:
                shutil.copyfileobj(f, out)

    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tgz, mode="r:gz") as tar:
        tar.extractall(EXTRACT_DIR, filter="data")

    shutil.rmtree(parts_dir, ignore_errors=True)  # reclaim ~365 MB per date
    return out_dir


def load_labels():
    """participant id -> binary label, from the corpus-level metadata CSV.

    Cached to disk: this is refetched on every process start otherwise, which
    makes the demo app slow to launch and useless without a network connection.
    """
    cached = COSWARA_DIR / "combined_data.csv"
    if cached.exists():
        raw = cached.read_text(encoding="utf-8", errors="replace")
    else:
        raw = _get(f"{RAW}/combined_data.csv").decode("utf-8", "replace")
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(raw, encoding="utf-8")
    labels = {}
    for row in csv.DictReader(io.StringIO(raw)):
        status = (row.get("covid_status") or "").strip()
        if status in POSITIVE:
            labels[row["id"]] = 1
        elif status in NEGATIVE:
            labels[row["id"]] = 0
    return labels


# Coswara's positive cases are concentrated in the later collection dates; the
# earliest dates are entirely healthy. Taking "the first N dates" yields a
# single-class dataset and a meaningless 100% accuracy, so the default set is
# curated for class balance: positive-rich late dates paired with healthy-rich
# early ones. Format: (date, positives, healthy).
DEFAULT_DATES = [
    "20220224",  # 159 pos /  29 healthy
    "20220116",  #  92 pos /  25 healthy
    "20210930",  #  85 pos /   7 healthy
    "20200413",  #   0 pos /  76 healthy
    "20200814",  #   1 pos /  67 healthy
    "20210406",  #  13 pos /  40 healthy
    "20200707",  #   3 pos /  32 healthy
    "20210507",  #  20 pos /  23 healthy
]


def local_dates():
    """Dates already extracted on disk."""
    if not EXTRACT_DIR.exists():
        return []
    return sorted(d.name for d in EXTRACT_DIR.iterdir() if d.is_dir())


def collect_clips(n_dates=3, dates=None, download=True):
    """Collect clips from the given dates (or the first n_dates of the curated set).

    Already-extracted dates are always included: downloads are flaky and slow,
    and work already on disk should never be ignored because a later fetch failed.
    """
    labels = load_labels()
    if dates is None:
        dates = sorted(set(DEFAULT_DATES[:n_dates]) | set(local_dates()))
    clips = []
    for d in dates:
        folder = EXTRACT_DIR / d
        if not folder.exists():
            if not download:
                continue
            try:
                folder = download_date(d)
            except Exception as e:
                print(f"    skipping {d}: {type(e).__name__}")
                continue
        if folder is None or not folder.exists():
            continue
        for user_dir in folder.iterdir():
            if not user_dir.is_dir():
                continue
            label = labels.get(user_dir.name)
            if label is None:
                continue
            for fname in COUGH_FILES:
                wav = user_dir / fname
                if wav.exists() and wav.stat().st_size > 8000:
                    clips.append((wav, label, user_dir.name))
    return clips


def load_audio(path, sr=config.SR, seconds=config.CLIP_SECONDS):
    """Load, trim leading/trailing silence, and fix to a constant length.

    Trimming is deliberately gentle (top_db=50): aggressive trimming would strip
    the quiet non-cough audio that the channel estimate depends on.
    """
    y, _ = librosa.load(path, sr=sr, mono=True)
    if len(y) == 0:
        return None
    y, _ = librosa.effects.trim(y, top_db=50)
    n = int(sr * seconds)
    if len(y) < sr * 0.5:
        return None
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    return y[:n]


def _demo():
    print("  listing dates...")
    dates = list_dates()
    assert len(dates) > 10, f"expected many date folders, got {len(dates)}"
    print(f"    {len(dates)} date folders, {dates[0]} .. {dates[-1]}")

    print("  loading labels...")
    labels = load_labels()
    pos = sum(labels.values())
    assert len(labels) > 500, f"too few labels: {len(labels)}"
    print(f"    {len(labels)} labelled participants  ({pos} positive, {len(labels)-pos} healthy)")

    have = local_dates()
    if have:
        print(f"  using already-extracted date {have[0]}")
        folder = EXTRACT_DIR / have[0]
    else:
        print("  downloading 1 date folder (~365 MB) to verify extraction...")
        folder = download_date(DEFAULT_DATES[0])
    assert folder and folder.exists(), "extraction failed"
    users = [u for u in folder.iterdir() if u.is_dir()]
    print(f"    extracted {len(users)} participant folders")
    assert len(users) > 10

    wavs = [w for u in users for f in COUGH_FILES if (w := u / f).exists()]
    print(f"    found {len(wavs)} cough recordings")
    assert wavs, "no cough recordings found"

    y = load_audio(wavs[0])
    assert y is not None and len(y) == int(config.SR * config.CLIP_SECONDS)
    print(f"    loaded sample: {len(y)} samples, RMS={np.sqrt((y**2).mean()):.4f}")

    # Real recordings must yield usable channel estimates -- the synthetic A0
    # gate proved the estimator, this proves real Coswara audio feeds it.
    from micshift import channel
    ok = 0
    for w in wavs[:20]:
        a = load_audio(w)
        if a is not None and channel.estimate_response(a) is not None:
            ok += 1
    print(f"    channel estimable on {ok}/20 real clips")
    assert ok >= 15, f"channel estimation failed on real audio ({ok}/20)"

    print("\ndata OK - real Coswara audio downloads, labels, and estimates.")


if __name__ == "__main__":
    _demo()
