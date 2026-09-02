"""Dataset caching and training.

Device simulation costs an ffmpeg subprocess per clip, far too slow to run
inside the training loop. Devices are therefore simulated once into an on-disk
cache (NFR-5), and epochs read from that cache.

The cache stores each clip under several simulated devices. Train devices come
from TRAIN_CODECS, test devices from the disjoint TEST_CODECS, so evaluation
genuinely measures unseen-device performance.
"""
import argparse
import hashlib
import json
import time

import numpy as np
import torch
import torch.nn as nn

import config
from micshift import data, features as F, model as M


def _cache_path(tag, dates, n_devices, invert):
    # The date list is part of the key: downloads land incrementally, and a cache
    # keyed only on a count would be silently reused against a different corpus.
    key = (f"{tag}_{','.join(sorted(dates))}_{n_devices}_{int(invert)}"
           f"_{config.SR}_{config.N_MELS}")
    return config.CACHE_DIR / f"{hashlib.md5(key.encode()).hexdigest()[:10]}_{tag}.npz"


def build_cache(n_dates=3, n_devices=3, invert=True, force=False, dates=None):
    """Simulate devices over every clip and cache features to disk.

    `dates` should be pinned by callers that compare several configurations:
    downloads landing mid-run would otherwise give each configuration a
    different corpus, and the ablation would compare data as much as method.

    Returns dict of split -> (X, y, mismatch, participant_ids).
    """
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    local = dates if dates is not None else data.local_dates()
    dates = local if local else data.DEFAULT_DATES[:n_dates]

    path = _cache_path("all", dates, n_devices, invert)
    if path.exists() and not force:
        z = np.load(path, allow_pickle=True)
        return {s: (z[f"{s}_X"], z[f"{s}_y"], z[f"{s}_m"], z[f"{s}_p"])
                for s in ("train", "val", "test", "test_matched")}

    clips = (data.collect_clips(dates=local, download=False) if local
             else data.collect_clips(n_dates=n_dates))
    if not clips:
        raise RuntimeError("no clips found - run `python -m micshift.data` first")
    tr_ids, va_ids, te_ids = F.split_participants(clips)

    # "test_matched" holds the SAME test participants rendered through the
    # TRAINING device pool. Retention must compare like with like: measuring
    # unseen-device accuracy against a different split's accuracy would confound
    # the device shift with how hard those particular participants happen to be.
    out = {s: ([], [], [], []) for s in ("train", "val", "test", "test_matched")}
    t0 = time.time()
    for i, (wav, label, pid) in enumerate(clips):
        y = data.load_audio(wav)
        if y is None:
            continue
        split = "train" if pid in tr_ids else "val" if pid in va_ids else "test"
        # Test clips see only codecs and mic responses never used in training.
        codecs = F.TRAIN_CODECS if split == "train" else F.TEST_CODECS
        # Offsetting the test seed keeps test mic responses out of the train stream.
        base = hash(pid) % 10000 + (0 if split == "train" else 500000)

        targets = [(split, codecs, base)]
        if split == "test":
            # Same participant, same audio, training-pool devices and seed stream.
            targets.append(("test_matched", F.TRAIN_CODECS, hash(pid) % 10000))

        for tgt, cdc, seed_base in targets:
            for d in range(n_devices):
                rng = np.random.default_rng(seed_base + d * 7919)
                dev = F.apply_device(y, rng, cdc)
                feat, mism = F.featurize(dev, invert_channel=invert)
                X, Y, Mm, P = out[tgt]
                X.append(feat); Y.append(label); Mm.append(mism); P.append(pid)

        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{len(clips)} clips  ({time.time()-t0:.0f}s)", flush=True)

    packed = {}
    for s, (X, Y, Mm, P) in out.items():
        packed[s] = (np.asarray(X, dtype=np.float32), np.asarray(Y),
                     np.asarray(Mm, dtype=np.float32), np.asarray(P))

    # A single-class split reports perfect accuracy while having learned nothing.
    # Coswara's early dates are entirely healthy, so this is a live hazard, not a
    # theoretical one -- fail loudly rather than publish a meaningless number.
    for s, (X, Y, _, _) in packed.items():
        counts = np.bincount(Y, minlength=2)
        if len(X) == 0 or counts.min() == 0:
            raise RuntimeError(
                f"split '{s}' has classes {counts.tolist()} - accuracy would be "
                f"meaningless. Use more/later dates (see data.DEFAULT_DATES).")
    np.savez_compressed(path, **{f"{s}_{k}": v for s, t in packed.items()
                                 for k, v in zip("Xymp", t)})
    return packed


def train(cached, epochs=config.EPOCHS, seed=config.SEED, verbose=True):
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr, ytr = cached["train"][0], cached["train"][1]
    Xva, yva = cached["val"][0], cached["val"][1]

    net = M.CoughCNN().to(dev)
    # Weight decay matters here: the corpus is small relative to the model, and
    # training balanced accuracy reaches ~0.96 while test sits near 0.56.
    opt = torch.optim.Adam(net.parameters(), lr=config.LR,
                           weight_decay=config.WEIGHT_DECAY)

    # Class weights: Coswara is roughly 2:1 healthy:positive, and an unweighted
    # loss would be optimized by predicting "healthy" for everyone.
    counts = np.bincount(ytr, minlength=2)
    w = torch.tensor(counts.sum() / (2 * np.maximum(counts, 1)), dtype=torch.float32, device=dev)
    lossf = nn.CrossEntropyLoss(weight=w)

    Xtr_t = torch.tensor(Xtr, device=dev)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=dev)
    Xva_t = torch.tensor(Xva, device=dev)
    yva_t = torch.tensor(yva, dtype=torch.long, device=dev)

    best_state, best_acc = None, -1
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        net.train()
        idx = rng.permutation(len(Xtr_t))
        for b in range(0, len(idx), config.BATCH_SIZE):
            sel = idx[b:b + config.BATCH_SIZE]
            if len(sel) < 2:
                continue
            opt.zero_grad()
            loss = lossf(net(Xtr_t[sel]), ytr_t[sel])
            loss.backward()
            opt.step()

        net.eval()
        with torch.no_grad():
            # Batched for the same memory reason as model.predict.
            pred = torch.cat([net(Xva_t[i:i + config.BATCH_SIZE]).argmax(1)
                              for i in range(0, len(Xva_t), config.BATCH_SIZE)])
        # Select on BALANCED accuracy. Plain accuracy is maximized by collapsing
        # to the majority class -- on a 68% healthy validation split, "always
        # healthy" scores 0.68 and wins selection while the model has learned
        # nothing. Balanced accuracy scores that degenerate solution at 0.500.
        per_class = [(pred[yva_t == c] == c).float().mean().item()
                     for c in (0, 1) if (yva_t == c).any()]
        acc = float(np.mean(per_class))
        # Select on validation, never on test -- test is touched once, at the end.
        if acc > best_acc:
            best_acc, best_state = acc, {k: v.clone() for k, v in net.state_dict().items()}
        if verbose and (ep + 1) % 10 == 0:
            print(f"    epoch {ep+1:3d}  loss={loss.item():.3f}  val_acc={acc:.3f}", flush=True)

    net.load_state_dict(best_state)
    return net, best_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", type=int, default=3)
    ap.add_argument("--devices", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--no-invert", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    print("building cache (device simulation)...")
    cached = build_cache(a.dates, a.devices, invert=not a.no_invert, force=a.force)
    for s, (X, y, m, p) in cached.items():
        print(f"  {s:5s}: {len(X):5d} samples, {len(set(p)):3d} participants, "
              f"{100*y.mean():.0f}% positive")

    print("training...")
    net, va = train(cached, epochs=a.epochs)
    print(f"  best val accuracy: {va:.3f}")

    Xte, yte = cached["test"][0], cached["test"][1]
    prob = M.predict(net, Xte)
    acc = (prob.argmax(1) == yte).mean()
    print(f"  unseen-device test accuracy: {acc:.3f}")

    # Calibrate the abstention bound on VALIDATION mismatch, never on test.
    from micshift import channel
    bound = channel.calibrate_bound(cached["val"][2])
    (config.DATA_DIR / "bound.json").write_text(json.dumps(
        {"bound": bound, "target_abstain": config.TARGET_ABSTAIN_RATE}))
    print(f"  calibrated abstention bound: {bound:.2f} dB "
          f"(targets {config.TARGET_ABSTAIN_RATE:.0%} in-domain abstention)")

    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), config.DATA_DIR / "model.pt")
    print(f"  saved -> {config.DATA_DIR / 'model.pt'}")


if __name__ == "__main__":
    main()
