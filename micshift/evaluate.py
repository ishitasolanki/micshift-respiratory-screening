"""Ablations, baselines, and the abstention analysis.

Produces the two numbers the project actually claims:

1. **Cross-device retention** — accuracy on unseen devices relative to matched
   conditions. This is the headline.
2. **Abstention precision** — whether rejected clips are genuinely the ones the
   model would have got wrong, rather than a random slice of the test set.

Includes the baselines a reviewer will demand ("why not just augment harder?").
Without those, the channel-inversion step is not shown to earn its place.
"""
import argparse
import json

import numpy as np
import torch

import config
from micshift import model as M, train as T


def specaugment(X, rng, n_freq=8, n_time=40):
    """Standard SpecAugment masking — a strong, cheap robustness baseline."""
    X = X.copy()
    for i in range(len(X)):
        f = rng.integers(0, n_freq)
        f0 = rng.integers(0, max(1, X.shape[1] - f))
        X[i, f0:f0 + f, :] = 0
        t = rng.integers(0, n_time)
        t0 = rng.integers(0, max(1, X.shape[2] - t))
        X[i, :, t0:t0 + t] = 0
    return X


def metrics(prob, y):
    pred = prob.argmax(1)
    acc = float((pred == y).mean())
    # Balanced accuracy: with skewed classes, plain accuracy flatters a model
    # that mostly predicts the majority class.
    accs = [float((pred[y == c] == c).mean()) for c in (0, 1) if (y == c).sum()]
    return acc, float(np.mean(accs))


def abstention_analysis(prob, y, mismatch, bound=config.MISMATCH_BOUND):
    """Does the reject option remove errors, or just remove data?"""
    keep = mismatch <= bound
    n_kept = int(keep.sum())
    if n_kept == 0:
        return dict(abstain_rate=1.0, kept_acc=float("nan"),
                    abstained_acc=float("nan"), precision=float("nan"))
    pred = prob.argmax(1)
    correct = pred == y
    rejected = ~keep
    return dict(
        abstain_rate=float(rejected.mean()),
        kept_acc=float(correct[keep].mean()),
        abstained_acc=float(correct[rejected].mean()) if rejected.any() else float("nan"),
        # Fraction of abstained clips that would indeed have been errors. Beating
        # the overall error rate is the bar: otherwise abstention is discarding
        # good predictions as readily as bad ones.
        precision=float((~correct[rejected]).mean()) if rejected.any() else float("nan"),
        base_error=float((~correct).mean()),
    )


def run_config(name, n_dates, n_devices, invert, augment_baseline=None,
               epochs=config.EPOCHS, seed=config.SEED, dates=None):
    cached = T.build_cache(n_dates, n_devices, invert=invert, dates=dates)
    if augment_baseline == "specaugment":
        rng = np.random.default_rng(seed)
        X, y, m, p = cached["train"]
        cached = dict(cached, train=(specaugment(X, rng), y, m, p))

    net, val_acc = T.train(cached, epochs=epochs, seed=seed, verbose=False)
    Xte, yte, mte, _ = cached["test"]
    prob = M.predict(net, Xte)
    acc, bacc = metrics(prob, yte)

    # Matched condition: the SAME test participants and the same source audio,
    # rendered through the training device pool. Only the device differs between
    # matched and unseen, so the ratio isolates the device effect.
    Xm, ym, _, _ = cached["test_matched"]
    macc, mbacc = metrics(M.predict(net, Xm), ym)

    ab = abstention_analysis(prob, yte, mte)
    return dict(name=name, invert=invert, baseline=augment_baseline,
                matched_acc=macc, matched_balanced_acc=mbacc,
                unseen_acc=acc, unseen_balanced_acc=bacc,
                # Retention on BALANCED accuracy: with a skewed test set, plain
                # accuracy can look stable purely because the majority class
                # dominates, hiding the very degradation being measured.
                retention=bacc / mbacc if mbacc > 0 else float("nan"),
                val_acc=val_acc, **ab)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", type=int, default=len(__import__("micshift.data", fromlist=["x"]).DEFAULT_DATES))
    ap.add_argument("--devices", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--fresh", action="store_true",
                    help="ignore saved per-config results and rerun everything")
    a = ap.parse_args()

    configs = [
        ("no-inversion (raw)",      False, None),
        ("SpecAugment baseline",    False, "specaugment"),
        ("MicShift (inversion)",    True,  None),
    ]

    # Pin the corpus once so every configuration sees identical data.
    from micshift import data as D
    dates = D.local_dates()
    print(f"corpus: {len(dates)} dates -> {', '.join(dates)}\n")

    # Write each configuration's result the moment it finishes. Each one costs
    # ~40 minutes on CPU, so batching all output to the end means an interrupt
    # (or a crash) discards every completed run.
    out = config.DATA_DIR / "results.json"
    rows = []
    if out.exists() and not a.fresh:
        prior = {r["name"]: r for r in json.loads(out.read_text())
                 if r.get("corpus") == len(dates)}
    else:
        prior = {}

    for name, inv, base in configs:
        if name in prior:
            print(f"reusing completed: {name}", flush=True)
            rows.append(prior[name])
            continue
        print(f"running: {name} ...", flush=True)
        r = run_config(name, a.dates, a.devices, inv, base,
                       epochs=a.epochs, dates=dates)
        r["corpus"] = len(dates)
        rows.append(r)
        out.write_text(json.dumps(rows, indent=2))
        print(f"  -> matched_bal={r['matched_balanced_acc']:.3f} "
              f"unseen_bal={r['unseen_balanced_acc']:.3f} "
              f"retention={r['retention']:.1%}  [saved]", flush=True)

    hdr = (f"{'configuration':24s} {'matched bal':>12s} {'unseen bal':>11s} "
           f"{'retention':>10s} {'unseen acc':>11s}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['name']:24s} {r['matched_balanced_acc']:12.3f} "
              f"{r['unseen_balanced_acc']:11.3f} {r['retention']:9.1%} "
              f"{r['unseen_acc']:11.3f}")
    print("(balanced accuracy: 0.500 = chance. Retention = unseen / matched, "
          "same participants.)")

    print(f"\n{'configuration':24s} {'abstain':>8s} {'kept acc':>9s} {'rejected acc':>13s} {'abst.prec':>10s}")
    print("-" * 68)
    for r in rows:
        print(f"{r['name']:24s} {r['abstain_rate']:8.1%} {r['kept_acc']:9.3f} "
              f"{r['abstained_acc']:13.3f} {r['precision']:10.3f}")

    out.write_text(json.dumps(rows, indent=2))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
