"""Runnable checks for every module. `python tests/test_channel.py`

Ordered by dependency, cheapest first. The A0 gate comes first because it is the
one that can invalidate the project: if blind estimation cannot recover a known
channel, nothing downstream is worth running.

Checks that need the Coswara download are skipped when the data is absent, so
this file stays runnable on a clean checkout.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from micshift import channel, augment, model


def main():
    failures = []

    checks = [
        ("A0 gate: blind channel estimation", channel._demo, True),
        ("model: shapes and gradients", model._demo, False),
        ("augment: real codecs", augment._demo, False),
    ]

    # Data-dependent checks only run if Coswara has actually been downloaded.
    if (config.DATA_DIR / "coswara" / "extracted").exists():
        from micshift import data, features
        checks += [("data: Coswara loading", data._demo, False),
                   ("features: splits and inversion", features._demo, False)]
    else:
        print("! Coswara not downloaded - skipping data/feature checks "
              "(run `python -m micshift.data`)\n")

    for name, fn, critical in checks:
        print(f"=== {name} ===")
        try:
            fn()
        except AssertionError as e:
            print(f"FAILED: {e}\n")
            failures.append(name)
            if critical:
                print("Critical gate failed - stopping.")
                break
        except Exception as e:
            print(f"ERROR: {type(e).__name__}: {e}\n")
            failures.append(name)
        else:
            print()

    if failures:
        print(f"{len(failures)} check(s) failed: {', '.join(failures)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
