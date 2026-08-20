"""Parity test: data_preparation.clean_gwl_values reproduces the abs()+caps
normalization that LSTMDataPreparation.load_gwl_readings applies inline, so every
inference GWL source (WRIS / NWDP / LocalCsv) cleans identically to training.

Run: pytest test_gwl_cleaning_parity.py    OR    python test_gwl_cleaning_parity.py
"""
from types import SimpleNamespace

import pandas as pd

from gwlcore.data_preparation import clean_gwl_values


def _cfg(max_gwl=100.0, min_gwl=-100.0):
    return SimpleNamespace(max_gwl=max_gwl, min_gwl=min_gwl)


def test_abs_sign_normalization():
    out = clean_gwl_values(pd.DataFrame({"gwl_value": [-5.0, 3.0, -2.5]}), _cfg())
    assert list(out["gwl_value"]) == [5.0, 3.0, 2.5]


def test_drops_nan():
    out = clean_gwl_values(pd.DataFrame({"gwl_value": [1.0, None, -2.0]}), _cfg())
    assert list(out["gwl_value"]) == [1.0, 2.0]


def test_max_cap_after_abs():
    # abs -> 50,150,200 ; max_gwl=100 keeps only 50 ; min branch inert (min_gwl=0)
    out = clean_gwl_values(pd.DataFrame({"gwl_value": [50.0, -150.0, 200.0]}),
                           _cfg(max_gwl=100.0, min_gwl=0.0))
    assert list(out["gwl_value"]) == [50.0]


def test_matches_training_inline_block():
    """clean_gwl_values must equal load_gwl_readings' inline abs+caps (the source-of-truth)."""
    df = pd.DataFrame({"gwl_value": [-5.0, 3.0, None, 250.0, -0.0, 12.3, -99.9]})
    cfg = _cfg(max_gwl=200.0, min_gwl=-1.0)
    t = df[df["gwl_value"].notna()].copy()
    t["gwl_value"] = t["gwl_value"].abs()
    if cfg.max_gwl > 0:
        t = t[t["gwl_value"] <= cfg.max_gwl]
    if cfg.min_gwl < 0:
        t = t[t["gwl_value"] >= cfg.min_gwl]
    out = clean_gwl_values(df, cfg)
    assert list(out["gwl_value"]) == list(t["gwl_value"])


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"OK: {len(fns)} parity tests passed")
