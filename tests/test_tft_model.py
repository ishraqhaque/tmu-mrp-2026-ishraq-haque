import numpy as np
import pandas as pd
from src.models.tft_model import run_tft

def _tiny_cfg():
    return {
        "split": {"train_frac": 0.70},
        "tft": {"window": 10, "hidden_size": 8, "attention_head_size": 1,
                "dropout": 0.1, "hidden_continuous_size": 8, "learning_rate": 0.01,
                "batch_size": 32, "max_epochs": 1, "patience": 2, "seeds": [0]},
    }

def test_run_tft_structure():
    n, nf = 200, 3
    idx = pd.date_range("2000-01-01", periods=n, freq="D")
    feats = pd.DataFrame(np.random.randn(n, nf).astype("float32"),
                         columns=[f"f{i}" for i in range(nf)], index=idx)
    target = pd.Series(np.random.randn(n).astype("float32"), index=idx)
    out = run_tft(feats, target, _tiny_cfg(), label="tft")
    assert set(out) == {"metrics", "seed_metrics", "predictions"}
    m = out["metrics"]
    assert {"mse", "rmse", "mae", "directional_accuracy", "f1", "n", "model"} <= set(m)
    assert m["model"] == "tft"
    preds = out["predictions"]
    assert {"y_true", "y_pred", "seed_0"} <= set(preds.columns)
    assert len(preds) == m["n"] > 0
    assert len(out["seed_metrics"]) == 1
    import pandas as _pd
    assert isinstance(preds.index, _pd.DatetimeIndex)
    assert preds.index.isin(feats.index).all()
    np.testing.assert_allclose(preds["y_true"].to_numpy(),
                               target.loc[preds.index].to_numpy(), rtol=1e-5)


def test_run_tft_multiseed_ensemble():
    cfg = _tiny_cfg()
    cfg["tft"]["seeds"] = [0, 1]
    n, nf = 200, 3
    idx = pd.date_range("2000-01-01", periods=n, freq="D")
    feats = pd.DataFrame(np.random.randn(n, nf).astype("float32"),
                         columns=[f"f{i}" for i in range(nf)], index=idx)
    target = pd.Series(np.random.randn(n).astype("float32"), index=idx)
    out = run_tft(feats, target, cfg, label="tft")
    preds = out["predictions"]
    assert {"seed_0", "seed_1"} <= set(preds.columns)
    assert len(out["seed_metrics"]) == 2
    # ensemble column is the mean of the per-seed columns
    np.testing.assert_allclose(
        preds["y_pred"].to_numpy(),
        preds[["seed_0", "seed_1"]].mean(axis=1).to_numpy(), rtol=1e-5)
