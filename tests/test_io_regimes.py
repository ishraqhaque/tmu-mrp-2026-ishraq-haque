import pandas as pd
from src.utils import io


def test_load_regime_posteriors_aligns_to_features():
    asset = io.target_keys()[0]
    feats = pd.read_parquet(io.PROCESSED_DIR / f"{asset}.parquet")
    post = io.load_regime_posteriors(asset, index=feats.index)
    assert len(post) == len(feats)                       # aligned 1:1
    assert list(post.index) == list(feats.index)
    assert all(c.startswith("regime_p") for c in post.columns)
    assert post.isna().sum().sum() == 0                  # no gaps after alignment
