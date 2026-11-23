"""使用回归 LightGBM 归档的 MAD 样本，复算固定正则的 Ridge 对照。"""
import json
from pathlib import Path
import shutil
import pandas as pd
import numpy as np
from src.scripts.run_all import new_run
from src.ml.models import compare_ridge
from src.backtest.engine import evaluate_signals
cache=Path('results/ml_study/daily_alpha158_comparison_20261121_105840_864298/preprocessing_test/sampling_mse')
cfg=json.loads((cache/'config.json').read_text())
assert cfg['alpha158_processing']=='mad' and cfg['ridge_alpha']==100 and cfg['training_label']=='zscore'
out=new_run(1,tag='mad_ridge')
out.mkdir()
print(out,flush=True)
shutil.copyfile(__file__,out/'replay.py')
cfg.update(feature_source=str(cache),ridge_strong_alphas=[cfg['ridge_alpha']])
(out/'config.json').write_text(json.dumps(cfg,indent=2,ensure_ascii=False))
scores=compare_ridge(cfg,cache,out).rename(columns={'ridge_strong':'ridge_mad'})
keys=['trade_date','ticker','split']; labels=['label','evaluable']
reference=pd.read_parquet(cache/'predictions.parquet',columns=keys+labels)
pd.testing.assert_frame_equal(reference.set_index(keys)[labels].sort_index(),scores.set_index(keys)[labels].sort_index())
assert np.isfinite(scores.ridge_mad).all()
scores.to_parquet(out/'predictions.parquet',index=False)
evaluate_signals(scores,cfg,out,('ridge_mad',))
old=pd.read_parquet(cache/'ledger.parquet').set_index('trade_date')
book=pd.read_parquet(out/'ledger.parquet').set_index('trade_date')
pd.testing.assert_series_equal(old.benchmark_return,book.benchmark_return)
np.testing.assert_allclose((1+book.daily_return).cumprod(),book.nav)
print((out/'portfolio_metrics.csv').read_text(),flush=True)
