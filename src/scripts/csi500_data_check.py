#%%
import pandas as pd
import numpy as np
from pathlib import Path

project_root_dir = Path("/home/lxy/quant/alpha-research")
csi_data_path = project_root_dir / "data" / "processed" / "daily" / "csi500_processed_daily.parquet"
csi_data = pd.read_parquet(csi_data_path)

value_judge = ((csi_data['low'] <= min(csi_data['open'], csi_data['close'])) &
         (csi_data['low'] > 0) &
         (csi_data['high'] >= max(csi_data['open'], csi_data['close'])) &
         (csi_data['open'] > 0) &
         (csi_data['close'] > 0) & 
         (csi_data['open'] < np.inf) &
         (csi_data['high'] < np.inf) & 
         (csi_data['close'] < np.inf) & 
         (csi_data['low'] < np.inf)
)

date_judge = 

if not value_judge.all(): 
    print("something wrong happened")
    print(csi_data.loc[value_judge])

if csi_data