import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path

# Resolve project root dynamically (parent of scripts/)
project_root = Path(__file__).resolve().parent.parent

models = ['A2C', 'PPO', 'DDPG', 'ensemble']
iters = [126, 189, 252, 315, 378, 441, 504, 567, 630, 693, 756, 819, 882, 945, 1008, 1071]

def get_mdd(returns):
    cumulative = (1 + returns).cumprod()
    peak = cumulative.expanding(min_periods=1).max()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()

for model in models:
    df_all = pd.DataFrame()
    for i in iters:
        file_path = project_root / f'results/account_value_trade_{model}_{i}.csv'
        if file_path.exists():
            temp = pd.read_csv(file_path, index_col=0)
            df_all = pd.concat([df_all, temp], ignore_index=True)
    if not df_all.empty:
        df_all.columns = ['account_value']
        # remove first row if needed, or just calculate returns
        returns = df_all['account_value'].pct_change(1).dropna()
        sharpe = (252**0.5) * returns.mean() / returns.std()
        mdd = get_mdd(returns)
        annual_return = (returns.mean() * 252)
        annual_vol = (returns.std() * np.sqrt(252))
        cum_ret = (df_all['account_value'].iloc[-1] - df_all['account_value'].iloc[0]) / df_all['account_value'].iloc[0]
        print(f"Model {model}:")
        print(f"  Cum Return: {cum_ret:.4f}")
        print(f"  Annual Return: {annual_return:.4f}")
        print(f"  Annual Volatility: {annual_vol:.4f}")
        print(f"  Sharpe Ratio: {sharpe:.4f}")
        print(f"  Max Drawdown: {mdd:.4f}")
        print("-" * 30)
