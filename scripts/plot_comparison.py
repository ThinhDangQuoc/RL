import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Resolve project root dynamically
project_root = Path(__file__).resolve().parent.parent

models = ['A2C', 'PPO', 'DDPG', 'ensemble']
iters = [126, 189, 252, 315, 378, 441, 504, 567, 630, 693, 756, 819, 882, 945, 1008, 1071]

plt.figure(figsize=(12, 6))

for model in models:
    df_all = pd.DataFrame()
    for i in iters:
        file_path = project_root / f'results/account_value_trade_{model}_{i}.csv'
        if file_path.exists():
            temp = pd.read_csv(file_path, index_col=0)
            df_all = pd.concat([df_all, temp], ignore_index=True)
            
    if not df_all.empty:
        df_all.columns = ['account_value']
        # Convert to cumulative return or keep as absolute value
        initial_val = df_all['account_value'].iloc[0]
        cum_returns = (df_all['account_value'] - initial_val) / initial_val * 100
        
        label_name = model.upper() if model != 'ensemble' else 'Ensemble'
        plt.plot(cum_returns, label=f"{label_name} ({cum_returns.iloc[-1]:.2f}%)")

plt.title("So sánh Lợi nhuận Lũy kế các Chiến lược DRL (2021--2025)\n(Mô phỏng khớp lệnh T+3 & Biên độ ±7% trên HOSE)", fontsize=12, fontweight='bold', pad=15)
plt.xlabel("Ngày giao dịch thực tế", fontsize=10, fontweight='bold')
plt.ylabel("Lợi nhuận lũy kế (%)", fontsize=10, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=10, loc='upper left')
plt.tight_layout()

# Save the plot
output_path = project_root / 'figs/strategy_comparison.png'
plt.savefig(output_path, dpi=300)
print(f"Comparison plot saved successfully to {output_path}")
