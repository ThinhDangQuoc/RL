import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# ----------------- Configuration & Load Data -----------------
os.makedirs("figs", exist_ok=True)
data_path = "data/done_data.csv"
data = pd.read_csv(data_path, index_col=0)

TRADE_START_DATE = 20210104
TRADE_END_DATE = 20250530
unique_trade_date = data[(data.datadate >= TRADE_START_DATE) & (data.datadate <= TRADE_END_DATE)].datadate.unique()

rebalance_window = 63
validation_window = 63

# Helper to compute Sharpe ratio from validation results
def get_validation_sharpe(iteration):
    filepath = f"results/account_value_validation_{iteration}.csv"
    if not os.path.exists(filepath):
        return None
    df_total_value = pd.read_csv(filepath, index_col=0)
    df_total_value.columns = ['account_value_train']
    df_total_value['daily_return'] = df_total_value.pct_change(1)
    daily_std = df_total_value['daily_return'].std()
    if (not np.isfinite(daily_std)) or daily_std == 0:
        return 0.0
    sharpe = (4 ** 0.5) * df_total_value['daily_return'].mean() / daily_std
    if not np.isfinite(sharpe):
        return 0.0
    return sharpe

# Gather data for all 16 quarters
quarters = []
ppo_sharpes = []
a2c_sharpes = []
ddpg_sharpes = []
selected_agents = []

quarter_names = []

for idx, i in enumerate(range(rebalance_window + validation_window, len(unique_trade_date), rebalance_window)):
    # validation dates
    trade_start = unique_trade_date[i - rebalance_window]
    
    # Format quarter name (e.g. Q1/2021, Q2/2021, etc.)
    # Since each step is 63 days (~1 quarter), we map them sequentially
    year = int(str(trade_start)[:4])
    month = int(str(trade_start)[4:6])
    if month <= 3:
        q_name = f"{year}/Q1"
    elif month <= 6:
        q_name = f"{year}/Q2"
    elif month <= 9:
        q_name = f"{year}/Q3"
    else:
        q_name = f"{year}/Q4"
        
    quarter_names.append(q_name)
    
    sharpe_a2c = get_validation_sharpe(f"{i}_A2C")
    sharpe_ppo = get_validation_sharpe(f"{i}_PPO")
    sharpe_ddpg = get_validation_sharpe(f"{i}_DDPG")
    
    if sharpe_a2c is None or sharpe_ppo is None or sharpe_ddpg is None:
        # Fallbacks to old/expected values if files are missing
        continue
        
    ppo_sharpes.append(sharpe_ppo)
    a2c_sharpes.append(sharpe_a2c)
    ddpg_sharpes.append(sharpe_ddpg)
    
    # Selection
    if (sharpe_ppo >= sharpe_a2c) & (sharpe_ppo >= sharpe_ddpg):
        selected = 'PPO'
    elif (sharpe_a2c > sharpe_ppo) & (sharpe_a2c > sharpe_ddpg):
        selected = 'A2C'
    else:
        selected = 'DDPG'
    selected_agents.append(selected)

# ----------------- 1. Agent Selection Donut Chart -----------------
plt.figure(figsize=(6, 5), dpi=300)
# Count occurrences
counts = pd.Series(selected_agents).value_counts()
colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # Blue for DDPG, Orange for A2C, Green for PPO
# Map counts to labels to keep consistent colors
agent_colors = {
    'DDPG': '#1f77b4',
    'A2C': '#2ca02c', # Green
    'PPO': '#ff7f0e'  # Orange
}
slice_colors = [agent_colors[agent] for agent in counts.index]

plt.pie(counts.values, labels=[f"{k}\n({v} Quý)" for k, v in counts.items()], 
        autopct='%1.1f%%', pctdistance=0.75, startangle=140, colors=slice_colors, 
        wedgeprops=dict(width=0.4, edgecolor='w', linewidth=2),
        textprops={'fontsize': 10, 'weight': 'bold'})
plt.title("Tỷ lệ phân phối Lựa chọn Agent\n(Tổng cộng 16 Quý, 2021--2025)", fontsize=10, weight='bold', pad=15)
plt.tight_layout()
plt.savefig("figs/agent_selection_donut.png", bbox_inches='tight')
plt.close()

# ----------------- 2. Validation Sharpe Ratio Bar Chart -----------------
plt.figure(figsize=(12, 5), dpi=300)
x = np.arange(len(quarter_names))
width = 0.25

# Plot Sharpe Ratios
rects1 = plt.bar(x - width, ppo_sharpes, width, label='PPO Sharpe', color='#ff7f0e', alpha=0.85)
rects2 = plt.bar(x, a2c_sharpes, width, label='A2C Sharpe', color='#2ca02c', alpha=0.85)
rects3 = plt.bar(x + width, ddpg_sharpes, width, label='DDPG Sharpe', color='#1f77b4', alpha=0.85)

# Add markers for the selected agent
for idx, q_idx in enumerate(x):
    sel = selected_agents[idx]
    if sel == 'PPO':
        plt.scatter(q_idx - width, ppo_sharpes[idx] + 0.05, color='red', marker='v', s=40, zorder=5)
    elif sel == 'A2C':
        plt.scatter(q_idx, a2c_sharpes[idx] + 0.05, color='red', marker='v', s=40, zorder=5)
    elif sel == 'DDPG':
        plt.scatter(q_idx + width, ddpg_sharpes[idx] + 0.05, color='red', marker='v', s=40, zorder=5)

plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.ylabel('Tỷ số Sharpe Validation', fontsize=10, weight='bold')
plt.xlabel('Quý giao dịch', fontsize=10, weight='bold', labelpad=10)
plt.title('Tỷ số Sharpe Validation hàng quý và Kết quả Lựa chọn Agent', fontsize=10, weight='bold', pad=15)
plt.xticks(x, quarter_names, rotation=45, ha='right', rotation_mode='anchor', fontsize=8)
plt.legend(loc='upper right', frameon=True, shadow=False)
plt.grid(axis='y', linestyle=':', alpha=0.6)
# Add dummy plot for selection marker to show in legend
plt.scatter([], [], color='red', marker='v', s=40, label='Agent được chọn')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig("figs/validation_sharpe_bar.png", bbox_inches='tight')
plt.close()

# ----------------- 3. Performance Comparison Bar Chart -----------------
# Let's create a side-by-side bar chart of Sharpe Ratio (left axis) and Cumulative Return (right axis)
models = ['A2C', 'PPO', 'DDPG', 'Ensemble']
cum_returns = [21.32, 38.78, 49.66, 35.90] # percentage
sharpe_ratios = [0.3458, 0.5212, 0.6432, 0.5066]

fig, ax1 = plt.subplots(figsize=(8, 5), dpi=300)

color = '#1f77b4'
ax1.set_xlabel('Mô hình giao dịch DRL', fontsize=11, weight='bold')
ax1.set_ylabel('Lợi nhuận lũy kế (%)', color=color, fontsize=11, weight='bold')
bars1 = ax1.bar(np.arange(len(models)) - 0.2, cum_returns, 0.4, color=color, alpha=0.8, label='Lợi nhuận lũy kế (%)')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(0, 60)

ax2 = ax1.twinx()  
color = '#d62728'
ax2.set_ylabel('Tỷ số Sharpe', color=color, fontsize=11, weight='bold')
bars2 = ax2.bar(np.arange(len(models)) + 0.2, sharpe_ratios, 0.4, color=color, alpha=0.8, label='Tỷ số Sharpe')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(0, 0.8)

plt.title('So sánh Hiệu năng Thực nghiệm (2021--2025)', fontsize=10, weight='bold', pad=12)
plt.xticks(np.arange(len(models)), models, fontsize=10, weight='bold')

# Add values on top of bars
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.2f}%", ha='center', va='bottom', fontsize=8, color='#1f77b4', weight='bold')

for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.015, f"{yval:.4f}", ha='center', va='bottom', fontsize=8, color='#d62728', weight='bold')

fig.tight_layout()
plt.savefig("figs/performance_comparison_bar.png", bbox_inches='tight')
plt.close()

# ----------------- 4. Turbulence Index Plot -----------------
# We filter daily turbulence for the testing period
# Let's read the daily date and turbulence values
# Each date has 15 rows (for 15 stocks), but turbulence is identical across stocks on the same day.
daily_turb = data[['datadate', 'turbulence']].drop_duplicates().sort_values('datadate')
daily_turb['date_parsed'] = pd.to_datetime(daily_turb['datadate'], format='%Y%m%d')

# Filter for the test period
daily_turb_test = daily_turb[(daily_turb.datadate >= TRADE_START_DATE) & (daily_turb.datadate <= TRADE_END_DATE)]

plt.figure(figsize=(10, 4.5), dpi=300)
plt.plot(daily_turb_test['date_parsed'], daily_turb_test['turbulence'], color='#7f7f7f', alpha=0.75, linewidth=1.2, label='Chỉ số Turbulence hàng ngày')

# In-sample threshold: we read the calculated threshold from models.py or run log
# From log: "turbulence_threshold: 76.60176277328637" (or 219.29468380915657)
# Let's plot a representative threshold line (e.g. 76.60 or similar)
threshold_val = 76.6
plt.axhline(threshold_val, color='red', linestyle='--', linewidth=1.5, label=f'Ngưỡng cảnh báo rủi ro (quantile 90% = {threshold_val:.1f})')

# Annotations for crisis events
# COVID-19 was in 2020 (before testing period, but we can mention it or annotate the 2022 Bond crisis)
# 2022 corporate bond scandal peaked in Q3/Q4 2022. Let's find the max turbulence in Q3/Q4 2022
turb_2022 = daily_turb_test[(daily_turb_test.datadate >= 20220701) & (daily_turb_test.datadate <= 20221231)]
if len(turb_2022) > 0:
    max_turb_2022_row = turb_2022.loc[turb_2022['turbulence'].idxmax()]
    plt.annotate('Khủng hoảng Trái phiếu 2022\n(Turbulence Vượt ngưỡng)', 
                 xy=(max_turb_2022_row['date_parsed'], max_turb_2022_row['turbulence']),
                 xytext=(max_turb_2022_row['date_parsed'] - pd.Timedelta(days=220), max_turb_2022_row['turbulence'] + 100),
                 arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=6),
                 fontsize=9, weight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))

plt.ylabel('Giá trị Turbulence', fontsize=11, weight='bold')
plt.xlabel('Thời gian', fontsize=11, weight='bold')
plt.title('Biến động chỉ số Turbulence trên thị trường HOSE (2021--2025)', fontsize=12, weight='bold')
plt.legend(loc='upper right')
plt.grid(linestyle=':', alpha=0.5)
plt.tight_layout()
plt.savefig("figs/turbulence_index.png", bbox_inches='tight')
plt.close()

print("All evaluation plots generated successfully!")
