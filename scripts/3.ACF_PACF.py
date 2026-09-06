from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / 'data' / 'processed' / '1-2.sf6_processed_data_with_exog.csv'
OUTPUT_FIGURE = PROJECT_ROOT / 'results' / 'figures' / '3.acf_pacf_period11.png'
START_DATE = '2025-10-23'
END_DATE = '2026-03-10'

df = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)
df_calm = df.loc[START_DATE:END_DATE, 'Players']

def print_adf_result(series, label):
    result = adfuller(series.dropna())
    print(f"=== {label} ===")
    print(f"ADF statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    return result

print_adf_result(df_calm, '原始資料 ADF 檢定')

# 將 d=1、D=1、s=7 視為候選差分策略。
df_diff = df_calm.diff(1).diff(7).dropna()
print()
print_adf_result(df_diff, '一般與季節差分後 ADF 檢定')

# 5. 繪製 ACF 與 PACF 圖 (用來判斷基礎 p, q, P, Q 參數)
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

axes[0].plot(df_calm.index, df_calm, color='steelblue', linewidth=2)
axes[0].set_title(f'Base Model Period (Period 11: {START_DATE} to {END_DATE})', fontsize=14)
axes[0].grid(True, alpha=0.3)

plot_acf(df_diff, lags=28, ax=axes[1], title="ACF (Differenced d=1, D=1, s=7)")
plot_pacf(df_diff, lags=28, ax=axes[2], title="PACF (Differenced d=1, D=1, s=7)")

plt.tight_layout()
plt.savefig(OUTPUT_FIGURE)
print(f"\n圖表已儲存：{OUTPUT_FIGURE}")

# output
# === 原始資料 ADF 檢定 ===
# p-value: 0.0388

# === 一階與季節差分後 ADF 檢定 ===
# p-value: 0.0002
