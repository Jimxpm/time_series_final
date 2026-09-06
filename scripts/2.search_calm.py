from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / 'data' / 'processed' / '1-2.sf6_processed_data_with_exog.csv'
OUTPUT_FIGURE = PROJECT_ROOT / 'results' / 'figures' / '2.calm_periods_datadriven.png'
MIN_DAYS = 30
TOP_N = 3

# 普通特價效果不明顯，因此只排除影響較大的事件。
event_columns = [
    'DLC_Peak', 'DLC_Tail',
    'Update_Peak', 'Update_Tail',
    'DeepSale_Start', 'DeepSale_Rest',
    'Tourney_Days', 'Tourney_After'
]

df = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)
df['Is_Calm'] = df[event_columns].eq(0).all(axis=1)

# 尋找至少 30 天的連續基準期。
clean_blocks = []
current_block = []

for date, row in df.iterrows():
    if row['Is_Calm']:
        current_block.append(date)
    else:
        if len(current_block) >= MIN_DAYS:
            clean_blocks.append(current_block)
        current_block = []
if len(current_block) >= MIN_DAYS:
    clean_blocks.append(current_block)

candidates = []
for i, block in enumerate(clean_blocks):
    start_date = block[0]
    end_date = block[-1]
    length = len(block)
    period_df = df.loc[start_date:end_date]
    std = period_df['Players'].std()
    mean = period_df['Players'].mean()
    cv = std / mean

    candidates.append({
        'Candidate': f"Period {i+1}",
        'Start': start_date.strftime('%Y-%m-%d'),
        'End': end_date.strftime('%Y-%m-%d'),
        'Days': length,
        'Mean': mean,
        'Volatility (CV)': cv
    })

if not candidates:
    raise ValueError(f'找不到長度至少 {MIN_DAYS} 天的基準期')

df_candidates = pd.DataFrame(candidates).sort_values(by='Days', ascending=False)
print("=== 平靜期候選（排除主要事件）===")
print(df_candidates.to_string(index=False))

# 顯示最長的三個候選期。
top_candidates = df_candidates.head(TOP_N)

plt.figure(figsize=(15, 8))
plt.plot(df.index, df['Players'], color='lightgray', label='All Data (Noisy)')

for _, row in top_candidates.iterrows():
    period_df = df.loc[row['Start']:row['End']]
    plt.plot(period_df.index, period_df['Players'], linewidth=2, label=f"{row['Candidate']} ({row['Days']} days)")

plt.title('SARIMA Base Modeling: Top 3 Calm Period Candidates (Data-Driven)', fontsize=14)
plt.ylabel('Peak Players', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_FIGURE)
print(f"\n圖表已儲存：{OUTPUT_FIGURE}")

# output
# === 平靜期候選（排除主要事件）===
# Candidate      Start        End  Days         Mean  Volatility (CV)
# Period 11 2025-10-23 2026-03-10   139 33546.431655         0.054500
# Period 4 2023-12-05 2024-02-20    78 21576.653846         0.080105
# Period 8 2024-12-06 2025-02-04    61 26087.032787         0.074080
# Period 7 2024-10-04 2024-12-01    59 27707.440678         0.056639
# Period 2 2023-08-03 2023-09-26    55 22012.036364         0.070261
# Period 1 2023-06-01 2023-07-23    53 39329.320755         0.245410
# Period 9 2025-03-18 2025-05-08    52 31567.653846         0.068484
# Period 5 2024-03-08 2024-04-26    50 24081.840000         0.066226
# Period 10 2025-06-14 2025-07-31    48 28488.791667         0.075074
# Period 3 2023-10-07 2023-11-21    46 19692.065217         0.061238
# Period 6 2024-08-15 2024-09-23    40 24849.650000         0.063140
# Period 12 2026-03-26 2026-04-30    36 35548.027778         0.062884
