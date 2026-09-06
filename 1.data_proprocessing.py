import pandas as pd
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

OUTPUT_FILE = '1-2.sf6_processed_data_with_exog.csv'

# 讀取並前處理資料
df = pd.read_csv('9-1.day_data.csv')
df['DateTime'] = pd.to_datetime(df['DateTime']).dt.normalize()
df = df.groupby('DateTime')['Players'].max().to_frame().sort_index().asfreq('D').ffill()

# 建立外生變數
cols = ['DLC_Peak', 'DLC_Tail', 'Update_Peak', 'Update_Tail',
        'Sale_Start', 'Sale_Rest', 'DeepSale_Start', 'DeepSale_Rest',
        'Tourney_Days', 'Tourney_After']
exog = pd.DataFrame(0, index=df.index, columns=cols)

def mark_period(column, start, end):
    """將資料範圍內的指定日期標記為 1。"""
    mask = (exog.index >= start) & (exog.index <= end)
    exog.loc[mask, column] = 1

# 1. DLC 處理
dlcs = ['2023-07-24', '2023-09-27', '2024-02-27', '2024-05-22', '2024-06-26', '2024-09-24', '2025-02-05', '2025-06-04', '2025-08-05', '2025-10-13', '2026-03-16', '2026-05-28']
for d in dlcs:
    dt = pd.to_datetime(d)
    mark_period('DLC_Peak', dt, dt + pd.Timedelta(days=2))
    mark_period('DLC_Tail', dt + pd.Timedelta(days=3), dt + pd.Timedelta(days=9))

# 2. Update 處理
updates = ['2023-12-01', '2024-12-02']
for d in updates:
    dt = pd.to_datetime(d)
    mark_period('Update_Peak', dt, dt + pd.Timedelta(days=1))
    mark_period('Update_Tail', dt + pd.Timedelta(days=2), dt + pd.Timedelta(days=3))

# 3. 特價與新史低列表
deep_sales_periods = [
    ('2023-11-21', '2023-11-28'), ('2024-06-03', '2024-06-17'),
    ('2025-09-23', '2025-10-06')
    # ('2025-08-11', '2025-08-24') 降價幅度太低
]

all_sales_periods = [
    ('2023-11-21', '2023-11-28'), ('2023-12-07', '2024-01-04'), ('2024-01-24', '2024-01-30'),
    ('2024-02-29', '2024-03-21'), ('2024-04-15', '2024-04-29'), ('2024-06-03', '2024-06-17'),
    ('2024-06-27', '2024-07-11'), ('2024-07-17', '2024-07-31'), ('2024-09-16', '2024-09-29'),
    ('2024-10-29', '2024-11-12'), ('2024-11-27', '2024-12-04'), ('2024-12-12', '2025-01-02'),
    ('2025-01-27', '2025-02-10'), ('2025-03-12', '2025-03-24'), ('2025-08-11', '2025-08-24'),
    ('2025-09-23', '2025-10-06'), ('2025-10-29', '2025-11-11'), ('2025-12-11', '2026-01-05'),
    ('2026-02-09', '2026-02-23'), ('2026-04-20', '2026-05-04'), ('2026-06-04', '2026-06-18')
]

normal_sales_periods = [p for p in all_sales_periods if p not in deep_sales_periods]

for s, e in normal_sales_periods:
    st = pd.to_datetime(s)
    en = pd.to_datetime(e)
    mark_period('Sale_Start', st + pd.Timedelta(days=1), st + pd.Timedelta(days=1))
    if en > st + pd.Timedelta(days=1):
        mark_period('Sale_Rest', st + pd.Timedelta(days=2), en)

for s, e in deep_sales_periods:
    st = pd.to_datetime(s)
    en = pd.to_datetime(e)
    mark_period('DeepSale_Start', st + pd.Timedelta(days=1), st + pd.Timedelta(days=1))
    if en > st + pd.Timedelta(days=1):
        mark_period('DeepSale_Rest', st + pd.Timedelta(days=2), en)

# 4. 賽事處理
tourneys = [
    ('2026-03-11', '2026-03-15'), ('2025-03-11', '2025-03-15'), ('2024-02-21', '2024-02-25'),
    ('2026-05-01', '2026-05-03'), ('2025-05-09', '2025-05-11'), ('2024-04-27', '2024-04-29'),
    ('2025-08-01', '2025-08-03'), ('2024-07-19', '2024-07-21'),
    ('2025-08-20', '2025-08-23'), ('2024-08-08', '2024-08-12')
]
for s, e in tourneys:
    st = pd.to_datetime(s)
    en = pd.to_datetime(e)
    mark_period('Tourney_Days', st, en)
    mark_period('Tourney_After', en + pd.Timedelta(days=1), en + pd.Timedelta(days=2))

# 合併人數與外生變數
final_dataset = pd.concat([df['Players'], exog], axis=1)
final_dataset.to_csv(OUTPUT_FILE, index=True)
print(f"處理後資料已儲存：{OUTPUT_FILE}")

# 模型配適
print(df['Players'].describe())
model = sm.tsa.SARIMAX(df['Players'], exog=exog, order=(0,1,1), seasonal_order=(1,1,1,7))
res = model.fit(disp=False)
print(res.summary().tables[1])

# output1：1-1 資料，早期探索模型 SARIMAX(1,1,1)x(0,1,1,7)
# ==================================================================================
#                      coef    std err          z      P>|z|      [0.025      0.975]
# ----------------------------------------------------------------------------------
# DLC_Peak         1.67e+04    192.242     86.887      0.000    1.63e+04    1.71e+04
# DLC_Tail        7398.1054    303.266     24.395      0.000    6803.714    7992.497
# Update_Peak     7526.1643   1024.404      7.347      0.000    5518.369    9533.960
# Update_Tail     3747.2675   1982.479      1.890      0.059    -138.321    7632.856
# Sale_Start       249.9669    765.784      0.326      0.744   -1250.942    1750.876
# Sale_Rest        473.2067    924.050      0.512      0.609   -1337.897    2284.311
# DeepSale_Start  1062.3511    789.241      1.346      0.178    -484.534    2609.236
# DeepSale_Rest   1480.0818    843.787      1.754      0.079    -173.710    3133.874
# Tourney_Days    1144.3199    600.113      1.907      0.057     -31.879    2320.519
# Tourney_After   2314.9333    544.514      4.251      0.000    1247.706    3382.161
# ar.L1              0.3763      0.053      7.110      0.000       0.273       0.480
# ma.L1             -0.6292      0.050    -12.573      0.000      -0.727      -0.531
# ma.S.L7           -0.9993      0.013    -79.334      0.000      -1.024      -0.975
# sigma2          7.647e+06      0.127   6.04e+07      0.000    7.65e+06    7.65e+06
# ==================================================================================

# output2：1-2 資料，早期探索模型 SARIMAX(1,1,1)x(0,1,1,7)
# ==================================================================================
#                      coef    std err          z      P>|z|      [0.025      0.975]
# ----------------------------------------------------------------------------------
# DLC_Peak        1.789e+04    185.132     96.610      0.000    1.75e+04    1.82e+04
# DLC_Tail        7850.3742    282.966     27.743      0.000    7295.772    8404.977
# Update_Peak     7509.6857   1093.303      6.869      0.000    5366.850    9652.521
# Update_Tail     3705.5117   2119.281      1.748      0.080    -448.203    7859.226
# Sale_Start       287.0314    756.711      0.379      0.704   -1196.094    1770.157
# Sale_Rest        737.0319    804.737      0.916      0.360    -840.224    2314.288
# DeepSale_Start  2504.5856   1048.087      2.390      0.017     450.374    4558.798
# DeepSale_Rest   1783.0189    974.155      1.830      0.067    -126.291    3692.329
# Tourney_Days    1130.0970    626.537      1.804      0.071     -97.892    2358.086
# Tourney_After   2268.6513    567.838      3.995      0.000    1155.709    3381.593
# ar.L1              0.3769      0.050      7.586      0.000       0.280       0.474
# ma.L1             -0.6407      0.047    -13.696      0.000      -0.732      -0.549
# ma.S.L7           -0.9994      0.013    -76.720      0.000      -1.025      -0.974
# sigma2          8.126e+06      0.118   6.89e+07      0.000    8.13e+06    8.13e+06
# ==================================================================================

# output3：1-2 資料，最終模型 SARIMAX(0,1,1)x(1,1,1,7)
# ==================================================================================
#                      coef    std err          z      P>|z|      [0.025      0.975]
# ----------------------------------------------------------------------------------
# DLC_Peak        1.789e+04    189.628     94.337      0.000    1.75e+04    1.83e+04
# DLC_Tail        7851.5958    281.346     27.907      0.000    7300.167    8403.025
# Update_Peak     7509.6578   1062.200      7.070      0.000    5427.784    9591.532
# Update_Tail     3704.9826   2070.391      1.790      0.074    -352.910    7762.875
# Sale_Start       284.5327    741.634      0.384      0.701   -1169.044    1738.109
# Sale_Rest        738.7664    768.228      0.962      0.336    -766.932    2244.465
# DeepSale_Start  2504.0253   1062.618      2.356      0.018     421.332    4586.719
# DeepSale_Rest   1782.3927    976.425      1.825      0.068    -131.366    3696.151
# Tourney_Days    1127.4214    585.341      1.926      0.054     -19.826    2274.669
# Tourney_After   2269.3319    551.646      4.114      0.000    1188.126    3350.538
# ma.L1             -0.2820      0.011    -25.117      0.000      -0.304      -0.260
# ar.S.L7            0.1655      0.013     12.726      0.000       0.140       0.191
# ma.S.L7           -0.9998      0.013    -77.681      0.000      -1.025      -0.975
# sigma2          8.112e+06      0.023   3.53e+08      0.000    8.11e+06    8.11e+06
# ==================================================================================
