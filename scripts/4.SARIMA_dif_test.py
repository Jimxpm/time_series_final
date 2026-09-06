from pathlib import Path

import pandas as pd
import statsmodels.api as sm
import warnings
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tools.sm_exceptions import ConvergenceWarning

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / 'data' / 'processed' / '1-2.sf6_processed_data_with_exog.csv'
START_DATE = '2025-10-23'
END_DATE = '2026-03-10'

df = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)
endog = df.loc[START_DATE:END_DATE, 'Players']

# 固定暫定的 AR/MA 階數，只比較 d、D。
test_cases = [
    {'name': '完全不差分', 'order': (1, 0, 1), 'seasonal': (0, 0, 1, 7)},
    {'name': '僅一般差分', 'order': (1, 1, 1), 'seasonal': (0, 0, 1, 7)},
    {'name': '僅季節差分', 'order': (1, 0, 1), 'seasonal': (0, 1, 1, 7)},
    {'name': '雙重差分(原設定)', 'order': (1, 1, 1), 'seasonal': (0, 1, 1, 7)},
]

print("=== 差分策略比較 ===")
print(f"{'模型策略':<15} | {'AIC':<10} | {'BIC':<10} | {'殘差白噪音檢定 (P值 > 0.05 為佳)':<25}")
print("-" * 75)

for case in test_cases:
    try:
        model = sm.tsa.SARIMAX(endog, order=case['order'], seasonal_order=case['seasonal'])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)
            res = model.fit(disp=False)

        aic = round(res.aic, 2)
        bic = round(res.bic, 2)

        lb_test = acorr_ljungbox(res.resid, lags=[7], return_df=True)
        lb_pval = round(lb_test['lb_pvalue'].iloc[0], 4)

        print(f"{case['name']:<13} | {aic:<10} | {bic:<10} | {lb_pval:<25}")
    except Exception as error:
        print(f"{case['name']:<13} | 執行失敗：{error}")

# === 差分策略網格搜索結果 ===
# 模型策略            | AIC        | BIC        | 殘差白噪音檢定 (P值 > 0.05 為佳)
# ---------------------------------------------------------------------------
# 完全不差分         | 2369.48    | 2381.21    | 0.9919
# 僅常規差分         | 2355.16    | 2366.87    | 0.9904
# 僅季節差分         | 2367.34    | 2378.87    | 0.0
# 雙重差分(原設定)     | 2265.75    | 2277.25    | 0.0016
