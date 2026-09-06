import pandas as pd
import statsmodels.api as sm
import warnings
from itertools import product
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tools.sm_exceptions import ConvergenceWarning

INPUT_FILE = '1-2.sf6_processed_data_with_exog.csv'
OUTPUT_FILE = '4.sarima_grid_search_results.csv'
START_DATE = '2025-10-23'
END_DATE = '2026-03-10'
SELECTED_MODEL = '(0,1,1)(1,1,1,7)'

df = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)
y = df.loc[START_DATE:END_DATE, 'Players']

results = []
failed_models = []

d = 1
D = 1
s = 7

# 固定 d、D，搜尋 p、q、P、Q。
for p, q, P, Q in product(range(3), range(3), range(2), range(2)):
    model_name = f"({p},{d},{q})({P},{D},{Q},{s})"
    try:
        model = sm.tsa.SARIMAX(
            y,
            order=(p, d, q),
            seasonal_order=(P, D, Q, s),
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)
            fit = model.fit(disp=False)

        lb = acorr_ljungbox(fit.resid, lags=[7], return_df=True)
        results.append({
            "Model": model_name,
            "AIC": fit.aic,
            "BIC": fit.bic,
            "LB_pvalue": lb["lb_pvalue"].iloc[0]
        })
    except Exception as error:
        failed_models.append((model_name, str(error)))

if not results:
    raise RuntimeError('所有模型皆配適失敗')

results_df = pd.DataFrame(results).sort_values("AIC")
adequate_models = results_df[results_df['LB_pvalue'] > 0.05]

print("\n=== Top 20 Models by AIC ===\n")
print(results_df.head(20).to_string(index=False))

print("\n=== Models with Ljung-Box p-value > 0.05 ===\n")
print(adequate_models.to_string(index=False))

# 最終模型綜合 AIC、ACF/PACF 與殘差檢定選定。
selected_result = results_df[results_df['Model'] == SELECTED_MODEL]
print("\n=== Selected Model ===\n")
print(selected_result.to_string(index=False))

if failed_models:
    print(f"\n配適失敗：{len(failed_models)} 組")

results_df.to_csv(OUTPUT_FILE, index=False)
print(f"\n結果已儲存至 {OUTPUT_FILE}")


#output
# === Top 20 Models by AIC ===

#                Model          AIC          BIC  LB_pvalue
# 11  (0,1,2)(1,1,1,7)  2075.670554  2089.649507   0.000142
# 33  (2,1,2)(0,1,1,7)  2078.495128  2095.269871   0.000678
# 23  (1,1,2)(1,1,1,7)  2078.740599  2095.515342   0.000004
# 9   (0,1,2)(0,1,1,7)  2080.090754  2091.273916   0.006670
# 35  (2,1,2)(1,1,1,7)  2080.135975  2099.706509   0.001570
# 21  (1,1,2)(0,1,1,7)  2080.865671  2094.844624   0.000493
# 19  (1,1,1)(1,1,1,7)  2091.732306  2105.752411   0.000056
# 27  (2,1,0)(1,1,1,7)  2091.868627  2105.888732   0.000088
# 34  (2,1,2)(1,1,0,7)  2094.466876  2111.291003   0.007908
# 5   (0,1,1)(0,1,1,7)  2095.496211  2103.908274   0.006824
# 17  (1,1,1)(0,1,1,7)  2096.524140  2107.740225   0.001150
# =========================================================
# 26  (2,1,0)(1,1,0,7)  2098.347410  2109.563494   0.108742
# 7   (0,1,1)(1,1,1,7)  2098.820221  2110.036305   0.137044
# =========================================================
# 29  (2,1,1)(0,1,1,7)  2101.305747  2115.325852   0.000078
# 30  (2,1,1)(1,1,0,7)  2102.108453  2116.128558   0.001286
# 31  (2,1,1)(1,1,1,7)  2103.303066  2120.127192   0.000065
# 15  (1,1,0)(1,1,1,7)  2107.355078  2118.603816   0.000100
# 3   (0,1,0)(1,1,1,7)  2109.401941  2117.838494   0.000143
# 22  (1,1,2)(1,1,0,7)  2111.800328  2125.861249   0.006262
# 13  (1,1,0)(0,1,1,7)  2111.943929  2120.380482   0.004755
