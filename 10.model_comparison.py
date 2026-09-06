import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning


INPUT_FILE = '1-2.sf6_processed_data_with_exog.csv'
HOLDOUT_OUTPUT = '10.model_comparison.csv'
ROLLING_OUTPUT = '10.ultimate_model_comparison.csv'

FORECAST_HORIZON = 7
STEP_SIZE = 7
NUM_ROLLS = 20
ROLLING_END = '2026-06-18'

HOLDOUT_SCENARIOS = [
    {
        'name': 'Calm Period',
        'test_start': '2026-05-13',
        'test_end': '2026-05-26',
    },
    {
        'name': 'Weak Event Period',
        'test_start': '2026-06-05',
        'test_end': '2026-06-18',
    },
]

MODELS = [
    {
        'name': 'Baseline: SARIMA(0,1,1)x(1,1,1)_7',
        'order': (0, 1, 1),
        'seasonal_order': (1, 1, 1, 7),
        'use_exog': False,
    },
    {
        'name': 'SARIMAX 1: (0,1,1)x(1,1,1)_7',
        'order': (0, 1, 1),
        'seasonal_order': (1, 1, 1, 7),
        'use_exog': True,
    },
    {
        'name': 'SARIMAX 2: (2,1,0)x(1,1,0)_7',
        'order': (2, 1, 0),
        'seasonal_order': (1, 1, 0, 7),
        'use_exog': True,
    },
]


def fit_model(train_df, exog_columns, model_info):
    exog_train = train_df[exog_columns] if model_info['use_exog'] else None
    model = sm.tsa.SARIMAX(
        train_df['Players'],
        exog=exog_train,
        order=model_info['order'],
        seasonal_order=model_info['seasonal_order'],
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        return model.fit(disp=False)


def forecast(result, test_df, exog_columns, use_exog):
    exog_test = test_df[exog_columns] if use_exog else None
    return result.get_forecast(
        steps=len(test_df),
        exog=exog_test,
    ).predicted_mean


def calculate_errors(actual, predicted):
    errors = actual.to_numpy() - predicted.to_numpy()
    rmse = np.sqrt(np.mean(errors ** 2))
    mae = np.mean(np.abs(errors))
    return rmse, mae


def validate_daily_window(test_df, expected_days, scenario_name):
    if len(test_df) != expected_days:
        raise ValueError(f'{scenario_name} 的測試資料不是 {expected_days} 天')

    expected_index = pd.date_range(test_df.index[0], test_df.index[-1], freq='D')
    if not test_df.index.equals(expected_index):
        raise ValueError(f'{scenario_name} 的測試日期不連續')


def run_holdout_scenarios(df, exog_columns):
    rows = []

    for scenario in HOLDOUT_SCENARIOS:
        test_start = pd.Timestamp(scenario['test_start'])
        test_end = pd.Timestamp(scenario['test_end'])
        train_df = df.loc[df.index < test_start]
        test_df = df.loc[test_start:test_end]
        validate_daily_window(test_df, 14, scenario['name'])

        print(
            f"\n{scenario['name']}："
            f'{test_start:%Y-%m-%d} 至 {test_end:%Y-%m-%d}'
        )

        for model_info in MODELS:
            result = fit_model(train_df, exog_columns, model_info)
            predicted = forecast(
                result,
                test_df,
                exog_columns,
                model_info['use_exog'],
            )
            rmse, mae = calculate_errors(test_df['Players'], predicted)

            rows.append({
                'Scenario': scenario['name'],
                'Test Start': test_start.strftime('%Y-%m-%d'),
                'Test End': test_end.strftime('%Y-%m-%d'),
                'Model': model_info['name'],
                'AICc': round(result.aicc, 2),
                'BIC': round(result.bic, 2),
                'RMSE': round(rmse, 2),
                'MAE': round(mae, 2),
            })

    return pd.DataFrame(rows)


def build_rolling_windows(df):
    windows = []
    required_rows = NUM_ROLLS * STEP_SIZE
    if len(df) < required_rows + 1:
        raise ValueError('資料長度不足以執行 20 折滾動驗證')

    for index in range(NUM_ROLLS):
        test_end_index = len(df) - index * STEP_SIZE
        test_start_index = test_end_index - FORECAST_HORIZON
        train_df = df.iloc[:test_start_index]
        test_df = df.iloc[test_start_index:test_end_index]
        validate_daily_window(test_df, FORECAST_HORIZON, f'Roll {index + 1}')
        windows.append((train_df, test_df))

    return windows


def run_rolling_comparison(df, exog_columns):
    rolling_df = df.loc[:ROLLING_END]
    windows = build_rolling_windows(rolling_df)
    rows = []

    print(
        f'\nLong-Term Rolling Backtest：{NUM_ROLLS} 折，'
        f'每折預測 {FORECAST_HORIZON} 天'
    )

    for model_info in MODELS:
        rmse_values = []
        mae_values = []

        for index, (train_df, test_df) in enumerate(windows, start=1):
            try:
                result = fit_model(train_df, exog_columns, model_info)
                predicted = forecast(
                    result,
                    test_df,
                    exog_columns,
                    model_info['use_exog'],
                )
            except Exception as error:
                period = (
                    f'{test_df.index[0]:%Y-%m-%d} 至 '
                    f'{test_df.index[-1]:%Y-%m-%d}'
                )
                raise RuntimeError(
                    f"{model_info['name']} 在 Roll {index}（{period}）配適失敗"
                ) from error

            rmse, mae = calculate_errors(test_df['Players'], predicted)
            rmse_values.append(rmse)
            mae_values.append(mae)

        full_result = fit_model(rolling_df, exog_columns, model_info)
        rows.append({
            'Model': model_info['name'],
            'AICc (Full Data)': round(full_result.aicc, 1),
            'Avg Rolling RMSE': round(np.mean(rmse_values), 2),
            'Avg Rolling MAE': round(np.mean(mae_values), 2),
        })
        print(f"完成：{model_info['name']}（{len(rmse_values)}/{NUM_ROLLS} 折）")

    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)
    df = df.sort_index()
    exog_columns = [column for column in df.columns if column != 'Players']

    holdout_results = run_holdout_scenarios(df, exog_columns)
    rolling_results = run_rolling_comparison(df, exog_columns)

    holdout_results.to_csv(HOLDOUT_OUTPUT, index=False)
    rolling_results.to_csv(ROLLING_OUTPUT, index=False)

    print('\n=== 14-Day Holdout Comparison ===')
    print(holdout_results.to_string(index=False))
    print('\n=== 20-Fold Rolling Comparison ===')
    print(rolling_results.to_string(index=False))
    print(f'\n結果已儲存：{HOLDOUT_OUTPUT}')
    print(f'結果已儲存：{ROLLING_OUTPUT}')


if __name__ == '__main__':
    main()
