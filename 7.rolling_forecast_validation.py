import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from statsmodels.tools.sm_exceptions import ConvergenceWarning


INPUT_FILE = '1-1.sf6_processed_data_with_exog.csv'
OUTPUT_CSV = '7.rolling_52folds_results.csv'
OUTPUT_FIGURE = '7.mega_52folds_focused_validation.png'
BACKTEST_END = '2026-05-18'
MODEL_ORDER = (0, 1, 1)
SEASONAL_ORDER = (1, 1, 1, 7)
FORECAST_HORIZON = 7
STEP_SIZE = 7
NUM_ROLLS = 52
HISTORY_WINDOW = 380


def fit_forecast(train_df, test_df, exog_columns):
    model = sm.tsa.SARIMAX(
        train_df['Players'],
        exog=train_df[exog_columns],
        order=MODEL_ORDER,
        seasonal_order=SEASONAL_ORDER,
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        result = model.fit(disp=False)

    forecast = result.get_forecast(
        steps=len(test_df),
        exog=test_df[exog_columns],
    )
    return forecast.predicted_mean


def run_backtest(df, exog_columns):
    results = []
    fig, ax = plt.subplots(figsize=(15, 8))

    history = df.iloc[-HISTORY_WINDOW:]
    ax.plot(
        history.index,
        history['Players'],
        color='black',
        alpha=0.3,
        linewidth=1,
    )

    for index in range(NUM_ROLLS):
        test_end_index = len(df) - index * STEP_SIZE
        test_start_index = test_end_index - FORECAST_HORIZON
        train_df = df.iloc[:test_start_index]
        test_df = df.iloc[test_start_index:test_end_index]

        if len(test_df) != FORECAST_HORIZON:
            raise ValueError('測試區間長度不正確')

        predicted_mean = fit_forecast(train_df, test_df, exog_columns)
        rmse = np.sqrt(mean_squared_error(test_df['Players'], predicted_mean))
        mape = mean_absolute_percentage_error(test_df['Players'], predicted_mean)
        roll = NUM_ROLLS - index

        results.append({
            'Roll': roll,
            'Start': test_df.index[0].strftime('%Y-%m-%d'),
            'End': test_df.index[-1].strftime('%Y-%m-%d'),
            'RMSE': rmse,
            'MAPE': mape,
        })
        ax.plot(predicted_mean.index, predicted_mean, linewidth=1.5, alpha=0.8)

        completed = index + 1
        if completed == 1 or completed % 5 == 0 or completed == NUM_ROLLS:
            print(
                f'完成 {completed}/{NUM_ROLLS}：'
                f'{test_df.index[0]:%Y-%m-%d} 至 {test_df.index[-1]:%Y-%m-%d}'
            )

    results_df = pd.DataFrame(results).sort_values('Roll').reset_index(drop=True)
    format_figure(ax, results_df, df.index[-1])
    return results_df, fig


def format_figure(ax, results_df, backtest_end):
    ax.set_title(f'SARIMAX 52-Fold Rolling Forecast Validation')
    ax.set_ylabel('Peak Concurrent Players')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlim(
        pd.Timestamp(results_df['Start'].iloc[0]) - pd.Timedelta(days=7),
        backtest_end + pd.Timedelta(days=7),
    )


def main():
    df = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)
    df = df.loc[:BACKTEST_END]
    exog_columns = [column for column in df.columns if column != 'Players']

    minimum_rows = NUM_ROLLS * STEP_SIZE + FORECAST_HORIZON
    if len(df) < minimum_rows:
        raise ValueError('資料長度不足以執行 52 組滾動驗證')

    results_df, fig = run_backtest(df, exog_columns)
    fig.tight_layout()
    fig.savefig(OUTPUT_FIGURE)
    plt.close(fig)
    results_df.to_csv(OUTPUT_CSV, index=False)

    print(f'\n平均 RMSE: {results_df["RMSE"].mean():.2f}')
    print(f'平均 MAPE: {results_df["MAPE"].mean() * 100:.2f}%')
    print(f'圖表已儲存：{OUTPUT_FIGURE}')
    print(f'結果已儲存：{OUTPUT_CSV}')


if __name__ == '__main__':
    main()
