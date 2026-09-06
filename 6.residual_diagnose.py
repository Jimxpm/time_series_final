import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from statsmodels.tools.sm_exceptions import ConvergenceWarning


INPUT_FILE = '1-1.sf6_processed_data_with_exog.csv'
RESIDUAL_FIGURE = '6.residual_diagnostics_4panel.png'
FORECAST_FIGURE = '6.forecast_performance.png'
MODEL_ORDER = (0, 1, 1)
SEASONAL_ORDER = (1, 1, 1, 7)
TRAIN_END = '2026-05-04'
TEST_START = '2026-05-05'
TEST_END = '2026-05-18'


def fit_model(endog, exog):
    model = sm.tsa.SARIMAX(
        endog,
        exog=exog,
        order=MODEL_ORDER,
        seasonal_order=SEASONAL_ORDER,
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        return model.fit(disp=False)


def plot_residual_diagnostics(result):
    fig = plt.figure(figsize=(12, 10))
    result.plot_diagnostics(fig=fig)
    fig.suptitle('SARIMAX(0,1,1)x(1,1,1,7) Residual Diagnostics', fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    fig.savefig(RESIDUAL_FIGURE)
    plt.close(fig)
    print(f'殘差診斷圖已儲存：{RESIDUAL_FIGURE}')


def evaluate_forecast(df, exog_columns):
    train_df = df.loc[:TRAIN_END]
    test_df = df.loc[TEST_START:TEST_END]

    if len(test_df) != 14:
        raise ValueError('測試資料不是 14 天')

    result = fit_model(train_df['Players'], train_df[exog_columns])
    forecast = result.get_forecast(
        steps=len(test_df),
        exog=test_df[exog_columns],
    )
    predicted_mean = forecast.predicted_mean
    actual = test_df['Players']

    rmse = np.sqrt(mean_squared_error(actual, predicted_mean))
    mae = mean_absolute_error(actual, predicted_mean)
    mape = mean_absolute_percentage_error(actual, predicted_mean)

    print('\n=== 14-day Forecast Performance ===')
    print(f'RMSE: {rmse:.2f}')
    print(f'MAE: {mae:.2f}')
    print(f'MAPE: {mape * 100:.2f}%')

    plot_data = df.loc[:TEST_END].iloc[-45:]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        plot_data.index,
        plot_data['Players'],
        label='Actual Players',
        color='black',
        marker='o',
    )
    ax.plot(
        predicted_mean.index,
        predicted_mean,
        label='Predicted Players',
        color='red',
        marker='x',
        linestyle='--',
    )
    ax.axvline(
        pd.Timestamp(TEST_START),
        color='blue',
        linestyle=':',
        label='Train/Test Split',
    )
    ax.set_title('Out-of-Sample Forecast Performance (14 Days)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FORECAST_FIGURE)
    plt.close(fig)
    print(f'預測績效圖已儲存：{FORECAST_FIGURE}')


def main():
    df = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)
    exog_columns = [column for column in df.columns if column != 'Players']

    diagnostic_df = df.loc[:TEST_END]
    full_result = fit_model(
        diagnostic_df['Players'],
        diagnostic_df[exog_columns],
    )
    plot_residual_diagnostics(full_result)
    evaluate_forecast(df, exog_columns)


if __name__ == '__main__':
    main()
