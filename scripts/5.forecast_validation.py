import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / 'data' / 'processed' / '1-3.sf6_processed_data_with_exog.csv'
FIGURE_DIR = PROJECT_ROOT / 'results' / 'figures'
MODEL_ORDER = (0, 1, 1)
SEASONAL_ORDER = (1, 1, 1, 7)
HISTORY_DAYS = 30

SCENARIOS = [
    {
        'name': 'Calm Period',
        'train_end': '2026-04-04',
        'test_start': '2026-04-05',
        'test_end': '2026-04-11',
        'event_date': None,
        'output': FIGURE_DIR / '5.forecast_calm_period.png',
    },
    {
        'name': 'Ingrid DLC Period',
        'train_end': '2026-05-24',
        'test_start': '2026-05-25',
        'test_end': '2026-05-31',
        'event_date': '2026-05-28',
        'output': FIGURE_DIR / '5.forecast_ingrid_dlc.png',
    },
]


def run_forecast(df, exog_columns, scenario):
    train_df = df.loc[:scenario['train_end']]
    test_df = df.loc[scenario['test_start']:scenario['test_end']]

    if len(test_df) != 7:
        raise ValueError(f"{scenario['name']} 的測試資料不是 7 天")

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
    predicted_mean = forecast.predicted_mean
    prediction_interval = forecast.conf_int(alpha=0.05)

    plot_forecast(
        train_df,
        test_df,
        predicted_mean,
        prediction_interval,
        scenario,
    )
    print_forecast_table(test_df, predicted_mean, prediction_interval, scenario)


def plot_forecast(
    train_df,
    test_df,
    predicted_mean,
    prediction_interval,
    scenario,
):
    fig, ax = plt.subplots(figsize=(14, 7))
    history = train_df.iloc[-HISTORY_DAYS:]

    ax.plot(
        history.index,
        history['Players'],
        label='Historical Data',
        color='black',
        marker='o',
    )
    ax.plot(
        predicted_mean.index,
        predicted_mean,
        label='Forecast (SARIMAX)',
        color='red',
        marker='x',
        linestyle='--',
    )
    ax.plot(
        test_df.index,
        test_df['Players'],
        label='Actual Players',
        color='green',
        marker='s',
        linewidth=2,
    )
    ax.fill_between(
        predicted_mean.index,
        prediction_interval.iloc[:, 0],
        prediction_interval.iloc[:, 1],
        color='red',
        alpha=0.2,
        label='95% Prediction Interval',
    )

    if scenario['event_date']:
        event_date = pd.Timestamp(scenario['event_date'])
        ax.axvline(
            event_date,
            color='blue',
            linestyle=':',
            linewidth=2,
            label='Ingrid DLC Release',
        )

    ax.set_title(
        f"SF6 Forecast Validation: {scenario['name']} "
        f"({scenario['test_start']} to {scenario['test_end']})"
    )
    ax.set_ylabel('Peak Concurrent Players')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(scenario['output'])
    plt.close(fig)


def print_forecast_table(test_df, predicted_mean, prediction_interval, scenario):
    print(f"\n=== {scenario['name']} ===")
    print(f"{'Date':<12} | {'Forecast':>10} | {'Actual':>10} | {'95% PI Upper':>12}")
    print('-' * 55)

    for date in test_df.index:
        forecast_value = int(predicted_mean.loc[date])
        actual_value = int(test_df.loc[date, 'Players'])
        upper_bound = int(prediction_interval.loc[date].iloc[1])
        print(
            f"{date:%Y-%m-%d} | {forecast_value:>10} | "
            f"{actual_value:>10} | {upper_bound:>12}"
        )

    print(f"圖表已儲存：{scenario['output']}")


def main():
    df = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)
    exog_columns = [column for column in df.columns if column != 'Players']

    for scenario in SCENARIOS:
        run_forecast(df, exog_columns, scenario)


if __name__ == '__main__':
    main()
