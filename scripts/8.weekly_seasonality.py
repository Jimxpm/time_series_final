from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.tsa.seasonal import STL


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / 'data' / 'processed' / '9-1.day_data.csv'
OUTPUT_FIGURE = PROJECT_ROOT / 'results' / 'figures' / '8.weekly_seasonality.png'
CALM_START = '2025-10-23'
CALM_END = '2026-03-10'
ZOOM_START = '2025-11-03'
ZOOM_END = '2025-12-07'
WEEKDAY_ORDER = [
    'Monday',
    'Tuesday',
    'Wednesday',
    'Thursday',
    'Friday',
    'Saturday',
    'Sunday',
]


def prepare_data():
    df = pd.read_csv(INPUT_FILE)
    df['DateTime'] = pd.to_datetime(df['DateTime']).dt.normalize()
    df = df.groupby('DateTime')['Players'].max().to_frame().sort_index()
    df['Weekday'] = pd.Categorical(
        df.index.day_name(),
        categories=WEEKDAY_ORDER,
        ordered=True,
    )

    calm_players = df.loc[CALM_START:CALM_END, 'Players'].asfreq('D')
    if calm_players.isna().any():
        raise ValueError('平靜期資料有缺日，無法進行 STL 分解')

    return df, calm_players


def plot_weekday_distribution(ax, df):
    sns.boxplot(
        data=df,
        x='Weekday',
        y='Players',
        hue='Weekday',
        palette='Set3',
        dodge=False,
        ax=ax,
    )
    legend = ax.get_legend()
    if legend:
        legend.remove()

    overall_mean = df['Players'].mean()
    ax.axhline(
        overall_mean,
        color='red',
        linestyle='--',
        alpha=0.7,
        label=f'Overall Mean ({int(overall_mean)})',
    )
    ax.set_title(
        'Distribution of Peak Players by Day of Week',
        fontsize=16,
        fontweight='bold',
    )
    ax.set_ylabel('Peak Players')
    ax.legend()


def plot_stl_seasonality(ax, calm_players):
    seasonal = STL(calm_players, period=7, robust=True).fit().seasonal
    zoomed = seasonal.loc[ZOOM_START:ZOOM_END]

    ax.plot(
        zoomed.index,
        zoomed,
        color='steelblue',
        marker='o',
        markersize=8,
        linewidth=2.5,
    )
    ax.set_title(
        'Extracted 7-Day Seasonality (Zoomed in 5 Weeks)',
        fontsize=16,
        fontweight='bold',
    )
    ax.set_ylabel('Seasonal Effect')
    ax.grid(True, alpha=0.4, linestyle='--')
    ax.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax.set_xticks(zoomed.index)
    ax.set_xticklabels(
        [f"{date:%m-%d}\n({date:%a})" for date in zoomed.index],
        rotation=45,
    )


def main():
    df, calm_players = prepare_data()
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    plot_weekday_distribution(axes[0], df)
    plot_stl_seasonality(axes[1], calm_players)
    fig.tight_layout()
    fig.savefig(OUTPUT_FIGURE, dpi=300)
    plt.close(fig)
    print(f'圖表已儲存：{OUTPUT_FIGURE}')


if __name__ == '__main__':
    main()
