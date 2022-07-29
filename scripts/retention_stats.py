import pandas as pd

import usfhn.datasets


def retention():
    df = usfhn.datasets.get_dataset_df('data', 'tenure-track-people_phd-granting-institutions')

    years = sorted(list(df['Year'].unique()))

    stats = []
    for year_one, year_two in zip(years[:-1], years[1:]):
        y1_pids = set(df[df['Year'] == year_one].PersonId.unique())
        y2_pids = set(df[df['Year'] == year_two].PersonId.unique())
        stats.append({
            'year': year_two,
            # 'previous': len(y1_pids),
            # 'current': len(y2_pids),
            'intersection': round(len(y1_pids & y2_pids) / max(len(y1_pids), len(y2_pids)), 2),
            'left': round(len(y1_pids - y2_pids) / len(y1_pids), 2),
            'joined': round(len(y2_pids - y1_pids) / len(y2_pids), 2),
        })

    stats = pd.DataFrame(stats)
    print(stats.head(len(years) - 1))

if __name__ == '__main__':
    retention()
