import numpy as np
import pandas as pd

import usfhn.null_models
import usfhn.datasets
import usfhn.views
import usfhn.stats

DIFFERENCE_THRESHOLD = .01


def make_null_steepness_df():
    steepness_df = usfhn.stats.runner.get('ranks/hierarchy-stats', rank_type='prestige')

    dfs = []
    for (level, value), rows in steepness_df.groupby(['TaxonomyLevel', 'TaxonomyValue']):
        edges = rows.iloc[0]['Edges']
        self_hires_fraction = rows[
            rows['MovementType'] == 'Self-Hire'
        ].iloc[0]['MovementFraction']

        steepness = rows[
            rows['MovementType'] == 'Downward'
        ].iloc[0]['MovementFraction']

        model = usfhn.null_models.Model(level=level, level_value=value, year=0)

        df = model.gathered_violations

        # the violations is an integer, and it is violations on a network that excludes self hires;
        # we convert that number into a fraction of violations on the self-hire including network
        df['NullViolations'] /= edges

        df['Interval'] += 1
        df['NullViolationsMean'] = df.groupby('Interval')['NullViolations'].transform(np.mean)
        df['NullSTD'] = df.groupby('Interval')['NullViolations'].transform(np.std)

        df['NullSteepness'] = 1 - df['NullViolations'] - self_hires_fraction
        df['NullSteepnessMean'] = 1 - df['NullViolationsMean'] - self_hires_fraction

        df['IsMoreHierarchical'] = steepness < df['NullSteepness']
        df['IsMoreHierarchical'] = df['IsMoreHierarchical'].apply(lambda x: 1 if x else 0)

        df['MoreHierarchicalCount'] = df.groupby('Interval')['IsMoreHierarchical'].transform('sum')
        df = df.drop(columns=['IsMoreHierarchical'])

        df['TaxonomyLevel'] = level
        df['TaxonomyValue'] = value
        dfs.append(df)

    return pd.concat(dfs)


def find_stable_interval(df):
    past_threshold_interval = 1
    for (level, value), rows in df.groupby(['TaxonomyLevel', 'TaxonomyValue']):
        last_interval = None
        last_mean = None
        last_std = None
        for interval, mean, std in zip(rows['Interval'], rows['NullViolationsMean'], rows['NullSTD']):
            if last_mean and last_std:
                mean_difference = abs(mean - last_mean) / last_mean
                std_difference = abs(std - last_std) / last_std

                if mean_difference > DIFFERENCE_THRESHOLD and std_difference > DIFFERENCE_THRESHOLD:
                    break

                if last_interval and past_threshold_interval < interval:
                    if mean_difference < DIFFERENCE_THRESHOLD and std_difference < DIFFERENCE_THRESHOLD:
                        past_threshold_interval = interval
                        break

            last_mean = mean
            last_std = std
            last_interval = interval

    return past_threshold_interval


def get_cleaned_violations(df=None, write=True):
    if df is None:
        df = make_null_steepness_df()

    stable_interval = find_stable_interval(df)
    print(stable_interval)
    df = df.copy()[
        df['Interval'] == stable_interval
    ].drop(columns=[
        'NullViolations',
        'NullSteepness',
        'Draw',
        'NullSTD',
        'Interval',
    ]).drop_duplicates()

    if write:
        df.to_csv(usfhn.datasets.CURRENT_DATASET.model_stats_path, index=False)

    return df


if __name__ == '__main__':
    df = get_cleaned_violations()
