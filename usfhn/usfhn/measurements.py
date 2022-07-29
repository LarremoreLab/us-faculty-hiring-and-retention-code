import numpy as np
import pandas as pd
from scipy.stats import kstest, ks_2samp

from functools import lru_cache
import SpringRank

import usfhn.utils as utils
import usfhn.constants as constants
import usfhn.views as views
import usfhn.datasets
import usfhn.fieldwork
import usfhn.gini
import usfhn.rank
import usfhn.stats_utils


################################################################################
#
#
# Production: Ginis and Lorenzs
#
#
################################################################################
def lorenz_curve(population):
    """
    args:
    - population: a list of integer counts
    - 


    returns:
        tuple:
            - xs: list of increasing fractions
            - ys: cumultative sum of fractional productions of elements in
              population

    calculated by way of the simplified version at:
    https://en.wikipedia.org/wiki/Gini_coefficient#Calculation
    """
    xs, xs_cum_sum = [0], 0
    ys, ys_cum_sum = [0], 0

    total_produced = sum(population)
    total_producers = len(population)
    fractions_produced = [amount / total_produced for amount in population]
    for fraction_produced in sorted(fractions_produced, reverse=True):
        xs_cum_sum += 1 / total_producers
        xs.append(xs_cum_sum)

        ys_cum_sum += fraction_produced
        ys.append(ys_cum_sum)

    ys = [round(y, 3) for y in ys]
    ys[-1] = 1
    xs = [round(x, 3) for x in xs]
    xs[-1] = 1

    curve = pd.DataFrame({'X': xs, 'Y': ys}).sort_values(
        by=['X'],
        ascending=False,
    )
    return curve


################################################################################
#
#
# Faculty Hiring
#
#
################################################################################
@lru_cache()
def ks_test_employment_vs_production():
    import usfhn.stats
    df = usfhn.stats.runner.get('basics/faculty-hiring-network')
    df = df[
        df['TaxonomyLevel'] != 'Taxonomy'
    ]
    groupby_columns = ['TaxonomyLevel', 'TaxonomyValue']

    production_df = df[
        groupby_columns + ['OutDegree', 'DegreeInstitutionId']
    ].rename(columns={
        'DegreeInstitutionId': 'InstitutionId',
    }).drop_duplicates()

    employment_df = df[
        groupby_columns + ['InDegree', 'InstitutionId']
    ].drop_duplicates()

    df = employment_df.merge(
        production_df,
        on=groupby_columns + ['InstitutionId']
    )

    self_hires = usfhn.stats.runner.get('self-hires/by-institution/df')[
        [
            'InstitutionId',
            'TaxonomyLevel',
            'TaxonomyValue',
            'SelfHires',
        ]
    ].drop_duplicates()

    df = df.merge(self_hires, on=groupby_columns + ['InstitutionId'], how='left')
    df['SelfHires'] = df['SelfHires'].fillna(0)

    df['SelfHirelessInDegree'] = df['InDegree'] - df['SelfHires']
    df['SelfHirelessOutDegree'] = df['OutDegree'] - df['SelfHires']

    new_dfs = []
    for exclude_self_hires in [True, False]:
        col_1 = 'InDegree'
        col_2 = 'OutDegree'

        if exclude_self_hires:
            col_1 = f"SelfHireless{col_1}"
            col_2 = f"SelfHireless{col_2}"

        new_df = run_ks_tests(
            df,
            groupby_cols=[
                'TaxonomyLevel',
                'TaxonomyValue',
            ],
            col_1=col_1,
            col_2=col_2,
        )

        new_df['ExcludeSelfHires'] = exclude_self_hires
        new_dfs.append(new_df)

    df = pd.concat(new_dfs)

    corrected_dfs = []
    for exclude_self_hires, rows in df.groupby('ExcludeSelfHires'):
        corrected_dfs.append(usfhn.stats_utils.correct_multiple_hypotheses(rows))

    df = pd.concat(corrected_dfs)
    df['Significant'] = df.apply(ks_test_is_significant, p_col='PCorrected', axis=1)

    df = df[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'ExcludeSelfHires',
            'Significant',
            'P',
            'PCorrected',
        ]
    ].drop_duplicates()

    return df


def run_ks_tests(
    df,
    groupby_cols,
    col_1,
    col_2,
    d_threshold=1.628,
    alpha=.05,
):
    results = []
    for (level, value), rows in df.groupby(groupby_cols):
        n_rows = len(rows)

        ks, p = ks_2samp(rows[col_1], rows[col_2])

        ks_threshold = d_threshold * (((2 * n_rows) / (n_rows ** 2)) **.5)

        results.append({
            **{c: rows.iloc[0][c] for c in groupby_cols},
            'KS': ks,
            'KS-Threshold': ks_threshold,
            'P': p,
        })

    df = pd.DataFrame(results)

    df['Significant'] = df.apply(ks_test_is_significant, axis=1)

    return df


def ks_test_is_significant(row, ks_col='KS', ks_threshold_col='KS-Threshold', p_col='P', alpha=.05):
    return row[ks_threshold_col] < abs(row[ks_col]) and row[p_col] < alpha


################################################################################
#
#
# Steepness
#
#
################################################################################
@lru_cache(maxsize=2)
def get_steepness_by_taxonomy(
    exclude_self_hires=False,
    ranks=tuple(constants.RANKS),
):
    columns = ['Year', 'TaxonomyLevel', 'TaxonomyValue', 'Gender']

    df = get_placements().copy()[
        columns + ['PersonId', 'RankDifference']
    ].drop_duplicates()

    person_ranks = usfhn.datasets.CURRENT_DATASET.data[
        [
            'PersonId',
            'Rank'
        ]
    ].drop_duplicates()

    person_ranks = person_ranks[
        person_ranks['Rank'].isin(ranks)
    ]

    df = df[
        df['PersonId'].isin(person_ranks['PersonId'].unique())
    ]

    upward = df[
        df['RankDifference'] > 0
    ].copy()

    upward['UpwardEdges'] = upward.groupby(columns)['PersonId'].transform('nunique')
    upward = upward[
        columns + ['UpwardEdges']
    ].drop_duplicates()

    self_hires = df[
        df['RankDifference'] == 0
    ].copy()

    self_hires['SelfHireEdges'] = self_hires.groupby(columns)['PersonId'].transform('nunique')
    self_hires = self_hires[
        columns + ['SelfHireEdges']
    ].drop_duplicates()

    downward = df[
        df['RankDifference'] < 0
    ].copy()

    downward['DownwardEdges'] = downward.groupby(columns)['PersonId'].transform('nunique')
    downward = downward[
        columns + ['DownwardEdges']
    ].drop_duplicates()

    # df['Edges'] = df.groupby(columns)['PersonId'].transform('nunique')

    df = df[
        # columns + ['Edges']
        columns
    ].drop_duplicates()

    df = df.merge(
        upward,
        on=columns,
        how='left',
    ).merge(
        self_hires,
        on=columns,
        how='left',
    ).merge(
        downward,
        on=columns,
        how='left'
    )

    df['UpwardEdges'] = df['UpwardEdges'].fillna(0)
    df['DownwardEdges'] = df['DownwardEdges'].fillna(0)
    df['SelfHireEdges'] = df['SelfHireEdges'].fillna(0)

    df['Edges'] = df['UpwardEdges'] + df['DownwardEdges'] + df['SelfHireEdges']

    df['Violations'] = df['UpwardEdges'] / df['Edges']
    df['Steepness'] = df['DownwardEdges'] / df['Edges']
    df['SelfHiresFraction'] = df['SelfHireEdges'] / df['Edges']

    return df


@lru_cache(maxsize=2)
def get_hierarchy_stats_by_movement_type():
    df = get_placements()
    df = usfhn.views.filter_exploded_df(df)
    df = df[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'PersonId',
            'RankDifference',
        ]
    ]

    return usfhn.rank.calculate_hierarchy_stats(
        df,
        'RankDifference',
        [
            'TaxonomyLevel',
            'TaxonomyValue',
        ],
    )



@lru_cache(maxsize=128)
def get_mean_placement_rank_change(direction=None, exclude_self_hires=True):
    df = get_placements().copy()

    if exclude_self_hires:
        df = df[
            df['RankDifference'] != 0
        ]

    if isinstance(direction, str):
        if direction.lower() == 'up':
            df = df[df['RankDifference'] > 0]
        elif direction.lower() == 'down':
            df = df[df['RankDifference'] < 0]

    groupby_columns = ['Year', 'TaxonomyLevel', 'TaxonomyValue', 'Gender']

    df['MeanRankChange'] = df.groupby(groupby_columns)['RankDifference'].transform('mean')
    df['MedianRankChange'] = df.groupby(groupby_columns)['RankDifference'].transform('median')
    df['NormalizedMeanRankChange'] = df.groupby(groupby_columns)['NormalizedRankDifference'].transform('mean')
    df['MedianNormalizedRankChange'] = df.groupby(groupby_columns)['NormalizedRankDifference'].transform('median')
    df['MeanPercentileRankChange'] = df.groupby(groupby_columns)['PercentileRankDifference'].transform('mean')
    df['MedianPercentileRankChange'] = df.groupby(groupby_columns)['PercentileRankDifference'].transform('median')

    df = df[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'Year',
            'Gender',
            'MeanRankChange',
            'MedianRankChange',
            'NormalizedMeanRankChange',
            'MedianNormalizedRankChange',
            'MeanPercentileRankChange',
            'MedianPercentileRankChange',
        ]
    ].drop_duplicates()

    return df
