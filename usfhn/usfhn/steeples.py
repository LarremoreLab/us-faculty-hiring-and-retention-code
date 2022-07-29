import pandas as pd
import numpy as np
import functools

import usfhn.measurements
import usfhn.views as views
import usfhn.constants as constants
import usfhn.institutions
import usfhn.datasets
import usfhn.stats


@functools.lru_cache(maxsize=128)
def get_absolute_steeples(rank_column='NormalizedRank'):
    """
    What is a steeple?

    for a field, construct a "high prestige" and a "low prestige" group. Either:
    - top 10 institutions
    - top decile of institutions

    Whichever is larger.

    A steeple is:
    - the institution average prestige is not in the top group.
    - the institution-field-prestige is in the top group
    """
    df = usfhn.stats.runner.get('ranks/df', rank_type='prestige')
    df = df[
        df['TaxonomyLevel'] == 'Field'
    ][
        [
            'InstitutionId',
            'TaxonomyLevel',
            'TaxonomyValue',
            'NormalizedRank',
            'Percentile',
            'OrdinalRank',
        ]
    ].drop_duplicates(subset=['InstitutionId', 'TaxonomyValue'])

    df['InstitutionsInTaxonomy'] = df.groupby('TaxonomyValue')['InstitutionId'].transform('nunique')
    df['Decile > 10'] = df['InstitutionsInTaxonomy'] > 100
    df['InstitutionFields'] = df.groupby('InstitutionId')['TaxonomyValue'].transform('nunique')

    top_annotated_rows = []
    for i, row in df.iterrows():
        # in_top_10 = True if row['Percentile'] <= 10 else False
        in_top_10 = True if row['OrdinalRank'] <= 10 else False

        top_annotated_rows.append({
            'TaxonomyValue': row['TaxonomyValue'],
            'InstitutionId': row['InstitutionId'],
            'InTop10': in_top_10,
        })

    df = df.merge(
        pd.DataFrame(top_annotated_rows),
        on=['TaxonomyValue', 'InstitutionId'],
    ).drop_duplicates()


    institutional_ranks = usfhn.stats.runner.get('ranks/df', rank_type='prestige')

    # institutional_ranks = views.filter_exploded_df(usfhn.datasets.CURRENT_DATASET.ranks)
    institutional_ranks = institutional_ranks[
        institutional_ranks['TaxonomyLevel'] == 'Academia'
    ][
        ['InstitutionId', 'Percentile', 'OrdinalRank']
    ].drop_duplicates().rename(columns={
        'Percentile': 'InstitutionalPercentileRank',
        'OrdinalRank': 'InstitutionalOrdinalRank',
    })

    df = df.merge(
        institutional_ranks,
        on='InstitutionId'
    )

    # df['Steeple'] = (df['InstitutionalPercentileRank'] > 10) & (df['InTop10'])
    df['Steeple'] = (df['InstitutionalOrdinalRank'] > 10) & (df['InTop10'])
    df['SteepleFromTop10'] = (df['InstitutionalOrdinalRank'] < 10) & (df['InTop10'])
    df['SteepleFromTop20'] = (df['InstitutionalOrdinalRank'] < 20) & (df['InTop10']) & (df['InstitutionalPercentileRank'] > 10)
    df['SteepleFromRest'] = (df['InTop10']) & (df['InstitutionalOrdinalRank'] > 20)

    df = df[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'InstitutionId',
            'Steeple',
            'InTop10',
            'OrdinalRank',
            'Percentile',
            'InstitutionalOrdinalRank',
            'InstitutionalPercentileRank',
        ]
    ].drop_duplicates()

    return df


def get_steeples_stats_by_umbrella():
    df = get_absolute_steeples().copy()
    df = df[
        df['TaxonomyLevel'] == 'Field'
    ]

    df = views.annotate_umbrella(df, taxonomization_level='Field')
    df['InTop10'] = df['InTop10'].apply(int)
    df['Steeple'] = df['Steeple'].apply(int)
    df['Top10InUmbrella'] = df.groupby('Umbrella')['InTop10'].transform('sum')
    df['SteeplesInUmbrella'] = df.groupby('Umbrella')['Steeple'].transform('sum')
    df['FractionOfSteeples'] = df['SteeplesInUmbrella'] / df['Top10InUmbrella']
    df = df[
        [
            'Umbrella',
            'Top10InUmbrella',
            'SteeplesInUmbrella',
            'FractionOfSteeples',
        ]
    ].drop_duplicates()

    return df


@functools.lru_cache(maxsize=128)
def get_steeples(std_difference=2, rank_column='NormalizedRank'):
    df = usfhn.datasets.CURRENT_DATASET.ranks
    df = df[
        (df['Year'] == constants.YEAR_UNION)
        &
        (df['Gender'] == constants.GENDER_AGNOSTIC)
        &
        (df['TaxonomyLevel'] == 'Field')
    ][
        [
            'InstitutionId',
            'TaxonomyValue',
            rank_column,
        ]
    ].drop_duplicates()

    df = views.annotate_umbrella_color(df, 'Field')

    df['InstitutionRankMean'] = df.groupby('InstitutionId')[rank_column].transform('mean')
    df['InstitutionRankStD'] = df.groupby('InstitutionId')[rank_column].transform('std')

    df['Steeple'] = df[rank_column] > (df['InstitutionRankMean'] + (std_difference * df['InstitutionRankStD']))

    df['Basement'] = df[rank_column] < (df['InstitutionRankMean'] - (std_difference * df['InstitutionRankStD']))

    return df


def get_summary_of_steeple_stats(threshold=10):
    df = get_absolute_steeples()
    institutions = set(df['InstitutionId'].unique())

    total_spots = len(df[df['OrdinalRank'] < threshold])

    top_df = df.copy()[
        df['OrdinalRank'] < threshold
    ]
    top_institutions = set(top_df['InstitutionId'].unique())

    bottom_df = df.copy()[
        df['OrdinalRank'] >= threshold
    ]
    bottom_institutions = set(bottom_df['InstitutionId'].unique())

    institutions_not_in_top = bottom_institutions - top_institutions

    percent_without_top = 100 * len(institutions_not_in_top) / len(institutions)

    top_df['Placements'] = top_df.groupby('InstitutionId')['TaxonomyValue'].transform('nunique')
    only_one_in_top = top_df.copy()[
        top_df['Placements'] == 1
    ]
    only_one_in_top_institutions = set(only_one_in_top['InstitutionId'].unique())
    percent_with_only_one_in_top = 100 * len(only_one_in_top_institutions) / len(institutions)
    percent_with_only_one_in_top = 100 * len(only_one_in_top_institutions) / len(institutions)

    df = df.merge(
        top_df[
            [
                'InstitutionId',
                'Placements',
            ]
        ],
        on='InstitutionId'
    )[
        [
            'InstitutionId',
            'Placements',
        ]
    ].drop_duplicates().sort_values(by='Placements', ascending=False)

    df['CumSum'] = df['Placements'].cumsum()
    df['CumSum%'] = df['CumSum'] / total_spots
    df['CumSum%'] *= 100
    df['CumSum%'] = df['CumSum%'].apply(round)
    df = usfhn.institutions.annotate_institution_name(df)
    df = df[
        [
            'InstitutionName',
            'Placements',
            'CumSum%',
        ]
    ]
    return df


@functools.lru_cache(maxsize=128)
def get_steeple_stats():
    df = get_absolute_steeples()

    steeples = df[
        df['Steeple'] == True
    ].copy()[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'InstitutionId',
        ]
    ].drop_duplicates()

    columns = ['TaxonomyLevel', 'TaxonomyValue']
    steeples['SteeplesCount'] = steeples.groupby(columns)['InstitutionId'].transform('nunique')
    steeples = steeples.drop(columns=['InstitutionId']).drop_duplicates()

    df['PossibleSteeplesCount'] = df.groupby(columns)['InstitutionId'].transform('nunique')

    df = df.merge(steeples, on=columns, how='left')
    df['SteeplesCount'] = df['SteeplesCount'].fillna(0)

    df['SteeplesFraction'] = df['SteeplesCount'] / df['PossibleSteeplesCount']

    df = df[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'SteeplesCount',
            'SteeplesFraction',
            'PossibleSteeplesCount',
        ]
    ].drop_duplicates()

    return df
