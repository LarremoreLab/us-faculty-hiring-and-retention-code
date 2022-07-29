from collections import defaultdict
from functools import lru_cache
import SpringRank
import numpy as np
import pandas as pd
import scipy.stats

import hnelib.pandas
import hnelib.stats

import usfhn.constants as constants
import usfhn.views
import usfhn.utils
import usfhn.datasets
import usfhn.stats_utils


def annotate_faculty_rank(df):
    assert('PersonId' in list(df.columns))

    if 'Rank' in df.columns:
        return df

    join_cols = [
        'PersonId',
    ]

    by_year = 'Year' in df.columns

    if by_year:
        join_cols.append('Year')

    rank_df = usfhn.datasets.get_dataset_df('closedness_data', by_year=by_year)

    rank_df = rank_df[
        ['Rank'] + join_cols
    ].drop_duplicates()

    df = df.merge(
        rank_df,
        on=join_cols,
    )

    return df


def annotate_seniority(df):
    df = annotate_faculty_rank(df)
    df['Senior'] = df['Rank'].isin(['Associate Professor', 'Professor'])
    df = df.drop(columns=['Rank']).drop_duplicates()

    return df


@lru_cache()
def get_ranks(dataset=None, by_year=False):
    """
    if `by_year` == True, will run ranks for each year
    """
    groupby_cols = ['TaxonomyLevel', 'TaxonomyValue']

    if by_year:
        groupby_cols.append('Year')

    df = usfhn.datasets.get_dataset_df('data', dataset)

    if 'PrimaryAppointment' in df.columns:
        df = df[
            df['PrimaryAppointment']
        ]

    df = usfhn.fieldwork.linearize_df_taxonomy_hierarchy(df, taxonomy_level_exclusions=['Taxonomy'])

    df = df[
        groupby_cols + [
            'InstitutionId',
            'DegreeInstitutionId',
            'PersonId',
        ]
    ].drop_duplicates()

    all_ranks = []
    for _, rows in df.groupby(groupby_cols):
        groupby_cols_info = {c: rows.iloc[0][c] for c in groupby_cols}

        rows = rows[
            [
                'DegreeInstitutionId',
                'InstitutionId',
                'PersonId',
            ]
        ].drop_duplicates()

        try:
            rows = filter_hiring(rows)
            ranks = recover_ranks(rows)
            ranks = ranks.sort_values(by='Rank', ascending=False)

            for groupby_col, value in groupby_cols_info.items():
                ranks[groupby_col] = value

            ranks['MaxRank'] = max(ranks['Rank'])
            ranks['NormalizedRank'] = ranks['Rank'] / ranks['MaxRank']
            ranks['OrdinalRank'] = [i for i in range(len(ranks))]
            ranks['Percentile'] = np.linspace(0, 100, len(ranks))
            all_ranks.append(ranks)
        except ValueError:
            pass

    df = pd.concat(all_ranks)
    df = df.sort_values(by=groupby_cols)

    return df


def filter_hiring(df, multigraph=False):
    """
    exclude faculty employed at level `X` who did not receive their degree from
    one of U.S. institutions that employs faculty at level `X`

    We require institutions to have at least one hire and at least one placement.

    20211213:
        Require institutions to employ at least one graduate in the field.
    
    If multigraph is false, will collapse the PersonIds into `HiredFaculty`. 
    Else leaves a row for each PersonId.
    """
    df = df.copy()

    df['HiredFaculty'] = df.groupby(['DegreeInstitutionId', 'InstitutionId'])['PersonId'].transform('nunique')
    df['TotalHires'] = df.groupby('InstitutionId')['PersonId'].transform('nunique')
    df['TotalPlacements'] = df.groupby('DegreeInstitutionId')['PersonId'].transform('nunique')

    df = df[
        (df['DegreeInstitutionId'].notnull())
        &
        (df['InstitutionId'].notnull())
        &
        (df['TotalHires'] > 0)
        &
        (df['TotalPlacements'] > 0)
        &
        (df['HiredFaculty'] > 0)
    ]

    while set(df['DegreeInstitutionId'].unique()) - set(df['InstitutionId'].unique()):
        df = df[
            df['DegreeInstitutionId'].isin(df['InstitutionId'].unique())
        ]

    if multigraph:
        df = df[
            [
                'DegreeInstitutionId',
                'InstitutionId',
                'PersonId',
            ]
        ].drop_duplicates()
    else:
        df = df[
            [
                'DegreeInstitutionId',
                'InstitutionId',
                'HiredFaculty',
            ]
        ].drop_duplicates()

    return df


def recover_ranks(df, recovered_rank_column='Rank'):
    """
    A[i, j] = x --> "institution j hired x faculty from institution i"

    1. give each InstitutionId a zero-index.
    2. map the DataFrame onto a matrix
    3. pass the matrix to SpringRank
    4. shift the ranks:
        - min(ranks) = 0
    5. map the ranks back from their indexes

    return the ranks
    """
    df = df.copy()

    identifiers = usfhn.utils.identifier_to_integer_id_map(df['InstitutionId'])

    df['i'] = df['DegreeInstitutionId'].apply(identifiers.get).astype(int)
    df['j'] = df['InstitutionId'].apply(identifiers.get).astype(int)
    matrix = np.zeros((len(identifiers), len(identifiers)), dtype=int)

    for i, j, w in zip(df['i'], df['j'], df['HiredFaculty']):
        matrix[i, j] = w

    index_sorted_identifiers = sorted(identifiers.keys(), key=lambda _id: identifiers[_id])
    ranks = pd.DataFrame({
        recovered_rank_column: SpringRank.get_scaled_ranks(matrix),
        'InstitutionId': index_sorted_identifiers,
    })

    min_rank = min(ranks[recovered_rank_column])
    ranks[recovered_rank_column] -= min_rank
    return ranks


################################################################################
# stuff for stats
################################################################################
def get_production_ranks_for_stats(df, groupby_cols):
    """
    df needs columns:
    - groupby_cols
    - PersonId
    - DegreeInstitutionId

    returns:
    - groupby_cols
    - InstitutionId
    - Percentile: rank (0 = most producingest; 1 = least)
    """
    production_ranks_dfs = []
    for _, rows in df.groupby(groupby_cols):
        rows = rows.copy()
        rows['Faculty'] = rows['PersonId'].nunique()
        rows['Production'] = rows.groupby('DegreeInstitutionId')['PersonId'].transform('nunique')
        rows['ProductionFraction'] = rows['Production'] / rows['Faculty']

        rows = rows[
            groupby_cols + [
                'ProductionFraction',
                'DegreeInstitutionId',
            ]
        ].drop_duplicates().rename(columns={
            'DegreeInstitutionId': 'InstitutionId',
        })

        rows = rows.sort_values(by=['ProductionFraction'], ascending=False)
        n_institutions = rows['InstitutionId'].nunique()
        rows['Percentile'] = [i / n_institutions for i in range(n_institutions)]
        production_ranks_dfs.append(rows)

    return pd.concat(production_ranks_dfs)


def get_placements(by_year=False, by_gender=False):
    import usfhn.gender
    import usfhn.stats
    groupby_cols = [
        'TaxonomyLevel',
        'TaxonomyValue',
    ]

    extra_columns = []

    if by_year:
        extra_columns.append('Year')

    df = usfhn.datasets.get_dataset_df('data', by_year=by_year)

    df = usfhn.fieldwork.linearize_df_taxonomy_hierarchy(df)

    df = df[
        groupby_cols + extra_columns + [
            'PersonId',
            'InstitutionId',
            'DegreeInstitutionId',
        ]
    ].drop_duplicates()

    df, _ = usfhn.stats.add_groupby_annotations_to_df(df, by_gender=by_gender)

    ranks = usfhn.stats.runner.get('ranks/df', rank_type='prestige')[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'InstitutionId',
            'Percentile',
        ]
    ].drop_duplicates()

    df = df.merge(
        ranks,
        on=['InstitutionId'] + groupby_cols,
    )

    df = df.merge(
        ranks.rename(columns={
            'InstitutionId': 'DegreeInstitutionId',
            'Percentile': 'DegreePercentile',
        }),
        on=['DegreeInstitutionId'] + groupby_cols,
    )

    df['NormalizedRankDifference'] = df['Percentile'] - df['DegreePercentile']
    df['NormalizedRankDifference'] *= -1
    return df


def get_ranks_df_for_stats(rank_type='production', by_year=False):
    """
    rank types: 
    - production
    - prestige

    we're going to convert all ranks to return in the following format:
    - TaxonomyLevel
    - TaxonomyValue
    - Year (if `by_year` == True)
    - InstitutionId
    - Percentile (0 = best rank, 1 = worst)
    """
    groupby_cols = [
        'TaxonomyLevel',
        'TaxonomyValue',
    ]

    if by_year:
        groupby_cols.append('Year')

    if rank_type == 'prestige':
        # we don't actually compute prestige ranks by year, so always pull the
        # ranks that aren't by year
        df = usfhn.datasets.get_dataset_df('ranks')

        if by_year:
            # however, for nice groupbys downstream, we'll fake like we actually
            # have year ranks
            dfs = []
            for year in usfhn.views.get_years():
                year_df = df.copy()
                year_df['Year'] = year
                dfs.append(year_df)

            df = pd.concat(dfs)

        df['Percentile'] /= 100
    elif rank_type == 'production':
        df = usfhn.datasets.get_dataset_df('data', by_year=by_year)
        df = usfhn.fieldwork.linearize_df_taxonomy_hierarchy(df)[
            groupby_cols + [
                'PersonId',
                'DegreeInstitutionId',
            ]
        ].drop_duplicates()

        df = get_production_ranks_for_stats(df, groupby_cols)

    return df


def get_rank_change_df(rank_type='production', by_year=False):
    """
    returns:
    - TaxonomyLevel
    - TaxonomyValue
    - Year (if `by_year` == True)
    - PersonId
    - RankDifference
    """
    import usfhn.stats

    groupby_cols = [
        'TaxonomyLevel',
        'TaxonomyValue',
    ]

    if by_year:
        groupby_cols.append('Year')
        ranks = usfhn.stats.runner.get('ranks/by-year/df', rank_type=rank_type)
    else: 
        ranks = usfhn.stats.runner.get('rank/df', rank_type=rank_type)
    
    df = usfhn.datasets.get_dataset_df('data', by_year=by_year)

    df = usfhn.fieldwork.linearize_df_taxonomy_hierarchy(df)[
        groupby_cols + [
            'PersonId',
            'InstitutionId',
            'DegreeInstitutionId',
        ]
    ].drop_duplicates()

    df = df.merge(
        ranks,
        on=groupby_cols + ['InstitutionId']
    )

    df = df.merge(
        ranks.rename(columns={
            'Percentile': 'DegreePercentile',
            'InstitutionId': 'DegreeInstitutionId',
        }),
        on=groupby_cols + ['DegreeInstitutionId']
    )

    df['RankDifference'] = df['Percentile'] - df['DegreePercentile']

    # because 0 is highest rank, but we want negative rank changes to be "down
    # the hierarchy"
    df['RankDifference'] = -1 * df['RankDifference']

    df = df[
        groupby_cols + [
            'PersonId',
            'RankDifference',
        ]
    ].drop_duplicates()

    return df


def get_hierarchy_stats(rank_type='production', by_year=False, by_gender=False, by_seniority=False):
    from usfhn.stats import add_groupby_annotations_to_df

    groupby_cols = [
        'TaxonomyLevel',
        'TaxonomyValue',
    ]

    if by_year:
        groupby_cols.append('Year')
        df = usfhn.stats.runner.get('ranks/by-year/change', rank_type=rank_type)
    else: 
        df = usfhn.stats.runner.get('ranks/change', rank_type=rank_type)

    df = df[
        groupby_cols + [
            'PersonId',
            'RankDifference',
        ]
    ].drop_duplicates()

    df, extra_groupby_cols = add_groupby_annotations_to_df(
        df,
        by_gender=by_gender,
        explode_gender=True,
        by_seniority=by_seniority,
    )

    groupby_cols += extra_groupby_cols

    return calculate_hierarchy_stats(df, 'RankDifference', groupby_cols)


def calculate_hierarchy_stats(df, rank_difference_column, groupby_cols):
    """
    expects df with columns:
    - `rank_difference_column`
    - PersonId

    returns:
    - Edges: edge count over all MovemenTypes
    - MeanDistance: mean distance travelled over all MovementTypes
    - MovementType: 'Self-Hire'/'Upward'/'Downward'
    - MovementEdges: # of edges in MovementType
    - MovementFraction: MovementEdges/Edge
    - MeanMovementDistance: mean distance travelled in MovementType


    probably 1 per employment in, say, a taxonomy or whatever
    """
    df = df.copy()

    up = df[df[rank_difference_column] > 0].copy()
    up['MovementType'] = 'Upward'

    self_hires = df[df[rank_difference_column] == 0].copy()
    self_hires['MovementType'] = 'Self-Hire'

    down = df[df[rank_difference_column] < 0].copy()
    down['MovementType'] = 'Downward'

    df = pd.concat([up, self_hires, down])

    df['MovementEdges'] = df.groupby(groupby_cols + ['MovementType'])['PersonId'].transform('nunique')

    df['MeanMovementDistance'] = df.groupby(
        groupby_cols + ['MovementType']
    )[rank_difference_column].transform('mean')

    df['MeanDistance'] = df.groupby(
        groupby_cols
    )[rank_difference_column].transform('mean')

    df = df[
        groupby_cols + [
            'MovementType',
            'MovementEdges',
            'MeanMovementDistance',
            'MeanDistance',
        ]
    ].drop_duplicates()

    df['Edges'] = df.groupby(groupby_cols)['MovementEdges'].transform('sum')

    df['MovementFraction'] = df['MovementEdges'] / df['Edges']

    df = df[
        groupby_cols +
        [
            'Edges',
            'MeanDistance',
            'MovementType',
            'MovementEdges',
            'MovementFraction',
            'MeanMovementDistance',
        ]
    ].drop_duplicates()

    return df


def get_gendered_movement_distance_significance(rank_type='prestige', alpha=.05):
    import usfhn.stats
    import usfhn.gender

    df = usfhn.stats.runner.get('ranks/change', rank_type=rank_type)
    df = usfhn.gender.annotate_gender(df, explode_gender=True)
    df = df[
        (df['Gender'].isin(['All', 'Male', 'Female']))
        &
        (df['RankDifference'] != 0)
    ]

    df = df[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'Gender',
            'PersonId',
            'RankDifference',
        ]
    ].drop_duplicates()

    up_df = df.copy()[
        df['RankDifference'] > 0
    ]

    up_df['MovementType'] = 'Up'

    down_df = df.copy()[
        df['RankDifference'] < 0
    ]

    down_df['MovementType'] = 'Down'

    df = pd.concat([up_df, down_df])

    new_rows = []
    for (level, value, movement_type), rows in df.groupby(['TaxonomyLevel', 'TaxonomyValue', 'MovementType']):
        new_row = {
            'TaxonomyLevel': level,
            'TaxonomyValue': value,
            'MovementType': movement_type,
        }

        p, significant = hnelib.stats.ks_test(
            rows[rows['Gender'] == 'Male']['RankDifference'],
            rows[rows['Gender'] == 'Female']['RankDifference'],
        )

        new_row['DistancePUncorrected'] = p
        new_row['DistanceSignificant'] = significant

        for gender, gendered_rows in rows.groupby('Gender'):
            gender_key = '' if gender == 'All' else gender
            new_row[f"Distance{gender_key}"] = abs(gendered_rows['RankDifference'].mean())

        new_rows.append(new_row)

    df = hnelib.pandas.aggregate_df_over_column(
        pd.DataFrame(new_rows),
        agg_col='MovementType',
        join_cols=[
            'TaxonomyLevel',
            'TaxonomyValue',
        ],
        value_cols=[
            'DistancePUncorrected',
            'DistanceSignificant',
            'Distance',
            'DistanceFemale',
            'DistanceMale',
        ],
    )

    for move in ['Up', 'Down']:
        p_col = f"{move}DistancePUncorrected"
        corrected_p_col = f"{move}DistanceP"

        df = usfhn.stats_utils.correct_multiple_hypotheses(
            df,
            p_col=p_col,
            corrected_p_col=corrected_p_col,
        )

        df[f"{move}DistanceSignificant"] &= df[corrected_p_col] < alpha

        df = df.drop(columns=[
            p_col,
        ])

    df['SignificantDistanceDifference'] = False
    for move in ['Up', 'Down']:
        df['SignificantDistanceDifference'] |= df[f'{move}DistanceSignificant']

    return df


def get_mean_rank_change(by_year=False, rank_type='prestige', by_gender=False, by_seniority=False):
    """
    returns:
    - TaxonomyLevel
    - TaxonomyValue
    - Year (if by_year)
    - MeanRankChange
    - MedianRankChange
    """
    from usfhn.stats import add_groupby_annotations_to_df

    groupby_cols = [
        'TaxonomyLevel',
        'TaxonomyValue',
    ]

    if by_year:
        groupby_cols.append('Year')
        df = usfhn.stats.runner.get('ranks/by-year/change', rank_type=rank_type)
    else: 
        df = usfhn.stats.runner.get('ranks/change', rank_type=rank_type)

    df['MeanRankChange'] = df.groupby(groupby_cols)['RankDifference'].transform('mean')
    df['MedianRankChange'] = df.groupby(groupby_cols)['RankDifference'].transform('median')

    df = df[
        groupby_cols + [
            'MeanRankChange',
            'MedianRankChange',
        ]
    ].drop_duplicates()

    df, _ = add_groupby_annotations_to_df(
        df,
        by_gender=by_gender,
        explode_gender=True,
        by_seniority=by_seniority,
    )

    return df


def get_field_to_field_rank_correlations(rank_type='prestige', p_threshold=.05):
    """
    df columns:
        TaxonomyLevel
        TaxonomyValueOne
        TaxonomyValueTwo
        Pearson
        P
    """
    import usfhn.stats
    df = usfhn.stats.runner.get('ranks/df', rank_type=rank_type)

    rank_col = 'NormalizedRank' if rank_type == 'prestige' else 'Percentile'

    df = df[
        [
            rank_col,
            'InstitutionId',
            'TaxonomyLevel',
            'TaxonomyValue',
        ]
    ].drop_duplicates()

    df_one = df.copy().rename(columns={
        rank_col: 'RankOne',
        'TaxonomyValue': 'TaxonomyValueOne',
    })

    df_two = df.copy().rename(columns={
        rank_col: 'RankTwo',
        'TaxonomyValue': 'TaxonomyValueTwo',
    })

    df = df_one.merge(
        df_two,
        on=['InstitutionId', 'TaxonomyLevel']
    )

    groupby_cols = ['TaxonomyLevel', 'TaxonomyValueOne', 'TaxonomyValueTwo']

    correlations = []
    for (level, value_one, value_two), rows in df.groupby(groupby_cols):
        if len(rows) < 5:
            continue

        pearson, p = scipy.stats.pearsonr(rows['RankOne'], rows['RankTwo'])
        correlations.append({
            'TaxonomyLevel': level,
            'TaxonomyValueOne': value_one,
            'TaxonomyValueTwo': value_two,
            'Pearson': pearson,
            'P': p,
            'Significant': p_threshold > p,
        })

    return pd.DataFrame(correlations)

def get_institution_fields():
    import usfhn.stats
    df = usfhn.stats.runner.get('ranks/df', rank_type='prestige')
    df = df[
        df['TaxonomyLevel'] == 'Field'
    ][
        [
            'TaxonomyValue',
            'InstitutionId',
        ]
    ].drop_duplicates()

    df['Count'] = df.groupby('InstitutionId')['TaxonomyValue'].transform('nunique')

    df = df[
        [
            'InstitutionId',
            'Count',
        ]
    ].drop_duplicates()

    return df

def get_institution_placement_stats():
    import usfhn.stats
    ranks = usfhn.stats.runner.get('ranks/df', rank_type='prestige')[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'OrdinalRank',
            'InstitutionId',
        ]
    ].drop_duplicates().rename(columns={
        'OrdinalRank': 'Rank',
    })

    df = usfhn.stats.runner.get('faculty-hiring-network')[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'InstitutionId',
            'DegreeInstitutionId',
            'Count',
        ]
    ].drop_duplicates()

    df = df.merge(
        ranks,
        on=[
            'TaxonomyLevel',
            'TaxonomyValue',
            'InstitutionId',
        ],
    ).merge(
        ranks.rename(columns={
            'InstitutionId': 'DegreeInstitutionId',
            'Rank': 'DegreeRank',
        }),
        on=[
            'TaxonomyLevel',
            'TaxonomyValue',
            'DegreeInstitutionId',
        ],
    )

    df = df[
        df['InstitutionId'] != df['DegreeInstitutionId']
    ]

    df['Up'] = df['Rank'] < df['DegreeRank']
    df['Down'] = ~df['Up']
    df['Up'] = df['Up'].astype(int)
    df['Down'] = df['Down'].astype(int)

    df['Up'] *= df['Count']
    df['Down'] *= df['Count']

    df = df.drop(columns=['Rank', 'DegreeRank', 'Count'])

    df['UpHires'] = df.groupby([
        'TaxonomyLevel',
        'TaxonomyValue',
        'InstitutionId',
    ])['Down'].transform('sum')

    df['DownHires'] = df.groupby([
        'TaxonomyLevel',
        'TaxonomyValue',
        'InstitutionId',
    ])['Up'].transform('sum')

    df['UpPlacements'] = df.groupby([
        'TaxonomyLevel',
        'TaxonomyValue',
        'DegreeInstitutionId',
    ])['Up'].transform('sum')

    df['DownPlacements'] = df.groupby([
        'TaxonomyLevel',
        'TaxonomyValue',
        'DegreeInstitutionId',
    ])['Down'].transform('sum')

    hires_df = df.copy()[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'InstitutionId',
            'UpHires',
            'DownHires',
        ]
    ].drop_duplicates()

    placements_df = df.copy()[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'DegreeInstitutionId',
            'UpPlacements',
            'DownPlacements',
        ]
    ].drop_duplicates().rename(columns={
        'DegreeInstitutionId': 'InstitutionId'
    })

    df = hires_df.merge(
        placements_df,
        on=[
            'TaxonomyLevel',
            'TaxonomyValue',
            'InstitutionId',
        ]
    )

    return df
