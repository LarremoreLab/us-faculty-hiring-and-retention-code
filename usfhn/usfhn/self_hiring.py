from functools import lru_cache
import itertools
import numpy as np
import pandas as pd
from patsy import dmatrices
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest

import hnelib.utils
import hnelib.model
import hnelib.pandas

import usfhn.measurements as measurements
import usfhn.fieldwork
import usfhn.datasets
import usfhn.constants as constants
import usfhn.rank
import usfhn.views as views
import usfhn.plot_utils
import usfhn.rank
import usfhn.stats_utils


def annotate_self_hires(df):
    self_hires_df = usfhn.datasets.get_dataset_df('data')[
        [
            'PersonId',
            'InstitutionId',
            'DegreeInstitutionId',
        ]
    ].drop_duplicates()


    self_hires_df['SelfHire'] = self_hires_df['InstitutionId'] == self_hires_df['DegreeInstitutionId']

    self_hires_df = self_hires_df[
        [
            'PersonId',
            'SelfHire',
        ]
    ].drop_duplicates()

    df = df.merge(
        self_hires_df,
        on='PersonId',
        how='left'
    )
    df['SelfHire'] = df['SelfHire'].fillna(False)

    return df

################################################################################
# stats dataframes
################################################################################
def get_taxonomy_self_hiring(
    by_year=False,
    by_faculty_rank=False,
    by_seniority=False,
    by_gender=False,
    by_institution=False,
    by_career_age=False,
):
    """
    note: this is _within closedness_ self hire rates (i.e., non-U.S. faculty
    are included in the denominator)
    
    also note: this is NOT 

    return columns:
    - TaxonomyLevel
    - TaxonomyValue
    - Year (if `by_year`)
    - Rank (if `by_faculty_rank`)

    and the following stats columns (which vary by the above columns):
    - Faculty
    - SelfHires
    - NonSelfHires
    - SelfHiresFraction
    - NonSelfHiresFraction
    """
    from usfhn.stats import add_groupby_annotations_to_df

    groupby_cols = [
        'TaxonomyLevel',
        'TaxonomyValue',
    ]

    if by_year:
        groupby_cols.append('Year')
    
    df = usfhn.datasets.get_dataset_df('closedness_data', by_year=by_year)

    df = usfhn.fieldwork.linearize_df_taxonomy_hierarchy(df)

    df['SelfHire'] = df['DegreeInstitutionId'] == df['InstitutionId']

    if by_institution:
        groupby_cols.append('InstitutionId')

    df = df[
        groupby_cols + [
            'PersonId',
            'SelfHire',
        ]
    ].drop_duplicates()

    df, extra_groupby_cols = add_groupby_annotations_to_df(
        df,
        by_gender=by_gender,
        explode_gender=True,
        by_faculty_rank=by_faculty_rank,
        by_seniority=by_seniority,
        by_career_age=by_career_age,
    )

    groupby_cols += extra_groupby_cols

    df['Faculty'] = df.groupby(groupby_cols)['PersonId'].transform('nunique')

    df = df[
        df['SelfHire']
    ]

    df['SelfHires'] = df.groupby(groupby_cols)['PersonId'].transform('nunique')

    df = df[
        groupby_cols + [
            'Faculty',
            'SelfHires',
        ]
    ].drop_duplicates()

    df['NonSelfHires'] = df['Faculty'] - df['SelfHires']
    df['SelfHiresFraction'] = df['SelfHires'] / df['Faculty']
    df['NonSelfHiresFraction'] = 1 - df['SelfHiresFraction']

    return df


################################################################################
# misc stuff
################################################################################
def compare_self_hire_rate_of_top_institutions_vs_rest(threshold=5):
    import usfhn.stats
    df = usfhn.stats.runner.get('self-hires/by-institution/df')
    ranks = usfhn.stats.runner.get('ranks/df', rank_type='prestige')

    df = df.merge(
        ranks,
        on=['InstitutionId', 'TaxonomyLevel', 'TaxonomyValue'],
    )[
        [
            'InstitutionId',
            'TaxonomyLevel',
            'TaxonomyValue',
            'SelfHiresFraction',
            'OrdinalRank',
        ]
    ].rename(columns={
        'SelfHiresFraction': 'SelfHireRate',
    })

    top_df = df.copy()[
        df['OrdinalRank'] < threshold
    ]

    bottom_df = df.copy()[
        df['OrdinalRank'] >= threshold
    ]

    top_df['TopSelfHireRateMean'] = top_df.groupby(
        ['TaxonomyLevel', 'TaxonomyValue']
    )['SelfHireRate'].transform('mean')

    bottom_df['BottomSelfHireRateMean'] = bottom_df.groupby(
        ['TaxonomyLevel', 'TaxonomyValue']
    )['SelfHireRate'].transform('mean')

    df = top_df[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'TopSelfHireRateMean',
        ]
    ].drop_duplicates().merge(
        bottom_df[
            [
                'TaxonomyLevel',
                'TaxonomyValue',
                'BottomSelfHireRateMean',
            ]
        ].drop_duplicates(),
        on=['TaxonomyLevel', 'TaxonomyValue']
    )

    df['Ratio'] = df['TopSelfHireRateMean'] / df['BottomSelfHireRateMean']

    return df


def compare_self_hire_rate_of_bottom_institutions_vs_rest(bottom_threshold=50):
    import usfhn.stats
    df = usfhn.stats.runner.get('self-hires/by-institution/df')
    ranks = usfhn.stats.runner.get('ranks/df', rank_type='prestige')

    df = df.merge(
        ranks,
        on=['InstitutionId', 'TaxonomyLevel', 'TaxonomyValue'],
    )[
        [
            'InstitutionId',
            'TaxonomyLevel',
            'TaxonomyValue',
            'SelfHireFraction',
            'OrdinalRank',
        ]
    ].rename(columns={
        'SelfHireFraction': 'SelfHireRate',
    })

    df['MaxOrdinalRank'] = df.groupby(['TaxonomyLevel', 'TaxonomyValue'])['OrdinalRank'].transform('max')
    
    top_df = df.copy()[
        df['OrdinalRank'] < df['MaxOrdinalRank'] - bottom_threshold
    ]

    bottom_df = df.copy()[
        df['OrdinalRank'] >= df['MaxOrdinalRank'] - bottom_threshold
    ]

    top_df['TopSelfHireRateMean'] = top_df.groupby(
        ['TaxonomyLevel', 'TaxonomyValue']
    )['SelfHireRate'].transform('mean')

    bottom_df['BottomSelfHireRateMean'] = bottom_df.groupby(
        ['TaxonomyLevel', 'TaxonomyValue']
    )['SelfHireRate'].transform('mean')

    df = top_df[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'TopSelfHireRateMean',
        ]
    ].drop_duplicates().merge(
        bottom_df[
            [
                'TaxonomyLevel',
                'TaxonomyValue',
                'BottomSelfHireRateMean',
            ]
        ].drop_duplicates(),
        on=['TaxonomyLevel', 'TaxonomyValue']
    )

    df['Ratio'] = df['TopSelfHireRateMean'] / df['BottomSelfHireRateMean']

    return df


def get_expected_self_hiring_rates_and_compare_to_actual(by_gender=False):
    import usfhn.stats

    groupby_cols = [
        'TaxonomyLevel',
        'TaxonomyValue',
    ]

    if by_gender:
        df = usfhn.stats.runner.get('basics/closedness-faculty-hiring-network/by-gender')
        groupby_cols.append('Gender')
    else:
        df = usfhn.stats.runner.get('basics/closedness-faculty-hiring-network/df')

    out_df = df.copy()[
        groupby_cols + [
            'DegreeInstitutionId',
            'OutDegree',
        ]
    ].drop_duplicates().rename(columns={
        'DegreeInstitutionId': 'InstitutionId',
    })

    in_df = df.copy()[
        groupby_cols + [
            'InstitutionId',
            'InDegree',
        ]
    ].drop_duplicates()

    df = out_df.merge(
        in_df,
        on=groupby_cols + ['InstitutionId'],
        how='outer',
    )

    df['OutDegree'] = df['OutDegree'].fillna(0)
    df['InDegree'] = df['InDegree'].fillna(0)

    df['SelfHires'] = df['OutDegree'] * df['InDegree']
    df['Edges'] = df.groupby(groupby_cols)['OutDegree'].transform('sum')
    df['E[SelfHires]'] = df.groupby(groupby_cols)['SelfHires'].transform('sum')
    df['E[SelfHires]'] /= df['Edges'] ** 2

    expected = df[
        groupby_cols + [
            'E[SelfHires]',
        ]
    ].drop_duplicates()

    if by_gender:
        actual = usfhn.stats.runner.get('self-hires/by-gender/df')
    else:
        actual = usfhn.stats.runner.get('self-hires/df')

    actual = actual[
        groupby_cols + [
            'SelfHiresFraction',
        ]
    ].drop_duplicates().rename(columns={
        'SelfHiresFraction': 'A[SelfHires]',
    })

    df = expected.merge(
        actual,
        on=groupby_cols,
    )

    df['Actual/Expected'] = df['A[SelfHires]'] / df['E[SelfHires]']
    df = df.sort_values(by='Actual/Expected')

    return df


def get_gendered_rank_change_significance(z_threshold=1.96, p_threshold=.05):
    import usfhn.stats
    df = usfhn.stats.runner.get('ranks/gender/hierarchy-stats', rank_type='prestige')

    movement_type_remap = {
        'Upward': 'Up',
        'Downward': 'Down',
        'Self-Hire': 'Self',
    }

    df['MovementType'] = df['MovementType'].apply(movement_type_remap.get)

    genders = ['All', 'Male', 'Female']

    df = df[
        df['Gender'].isin(genders)
    ]

    edges_df = df.copy()[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'Gender',
            'Edges',
        ]
    ].drop_duplicates()

    df = df.drop(columns=['Edges']).rename(columns={
        'MovementEdges': 'Edges',
        'MovementFraction': 'Fraction',
    })

    value_cols = [
        'Edges',
        'Fraction',
    ]

    df = hnelib.pandas.aggregate_df_over_column(
        df,
        agg_col='MovementType',
        join_cols=[
            'TaxonomyLevel',
            'TaxonomyValue',
            'Gender',
        ],
        value_cols=value_cols,
    )

    df = df.merge(
        edges_df,
        on=[
            'TaxonomyLevel',
            'TaxonomyValue',
            'Gender',
        ]
    )

    movement_types = [
        'Up',
        'Down',
        'Self',
    ]

    movement_value_cols = ['Edges']
    for movement_type, value_col in itertools.product(movement_types, value_cols):
        movement_value_cols.append(f"{movement_type}{value_col}")

    df = hnelib.pandas.aggregate_df_over_column(
        df,
        agg_col='Gender',
        join_cols=[
            'TaxonomyLevel',
            'TaxonomyValue',
        ],
        value_cols=movement_value_cols,
    )

    df = df.drop(columns='AllEdges')

    # df['DiffUpFrac'] = df['FemaleUpFrac'] - df['MaleUpFrac']
    # df['DiffDownFrac'] = df['FemaleDownFrac'] - df['MaleDownFrac']
    # df['DiffSelfFrac'] = df['FemaleSelfFrac'] - df['MaleSelfFrac']

    cols = list(df.columns)

    new_df = []
    for i, row in df.iterrows():
        new_row = {c: row[c] for c in cols}
        new_row = {
            'TaxonomyLevel': row['TaxonomyLevel'],
            'TaxonomyValue': row['TaxonomyValue'],
        }

        significant_difference = False
        for move in movement_types:
            for gender in genders:
                gender_key = '' if gender == 'All' else gender
                new_row[f'{move}Fraction{gender_key}'] = row[f'{gender}{move}Fraction']

            z_value, p_value = proportions_ztest(
                [row[f'Male{move}Edges'], row[f'Female{move}Edges']],
                [row['MaleEdges'], row['FemaleEdges']],
            )

            p_significant = p_value < p_threshold
            z_significant = abs(z_value) > z_threshold

            new_row[f'{move}FractionPUncorrected'] = p_value
            new_row[f'{move}FractionPSignificant'] = p_significant
            new_row[f'{move}FractionZSignificant'] = z_significant

        new_df.append(new_row)

    df = pd.DataFrame(new_df)
    for move in movement_types:
        df = usfhn.stats_utils.correct_multiple_hypotheses(
            df,
            p_col=f"{move}FractionPUncorrected",
            corrected_p_col=f"{move}FractionP",
        )

        df[f'{move}FractionSignificant'] = (
            (df[f'{move}FractionP'] < p_threshold)
            &
            (df[f'{move}FractionZSignificant'])
        )

        df = df.drop(columns=[
            f'{move}FractionPUncorrected',
            f'{move}FractionPSignificant',
            f'{move}FractionZSignificant',
        ])

    df['SignificantFractionDifference'] = False
    for move in movement_types:
        df['SignificantFractionDifference'] |= df[f'{move}FractionSignificant']

    df = df.merge(
        usfhn.stats.runner.get('ranks/movement-distance', rank_type='prestige'),
        on=[
            'TaxonomyLevel',
            'TaxonomyValue',
        ]
    )

    cols = [
        'TaxonomyLevel',
        'TaxonomyValue',
    ]

    for move in movement_types:
        move_cols = []
        for col in df.columns:
            if col.startswith(move):
                move_cols.append(col)

        move_cols.sort()
        cols += move_cols

    cols += [
        'SignificantFractionDifference',
        'SignificantDistanceDifference',
    ]

    df = df[cols]

    return df


def get_rank_change_table(
    z_threshold=1.96,
    p_threshold=.05,
    include_significant_difference=False,
):
    df = get_gendered_rank_change_significance(z_threshold=z_threshold, p_threshold=p_threshold)

    movement_types = [
        'Up',
        'Down',
        'Self',
    ]

    movement_to_string = {
        'Up': 'up',
        'Down': 'down',
        'Self': 'self-hires',
    }

    rows = []
    for i, row in df.iterrows():
        new_row = {
            'TaxonomyValue': row['TaxonomyValue'],
            'TaxonomyLevel': row['TaxonomyLevel'],
        }

        for move, move_string in movement_to_string.items():
            for measure in ['Fraction', 'Distance']:
                if move == 'Self' and measure == 'Distance':
                    continue

                string = str(round(100 * row[f'{move}{measure}'])) + '%'

                if row[f'{move}{measure}Significant']:
                    female_val = row[f"{move}{measure}Female"]
                    male_val = row[f"{move}{measure}Male"]

                    difference = hnelib.utils.fraction_to_percent(female_val - male_val)

                    if difference != 0:
                        sign = '+' if difference > 0 else ''
                        string = f"{string} ({sign}{str(difference)}%)"

                col = move_string

                if measure == 'Distance':
                    col = f'ranks {col}'

                new_row[col] = string

        rows.append(new_row)

    df = pd.DataFrame(rows)

    umbrella_df = usfhn.views.filter_to_academia_and_domain(df.copy()).rename(columns={
        'TaxonomyValue': 'Domain',
    })
    umbrella_df['Field'] = ''

    field_df = usfhn.views.filter_by_taxonomy(df.copy(), 'Field')
    field_df = usfhn.views.annotate_umbrella(field_df, 'Field').drop(columns=['Field'])
    field_df = field_df.rename(columns={
        'TaxonomyValue': 'Field',
        'Umbrella': 'Domain',
    })

    field_df['Field'] = field_df['Field'].apply(usfhn.plot_utils.clean_taxonomy_string)

    df = pd.concat([umbrella_df, field_df])

    df['Domain'] = df['Domain'].apply(usfhn.plot_utils.clean_taxonomy_string)

    df = df[
        [
            'Domain',
            'Field',
            'up',
            'down',
            'self-hires',
            "ranks up",
            "ranks down",
        ]
    ].drop_duplicates()

    df = df.sort_values(by=[
        'Domain',
        'Field',
    ])

    return df



def get_significance_of_gendered_self_hiring_rates(z_threshold=1.96, p_threshold=.05):
    df = usfhn.stats.runner.get('self-hires/by-gender/df')

    gender_df = views.get_gender_df_and_rename_non_join_columns(df, 'Male').merge(
        views.get_gender_df_and_rename_non_join_columns(df, 'Female'),
        on=['TaxonomyLevel', 'TaxonomyValue']
    )

    new_df = []
    for i, row in gender_df.iterrows():
        z_value, p_value = proportions_ztest(
            [row['MaleSelfHires'], row['FemaleSelfHires']],
            [row['MaleFaculty'], row['FemaleFaculty']],
        )

        male_fraction = row['MaleSelfHiresFraction']
        female_fraction = row['FemaleSelfHiresFraction']

        z_significant = z_threshold < abs(z_value)
        p_significant = p_value < p_threshold

        new_df.append({
            'TaxonomyLevel': row['TaxonomyLevel'],
            'TaxonomyValue': row['TaxonomyValue'],
            'WomenHiredMoreThanMen': male_fraction < female_fraction,
            'ZSignificant': z_significant,
            'P': p_value,
            'Significant': z_significant and p_significant,
        })

    df = pd.DataFrame(new_df)

    df = usfhn.stats_utils.correct_multiple_hypotheses(df)
    df['Significant'] = (
        (df['ZSignificant'])
        &
        (df['PCorrected'] < p_threshold)
    )

    return df


@lru_cache()
def null_model(null_rate='uniform'):
    import usfhn.stats
    df = usfhn.stats.runner.get('basics/faculty-hiring-network').merge(
        usfhn.stats.runner.get('ranks/df', rank_type='prestige'),
        on=['InstitutionId', 'TaxonomyLevel', 'TaxonomyValue']
    ).merge(
        usfhn.stats.runner.get('self-hires/institution'),
        on=['InstitutionId', 'TaxonomyLevel', 'TaxonomyValue']
    )

    # some of these are too small
    df = df[
        df['TaxonomyLevel'] != 'Taxonomy'
    ]

    rate_to_function = {
        'uniform': uniform_null,
        'max_other': max_other_null,
        'chung_lu': chung_lu_null,
        'prestige_neighbor': get_average_self_hiring_rate_by_prestige_proximity,
    }

    rate_function = rate_to_function[null_rate]

    nulls = []
    for (year, gender, level, value), rows in df.groupby(['Year', 'Gender', 'TaxonomyLevel', 'TaxonomyValue']):
        if len(rows) == 1:
            continue

        value_nulls = rate_function(rows)[
            ['InstitutionId', 'NullRate']
        ].drop_duplicates()
        value_nulls['Year'] = year
        value_nulls['Gender'] = gender
        value_nulls['TaxonomyLevel'] = level
        value_nulls['TaxonomyValue'] = value
        nulls.append(value_nulls)

    df = df[
        [
            'Year',
            'Gender',
            'TaxonomyLevel',
            'TaxonomyValue',
            'InstitutionId',
            'SelfHires',
            'SelfHireFraction',
            'Hires',
            'OrdinalRank',
        ]
    ]

    df = df.merge(
        pd.concat(nulls),
        on=[
            'Year',
            'Gender',
            'TaxonomyLevel',
            'TaxonomyValue',
            'InstitutionId',
        ],
    )

    df['SelfHiresExpectedFromNull'] = df['SelfHires'] * df['NullRate']
    df['SelfHiresDelta'] = df['SelfHires'] - df['SelfHiresExpectedFromNull'] 
    df['ActualOverExpected'] = df['SelfHireFraction'] / df['NullRate']
    # df['PrestigeNeighborNullSelfHires'] = df['SelfHires'] * df['PrestigeNeighborNullRate']
    # df['PrestigeNeighborNullSelfHiresDelta'] = df['SelfHires'] - df['PrestigeNeighborNullSelfHires']

    return df

def uniform_null(df):
    df = df.copy()
    institution_count = df['InstitutionId'].nunique()
    df['NullRate'] = 1 / institution_count
    return df


def max_other_null(df):
    """
    given some rows with:
    - InstitutionId
    - DegreeInstitutionId
    - Count
    - TotalCount

    returns the highest rate at which an institution hires from some other institution
    """
    df = df.copy()
    df = df[
        df['DegreeInstitutionId'] != df['InstitutionId']
    ]
    df['MaxCount'] = df.groupby('InstitutionId')['Count'].transform('max')
    df['NullRate'] = df['MaxCount'] / df['Hires']
    return df


def chung_lu_null(df):
    """
    given some rows with:
    - InstitutionId
    - DegreeInstitutionId
    - Count
    - TotalCount
    - InstitutionId

    expect is:
    k_out * k_in / 2m
    """
    df = df.copy()
    df = df[
        df['InstitutionId'] == df['DegreeInstitutionId']
    ]
    df['NullRate'] = df['OutDegree'] * df['InDegree']
    df['NullRate'] /= df['TotalCount']
    # df['NullRate'] /= 2 * df['TotalCount']
    return df


def get_average_self_hiring_rate_by_prestige_proximity(df):
    """
    given a df of N rows with columns:
    - InstitutionId
    - SelfHireRate
    - OrdinalRank

    returns the average ratio of the self hire rate each institution over the
    average of the self hire rates of the institutions one higher and one lower
    in the ranking

    df[i]['SelfHireRate'] / (df[i - 1]['SelfHireRate'] + df[i + 1]['SelfHireRate']) / 2)

    Then, how we turn this into a ratio is... ugly (to hunter).

    For each institution:
    1. take its True-SelfHireRate and multiply it by the average
    2. take the average of the neighbor-SelfHireRate(s)
    3. give it a ratio of: #1 / #2

    """
    df = df.copy()
    df = df[
        ['InstitutionId', 'SelfHireRate', 'OrdinalRank']
    ].drop_duplicates().sort_values(by='OrdinalRank')
    self_hire_rates = list(df['SelfHireRate'])
    values = []
    last_index = len(self_hire_rates) - 1
    for i, rate in enumerate(self_hire_rates):
        numerator = rate
        denominator = get_neighbor_rate(self_hire_rates, i)

        if denominator:
            values.append(numerator / denominator)
        else:
            values.append(0)

    mean = np.mean(values)

    null_ratios = []
    for i, institution_id in enumerate(list(df['InstitutionId'])):
        self_hire_rate = self_hire_rates[i]

        neighbor_rate = get_neighbor_rate(self_hire_rates, i)
        if neighbor_rate:
            rate = (mean * self_hire_rate) / neighbor_rate
        else:
            rate = 0

        null_ratios.append({
            'InstitutionId': institution_id,
            'NullRate': neighbor_rate,
        })

    return pd.DataFrame(null_ratios)


def get_neighbor_rate(values, i):
    last_index = len(values) - 1
    if i:
        neighbors = values[i - 1]

    if i < last_index:
        neighbors = values[i + 1]

    if neighbors:
        return np.mean(neighbors)
        
    return 0


@lru_cache(maxsize=1)
def get_binomial_regression_data():
    df = usfhn.datasets.CURRENT_DATASET.data[
        ['InstitutionId', 'DegreeInstitutionId', 'Taxonomy', 'Year', 'PersonId', 'Gender']
    ].drop_duplicates()

    df = usfhn.fieldwork.linearize_df_taxonomy_hierarchy(df)
    df = views.explode_gender(df)
    df = df[
        df['Gender'].isin(['Male', 'Female'])
    ]

    ranks = usfhn.datasets.get_dataset_df('ranks')
    ranks = ranks.drop(columns=['OrdinalRank', 'Percentile', 'MaxRank'])

    df = df.merge(
        ranks,
        on=['InstitutionId', 'Year', 'TaxonomyLevel', 'TaxonomyValue']
    )

    # testing on only binomial
    df = df[
        (df['Year'] == usfhn.constants.YEAR_UNION)
        &
        (df['NormalizedRank'].notnull())
    ]

    prestige_labels = np.linspace(.1, 1, 10)
    prestige_bins = np.linspace(0, 1, 11)
    df['PrestigeBin'] = pd.cut(df['NormalizedRank'], prestige_bins, labels=prestige_labels)

    df['SelfHire'] = df['InstitutionId'] == df['DegreeInstitutionId']
    df['SelfHire'] = df['SelfHire'].apply(int)
    df['Gender'] = df['Gender'].apply({'Male': 0, 'Female': 1}.get)

    return df


def binomial_regression():
    df = get_binomial_regression_data()
    regression_columns = ['TaxonomyLevel', 'TaxonomyValue', 'Gender', 'PrestigeBin']

    df = df[
        df['TaxonomyLevel'] == 'Academia'
    ]

    df = df[
        regression_columns + ['PersonId', 'SelfHire']
    ].drop_duplicates()

    df['Hires'] = df.groupby(regression_columns)['PersonId'].transform('nunique')
    df['SelfHires'] = df.groupby(regression_columns)['SelfHire'].transform('sum')
    df['NonSelfHires'] = df['Hires'] - df['SelfHires']
    df = df[
        ['SelfHires', 'NonSelfHires', 'PrestigeBin', 'Gender']
    ].drop_duplicates()

    formula = 'SelfHires + NonSelfHires ~ PrestigeBin + Gender'

    y_train, _ = dmatrices(formula, df, return_type='dataframe')
    X_train = df.copy()[
        ['PrestigeBin', 'Gender']
    ]

    X_train['Intercept'] = 1
    X_train = X_train.dropna()

    model = sm.GLM(y_train, X_train, family=sm.families.Binomial())
    results = model.fit()
    print(results.summary())
    for param, coef in results.params.items():
        print(f'{param}: {coef}')

    return df


def rank_vs_self_hires_logit(rank_type='prestige', by_new_hire=False, existing_hires=False):
    import usfhn.datasets
    df = usfhn.datasets.get_dataset_df('closedness_data')[
        [
            'PersonId',
            'Taxonomy',
            'InstitutionId',
            'DegreeInstitutionId',
        ]
    ].drop_duplicates()

    df['SelfHire'] = df['DegreeInstitutionId'] == df['InstitutionId']

    df['SelfHire'] = df['SelfHire'].apply(int)

    df = usfhn.fieldwork.linearize_df_taxonomy_hierarchy(df)

    ranks = usfhn.stats.runner.get('ranks/df', rank_type=rank_type)

    df = df.merge(
        ranks,
        on=[
            'TaxonomyLevel',
            'TaxonomyValue',
            'InstitutionId',
        ]
    )

    if by_new_hire:
        import usfhn.new_hires
        df = usfhn.new_hires.annotate_new_hires(df)

        if existing_hires:
            df = df[
                ~df['NewHire']
            ]
        else:
            df = df[
                df['NewHire']
            ]

    df = hnelib.model.get_logits(
        df,
        endog='SelfHire',
        exog='Percentile',
        groupby_cols=[
            'TaxonomyLevel',
            'TaxonomyValue',
        ],
    )

    df = usfhn.stats_utils.correct_multiple_hypotheses(df, p_col='Percentile-P', corrected_p_col='Percentile-P')

    return df


def get_self_hire_non_self_hire_risk_ratios():
    import usfhn.stats
    import usfhn.attrition

    df = usfhn.stats.runner.get('attrition/risk/self-hires')
    df = hnelib.pandas.aggregate_df_over_column(
        df,
        agg_col='SelfHire',
        join_cols=[
            'TaxonomyLevel',
            'TaxonomyValue',
        ],
        value_cols=[
            'AttritionRisk',
            'Events',
            'AttritionEvents'
        ],
        agg_value_to_label={
            True: 'SelfHire',
            False: 'NonSelfHire'
        }
    )

    df = df.merge(
        usfhn.attrition.compute_attrition_risk_significance(
            df,
            'SelfHireAttritionEvents',
            'SelfHireEvents',
            'NonSelfHireAttritionEvents',
            'NonSelfHireEvents',
        ),
        on=[
            'TaxonomyLevel',
            'TaxonomyValue',
        ]
    )

    df['Ratio'] = df['SelfHireAttritionRisk'] / df['NonSelfHireAttritionRisk']
    return df


def self_hiring_binned_by_prestige_attrition_risk(
    rank_type='production',
    p_threshold=.05,
    by_self_hire=False,
):
    import usfhn.stats

    df = usfhn.stats.runner.get('attrition/by-rank/df', rank_type=rank_type)[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'PersonId',
            'Year',
            'AttritionYear',
            'InstitutionId',
            'DegreeInstitutionId',
            'Percentile',
        ]
    ]

    df['SelfHire'] = df['InstitutionId'] == df['DegreeInstitutionId']
    df['SelfHire'] = df['SelfHire'].apply(int)
    df['AttritionYear'] = df['AttritionYear'].apply(int)

    df = df[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'Year',
            'AttritionYear',
            'SelfHire',
            'Percentile',
        ]
    ]

    df['Decile'] = df['Percentile'] * 100 // 10

    cols = [
        'TaxonomyLevel',
        'TaxonomyValue',
        'Decile',
        'SelfHire',
    ]

    df['Events'] = df.groupby(cols)['Year'].transform('count')
    df['AttritionEvents'] = df.groupby(cols)['AttritionYear'].transform('sum')
    df['Risk'] = df['AttritionEvents'] / df['Events']

    df = df[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'Decile',
            'SelfHire',
            'Events',
            'AttritionEvents',
            'Risk',
        ]
    ].drop_duplicates()


    df = hnelib.pandas.aggregate_df_over_column(
        df,
        agg_col='SelfHire',
        join_cols=[
            'TaxonomyLevel',
            'TaxonomyValue',
            'Decile',
        ],
        value_cols=[
            'Risk',
            # 'AttritionEvents',
            # 'Events',
        ],
        agg_value_to_label={
            True: 'SelfHire',
            False: 'NonSelfHire',
        }
    )

    df['Ratio'] = df['SelfHireRisk'] / df['NonSelfHireRisk']

    df = df[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'Decile',
            'Ratio',
        ]
    ].drop_duplicates().sort_values(by=[
        'TaxonomyLevel',
        'TaxonomyValue',
        'Decile',
    ])

    return df


def self_hiring_by_career_age_logistics(rank_type='production'):
    import usfhn.stats
    import usfhn.careers

    df = usfhn.stats.runner.get('attrition/by-rank/df', rank_type=rank_type)[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'PersonId',
            'Year',
            'AttritionYear',
            'InstitutionId',
            'DegreeInstitutionId',
            'Percentile',
        ]
    ]

    df['SelfHire'] = df['InstitutionId'] == df['DegreeInstitutionId']
    df['SelfHire'] = df['SelfHire'].apply(int)

    df['AttritionYear'] = df['AttritionYear'].apply(int)

    df = usfhn.careers.annotate_career_age(df)

    df = df[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'Year',
            'AttritionYear',
            'SelfHire',
            'Percentile',
            'CareerAge',
        ]
    ]

    def bin_career_age(age):
        if 0 <= age <= 10:
            return "0-10"
        elif 10 < age <= 20:
            return "11-20"
        elif 20 < age <= 30:
            return "21-30"
        elif 30 < age <= 40:
            return "31-40"

    df['CareerAgeBin'] = df['CareerAge'].apply(bin_career_age)

    results_dfs = []
    for career_age_bin, rows in df.groupby('CareerAgeBin'):
        rows = hnelib.model.get_logits(
            rows,
            endog='SelfHire',
            exog='Percentile',
            groupby_cols=[
                'TaxonomyLevel',
                'TaxonomyValue',
            ],
        )

        rows = usfhn.stats_utils.correct_multiple_hypotheses(
            rows,
            p_col='Percentile-P',
            corrected_p_col='Percentile-P',
        )

        rows['CareerAgeBin'] = career_age_bin
        results_dfs.append(rows)

    return pd.concat(results_dfs)

