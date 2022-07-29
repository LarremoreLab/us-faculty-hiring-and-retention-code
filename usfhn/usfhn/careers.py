import pandas as pd
from statsmodels.stats.proportion import proportions_ztest

import hnelib.model

import usfhn.constants
import usfhn.datasets
import usfhn.gini
import usfhn.rank


def annotate_career_age(df):
    join_cols = [
        'PersonId'
    ]

    by_year = 'Year' in df.columns

    if by_year:
        join_cols.append('Year')

    career_df = usfhn.datasets.get_dataset_df('closedness_data', by_year=by_year)

    career_df = career_df[
        [
            'PersonId',
            'DegreeYear',
            'Year',
        ]
    ].drop_duplicates()

    career_df['DegreeYear'] = career_df['DegreeYear'].fillna(0)

    career_df = career_df[
        career_df['DegreeYear'] >= usfhn.constants.EARLIEST_DEGREE_YEAR
    ]

    career_df['CareerAge'] = career_df['Year'] - career_df['DegreeYear']

    career_df = career_df[
        ['CareerAge'] + join_cols
    ].drop_duplicates()

    df = df.merge(
        career_df,
        on=join_cols,
    )

    return df


def get_career_moves_df():
    """
    we want:
    - PersonId
    - InstitutionId
    - CareerStep
    - CareerYears
    - Year
    - FirstYearAtInstitution
    """
    df = usfhn.datasets.get_dataset_df('closedness_data', 'careers')[
        [
            'PersonId',
            'InstitutionId',
            'Year',
        ]
    ].drop_duplicates()

    df['YearsAtInstitution'] = df.groupby(['PersonId', 'InstitutionId'])['Year'].transform('nunique')
    df['FirstYearAtInstitution'] = df.groupby(['PersonId', 'InstitutionId'])['Year'].transform('min')

    career_lengths_df = df.copy()[
        [
            'PersonId',
            'Year',
        ]
    ].drop_duplicates()

    career_lengths_df['CareerYears'] = career_lengths_df.groupby('PersonId')['Year'].transform('nunique')
    career_lengths_df = career_lengths_df[
        [
            'PersonId',
            'CareerYears',
        ]
    ].drop_duplicates()

    df = df.drop(columns=['Year']).drop_duplicates()

    df['CareerStep'] = df.groupby('PersonId')['FirstYearAtInstitution'].transform('cumcount')
    df['CareerStep'] += 1

    career_starts = usfhn.datasets.get_dataset_df('closedness_data', 'careers')[
        [
            'PersonId',
            'DegreeInstitutionId',
        ]
    ].drop_duplicates().rename(columns={
        'DegreeInstitutionId': 'InstitutionId'
    })

    career_starts['FirstYearAtInstitution'] = 0
    career_starts['YearsAtInstitution'] = 1
    career_starts['CareerStep'] = 0

    df = pd.concat([df, career_starts])

    df = df.merge(
        career_lengths_df,
        on='PersonId',
    )

    n_people = df['PersonId'].nunique()
    n_moves_people = df[
        df['CareerStep'] > 1
    ]['PersonId'].nunique()

    moves_percent = round(100 * (n_moves_people / n_people), 2)

    print(f"{n_people} people")
    print(f"{n_moves_people} people move institutions ({moves_percent}%)")

    return df


def get_move_risk_by_gender():
    """
    risk is:
    - numerator: # of moves
    - denominator: # of non-moves (years at a place)

    So if a person's career looks like this:
    
    YEAR: 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020
    INST:    A    A    A    A    B    B    C    C    C    C

    there are 10 years, 3 of them are moves, 7 are non-moves

    columns:
    - Gender
    - MoveRisk
    - EventCount: total career years of gender
    - MoveCount: # of moves
    """
    import usfhn.stats
    import usfhn.gender
    df = usfhn.stats.runner.get('careers/df')
    df = usfhn.gender.annotate_gender(df, explode_gender=True)

    df['PersonMoves'] = df.groupby('PersonId')['CareerStep'].transform('max')
    df['PersonMoves'] -= 1
    df['CareerYears'] -= 1

    df = df[
        [
            'PersonId',
            'PersonMoves',
            'CareerYears',
            'Gender',
        ]
    ].drop_duplicates()

    df['EventCount'] = df.groupby('Gender')['CareerYears'].transform('sum')
    df['MoveCount'] = df.groupby('Gender')['PersonMoves'].transform('sum')

    df['MoveRisk'] = df['MoveCount'] / df['EventCount']

    df = df[
        [
            'Gender',
            'MoveRisk',
            'EventCount',
            'MoveCount',
        ]
    ].drop_duplicates()

    return df

def get_move_risk_by_gender_and_year():
    """
    columns:
    - Year
    - Gender
    - MoveRisk
    - EventCount: total career years of gender
    - MoveCount: # of moves
    """
    import usfhn.stats
    import usfhn.gender
    df = usfhn.stats.runner.get('careers/df')

    years_df = usfhn.datasets.get_dataset_df('data', 'careers')[
        [
            'PersonId',
            'InstitutionId',
            'Year',
        ]
    ].drop_duplicates()

    years_df['FirstYear'] = years_df.groupby('PersonId')['Year'].transform('min')

    years_df['FirstYearAtInstitution'] = years_df.groupby([
        'PersonId',
        'InstitutionId',
    ])['Year'].transform('min')

    # You can't make a career move in your first year, so exclude these from the
    # count
    years_df = years_df[
        years_df['Year'] != years_df['FirstYear']
    ]

    years_df['CareerMove'] = years_df['FirstYearAtInstitution'] == years_df['Year']
    years_df['CareerMove'] = years_df['CareerMove'].apply(int)

    df = usfhn.gender.annotate_gender(df, explode_gender=True)

    df = df.merge(
        years_df,
        on=[
            'PersonId',
            'InstitutionId',
        ]
    )

    df = df[
        [
            'PersonId',
            'Gender',
            'Year',
            'CareerMove',
        ]
    ].drop_duplicates()

    df['EventCount'] = df.groupby([
        'Gender',
        'Year',
    ])['PersonId'].transform('nunique')

    df['MoveCount'] = df.groupby([
        'Gender',
        'Year',
    ])['CareerMove'].transform('sum')

    df = df[
        [
            'Gender',
            'Year',
            'MoveCount',
            'EventCount',
        ]
    ].drop_duplicates()

    df['MoveRisk'] = df['MoveCount'] / df['EventCount']

    df = df[
        [
            'Year',
            'Gender',
            'MoveRisk',
            'EventCount',
            'MoveCount',
        ]
    ].drop_duplicates()

    return df


def get_institution_move_risk():
    df = usfhn.datasets.get_dataset_df('data', 'careers')[
        [
            'PersonId',
            'InstitutionId',
            'Year',
        ]
    ].drop_duplicates()


    df['FirstYear'] = df.groupby('PersonId')['Year'].transform('min')
    df['LastYear'] = df.groupby('PersonId')['Year'].transform('max')

    df['FirstYearAtInstitution'] = df.groupby([
        'PersonId',
        'InstitutionId',
    ])['Year'].transform('min')

    df['LastYearAtInstitution'] = df.groupby([
        'PersonId',
        'InstitutionId',
    ])['Year'].transform('max')

    df['Leaves'] = (df['LastYearAtInstitution'] == df['Year']) & (df['Year'] != df['LastYear'])
    df['Leaves'] = df['Leaves'].apply(int)

    df['Joins'] = (df['FirstYearAtInstitution'] == df['Year']) & (df['Year'] != df['FirstYear'])
    df['Joins'] = df['Joins'].apply(int)

    df = df[
        [
            'Year',
            'PersonId',
            'InstitutionId',
            'FirstYear',
            'Joins',
            'LastYear',
            'Leaves',
        ]
    ].drop_duplicates()

    leaving_df = df.copy()

    # exclude last years from the leaving df
    leaving_df = leaving_df[
        leaving_df['Year'] != leaving_df['LastYear']
    ].drop(columns=['Year']).drop_duplicates()

    leaving_df['EventCount-Leaving'] = leaving_df.groupby('InstitutionId')['PersonId'].transform('nunique')
    leaving_df['MoveCount-Leaving'] = leaving_df.groupby('InstitutionId')['Leaves'].transform('sum')

    leaving_df = leaving_df[
        [
            'InstitutionId',
            'EventCount-Leaving',
            'MoveCount-Leaving'
        ]
    ].drop_duplicates()

    joining_df = df.copy()

    # exclude last years from the joining df
    joining_df = joining_df[
        joining_df['Year'] != joining_df['FirstYear']
    ].drop(columns=['Year']).drop_duplicates()

    joining_df['EventCount-Joining'] = joining_df.groupby('InstitutionId')['PersonId'].transform('nunique')
    joining_df['MoveCount-Joining'] = joining_df.groupby('InstitutionId')['Joins'].transform('sum')

    joining_df = joining_df[
        [
            'InstitutionId',
            'EventCount-Joining',
            'MoveCount-Joining',
        ]
    ].drop_duplicates()

    df = leaving_df.merge(
        joining_df,
        on=[
            'InstitutionId',
        ],
        how='outer',
    )

    df['JoiningRisk'] = df['MoveCount-Joining'] / df['EventCount-Joining']
    df['LeavingRisk'] = df['MoveCount-Leaving'] / df['EventCount-Leaving']
    df['JoiningRisk'] = df['JoiningRisk'].fillna(0)
    df['LeavingRisk'] = df['LeavingRisk'].fillna(0)

    return df

def get_mid_career_movers():
    import usfhn.stats
    df = usfhn.stats.runner.get('careers/df')

    df = df[
        df.groupby('PersonId')['CareerStep'].transform('max') > 1
    ]

    return list(df['PersonId'].unique())

def get_first_jobs_of_mid_career_movers():
    import usfhn.stats
    df = usfhn.stats.runner.get('careers/df')

    df = df[
        df.groupby('PersonId')['CareerStep'].transform('max') > 1
    ]

    return df[
        df['CareerStep'] == 1
    ][
        [
            'PersonId',
            'InstitutionId',
        ]
    ].drop_duplicates()

def get_last_jobs_of_mid_career_movers():
    import usfhn.stats
    df = usfhn.stats.runner.get('careers/df')

    df = df[
        df.groupby('PersonId')['CareerStep'].transform('max') > 1
    ]

    return df[
        df.groupby('PersonId')['CareerStep'].transform('max') == df['CareerStep']
    ][
        [
            'PersonId',
            'InstitutionId',
        ]
    ].drop_duplicates()


def get_taxonomy_hires_before_and_after_mcms():
    """
    when looking at how MCMs affect the hierarchy, we are going to do the following:
    - define "before" as: everyone in the taxonomy at move 1
    - define "after" as: everyone in the taxonomy at final move

    or:
    - define "before" as: MCMs in the taxonomy at move 1
    - define "after" as: MCMs in the taxonomy at final move
    """
    df = usfhn.stats.runner.get('careers/df')[
        [
            'PersonId',
            'InstitutionId',
            'CareerStep',
        ]
    ].drop_duplicates()

    taxonomies = usfhn.datasets.get_dataset_df('data', 'careers')[
        [
            'PersonId',
            'Taxonomy',
        ]
    ].drop_duplicates()

    df = df.merge(
        taxonomies,
        on='PersonId',
    )

    df = df[
        df['CareerStep'] > 0
    ]

    df['FirstStep'] = df.groupby('PersonId')['CareerStep'].transform('min')
    df['LastStep'] = df.groupby('PersonId')['CareerStep'].transform('max')

    first_jobs = df.copy()[
        df['CareerStep'] == df['FirstStep']
    ][
        [
            'PersonId',
            'InstitutionId',
            'Taxonomy',
        ]
    ].drop_duplicates()
    first_jobs['JobType'] = 'FirstJob'

    last_jobs = df.copy()[
        df['CareerStep'] == df['LastStep']
    ][
        [
            'PersonId',
            'InstitutionId',
            'Taxonomy',
        ]
    ].drop_duplicates()

    last_jobs['JobType'] = 'LastJob'

    df = pd.concat([first_jobs, last_jobs])
    df = usfhn.fieldwork.linearize_df_taxonomy_hierarchy(df)

    return df


def get_mcm_stats():
    import usfhn.stats
    df = usfhn.stats.runner.get('careers/taxonomy-hires-before-and-after-mcms')

    degrees = usfhn.datasets.get_dataset_df('data')[
        [
            'PersonId',
            'DegreeInstitutionId',
        ]
    ].drop_duplicates()

    df = df.merge(
        degrees,
        on='PersonId',
    )

    ranks = usfhn.stats.runner.get('ranks/df', rank_type='prestige')[
        [
            'InstitutionId',
            'TaxonomyLevel',
            'TaxonomyValue',
            'Percentile',
        ]
    ].drop_duplicates().rename(columns={
        'Percentile': 'Rank',
    })

    df = df.merge(
        ranks,
        on=[
            'InstitutionId',
            'TaxonomyLevel',
            'TaxonomyValue',
        ]
    ).merge(
        ranks.rename(columns={
            'Rank': 'DegreeRank',
            'InstitutionId': 'DegreeInstitutionId',
        }),
        on=[
            'DegreeInstitutionId',
            'TaxonomyLevel',
            'TaxonomyValue',
        ]
    )

    df['RankDifference'] = df['Rank'] - df['DegreeRank']

    df = df[
        [
            'PersonId',
            'RankDifference',
            'TaxonomyLevel',
            'TaxonomyValue',
            'JobType',
        ]
    ].drop_duplicates()

    stats_dfs = []
    groupby_cols = ['TaxonomyLevel', 'TaxonomyValue']

    stats_columns = ['Edges', 'MeanDistance', 'MovementEdges', 'MovementFraction', 'MeanMovementDistance']

    pre_df = usfhn.rank.calculate_hierarchy_stats(
        df[
            df['JobType'] == 'FirstJob'
        ].drop(columns=['JobType']),
        'RankDifference',
        groupby_cols=groupby_cols
    ).rename(columns={c: f"{c}-Pre" for c in stats_columns})

    post_df = usfhn.rank.calculate_hierarchy_stats(
        df[
            df['JobType'] == 'LastJob'
        ].drop(columns=['JobType']),
        'RankDifference',
        groupby_cols=groupby_cols
    ).rename(columns={c: f"{c}-Post" for c in stats_columns})

    df = pre_df.merge(
        post_df,
        on=[
            'TaxonomyLevel',
            'TaxonomyValue',
            'MovementType',
        ]
    )

    for column in stats_columns:
        df[f"{column}-Difference"] = df[f'{column}-Post'] - df[f'{column}-Pre']

    return df


def get_mcm_stats_comparison(z_threshold=1.96, p_threshold=.05):
    import usfhn.stats
    df = usfhn.stats.runner.get('careers/hierarchy-changes-from-mcms')

    significance_rows = []
    for i, row in df.iterrows():
        z_value, p_value = proportions_ztest(
            [row['MovementEdges-Pre'], row['MovementEdges-Post']],
            [row['Edges-Pre'], row['Edges-Post']],
        )

        reject_null = p_value < p_threshold
        significant = abs(z_value) > z_threshold


        significance_rows.append({
            'TaxonomyLevel': row['TaxonomyLevel'],
            'TaxonomyValue': row['TaxonomyValue'],
            'MovementType': row['MovementType'],
            'P': p_value,
            'SignificantDifference': reject_null and significant,
        })

    significance_df = pd.DataFrame(significance_rows)

    df = df.merge(
        significance_df,
        on=[
            'TaxonomyLevel',
            'TaxonomyValue',
            'MovementType',
        ]
    )

    return df


def get_career_lengths_df(by_year=False):
    """
    we're mostly talking about `by_year` == True, but when `by_year` == False,
    we're going to only include the last year
    """
    df = usfhn.datasets.get_dataset_df('closedness_data', by_year=by_year)

    df = usfhn.fieldwork.linearize_df_taxonomy_hierarchy(df)

    df['CareerYear'] = df['Year'] - df['DegreeYear']

    if by_year:
        df = df.drop(columns=['DegreeYear']).drop_duplicates()
    else:
        df['MaxPersonYear'] = df.groupby('PersonId')['Year'].transform('max')
        df = df[
            df['Year'] == df['MaxPersonYear']
        ]

        df = df.drop(columns=[
            'MaxPersonYear',
            'DegreeYear',
            'Year',
        ]).drop_duplicates()

    return df


def career_length_by_gender(by_year=False):
    """
    returns:
    - Gender
    - TaxonomyLevel
    - TaxonomyValue
    - Year (if by_year == True)
    - MeanCareerLength
    - MedianCareerLength
    """
    import usfhn.stats
    import usfhn.gender

    groupby_cols = [
        'TaxonomyLevel',
        'TaxonomyValue',
        'Gender',
    ]

    if by_year:
        df = usfhn.stats.runner.get('careers/length/by-year/df')
        groupby_cols.append('Year')
    else:
        df = usfhn.stats.runner.get('careers/length/df')

    df = usfhn.gender.annotate_gender(df)

    df['MeanCareerLength'] = df.groupby(groupby_cols)['CareerYear'].transform('mean')
    df['MedianCareerLength'] = df.groupby(groupby_cols)['CareerYear'].transform('median')

    df = df[
        groupby_cols + [
            'MeanCareerLength',
            'MedianCareerLength',
        ]
    ].drop_duplicates()

    return df


def career_length_by_taxonomy(by_year=False):
    """
    returns:
    - TaxonomyLevel
    - TaxonomyValue
    - Year (if by_year == True)
    - MeanCareerLength
    - MedianCareerLength
    """
    import usfhn.stats

    groupby_cols = [
        'TaxonomyLevel',
        'TaxonomyValue',
    ]

    if by_year:
        df = usfhn.stats.runner.get('careers/length/by-year/df')
        groupby_cols.append('Year')
    else:
        df = usfhn.stats.runner.get('careers/length/df')

    df['MeanCareerLength'] = df.groupby(groupby_cols)['CareerYear'].transform('mean')
    df['MedianCareerLength'] = df.groupby(groupby_cols)['CareerYear'].transform('median')

    df = df[
        groupby_cols + [
            'MeanCareerLength',
            'MedianCareerLength',
        ]
    ].drop_duplicates()

    return df


def get_gini_coefficients_by_career_length(by_year=False):
    import usfhn.stats

    groupby_cols = [
        'TaxonomyLevel',
        'TaxonomyValue',
        'CareerYear',
    ]

    if by_year:
        groupby_cols.append('Year')
        df = usfhn.stats.runner.get('careers/length/by-year/df')
    else:
        df = usfhn.stats.runner.get('careers/length/df')

    return usfhn.gini.get_gini_coefficients_for_df(df, groupby_cols)


def rank_vs_career_length_logit(rank_type='prestige'):
    import usfhn.datasets
    df = usfhn.datasets.get_dataset_df('closedness_data')[
        [
            'PersonId',
            'Taxonomy',
            'InstitutionId',
            'DegreeInstitutionId',
        ]
    ].drop_duplicates()

    df = usfhn.fieldwork.linearize_df_taxonomy_hierarchy(df)

    df = annotate_career_age(df)

    df['MaxCareerLength'] = df.groupby([
        'PersonId',
        'TaxonomyLevel',
        'TaxonomyValue',
    ])['CareerAge'].transform('max')

    df = df[
        df['CareerAge'] == df['MaxCareerLength']
    ]

    df = df[
        df['CareerAge'] <= 50
    ]

    df['CareerAgeFraction'] = df['CareerAge'] / 50

    ranks = usfhn.stats.runner.get('ranks/df', rank_type=rank_type)

    df = df.merge(
        ranks,
        on=[
            'TaxonomyLevel',
            'TaxonomyValue',
            'InstitutionId',
        ]
    )

    return hnelib.model.get_logits(
        df,
        endog='CareerAgeFraction',
        exog='Percentile',
        groupby_cols=[
            'TaxonomyLevel',
            'TaxonomyValue',
        ],
    )
