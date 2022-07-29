import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats.contingency import relative_risk


import hnelib.pandas

import usfhn.fieldwork
import usfhn.views
import usfhn.datasets
import usfhn.constants
import usfhn.stats_utils


def get_attritions_df():
    """
    returns:
    - PersonId
    - LastYear: year preceding attrition.
    - CareerLength: number of years you are present in dataset.
    - Attrition: boolean. True if you EVER attrited.
    - AttritionYear: boolean. True if you attrited in `Year`.
    - Year
    - TaxonomyLevel
    - TaxonomyValue
    - InstitutionId
    - DegreeInstitutionId
    """
    df = usfhn.datasets.get_dataset_df('closedness_data', by_year=True)

    df = df[
        [
            'PersonId',
            'Taxonomy',
            'Year',
            'InstitutionId',
            'DegreeInstitutionId',
        ]
    ].drop_duplicates()

    max_year = df['Year'].max()

    df['LastYear'] = df.groupby('PersonId')['Year'].transform('max')
    df['CareerLength'] = df.groupby('PersonId')['Year'].transform('nunique')

    df['Attrition'] = df['LastYear'] != max_year
    df['AttritionYear'] = df['Attrition'] & (df['LastYear'] == df['Year'])

    df = usfhn.fieldwork.linearize_df_taxonomy_hierarchy(df)

    df = df[
        df['TaxonomyLevel'].isin(['Academia', 'Umbrella', 'Field'])
    ]

    df = df[
        [
            'PersonId',
            'LastYear',
            'CareerLength',
            'Attrition',
            'AttritionYear',
            'Year',
            'TaxonomyLevel',
            'TaxonomyValue',
            'InstitutionId',
            'DegreeInstitutionId',
        ]
    ].drop_duplicates()

    return df


def get_attrition_risk_df(df, groupby_cols, attrition_col='Attrition'):
    """
    Expects df with columns:
    - PersonId
    - Attrition: boolean. did the person Attrite?
    - groupby_cols
    
    and then whatever else is in groupby_cols

    returns:
    - Events: # of possible attritions
    - AttritionEvents: # of attritions
    - AttritionRisk: AttritionEvents/Events
    - groupby_cols
    """
    df = df.copy()

    df['Events'] = df.groupby(groupby_cols)['PersonId'].transform('nunique')

    df = df[
        df[attrition_col]
    ].drop(columns=[attrition_col])

    df['AttritionEvents'] = df.groupby(groupby_cols)['PersonId'].transform('nunique')

    df = df[
        groupby_cols + [
            'Events',
            'AttritionEvents',
        ]
    ].drop_duplicates()

    df['AttritionRisk'] = df['AttritionEvents'] / df['Events']

    return df


def annotate_career_stage(df, attrition_col, year_threshold=usfhn.constants.CAREER_LENGTH_THRESHOLD):
    """
    Attritions are 'retirements' if they happen more than
    usfhn.constants.CAREER_LENGTH_THRESHOLD years after the person received
    their degree
    """
    columns = df.columns

    degree_years = usfhn.datasets.get_dataset_df('closedness_data', by_year=True)

    degree_years = degree_years[
        [
            'PersonId',
            'Year',
            'DegreeYear',
        ]
    ].drop_duplicates()

    degree_years = degree_years[
        degree_years['DegreeYear'].notna()
    ]

    degree_years['EligibleToRetire'] = (degree_years['DegreeYear'] + year_threshold) < degree_years['Year']
    degree_years['CareerStage'] = degree_years['EligibleToRetire'].apply({
        True: 'Late',
        False: 'Early',
    }.get)

    degree_years = degree_years.drop(columns=['DegreeYear'])

    df = df.merge(
        degree_years,
        on=[
            'PersonId',
            'Year',
        ]
    )

    df = df[
        list(columns) + ['CareerStage']
    ].drop_duplicates()

    return df


def get_attrition_df_by_criteria(
    by_year=False,
    by_career_stage=False,
    by_degree_year=False,
    by_non_us=False,
    df=None,
):
    import usfhn.stats
    if not df:
        df = usfhn.stats.runner.get('attrition/df')

    groupby_cols = []

    if by_year:
        attrition_col = 'AttritionYear'
        groupby_cols.append('Year')
    else:
        attrition_col = 'Attrition'

    if by_degree_year:
        groupby_cols.append('DegreeYear')

        if 'DegreeYear' not in df.columns:
            degree_year_df = usfhn.datasets.CURRENT_DATASET.closedness_data[
                [
                    'PersonId',
                    'DegreeYear'
                ]
            ].drop_duplicates()

            df = df.merge(
                degree_year_df,
                on='PersonId',
            )

    if by_career_stage:
        df = annotate_career_stage(df, attrition_col)
        groupby_cols.append('CareerStage')

    if by_non_us:
        import usfhn.non_us
        df = usfhn.non_us.annotate_non_us(df)
        groupby_cols.append('NonUS')

    return [
        df,
        groupby_cols,
        attrition_col,
    ]


def get_taxonomy_attrition_risk(by_year=False, by_career_stage=False, by_degree_year=False):
    (df, groupby_cols, attrition_col) = get_attrition_df_by_criteria(
        by_year=by_year,
        by_career_stage=by_career_stage,
        by_degree_year=by_degree_year,
    )

    groupby_cols += [
        'TaxonomyLevel',
        'TaxonomyValue',
    ]

    df = df[
        groupby_cols +
        [
            'PersonId',
            attrition_col,
        ]
    ].drop_duplicates()

    return get_attrition_risk_df(
        df,
        groupby_cols=groupby_cols,
        attrition_col=attrition_col,
    )


def get_rank_change_attrition_risk(
    by_year=False,
    by_career_stage=False,
    by_degree_year=False,
    rank_type='prestige',
):
    import usfhn.stats

    (df, groupby_cols, attrition_col) = get_attrition_df_by_criteria(
        by_year=by_year,
        by_career_stage=by_career_stage,
        by_degree_year=by_degree_year,
    )

    groupby_cols += [
        'TaxonomyLevel',
        'TaxonomyValue',
    ]

    df = df[
        groupby_cols +
        [
            'PersonId',
            attrition_col,
        ]
    ].drop_duplicates()

    if by_year:
        ranks_df_name = 'ranks/by-year/change'
    else:
        ranks_df_name = 'ranks/change'

    ranks = usfhn.stats.runner.get(ranks_df_name, rank_type=rank_type, by_year=by_year)

    rank_groupby_cols = [c for c in groupby_cols if c in list(ranks.columns)]

    ranks = ranks[
        rank_groupby_cols + [
            'RankDifference',
            'PersonId',
        ]
    ].drop_duplicates()

    df = df.merge(
        ranks,
        on=['PersonId'] + rank_groupby_cols,
    )
    df['RankDifference'] *= -1

    up = df[df['RankDifference'] > 0].copy()
    up['MovementType'] = 'Upward'

    self_hires = df[df['RankDifference'] == 0].copy()
    self_hires['MovementType'] = 'Self-Hire'

    down = df[df['RankDifference'] < 0].copy()
    down['MovementType'] = 'Downward'

    df = pd.concat([up, self_hires, down])

    df = df[
        groupby_cols + [
            'PersonId',
            attrition_col,
            'MovementType',
        ]
    ].drop_duplicates()

    groupby_cols.append('MovementType')

    return get_attrition_risk_df(
        df,
        groupby_cols=groupby_cols,
        attrition_col=attrition_col,
    )


def get_gender_attrition_risk(by_year=False, by_career_stage=False, by_degree_year=False, df=None):
    from usfhn.stats import add_groupby_annotations_to_df

    (df, groupby_cols, attrition_col) = get_attrition_df_by_criteria(
        by_year=by_year,
        by_career_stage=by_career_stage,
        by_degree_year=by_degree_year,
        df=df,
    )

    groupby_cols += [
        'TaxonomyLevel',
        'TaxonomyValue',
    ]

    df = df[
        groupby_cols +
        [
            'PersonId',
            attrition_col,
        ]
    ].drop_duplicates()

    df, extra_groupby_cols = add_groupby_annotations_to_df(df, by_gender=True)
    groupby_cols += extra_groupby_cols

    return get_attrition_risk_df(
        df,
        groupby_cols=groupby_cols,
        attrition_col=attrition_col,
    )


def get_gender_balance_given_attrition_and_balanced_hiring():
    """
    returns df:
    - TaxonomyLevel
    - TaxonomyValue
    - CareerYear
    - MaleEvents
    - MaleAttritionEvents
    - MaleAttritionRisk
    - MaleRetention
    - FemaleEvents
    - FemaleAttritionEvents
    - FemaleAttritionRisk
    - FemaleRetention
    - TotalRetained
    - FractionFemaleRetained
    """
    import usfhn.stats
    df = usfhn.stats.runner.get('attrition/risk/by-time/by-degree-year/gender')

    groupby_cols = [
        'TaxonomyLevel',
        'TaxonomyValue',
        'Gender',
        'CareerYear',
    ]

    df['CareerYear'] = df['Year'] - df['DegreeYear']
    df['CareerEvents'] = df.groupby(groupby_cols)['Events'].transform('sum')
    df['CareerAttritionEvents'] = df.groupby(groupby_cols)['AttritionEvents'].transform('sum')

    df = df[
        groupby_cols + [
            'CareerEvents',
            'CareerAttritionEvents',
        ]
    ].drop_duplicates().rename(columns={
        'CareerEvents': 'Events',
        'CareerAttritionEvents': 'AttritionEvents',
    })

    df['AttritionRisk'] = df['AttritionEvents'] / df['Events']

    df['RetentionRate'] = 1 - df['AttritionRisk']

    male_df = df.copy()[
        df['Gender'] == 'Male'
    ].drop(columns=['Gender']).rename(columns={
        'Events': 'MaleEvents',
        'AttritionEvents': 'MaleAttritionEvents',
        'AttritionRisk': 'MaleAttritionRisk',
        'RetentionRate': 'MaleRetentionRate',
    })

    female_df = df.copy()[
        df['Gender'] == 'Female'
    ].drop(columns=['Gender']).rename(columns={
        'Events': 'FemaleEvents',
        'AttritionEvents': 'FemaleAttritionEvents',
        'AttritionRisk': 'FemaleAttritionRisk',
        'RetentionRate': 'FemaleRetentionRate',
    })

    df = male_df.merge(
        female_df,
        on=[
            'TaxonomyLevel',
            'TaxonomyValue',
            'CareerYear',
        ]
    )

    balanced_retention_rows = []
    for (level, value), rows in df.groupby(['TaxonomyLevel', 'TaxonomyValue']):
        rows = rows.sort_values(by=['CareerYear'])

        cols = {
            'TaxonomyLevel': level,
            'TaxonomyValue': value,
        }

        new_rows = []
        iterator = zip(rows['CareerYear'], rows['MaleRetentionRate'], rows['FemaleRetentionRate'])
        for i, (career_year, male_retention_rate, female_retention_rate) in enumerate(iterator):
            previous_male_fraction = new_rows[i - 1]['MaleFractionRetained'] if i else 1
            previous_female_fraction = new_rows[i - 1]['FemaleFractionRetained'] if i else 1

            new_rows.append({
                'TaxonomyLevel': level,
                'TaxonomyValue': value,
                'CareerYear': career_year,
                'MaleFractionRetained': previous_male_fraction * male_retention_rate,
                'FemaleFractionRetained': previous_female_fraction * female_retention_rate,
            })

        balanced_retention_rows += new_rows

    df = df.merge(
        pd.DataFrame(balanced_retention_rows),
        on=[
            'TaxonomyLevel',
            'TaxonomyValue',
            'CareerYear',
        ]
    )

    df['TotalRetained'] = df['MaleFractionRetained'] + df['FemaleFractionRetained']
    df['FractionFemaleRetained'] = df['FemaleFractionRetained'] / df['TotalRetained']

    return df


def get_self_hire_attrition_risk(by_year=False, by_career_stage=False, by_degree_year=False):
    (df, groupby_cols, attrition_col) = get_attrition_df_by_criteria(
        by_year=by_year,
        by_career_stage=by_career_stage,
        by_degree_year=by_degree_year,
    )
    df['SelfHire'] = df['InstitutionId'] == df['DegreeInstitutionId']

    groupby_cols += [
        'TaxonomyLevel',
        'TaxonomyValue',
        'SelfHire'
    ]

    df = df[
        groupby_cols +
        [
            'PersonId',
            attrition_col,
        ]
    ].drop_duplicates()

    return get_attrition_risk_df(
        df,
        groupby_cols=groupby_cols,
        attrition_col=attrition_col,
    )

def get_non_us_attrition_risk(by_year=False, by_career_stage=False, by_degree_year=False):
    import usfhn.institutions

    (df, groupby_cols, attrition_col) = get_attrition_df_by_criteria(
        by_year=by_year,
        by_career_stage=by_career_stage,
        by_degree_year=by_degree_year,
    )

    df = usfhn.institutions.annotate_us(df)

    groupby_cols += [
        'TaxonomyLevel',
        'TaxonomyValue',
        'US',
    ]

    df = df[
        groupby_cols +
        [
            'PersonId',
            attrition_col,
        ]
    ].drop_duplicates()

    return get_attrition_risk_df(
        df,
        groupby_cols=groupby_cols,
        attrition_col=attrition_col,
    )

def get_non_us_by_is_english_attrition_risk(by_year=False, by_career_stage=False, by_degree_year=False):
    import usfhn.institutions

    (df, groupby_cols, attrition_col) = get_attrition_df_by_criteria(
        by_year=by_year,
        by_career_stage=by_career_stage,
        by_degree_year=by_degree_year,
    )

    df = usfhn.institutions.annotate_us(df)
    df = usfhn.institutions.annotate_highly_productive_non_us_countries(df)

    groupby_cols += [
        'TaxonomyLevel',
        'TaxonomyValue',
        'US',
        'IsHighlyProductiveNonUSCountry',
    ]

    df = df[
        groupby_cols +
        [
            'PersonId',
            attrition_col,
        ]
    ].drop_duplicates()

    return get_attrition_risk_df(
        df,
        groupby_cols=groupby_cols,
        attrition_col=attrition_col,
    )


def compute_attrition_risk_significance(
    df,
    attrition_events_col_1,
    events_col_1,
    attrition_events_col_2,
    events_col_2,
    alpha=.05,
):
    from statsmodels.stats.multitest import multipletests
    from scipy.stats.contingency import chi2_contingency

    df = df.copy()[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            events_col_1,
            events_col_2,
            attrition_events_col_1,
            attrition_events_col_2,
        ]
    ].drop_duplicates()

    p_rows = []
    for i, row in df.iterrows():
        g1_attrition_events = row[attrition_events_col_1]
        g1_non_attrition_events = row[events_col_1] - g1_attrition_events

        g2_attrition_events = row[attrition_events_col_2]
        g2_non_attrition_events = row[events_col_2] - g2_attrition_events

        _, p, _, _ = chi2_contingency(
            np.array([
                [g1_attrition_events, g2_attrition_events],
                [g1_non_attrition_events, g2_non_attrition_events],
            ]),
        )

        p_rows.append(p)

    df['P'] = p_rows

    corrected_df = usfhn.stats_utils.correct_multiple_hypotheses(df, alpha=alpha)

    significance_rows = []
    for i, row in corrected_df.iterrows():
        result = relative_risk(
            row[attrition_events_col_1],
            row[events_col_1],
            row[attrition_events_col_2],
            row[events_col_2],
        )

        confidence_interval = result.confidence_interval(1 - alpha)

        if confidence_interval.low < 1 < confidence_interval.high:
            conf_interval_significant = False
            significant = False
        else:
            conf_interval_significant = True
            significant = True

        p_value_significant = row['PCorrected'] < alpha

        significant = conf_interval_significant and p_value_significant

        significance_rows.append({
            'TaxonomyLevel': row['TaxonomyLevel'],
            'TaxonomyValue': row['TaxonomyValue'],
            'Significant': significant,
            'P': row['P'],
            'PCorrected': row['PCorrected'],
        })

    return pd.DataFrame(significance_rows)


def get_institution_attrition_risk(institution_column, by_year=False, by_career_stage=False, by_degree_year=False):
    (df, groupby_cols, attrition_col) = get_attrition_df_by_criteria(
        by_year=by_year,
        by_career_stage=by_career_stage,
        by_degree_year=by_degree_year,
    )
    groupby_cols += [
        'TaxonomyLevel',
        'TaxonomyValue',
        institution_column,
    ]

    df = df[
        groupby_cols +
        [
            'PersonId',
            attrition_col,
        ]
    ].drop_duplicates()

    return get_attrition_risk_df(
        df,
        groupby_cols=groupby_cols,
        attrition_col=attrition_col,
    )


def institutions_with_ranks(rank_type='production'):
    """
    rank_types:
    - production
    - doctoral-institution
    - employing-institution

    """
    import usfhn.stats

    ranks = usfhn.stats.runner.get(
        'ranks/df',
        rank_type='production' if rank_type == 'production' else 'prestige',
    )

    if rank_type == 'doctoral-institution':
        institution_column = 'DegreeInstitutionId'
        ranks = ranks.rename(columns={
            'InstitutionId': 'DegreeInstitutionId',
        })
    else:
        institution_column = 'InstitutionId'

    df = usfhn.stats.runner.get('attrition/risk/institution', institution_column=institution_column)

    ranks = ranks[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            institution_column,
            'Percentile',
        ]
    ].drop_duplicates()

    df = df.merge(
        ranks,
        on=[
            'TaxonomyLevel',
            'TaxonomyValue',
            institution_column,
        ]
    )

    return df


def attritions_df_with_ranks(rank_type='production', academia_rank_only=False):
    """
    rank_types:
    - production
    - doctoral-institution
    - employing-institution
    """
    import usfhn.stats

    ranks = usfhn.stats.runner.get(
        'ranks/df',
        rank_type='production' if rank_type == 'production' else 'prestige',
    )

    if rank_type == 'doctoral-institution':
        institution_column = 'DegreeInstitutionId'
        ranks = ranks.rename(columns={
            'InstitutionId': 'DegreeInstitutionId',
        })
    else:
        institution_column = 'InstitutionId'

    df = usfhn.stats.runner.get('attrition/df')

    ranks = ranks[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            institution_column,
            'Percentile',
        ]
    ].drop_duplicates()

    if academia_rank_only:
        ranks = usfhn.views.filter_by_taxonomy(ranks, 'Academia')[
            [
                institution_column,
                'Percentile',
            ]
        ].drop_duplicates()

        df = df.merge(
            ranks,
            on=institution_column,
        )
    else:
        df = df.merge(
            ranks,
            on=[
                'TaxonomyLevel',
                'TaxonomyValue',
                institution_column,
            ]
        )

    return df


def rank_vs_attrition_risk_logit(
    rank_type='production',
    p_threshold=.05,
    by_gender=False,
    by_self_hire=False,
    academia_rank_only=False,
):
    import usfhn.stats

    if academia_rank_only:
        df = usfhn.stats.runner.get('attrition/by-rank/academia-ranks/df', rank_type=rank_type)
    else:
        df = usfhn.stats.runner.get('attrition/by-rank/df', rank_type=rank_type)

    groupby_cols = [
        'TaxonomyLevel',
        'TaxonomyValue',
    ]

    df, extra_groupby_cols = usfhn.stats.add_groupby_annotations_to_df(
        df,
        by_gender=by_gender,
        by_self_hire=by_self_hire,
    )

    groupby_cols += extra_groupby_cols

    df['AttritionYear'] = df['AttritionYear'].apply(int)

    # don't drop duplicates, because we want them by year and PersonId
    logit_df = df[
        groupby_cols + [
            'AttritionYear',
            'Percentile',
        ]
    ]

    df = hnelib.model.get_logits(
        df,
        endog='AttritionYear',
        exog='Percentile',
        groupby_cols=[
            'TaxonomyLevel',
            'TaxonomyValue',
        ],
    )

    df = usfhn.stats_utils.correct_multiple_hypotheses(df, p_col='Percentile-P', corrected_p_col='Percentile-P')

    return df


def mean_age_of_attrition():
    import usfhn.stats
    df = usfhn.stats.runner.get('attrition/df')[
        [
            'PersonId',
            'Attrition',
            'TaxonomyLevel',
            'TaxonomyValue',
        ]
    ].drop_duplicates()

    df = df[
        df['Attrition']
    ]

    df, _ = usfhn.stats.add_groupby_annotations_to_df(
        df,
        by_gender=True,
        explode_gender=True,
        by_career_age=True,
    )

    df['MeanCareerAgeOfAttrition'] = df.groupby('Gender')['CareerAge'].transform('mean')

    df = df[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'Gender',
            'MeanCareerAgeOfAttrition',
        ]
    ].drop_duplicates()

    return df

def self_hire_to_non_self_hire_risk_ratio():
    import usfhn.stats

    df = usfhn.stats.runner.get('attrition/risk/self-hires')

    print(df.head())

    df = hnelib.pandas.aggregate_df_over_column(
        df,
        agg_col='SelfHire',
        join_cols=[
            'TaxonomyLevel',
            'TaxonomyValue',
        ],
        value_cols=['AttritionRisk'],
        agg_value_to_label={
            True: 'SelfHire',
            False: 'NonSelfHire',
        }
    )

    df['RiskRatio'] = df['SelfHireAttritionRisk'] / df['NonSelfHireAttritionRisk']

    df = df[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'RiskRatio',
        ]
    ].drop_duplicates()

    return df
