from functools import lru_cache
import json
import time
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from statsmodels.stats.proportion import proportions_ztest

import hnelib.utils
import hnelib.pandas

import usfhn.constants
import usfhn.utils
import usfhn.datasets
import usfhn.views
import usfhn.closedness
import usfhn.self_hiring
import usfhn.null_models
import usfhn.steeples
import usfhn.fieldwork
import usfhn.plot_utils
import usfhn.pool_reduction
import usfhn.gender
import usfhn.stats
import usfhn.institutions
import usfhn.careers
import usfhn.stats_utils

def get_number_or_percent(get, numerator, denominator, round_to=0):
    if get == 'number':
        return numerator
    elif get == 'percent':
        return hnelib.utils.fraction_to_percent(numerator / denominator, round_to=round_to)

################################################################################
# general
################################################################################
def fields_in_engineering():
    df = usfhn.views.get_taxonomization()
    df = df[
        df['Umbrella'] == 'Engineering'
    ]
    return int(df['Field'].nunique())

def fields_in_level_and_value(level='Academia', value='Academia'):
    df = usfhn.views.get_taxonomization()
    df = df[
        (df[level] == value)
        &
        (df['Field'].notnull())
    ]

    return len(df)

def non_arts_fields_in_humanities():
    df = usfhn.views.get_taxonomization()
    df = df[
        (df['Umbrella'] == 'Humanities')
        &
        (df['Field'].notnull())
        &
        (~df['Field'].isin(usfhn.constants.ARTS_HUMANITIES_FIELDS))
    ]

    return len(df)


def gender_stats(gender='Unknown', get='number'):
    df = usfhn.datasets.get_dataset_df('closedness_data')
    df = df[
        [
            'PersonId',
            'Gender',
        ]
    ].drop_duplicates()

    total = df['PersonId'].nunique()

    if gender == 'Known':
        df = df[
            df['Gender'] != 'Unknown'
        ]
    else:
        df = df[
            df['Gender'] == gender
        ]

    of_gender = df['PersonId'].nunique()

    if get == 'number':
        return of_gender
    elif get == 'percent':
        return hnelib.utils.fraction_to_percent(of_gender / total)

def multi_taxonomy_faculty(get='percent'):
    df = usfhn.datasets.get_dataset_df('closedness_data')
    df = usfhn.fieldwork.linearize_df_taxonomy_hierarchy(df)[
        [
            'PersonId',
            'TaxonomyLevel',
            'TaxonomyValue',
        ]
    ].drop_duplicates()

    denominator = df['PersonId'].nunique()

    df['TaxonomyCount'] = df.groupby(['PersonId', 'TaxonomyLevel'])['TaxonomyValue'].transform('nunique')

    df = df[
        df['TaxonomyCount'] > 1
    ]

    numerator = df['PersonId'].nunique()

    return get_number_or_percent(get, numerator, denominator)

def get_n_ranked_people(gendered=False):
    df = usfhn.stats.runner.get('ranks/change', rank_type='prestige')

    if gendered:
        df = usfhn.gender.annotate_gender(df, explode_gender=True)
        df = df[
            df['Gender'].isin(['Male', 'Female'])
        ]

    return df['PersonId'].nunique()


################################################################################
# production
################################################################################
def eighty_percent_of_faculty():
    df = usfhn.stats.runner.get('basics/lorenz')
    df = usfhn.views.filter_by_taxonomy(df, 'Academia')
    df = df[
        df['Y'] > .8
    ].sort_values(by='X')

    return hnelib.utils.fraction_to_percent(df.iloc[0]['X'], 1)


def percent_produced_by_top_five_institutions():
    df = usfhn.stats.runner.get('basics/lorenz')
    df = usfhn.views.filter_by_taxonomy(df, 'Academia').sort_values(by='X')

    return hnelib.utils.fraction_to_percent(df.iloc[5]['Y'], 1)

def top_five_most_producing_institutions(level='Academia', value=None):
    df = usfhn.stats.runner.get('basics/production')
    df = usfhn.views.filter_by_taxonomy(df, level=level, value=value)

    df = df[
        [
            'DegreeInstitutionId',
            'ProductionPercent',
        ]
    ].drop_duplicates()

    df = df.sort_values(by='ProductionPercent', ascending=False)
    df = usfhn.institutions.annotate_institution_name(df, id_col='DegreeInstitutionId')

    df = df[
        [
            'InstitutionName',
            'ProductionPercent',
        ]
    ].drop_duplicates()

    institutions = list(df['InstitutionName'])[:5]

    string = ", ".join(institutions[:-1]) + ", and " + institutions[-1]
    return string


def n_institutions_producing_twenty_percent_of_faculty(twenty_percent=0):
    df = usfhn.stats.runner.get('basics/lorenz')
    df = usfhn.views.filter_by_taxonomy(df, level='Academia', value='Academia')

    lower_bound = twenty_percent * .2
    upper_bound = (twenty_percent + 1) * .2

    df = df[
        (df['Y'] > lower_bound)
        &
        (df['Y'] <= upper_bound)
    ]

    value = len(df)

    if upper_bound == 1:
        value -= 1

    return value


def range_of_percents_of_institutions_producing_eighty_percent_of_faculty_at_domain_level():
    df = usfhn.stats.runner.get('basics/lorenz')
    df = usfhn.views.filter_by_taxonomy(df, 'Umbrella')
    df = df[
        df['Y'] > .8
    ]

    percents_of_institutions_producing_eighty_percent = []
    for umbrella, rows in df.groupby('TaxonomyValue'):
        rows = rows.sort_values(by='X')
        percents_of_institutions_producing_eighty_percent.append(rows.iloc[0]['X'])

    min_percent = hnelib.utils.fraction_to_percent(min(percents_of_institutions_producing_eighty_percent))
    max_percent = hnelib.utils.fraction_to_percent(max(percents_of_institutions_producing_eighty_percent))
    return f"{min_percent}-{max_percent}"


def correlation_between_institution_size_and_production():
    df = usfhn.stats.runner.get('basics/faculty-hiring-network')
    df = usfhn.views.filter_by_taxonomy(df, 'Academia')

    employment = df.copy()[
        [
            'InstitutionId',
            'InDegree',
        ]
    ].drop_duplicates()

    production = df.copy()[
        [
            'DegreeInstitutionId',
            'OutDegree',
        ]
    ].rename(columns={
        'DegreeInstitutionId': 'InstitutionId'
    }).drop_duplicates()

    df = employment.merge(production, on='InstitutionId')

    corr, p = pearsonr(df['InDegree'], df['OutDegree'])

    return {
        'correlation': round(corr, 2),
        'p': round(p, 2),
    }

def gini_at_level_and_value(level, value):
    df = usfhn.stats.runner.get('ginis/df')
    df = usfhn.views.filter_by_taxonomy(df, level, value)

    return round(df.iloc[0]['GiniCoefficient'], 2)

def gini_stats_engineering_fields():
    df = usfhn.stats.runner.get('ginis/df')
    fields = df.copy()[
        (df['TaxonomyLevel'] == 'Field')
    ]
    fields = usfhn.views.annotate_umbrella(fields, 'Field')
    fields = fields[
        fields['Umbrella'] == 'Engineering'
    ]

    return {
        'min': round(min(fields['GiniCoefficient']), 2),
        'max': round(max(fields['GiniCoefficient']), 2),
    }

def number_of_fields_less_unequal_than_their_umbrella(get='number'):
    df = usfhn.stats.runner.get('ginis/df')
    umbrellas = df.copy()[
        (df['TaxonomyLevel'] == 'Umbrella')
    ].rename(columns={
        'TaxonomyValue': 'Umbrella',
        'GiniCoefficient': 'UmbrellaGiniCoefficient',
    })[
        [
            'Umbrella',
            'UmbrellaGiniCoefficient',
        ]
    ].drop_duplicates()

    fields = df.copy()[
        (df['TaxonomyLevel'] == 'Field')
    ][
        [
            'TaxonomyValue',
            'GiniCoefficient',
        ]
    ].drop_duplicates()

    fields = usfhn.views.annotate_umbrella(fields, 'Field')
    df = fields.merge(umbrellas, on='Umbrella')

    df = df[
        df['GiniCoefficient'] < df['UmbrellaGiniCoefficient']
    ]

    if get == 'number':
        value = len(df)
    else:
        value = hnelib.utils.fraction_to_percent(
            len(df) / len(usfhn.views.get_fields())
        )

    return value


def number_of_umbrellas_less_unequal_than_academia(get='number'):
    df = usfhn.stats.runner.get('ginis/df')

    academia_gini = df.copy()[
        (df['TaxonomyLevel'] == 'Academia')
    ][
        [
            'TaxonomyValue',
            'GiniCoefficient',
        ]
    ].drop_duplicates().iloc[0]['GiniCoefficient']

    df = df[
        (df['TaxonomyLevel'] == 'Umbrella')
    ][
        [
            'TaxonomyValue',
            'GiniCoefficient',
        ]
    ].drop_duplicates()

    df = df[
        df['GiniCoefficient'] < academia_gini
    ]

    if get == 'number':
        value = len(df)
    else:
        value = hnelib.utils.fraction_to_percent(
            len(df) / len(usfhn.views.get_umbrellas()),
            round_to=1,
        )

    return value


def number_of_taxonomies_where_new_hire_gini_less_than_existing(level=None, get='number'):
    df = usfhn.stats.runner.get('ginis/by-new-hire/df')
    df = hnelib.pandas.aggregate_df_over_column(
        df,
        agg_col='NewHire',
        join_cols=['TaxonomyLevel', 'TaxonomyValue'],
        value_cols=['GiniCoefficient'],
        agg_value_to_label={
            True: 'NewHire',
            False: 'ExistingHire',
        },
    )

    if level:
        df = usfhn.views.filter_by_taxonomy(df, level)

    n_rows_start = len(df)

    df = df[
        df['NewHireGiniCoefficient'] < df['ExistingHireGiniCoefficient']
    ]

    n_rows = len(df)

    if get == 'number':
        return n_rows
    elif get == 'percent':
        return hnelib.utils.fraction_to_percent(n_rows / n_rows_start)


def get_gini_ols_models_stat(stat='significant', get='number', which=min):
    df = usfhn.stats.runner.get('ginis/slopes')

    n_rows_total = len(df)

    if stat == 'total':
        return n_rows_total

    df = df[
        df['Significant']
    ]

    n_rows_significant = len(df)

    if stat == 'significant':
        return get_number_or_percent(get, n_rows_significant, n_rows_total)

    if stat == 'downward':
        df = df[
            df['Slope'] < 0
        ]

        n_rows_downward = len(df)

        if get == 'range':
            slopes = [abs(s) for s in df['Slope']]
            min_slope = round(min(slopes), 4)
            max_slope = round(max(slopes), 4)

            return f"-{min_slope} and -{max_slope}"
        else:
            return get_number_or_percent(get, n_rows_downward, n_rows_significant)

    if stat == 'years-to-explain-differences':
        df = usfhn.stats.runner.get('ginis/stasis')
        df = df[
            df['YearsToProduceChange'] > 0
        ]


        if get == 'range':
            min_years = round(min(df['YearsToProduceChange']))
            max_years = round(max(df['YearsToProduceChange']))

            return f"{min_years} and {max_years}"
        elif get == 'number':
            return round(which(df['YearsToProduceChange']))


def get_gender_ols_models_stat(
    stat='significant',
    get='number',
    level=None,
    value=None,
    which=max,
    by_new_hire=False,
    significant=True,
):
    if by_new_hire:
        df = usfhn.stats.runner.get('gender/by-new-hire/slopes')
        df = df[
            df['NewHire']
        ]
    else:
        df = usfhn.stats.runner.get('gender/slopes')

    df = usfhn.views.filter_by_taxonomy(df, level)

    n_rows_total = len(df)

    if significant:
        df = df[
            df['Significant']
        ]
    else:
        df = df[
            ~df['Significant']
        ]

    n_rows_significant = len(df)

    if stat == 'total' and get == 'number':
        return n_rows_significant

    if stat == 'significant':
        return get_number_or_percent(get, n_rows_significant, n_rows_total)

    if stat == 'slopes':
        if get == 'range':
            slopes = [abs(s) for s in df['Slope']]
            min_slope = round(min(slopes), 4)
            max_slope = round(max(slopes), 4)

            return f"-{min_slope} and -{max_slope}"
        else:
            return get_number_or_percent(get, n_rows_downward, n_rows_significant)

    if stat == 'downward':
        df = df[
            df['Slope'] < 0
        ]

        n_rows_downward = len(df)

        if get == 'range':
            slopes = [abs(s) for s in df['Slope']]
            min_slope = round(min(slopes), 4)
            max_slope = round(max(slopes), 4)

            return f"-{min_slope} and -{max_slope}"
        else:
            return get_number_or_percent(get, n_rows_downward, n_rows_significant)

    if stat == 'years-to-explain-differences':
        df = usfhn.stats.runner.get('gender/stasis')

        df = usfhn.views.filter_by_taxonomy(df, level, value)

        df = df[
            df['YearsToProduceChange'] > 0
        ]

        if get == 'range':
            min_years = round(min(df['YearsToProduceChange']))
            max_years = round(max(df['YearsToProduceChange']))

            return f"{min_years}-{max_years}"

        row = df[
            which(df['YearsToProduceChange']) == df['YearsToProduceChange']
        ].iloc[0]

        if get == 'string':
            return usfhn.plot_utils.main_text_taxonomy_string(row['TaxonomyValue'])
        else:
            return round(row['YearsToProduceChange'])


def get_production_to_attrition_significant_stat(
    level='Field',
    get='number',
    significant=True,
    academia_rank_only=False,
):
    if academia_rank_only:
        dataframe_name = 'attrition/by-rank/academia-ranks/institution/logits'
    else:
        dataframe_name = 'attrition/by-rank/institution/logits'

    df = usfhn.stats.runner.get(dataframe_name, rank_type='production')

    df['Significant'] = df['Percentile-P'] <= .05

    if level:
        df = usfhn.views.filter_by_taxonomy(df, level)

    n_rows_start = len(df)

    df = df[
        df['Significant'] == significant
    ]

    n_rows = len(df)

    if get == 'number':
        return n_rows
    elif get == 'percent':
        return hnelib.utils.fraction_to_percent(n_rows / n_rows_start)
    elif get == 'max-p':
        return round(max(df['P']) , 2)


def years_to_go_from_existing_hire_gini_to_new_hire_gini(which=min):
    df = usfhn.stats.runner.get('ginis/stasis')

    row = df[
        which(df['YearsToProduceChange']) == df['YearsToProduceChange']
    ].iloc[0]

    return round(row['YearsToProduceChange'])


def fields_production_and_employment_distribution_stat(
    level=None,
    exclude_self_hires=False,
    get='number',
    different=False,
):
    df = usfhn.measurements.ks_test_employment_vs_production()
    df = usfhn.views.filter_by_taxonomy(df, level=level)

    df = df[
        df['ExcludeSelfHires'] == exclude_self_hires
    ]

    denominator = len(df)

    if different:
        df = df[
            df['Significant']
        ]
    else:
        df = df[
            ~df['Significant']
        ]

    numerator = len(df)

    return get_number_or_percent(get, numerator, denominator)

################################################################################
# closedness
################################################################################
def percent_of_faculty_who_hold_a_phd():
    row = usfhn.closedness.get_closedness_across_levels_at_level('Academia')
    return hnelib.utils.fraction_to_percent(row['PhD'] / row['FacultyCount'], 1)


def percent_faculty_without_a_phd(level='Umbrella', get='percent', which=max):
    df = usfhn.stats.runner.get('closedness/df')
    df = usfhn.views.filter_by_taxonomy(df, level)

    row = df[
        which(df['NonPhDFraction']) == df['NonPhDFraction']
    ].iloc[0]

    if get == 'percent':
        return hnelib.utils.fraction_to_percent(row['NonPhDFraction'])
    elif get == 'string':
        return usfhn.plot_utils.main_text_taxonomy_string(row['TaxonomyValue'])


def faculty_from_outside_the_us(get='percent'):
    df = usfhn.stats.runner.get('non-us/df')
    row = usfhn.views.filter_by_taxonomy(df, 'Academia').iloc[0]
    value = row['NonUSFraction']

    if get == 'percent':
        return hnelib.utils.fraction_to_percent(value)
    
    return value


def percent_of_in_field_faculty_at_level(level):
    row = usfhn.closedness.get_closedness_across_levels_at_level(level)
    return hnelib.utils.fraction_to_percent(row['USPhDInField'] / row['FacultyCount'])


def percent_of_out_of_field_faculty_at_level(level):
    row = usfhn.closedness.get_closedness_across_levels_at_level(level)
    return hnelib.utils.fraction_to_percent(row['USPhDOutOfField'] / row['FacultyCount'])


def in_field_to_out_of_field_odds(level):
    row = usfhn.closedness.get_closedness_across_levels_at_level(level)

    odds = row['USPhDInField'] / row['USPhDOutOfField']
    return round(odds, 1)


def percent_non_us_at_level_and_value(level, value):
    row = usfhn.closedness.get_closedness_at_level_and_value(level, value)
    return hnelib.utils.fraction_to_percent(row['NonUSPhD'] / row['FacultyCount'])


def max_percent_non_us_in_non_arts_humanities_fields():
    df = usfhn.views.get_taxonomization()
    df = df[
        (df['Umbrella'] == 'Humanities')
        &
        (df['Field'].notna())
        &
        (~df['Field'].isin(usfhn.constants.ARTS_HUMANITIES_FIELDS))
    ]

    max_fraction_out_of_field = 0
    for field in list(df['Field']):
        row = usfhn.closedness.get_closedness_at_level_and_value('Field', field)
        fraction_out_of_field = row['NonPhD'] / row['FacultyCount']
        max_fraction_out_of_field = max(max_fraction_out_of_field, fraction_out_of_field)

    return hnelib.utils.fraction_to_percent(max_fraction_out_of_field)


def percent_out_of_field_at_level_and_value(level, value):
    row = usfhn.closedness.get_closedness_at_level_and_value(level, value)
    return hnelib.utils.fraction_to_percent(row['USPhDOutOfField'] / row['FacultyCount'])


def people_out_of_us_sample(get='percent'):
    df = usfhn.closedness.get_closednesses()
    row = usfhn.views.filter_by_taxonomy(df, 'Academia', 'Academia').iloc[0]
    if get == 'percent':
        return hnelib.utils.fraction_to_percent(
            row['USPhDOutOfField'] / (row['USPhDOutOfField'] + row['USPhDInField'])
        )
    else:
        return row['USPhDOutOfField']

def institutions_out_of_sample(get='number', to_sample='out'):
    """
    what percentage of institutions were in of sample?
    what percentage of employing institutions were out of sample?
    how many employing institutions were out of sample?
    what percentage of producing institutions lacked employment records?
    how many employing institutions lacked employment records?
    """
    df = usfhn.datasets.get_dataset_df('data')
    employing = set(df['InstitutionId'].unique())
    producing = set(df['DegreeInstitutionId'].unique())

    institutions = employing | producing

    in_sample = employing
    out_of_sample = institutions - employing

    employing_not_producing = employing - producing
    producing_not_employing = producing - employing

    if to_sample == 'employing':
        sample = employing
    elif to_sample == 'out':
        sample = out_of_sample
    elif to_sample == 'producing not employing':
        sample = producing_not_employing
    elif to_sample == 'employing not producing':
        sample = employing_not_producing

    sample = len(sample)
    total = len(institutions)

    if get == 'number':
        return sample
    elif get == 'percent':
        return hnelib.utils.fraction_to_percent(sample / total)


def highly_productive_attrition_risk_stat(level='Field', get='number', highly_productive=False):
    df = usfhn.stats.runner.get('attrition/risk/non-us-by-is-english')[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'US',
            'IsHighlyProductiveNonUSCountry',
            'AttritionRisk',
            'Events',
            'AttritionEvents',
        ]
    ].drop_duplicates()

    df = usfhn.views.filter_by_taxonomy(df, level=level)

    if highly_productive:
        selector = df['IsHighlyProductiveNonUSCountry']
    else:
        selector = ~df['IsHighlyProductiveNonUSCountry']

    df = df[selector]

    df = hnelib.pandas.aggregate_df_over_column(
        df,
        agg_col='US',
        join_cols=[
            'TaxonomyLevel',
            'TaxonomyValue',
        ],
        value_cols=[
            'AttritionRisk',
            'Events',
            'AttritionEvents',
        ],
        agg_value_to_label={
            True: 'US',
            False: 'NonUS',
        }
    )

    df = df.merge(
        usfhn.attrition.compute_attrition_risk_significance(
            df,
            'NonUSAttritionEvents',
            'NonUSEvents',
            'USAttritionEvents',
            'USEvents',
        ),
        on=[
            'TaxonomyLevel',
            'TaxonomyValue',
        ]
    )

    denominator = len(df)

    df = df[
        df['Significant']
    ]

    df['Ratio'] = df['NonUSAttritionRisk'] / df['USAttritionRisk']

    df = df[
        df['Ratio'] > 1
    ]

    numerator = len(df)

    return get_number_or_percent(get, numerator, denominator)


def non_us_attrition_risk_stat(level='Field', get='number'):
    df = usfhn.stats.runner.get('attrition/risk/us')[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'US',
            'AttritionRisk',
            'Events',
            'AttritionEvents',
        ]
    ].drop_duplicates()

    df = hnelib.pandas.aggregate_df_over_column(
        df,
        agg_col='US',
        join_cols=[
            'TaxonomyLevel',
            'TaxonomyValue',
        ],
        value_cols=[
            'AttritionRisk',
            'Events',
            'AttritionEvents',
        ],
        agg_value_to_label={
            True: 'US',
            False: 'NonUS',
        }
    )

    df['Ratio'] = df['NonUSAttritionRisk'] / df['USAttritionRisk']

    df = df.merge(
        usfhn.attrition.compute_attrition_risk_significance(
            df,
            'NonUSAttritionEvents',
            'NonUSEvents',
            'USAttritionEvents',
            'USEvents',
        ),
        on=[
            'TaxonomyLevel',
            'TaxonomyValue',
        ]
    )

    denominator = len(df)

    df = df[
        df['Significant']
    ]

    df = usfhn.views.filter_by_taxonomy(df, level=level)

    denominator = len(df)

    df = df[
        df['Ratio'] > 1
    ]

    numerator = len(df)

    return get_number_or_percent(get, numerator, denominator)


def non_us_nubbins_stat(get='percent'):
    df = usfhn.datasets.get_dataset_df('closedness_data')[
        [
            'PersonId',
            'DegreeInstitutionId'
        ]
    ].drop_duplicates()

    
    df = usfhn.institutions.filter_to_non_us(df)
    df = usfhn.institutions.annotate_highly_productive_non_us_countries(df)
    df = usfhn.institutions.annotate_continent(df)

    n_non_us = len(df)

    df = df[
        ~df['IsHighlyProductiveNonUSCountry']
    ]

    n_nubbins = len(df)

    return get_number_or_percent(get, n_nubbins, n_non_us, round_to=1)

def english_nubbins_stat(get='percent'):
    df = usfhn.datasets.get_dataset_df('closedness_data')[
        [
            'PersonId',
            'DegreeInstitutionId'
        ]
    ].drop_duplicates()

    df = usfhn.institutions.filter_to_non_us(df)
    df = usfhn.institutions.annotate_highly_productive_non_us_countries(df)

    n_non_us = len(df)

    df = df[
        df['IsHighlyProductiveNonUSCountry']
    ]

    n_nubbins = len(df)

    return get_number_or_percent(get, n_nubbins, n_non_us, round_to=1)

def p_value_of_proportionality_between_out_of_field_and_non_us_at_level(level):
    df = usfhn.closedness.get_proportionality_tests_for_non_us_vs_out_of_field_at_level('Umbrella')
    value = max(df['P'])
    return int(round(value))


def get_null_models_stat(
    level=None,
    value=None,
    alpha=.05,
    get='number',
    iloc=0,
    significant=False,
    max_violations=None,
):
    df = usfhn.null_models.get_stats().copy()

    df = usfhn.views.filter_by_taxonomy(df, level)

    df['P'] = df['MoreHierarchicalCount'] / usfhn.constants.NULL_MODEL_DRAWS
    df['Significant'] = df['P'] < alpha

    df = usfhn.stats_utils.correct_multiple_hypotheses(df, alpha=alpha)
    df['Significant'] = df['PCorrected'] < alpha

    denominator = len(df)

    if significant:
        df = df[
            df['Significant']
        ]

    if max_violations != None:
        df = df[
            df['MoreHierarchicalCount'] <= max_violations
        ]

    df = df.sort_values(by=['P'], ascending=False)

    numerator = len(df)

    row = df.iloc[iloc]

    if get == 'p':
        return round(row['PCorrected'], 2)
    elif get == 'string':
        return usfhn.plot_utils.main_text_taxonomy_string(row['TaxonomyValue'])

    return get_number_or_percent(get, numerator, denominator)


################################################################################
# Field to Field Correlations
################################################################################
def get_field_to_field_correlations_stat(
    threshold=0,
    rank_type='prestige',
    get='number',
    round_to=0,
    threshold_type='above',
):
    df = usfhn.stats.runner.get('ranks/institution-rank-correlations', rank_type=rank_type)

    n_rows_start = len(df)

    if threshold_type == 'above':
        df = df[
            df['Pearson'] > threshold
        ]
    elif threshold_type == 'below':
        df = df[
            df['Pearson'] < threshold
        ]
    
    n_rows = len(df)

    if get == 'number':
        return n_rows
    elif get == 'percent':
        return hnelib.utils.fraction_to_percent(n_rows / n_rows_start)


def mean_correlation_of_field_to_field(field, rank_type='prestige'):
    df = usfhn.stats.runner.get('ranks/institution-rank-correlations', rank_type=rank_type)

    df = df[
        df['TaxonomyValueOne'] == field
    ]

    return round(df['Pearson'].mean(), 2)

################################################################################
# gender
################################################################################
def count_of_gender(gender):
    df = usfhn.datasets.get_dataset_df('closedness_data')
    df = df[
        df['Gender'] == gender
    ]
    return df['PersonId'].nunique()

def fraction_of_gender(gender):
    df = usfhn.datasets.get_dataset_df('closedness_data')
    df = df[
        df['Gender'].isin(['Male', 'Female'])
    ]
    
    fraction = count_of_gender(gender) / df['PersonId'].nunique()
    return round(fraction, 2)

def fraction_of_gender_not_known():
    df = usfhn.datasets.get_dataset_df('closedness_data')
    known = df[
        df['Gender'].isin(['Male', 'Female'])
    ]['PersonId'].nunique()

    total = df['PersonId'].nunique()
    unknown = total - known

    return hnelib.utils.fraction_to_percent(unknown / total)


def fraction_of_gender_annotated():
    df = usfhn.datasets.get_dataset_df('closedness_data')[
        [
            'PersonId',
            'Gender',
        ]
    ].drop_duplicates()

    df = df.merge(
        pd.read_csv(usfhn.constants.AA_2022_PEOPLE_GENDERED_PEOPLE_PATH),
        on='PersonId',
    )
    
    n_people = df['PersonId'].nunique()

    df = df[
        df['Source'] == 'Annotation'
    ]

    n_annotated = df['PersonId'].nunique()

    return hnelib.utils.fraction_to_percent(n_annotated / n_people)

def fraction_of_gender_known():
    df = usfhn.datasets.get_dataset_df('closedness_data')
    known = df[
        df['Gender'].isin(['Male', 'Female'])
    ]['PersonId'].nunique()
    
    total = df['PersonId'].nunique()

    return hnelib.utils.fraction_to_percent(known / total)

def percent_of_gender_in_domain(gender, domain):
    df = usfhn.stats.runner.get('gender/df')

    row = usfhn.views.filter_by_taxonomy(df, level='Umbrella', value=domain).iloc[0]
    value = row['FractionMale'] if gender == 'Male' else row['FractionFemale']

    return hnelib.utils.fraction_to_percent(value)


def range_of_percents_of_gender_in_fields_within_domain(gender, domain):
    df = usfhn.stats.runner.get('gender/df')

    df = usfhn.views.filter_by_taxonomy(df, level='Field')
    df = usfhn.views.annotate_umbrella(df, 'Field')

    df = df[
        df['Umbrella'] == domain
    ]

    df['Value'] = df['FractionMale'] if gender == 'Male' else df['FractionFemale']
    df['Percent'] = df['Value'].apply(hnelib.utils.fraction_to_percent)

    min_percent = min(df['Percent'])
    max_percent = max(df['Percent'])

    return f"{min_percent}-{max_percent}"


def percent_female_in_stem():
    df = usfhn.datasets.get_dataset_df('data')
    df = df[
        [
            'PersonId',
            'Gender',
            'Area',
            'Umbrella',
        ]
    ].drop_duplicates()

    df = df[
        (df['Umbrella'].isin(['Natural Sciences', 'Mathematics and Computing', 'Engineering']))
        |
        (df['Area'].isin(['Medical Sciences']))
    ]

    df = df.drop(columns=['Umbrella', 'Area']).drop_duplicates()

    df = df[
        df['Gender'].isin(['Male', 'Female'])
    ]

    n_men = df[
        df['Gender'] == 'Male'
    ]['PersonId'].nunique()

    n_women = df[
        df['Gender'] == 'Female'
    ]['PersonId'].nunique()

    total = n_men + n_women

    return hnelib.utils.fraction_to_percent(n_women / total)

def get_number_of_fields_with_more_junior_women_than_senior_women(get='number'):
    df = usfhn.stats.runner.get('gender/by-seniority/df')

    senior_faculty_df = df.copy()[
        df['Senior']
    ].drop(columns=['Senior'])
    senior_faculty_df = usfhn.views.filter_by_taxonomy(senior_faculty_df, level='Field').rename(columns={
        'FractionFemale': 'SeniorFacultyFractionFemale'
    })

    junior_faculty_df = df.copy()[
        ~df['Senior']
    ].drop(columns=['Senior'])
    junior_faculty_df = usfhn.views.filter_by_taxonomy(junior_faculty_df, level='Field').rename(columns={
        'FractionFemale': 'JuniorFacultyFractionFemale'
    })

    df = senior_faculty_df.merge(
        junior_faculty_df,
        on=['TaxonomyLevel', 'TaxonomyValue'],
    )

    df = df[
        df['SeniorFacultyFractionFemale'] <= df['JuniorFacultyFractionFemale']
    ]

    if get == 'number':
        value = len(df)
    elif get == 'percent':
        value = hnelib.utils.fraction_to_percent(
            len(df) / len(usfhn.views.get_fields())
        )

    return value


def get_gender_change_df_with_significance(
    increasing=None,
    p_threshold=.05,
    z_threshold=1.96,
):
    df = usfhn.stats.runner.get('gender/by-year/df')[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'Year',
            'FractionFemale',
            'Faculty',
            'FemaleFaculty',
        ]
    ].drop_duplicates()

    df = hnelib.pandas.aggregate_df_over_column(
        df,
        agg_col='Year',
        join_cols=['TaxonomyLevel', 'TaxonomyValue'],
        value_cols=[
            'FractionFemale',
            'Faculty',
            'FemaleFaculty',
        ],
        agg_value_to_label={
            min(df['Year']): 'Start',
            max(df['Year']): 'End',
        },
    )

    proportion_test_rows = []
    for i, row in df.iterrows():
        alternative = 'two-sided'

        if increasing != None:
            alternative = 'smaller' if increasing else 'larger'

        z_value, p_value = proportions_ztest(
            [row[f'StartFemaleFaculty'], row[f'EndFemaleFaculty']],
            [row['StartFaculty'], row['EndFaculty']],
            alternative=alternative,
        )

        reject_null = p_value < p_threshold
        significant = abs(z_value) > z_threshold

        proportion_test_rows.append({
            'TaxonomyValue': row['TaxonomyValue'],
            'TaxonomyLevel': row['TaxonomyLevel'],
            'Significant': reject_null and significant,
            'P': p_value,
            'ZTestSignificant': abs(z_value) > z_threshold,
        })

    proportions_df = pd.DataFrame(proportion_test_rows)

    proportions_df = usfhn.stats_utils.correct_multiple_hypotheses(proportions_df)

    proportions_df['Significant'] = (
        (proportions_df['ZTestSignificant'])
        &
        (proportions_df['PCorrected'] < p_threshold)
    )

    df = df.merge(
        proportions_df,
        on=[
            'TaxonomyLevel',
            'TaxonomyValue',
        ]
    )

    return df


def places_where_representation_of_women_is_increasing(
    get='number',
    level=None,
    increasing=True,
    significant=True,
):
    df = get_gender_change_df_with_significance(increasing=increasing)
    df = usfhn.views.filter_by_taxonomy(df, level)

    df['WomenIncreasing'] = df['StartFractionFemale'] < df['EndFractionFemale']

    n_rows_start = len(df)

    df = df[
        df['Significant']
    ]

    if increasing:
        df = df[
            df['WomenIncreasing']
        ]
    else:
        df = df[
            ~df['WomenIncreasing']
        ]

    n_rows_end = len(df)

    if get == 'string':
        fields = list(df['TaxonomyValue'].unique())
        if len(fields) == 1:
            return fields[0]
        elif len(fields) == 2:
            return f"{fields[0]} and {fields[1]}"
        elif len(fields) > 2:
            first_fields = fields[:-1]
            last_field = fields[-1]

            return ", ".join(first_fields) + ', and ' + last_field

    return get_number_or_percent(get, n_rows_end, n_rows_start)
    

def places_where_gender_parity_is_increasing(
    get='number',
    level=None,
    parity_direction=None,
):
    df = get_gender_change_df_with_significance()
    df = usfhn.views.filter_by_taxonomy(df, level)

    df['StartDistFromParity'] = .5 - df['StartFractionFemale']
    df['StartDistFromParity'] = df['StartDistFromParity'].apply(abs)

    df['EndDistFromParity'] = .5 - df['EndFractionFemale']
    df['EndDistFromParity'] = df['EndDistFromParity'].apply(abs)

    df['ParityIncreasing'] = df['StartDistFromParity'] > df['EndDistFromParity']

    n_rows_start = len(df)

    df = df[
        df['Significant']
    ]

    n_rows_significant = len(df)

    if parity_direction == 'increasing':
        df = df[
            df['ParityIncreasing']
        ]
    elif parity_direction == 'decreasing':
        df = df[
            ~df['ParityIncreasing']
        ]

    n_rows_end = len(df)

    if get == 'string':
        fields = list(df['TaxonomyValue'].unique())
        if len(fields) == 1:
            return fields[0]
        elif len(fields) == 2:
            return f"{fields[0]} and {fields[1]}"
        elif len(fields) > 2:
            first_fields = fields[:-1]
            last_field = fields[-1]

            return ", ".join(first_fields) + ', and ' + last_field

    return get_number_or_percent(get, n_rows_end, n_rows_significant)


def get_start_or_end_gender_stat(which, level=None, value=None):
    df = usfhn.stats.runner.get('gender/by-year/df')
    df = usfhn.views.filter_by_taxonomy(df, level, value)

    df = hnelib.pandas.aggregate_df_over_column(
        df,
        agg_col='Year',
        join_cols=['TaxonomyLevel', 'TaxonomyValue'],
        value_cols=['FractionFemale'],
        agg_value_to_label={
            min(df['Year']): 'Start',
            max(df['Year']): 'End',
        },
    )

    return hnelib.utils.fraction_to_percent(df.iloc[0][f'{which}FractionFemale'], 1)


def percent_female_new_hires(level='Academia', value=None, which=None, get='number', iloc=0):
    df = usfhn.stats.runner.get('gender/by-new-hire/df')
    df = df[
        df['NewHire']
    ]
    df = usfhn.views.filter_by_taxonomy(df, level, value)

    if which == min:
        df = df.sort_values(by='FractionFemale')
    elif which == max:
        df = df.sort_values(by='FractionFemale', ascending=False)

    row = df.iloc[iloc]

    if get == 'string':
        return usfhn.plot_utils.main_text_taxonomy_string(row['TaxonomyValue'])
    
    return hnelib.utils.fraction_to_percent(row['FractionFemale'])


def minority_women_places(get='number', level='Field'):
    df = usfhn.stats.runner.get('gender/by-new-hire/df')
    df = df[
        df['NewHire']
    ]
    df = usfhn.views.filter_by_taxonomy(df, level)

    n_rows_start = len(df)

    df = df[
        df['FractionFemale'] < .5
    ]

    n_rows_end = len(df)

    return get_number_or_percent(get, n_rows_end, n_rows_start)


################################################################################
# Prestige
################################################################################
def min_violations_percent():
    df = usfhn.stats.runner.get('ranks/hierarchy-stats', rank_type='prestige')
    df = df[
        df['MovementType'] == 'Upward'
    ]
    return hnelib.utils.fraction_to_percent(min(df['MovementFraction']))


def max_violations_percent():
    df = usfhn.stats.runner.get('ranks/hierarchy-stats', rank_type='prestige')
    df = df[
        df['MovementType'] == 'Upward'
    ]
    df = df[
        df['MovementFraction'] == max(df['MovementFraction'])
    ]
    return hnelib.utils.fraction_to_percent(max(df['MovementFraction']))


def percent_steepness_at_level_and_value(level, value):
    df = usfhn.stats.runner.get('ranks/hierarchy-stats', rank_type='prestige')
    df = df[
        df['MovementType'] == 'Upward'
    ]
    df = usfhn.views.filter_by_taxonomy(df, level, value)

    return hnelib.utils.fraction_to_percent(df.iloc[0]['MovementFraction'])


def mean_movement_on_hierarchy():
    df = usfhn.stats.runner.get('ranks/mean-change', rank_type='prestige')
    value = usfhn.views.filter_by_taxonomy(df, 'Academia', 'Academia').iloc[0]['MeanRankChange']
    value = abs(value)
    value *= 100
    return round(value)


def median_movement_on_hierarchy():
    df = usfhn.stats.runner.get('ranks/mean-change', rank_type='prestige')
    value = usfhn.views.filter_by_taxonomy(df, 'Academia', 'Academia').iloc[0]['MedianRankChange']
    value = abs(value)
    value *= 100
    return round(value)


def mean_production_difference_from_advisor():
    df = usfhn.stats.runner.get('ranks/mean-change', rank_type='prestige')
    mean_movement = usfhn.views.filter_by_taxonomy(df, 'Academia', 'Academia').iloc[0]['MeanRankChange']
    mean_movement = abs(mean_movement)

    ranks = usfhn.stats.runner.get('ranks/df', rank_type='prestige')
    ranks = usfhn.views.filter_by_taxonomy(ranks, 'Academia', 'Academia')[
        [
            'InstitutionId',
            'Percentile',
        ]
    ].drop_duplicates()

    employment = usfhn.stats.runner.get('basics/faculty-hiring-network')
    employment = usfhn.views.filter_by_taxonomy(employment, 'Academia', 'Academia')[
        [
            'InstitutionId',
            'InDegree',
        ]
    ].drop_duplicates().rename(columns={
        'InDegree': 'EmploymentCount',
    })

    production = usfhn.stats.runner.get('basics/production')
    production = usfhn.views.filter_by_taxonomy(production, 'Academia', 'Academia')[
        [
            'DegreeInstitutionId',
            'ProductionCount',

        ]
    ].rename(columns={
        'DegreeInstitutionId': 'InstitutionId',
    }).drop_duplicates()

    df = ranks.merge(
        employment,
        on='InstitutionId',
    ).merge(
        production,
        on='InstitutionId',
    ).sort_values(by='Percentile')

    df['FacultyPerAdvisor'] = df['ProductionCount'] / df['EmploymentCount']

    difference_between_advisory_and_faculty_production = []
    for i, row in df.iterrows():
        advisor_value = row['FacultyPerAdvisor']

        mean_placement_df = df[
            df['Percentile'] > row['Percentile'] + mean_movement
        ]

        placement_value = 0
        if mean_placement_df.empty:
            continue

        placement_value = mean_placement_df.sort_values(by='Percentile').iloc[0]['FacultyPerAdvisor']
    
        difference_between_advisory_and_faculty_production.append(advisor_value / placement_value)

    mean = np.mean(difference_between_advisory_and_faculty_production)

    return round(mean, 1)


def mean_movement_in_direction(direction, max_or_min, get='value'):
    df = usfhn.stats.runner.get('ranks/hierarchy-stats', rank_type='prestige')

    movement_type = 'Upward' if direction == 'up' else 'Downward'

    df = df[
        df['MovementType'] == movement_type
    ]

    df = usfhn.views.filter_by_taxonomy(df, level='Field')

    df['MeanMovementDistance'] = df['MeanMovementDistance'].apply(abs)
    df['MeanMovementDistance'] *= 100
    values = list(df['MeanMovementDistance'])

    if max_or_min == 'max':
        value = max(values)
    else:
        value = min(values)

    if get == 'value':
        return round(value)
    elif get == 'field':
        row = df[
            df['MeanMovementDistance'] == value
        ].iloc[0]
    
        return usfhn.plot_utils.main_text_taxonomy_string(row['TaxonomyValue'])


def get_mobility_stat(
    level='Field',
    value=None,
    domain=None,
    get='number',
    significant=None,
    direction=None,
):
    df = usfhn.self_hiring.get_gendered_rank_change_significance()
    df = usfhn.views.filter_by_taxonomy(df, level, value)

    if domain:
        df = usfhn.views.annotate_umbrella(df, level)
        df = df[
            df['Umbrella'] == domain
        ]

    denominator = len(df)

    if significant != None:
        if direction:
            selector = df[f"{direction}Significant"]
        else:
            selector = df['SignificantFractionDifference']

        if not significant:
            selector = ~selector

        df = df[
            selector
        ]

    numerator = len(df)

    return get_number_or_percent(get, numerator, denominator)
    

def p_movement_at_level_and_value(direction='up', level='Academia', value='Academia', round_to=0):
    df = usfhn.stats.runner.get('ranks/hierarchy-stats', rank_type='prestige')
    df = usfhn.views.filter_by_taxonomy(df, level=level, value=value)

    movement_type = 'Self-Hire'
    if direction == 'up':
        movement_type = 'Upward'
    elif direction == 'down':
        movement_type = 'Downward'

    row = df[
        df['MovementType'] == movement_type
    ].iloc[0]

    fraction = row['MovementFraction']

    return hnelib.utils.fraction_to_percent(fraction, round_to=round_to)


################################################################################
# Top Spots
################################################################################
def number_of_top_10_departments():
    df = usfhn.steeples.get_absolute_steeples()
    df = df[
        df['OrdinalRank'] < 10
    ]

    return len(df)


def top_10_departments_at_top_five_insts(get='number'):
    df = usfhn.steeples.get_absolute_steeples()
    df = df[
        df['OrdinalRank'] < 10
    ]

    denominator = len(df)

    df = df[
        df['InTop10']
    ]

    df['InstTop10Count'] = df.groupby('InstitutionId')['InTop10'].transform('count')

    df = df[
        [
            'InstitutionId',
            'InstTop10Count',
        ]
    ].drop_duplicates()

    df = df.sort_values(by=['InstTop10Count'], ascending=False)

    df = df.head(5)
    numerator = df['InstTop10Count'].sum()

    return get_number_or_percent(get, numerator, denominator, 1)


def number_of_institutions_without_top_10_departments():
    df = usfhn.steeples.get_absolute_steeples()
    institutions = set(df['InstitutionId'].unique())
    df = df[
        df['OrdinalRank'] < 10
    ]
    top_institutions = set(df['InstitutionId'].unique())

    return len(institutions - top_institutions)


def percent_of_institutions_without_top_10_departments():
    institutions = len(usfhn.institutions.get_institution_id_to_name_map(producing=False))
    fraction = number_of_institutions_without_top_10_departments() / institutions
    return hnelib.utils.fraction_to_percent(fraction)

def percent_of_institutions(n):
    df = usfhn.datasets.get_dataset_df('data')
    n_institutions = df['InstitutionId'].nunique()

    return hnelib.utils.fraction_to_percent(n / n_institutions, 1)

def number_of_institutions_with_1_top_10_department():
    df = usfhn.steeples.get_absolute_steeples()
    df = df[
        df['OrdinalRank'] < 10
    ].copy()
    df['Top10Departments'] = df.groupby('InstitutionId')['InstitutionId'].transform('count')
    df = df[
        df['Top10Departments'] == 1
    ]

    return df['InstitutionId'].nunique()


def percent_of_institutions_with_1_top_10_department():
    fraction = number_of_institutions_with_1_top_10_department() / number_of_top_10_departments()
    return hnelib.utils.fraction_to_percent(fraction)


def number_of_top_10_departments_from_top_five_schools():
    return usfhn.steeples.get_summary_of_steeple_stats().iloc[4]['Placements']

def get_names_of_top_five_institutional_producers_of_top_ten_departments():
    df = usfhn.steeples.get_absolute_steeples().copy()

    df = df[
        df['OrdinalRank'] < 10
    ]

    df['Top10Departments'] = df.groupby('InstitutionId')['InstitutionId'].transform('count')

    df = usfhn.institutions.annotate_institution_name(df)

    df = df[
        [
            'InstitutionName',
            'Top10Departments',
        ]
    ].drop_duplicates()

    df = df.sort_values(by=['Top10Departments'], ascending=False)

    top_institutions = list(df['InstitutionName'])[:5]
    first_institutions = top_institutions[:-1]
    last_institution = top_institutions[-1]

    return ", ".join(first_institutions) + ', and ' + last_institution


def percent_of_top_10_departments_from_top_five_schools():
    return usfhn.steeples.get_summary_of_steeple_stats().iloc[4]['CumSum%']


def percent_of_top_10_departments_not_from_top_five_schools():
    return 1 - (usfhn.steeples.get_summary_of_steeple_stats().iloc[4]['CumSum%'] / 100)


def number_of_institutions_with_a_top_10_department_but_not_top_5():
    df = usfhn.steeples.get_absolute_steeples()
    df = df[
        df['OrdinalRank'] < 10
    ]
                
    return df['InstitutionId'].nunique() - 5


def percent_of_institutions_with_a_top_10_department_but_not_top_5():
    df = usfhn.steeples.get_absolute_steeples()
    fraction = number_of_institutions_with_a_top_10_department_but_not_top_5() / df['InstitutionId'].nunique()
    return hnelib.utils.fraction_to_percent(fraction)


def number_of_institutions_with_more_than_one_top_10_department_but_not_top_5():
    df = usfhn.steeples.get_absolute_steeples()
    df = df[
        df['OrdinalRank'] < 10
    ].copy()
    df['Top10Departments'] = df.groupby('InstitutionId')['InstitutionId'].transform('count')
    df = df[
        df['Top10Departments'] > 1
    ]

    return df['InstitutionId'].nunique() - 5


def percent_of_institutions_with_more_than_one_top_10_department_but_not_top_5():
    n_institutions = usfhn.steeples.get_absolute_steeples()['InstitutionId'].nunique()
    fraction = number_of_institutions_with_more_than_one_top_10_department_but_not_top_5() / n_institutions
    return hnelib.utils.fraction_to_percent(fraction)

def number_of_institutions_with_a_top_10_department_academia():
    inst_df = usfhn.stats.runner.get('ranks/df', rank_type='prestige')

    inst_df = usfhn.views.filter_by_taxonomy(inst_df, level='Academia', value='Academia')

    institutions = set(inst_df['InstitutionId'].unique())
    n_institutions = len(institutions)

    df = usfhn.steeples.get_absolute_steeples()
    df = usfhn.views.filter_exploded_df(df)
    df = usfhn.views.filter_by_taxonomy(df, level='Field')
    df = usfhn.views.annotate_umbrella(df, 'Field')
    
    df = df[
        df['InTop10'] == True
    ]

    df['InstitutionCount'] = df.groupby('InstitutionId')['InstitutionId'].transform('count')
    df = df[
        [
            'InstitutionId',
            'InstitutionCount',
        ]
    ].drop_duplicates()

    return df['InstitutionId'].nunique()

def percent_of_institutions_with_a_top_10_department_academia():
    inst_df = usfhn.stats.runner.get('ranks/df', rank_type='prestige')

    inst_df = usfhn.views.filter_by_taxonomy(inst_df, level='Academia', value='Academia')

    institutions = set(inst_df['InstitutionId'].unique())
    n_institutions = len(institutions)

    df = usfhn.steeples.get_absolute_steeples()
    df = usfhn.views.filter_exploded_df(df)
    df = usfhn.views.filter_by_taxonomy(df, level='Field')
    df = usfhn.views.annotate_umbrella(df, 'Field')
    
    df = df[
        df['InTop10'] == True
    ]

    df['InstitutionCount'] = df.groupby('InstitutionId')['InstitutionId'].transform('count')
    df = df[
        [
            'InstitutionId',
            'InstitutionCount',
        ]
    ].drop_duplicates()

    institutions_without_steeples = institutions - set(df['InstitutionId'].unique())

    fraction = len(institutions_without_steeples) / n_institutions
    return hnelib.utils.fraction_to_percent(fraction)

def percent_of_institutions_with_top_10_domain_level(function='max', get='fraction'):
    fractions = []
    for umbrella in usfhn.views.get_umbrellas():
        inst_df = usfhn.stats.runner.get('ranks/df', rank_type='prestige')

        inst_df = usfhn.views.filter_by_taxonomy(inst_df, level='Umbrella', value=umbrella)

        institutions = set(inst_df['InstitutionId'].unique())
        n_institutions = len(institutions)

        df = usfhn.steeples.get_absolute_steeples()
        df = usfhn.views.filter_exploded_df(df)
        df = usfhn.views.filter_by_taxonomy(df, level='Field')
        df = usfhn.views.annotate_umbrella(df, 'Field')
        
        df = df[
            df['Umbrella'] == umbrella
        ]

        df = df[
            df['InTop10'] == True
        ]

        df['InstitutionCount'] = df.groupby('InstitutionId')['InstitutionId'].transform('count')
        df = df[
            [
                'InstitutionId',
                'InstitutionCount',
            ]
        ].drop_duplicates()

        institutions_with_steeples = df['InstitutionId'].nunique()

        fraction = institutions_with_steeples / n_institutions
        fractions.append((fraction, umbrella, institutions_with_steeples))

    fractions = sorted(fractions)

    if function == 'min':
        fraction, umbrella, number = fractions[0]
    else:
        fraction, umbrella, number = fractions[-1]

    if get == 'fraction':
        return hnelib.utils.fraction_to_percent(fraction)
    elif get == 'name':
        return usfhn.plot_utils.main_text_taxonomy_string(umbrella)
    elif get == 'number':
        return number

################################################################################
# Self Hiring
################################################################################
def self_hire_rate_at_level_and_value_and_gender(
    level='Academia',
    value='Academia',
    gender=usfhn.constants.GENDER_AGNOSTIC,
    get='percent',
    column='SelfHires',
):
    df = usfhn.stats.runner.get('self-hires/by-gender/df')
    df = usfhn.views.filter_exploded_df(df, gender=gender)
    df = usfhn.views.filter_by_taxonomy(df, level=level, value=value)
    row = df.iloc[0]
    return get_number_or_percent(get, row[column], row['Faculty'], round_to=1)


def number_of_fields_self_hiring_less_than_top_5_departments():
    df = usfhn.self_hiring.compare_self_hire_rate_of_top_institutions_vs_rest()
    df = usfhn.views.filter_by_taxonomy(df, level='Field')

    df = df[
        df['Ratio'] < 1
    ]
    return len(df)


def fields_where_prestigeous_institutions_self_hire_more(get='number'):
    df = usfhn.self_hiring.compare_self_hire_rate_of_top_institutions_vs_rest()
    df = usfhn.views.filter_by_taxonomy(df, level='Field')

    df = df[
        df['Ratio'] >= 1
    ]

    if get == 'number':
        value = len(df)
    else:
        value = hnelib.utils.fraction_to_percent(
            len(df) / len(usfhn.views.get_fields())
        )

    return value



def get_self_hiring_differs_by_gender_stat(
    level='Field',
    get='number',
    women_more_than_men=False,
    domain=None,
):
    df = usfhn.self_hiring.get_significance_of_gendered_self_hiring_rates()
    df = usfhn.views.filter_by_taxonomy(df, level=level)

    denominator = len(df)

    df = df[
        df['Significant']
    ]

    if women_more_than_men:
        df = df[
            df['WomenHiredMoreThanMen']
        ]

    if domain:
        df = usfhn.views.annotate_umbrella(df, level)
        df = df[
            df['Umbrella'] == domain
        ]

    numerator = len(df)

    if get == 'p':
        return round(max(df['P']), 2)

    return get_number_or_percent(get, numerator, denominator)


def self_hiring_actual_over_expected(function='max', value='value'):
    df = usfhn.self_hiring.get_expected_self_hiring_rates_and_compare_to_actual()

    selection_value = min(df['Actual/Expected']) if function == 'min' else max(df['Actual/Expected'])

    row = df[
        df['Actual/Expected'] == selection_value
    ].iloc[0]

    if value == 'value':
        return round(row['Actual/Expected'], 1)
    else:
        return usfhn.plot_utils.main_text_taxonomy_string(row['TaxonomyValue'])


def self_hiring_correlation(level='Academia', value='Academia', get='pearson'):
    df = usfhn.stats.runner.get('ranks/df', rank_type='prestige')
    df = usfhn.views.filter_by_taxonomy(df, level=level, value=value)[
        [
            'InstitutionId',
            'Percentile',
        ]
    ].drop_duplicates()

    sh = usfhn.stats.runner.get('self-hires/by-institution/df')
    sh = usfhn.views.filter_by_taxonomy(sh, level=level, value=value)[
        [
            'InstitutionId',
            'SelfHiresFraction',
        ]
    ].drop_duplicates()

    df = df.merge(
        sh,
        on='InstitutionId',
    )

    pearson, p = pearsonr(df['Percentile'], df['SelfHiresFraction'])

    if get == 'pearson':
        return round(pearson, 2)
    elif get == 'p':
        p = round(p, 2)
        if p == 0.00000:
            p = r'$p < 10^-5$'
        else:
            p = r'$p =' + str(p) + r'$'

        return p

################################################################################
# Time
################################################################################
def get_professor_count_at_point(when='start'):
    df = usfhn.datasets.get_dataset_df('closedness_data', by_year=True)[
        [
            'PersonId',
            'Year',
        ]
    ].drop_duplicates()

    df['PeopleInYear'] = df.groupby('Year')['PersonId'].transform('nunique')

    if when == 'start':
        row = df[
            df['Year'] == min(df['Year'])
        ].iloc[0]
    else:
        row = df[
            df['Year'] == max(df['Year'])
        ].iloc[0]

    return row['PeopleInYear']


@lru_cache
def get_professor_count_df():
    df = usfhn.datasets.get_dataset_df('closedness_data', by_year=True)[
        [
            'PersonId',
            'Year',
            'Rank',
        ]
    ].drop_duplicates()

    df['PeoplePerYearByRank'] = df.groupby(['Year', 'Rank'])['PersonId'].transform('nunique')
    df['TotalInYear'] = df.groupby(['Year'])['PersonId'].transform('nunique')
    df['PercentInYear'] = df['PeoplePerYearByRank'] / df['TotalInYear']

    df = df.drop(columns=['PersonId']).drop_duplicates()
    return df


def get_professor_count(kind='max', rank='Professor', get='number'):
    df = get_professor_count_df().copy()

    df = df[
        df['Rank'] == rank
    ]

    col = 'PercentInYear' if get == 'percent' else 'PeoplePerYearByRank'

    if kind == 'max':
        df = df[
            df[col] == max(df[col])
        ]
    else:
        df = df[
            df[col] == min(df[col])
        ]

    value = df.iloc[0][col]

    if get == 'percent':
        return hnelib.utils.fraction_to_percent(value)
    else:
        return value

################################################################################
# mid-career moves
################################################################################
def get_mcm_significance(level=None, value=None, get='number', movement_type='Self-Hire'):
    df = usfhn.stats.runner.get('careers/hierarchy-changes-from-mcms-compare-to-normal')
    if level:
        df = usfhn.views.filter_by_taxonomy(df, level, value)

    df = df[
        df['MovementType'] == movement_type
    ]

    n_level = df['TaxonomyValue'].nunique()

    df = df[
        df['SignificantDifference']
    ]

    n_significant = df['TaxonomyValue'].nunique()

    if get == 'number':
        return n_significant
    elif get == 'percent':
        return hnelib.utils.fraction_to_percent(n_significant / n_level)
    elif get == 'p':
        return round(df.iloc[0]['P'], 2)


def get_mcm_difference(level, value, movement_type='Upward', get='value'):
    df = usfhn.stats.runner.get('careers/hierarchy-changes-from-mcms')
    df = usfhn.views.filter_by_taxonomy(df, level, value)

    row = df[
        df['MovementType'] == movement_type
    ].iloc[0]

    if get == 'value':
        pre_value = row['MovementFraction-Pre']
        post_value = row['MovementFraction-Post']

        difference = post_value - pre_value 

        return hnelib.utils.fraction_to_percent(difference, 1)
    elif get == 'p':
        return round(row['P'], 2)

################################################################################
# Attrition
################################################################################
def self_hire_attrition_stat(
    level='Field',
    stat='higher-risk-count',
    ascending=False,
    get='number',
    iloc=0,
):
    df = usfhn.self_hiring.get_self_hire_non_self_hire_risk_ratios()

    df = usfhn.views.filter_by_taxonomy(df, level)

    df = df[
        df['Significant']
    ]

    n_rows_start = len(df)

    if stat == 'higher-risk-count':
        df = df[
            df['Ratio'] > 1
        ]

        return get_number_or_percent(get, len(df), n_rows_start)

    df = df.sort_values(by=['Ratio'], ascending=ascending)

    row = df.iloc[iloc]

    if get == 'number':
        return round(row['Ratio'], 1)
    elif get == 'string':
        return usfhn.plot_utils.main_text_taxonomy_string(row['TaxonomyValue'])


################################################################################
# Market share
################################################################################
def get_growth_stat(
    level='Academia',
    value=None,
    umbrella=None,
    get='percent',
    change_type=None,
    ascending=False,
    stat='market-share',
    iloc=0,
    round_to=0,
):
    df = usfhn.stats.runner.get('taxonomy/by-year/df')[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'Year',
            'Fraction',
            'Count',
        ]
    ].drop_duplicates()

    min_year_df = df[
        df['Year'] == min(df['Year'])
    ].copy().rename(columns={
        'Fraction': 'StartingFraction',
        'Count': 'StartingCount',
    }).drop(columns=['Year'])

    df = df.merge(
        min_year_df,
        on=[
            'TaxonomyLevel',
            'TaxonomyValue',
        ]
    )

    df = df[
        df['Year'] == max(df['Year'])
    ]

    df['Growth'] = df['Fraction'] / df['StartingFraction']
    df['FacultyCountGrowth'] = df['Count'] / df['StartingCount']

    growth_column = 'FacultyCountGrowth' if stat == 'faculty-count' else 'Growth'

    df = usfhn.views.filter_by_taxonomy(df, level, value)

    if umbrella:
        df = usfhn.views.annotate_umbrella(df, level)
        df = df[
            df['Umbrella'] == umbrella
        ]

    n_rows_start = len(df)

    if change_type == 'loss':
        df = df[
            df[growth_column] < 1
        ]

        df[growth_column] = 1 - df[growth_column]
    elif change_type == 'growth':
        df = df[
            df[growth_column] > 1
        ]

        df[growth_column] = df[growth_column] - 1

    n_rows_end = len(df)

    if stat == 'field-count':
        return get_number_or_percent(get, n_rows_end, n_rows_start, round_to=round_to)

    df = df.sort_values(by=growth_column, ascending=ascending)
    row = df.iloc[iloc]

    if get == 'string':
        return usfhn.plot_utils.main_text_taxonomy_string(row['TaxonomyValue'])

    value = row[growth_column]

    return hnelib.utils.fraction_to_percent(value, round_to=round_to)


################################################################################
# Stats
################################################################################
def general_stats_generator():
    df = usfhn.datasets.get_dataset_df('data')
    closedness_df = usfhn.datasets.get_dataset_df('closedness_data')

    return {
        'nPeople': {
            'comment': 'how many people are there?',
            'value': int(closedness_df['PersonId'].nunique()),
        },
        'nUSPeople': {
            'comment': 'how many U.S.-trained people are there?',
            'value': int(df['PersonId'].nunique()),
        },
        'nDepartments': {
            'comment': 'how many "departments" are there?',
            'value': int(df['DepartmentId'].nunique()),
        },
        'nEmployingInstitutions': {
            'comment': 'how many employing institutions are there?',
            'value': int(df['InstitutionId'].nunique()),
        },
        'nProducingInstitutions': {
            'comment': 'how many producing institutions are there?',
            'value': int(closedness_df['DegreeInstitutionId'].nunique()),
        },
        'nUSProducingInstitutions': {
            'comment': 'how many U.S. producing institutions are there?',
            'value': int(df['DegreeInstitutionId'].nunique()),
        },
        'nFields': {
            'comment': 'how many fields are there?',
            'value': int(usfhn.views.get_taxonomization()['Field'].nunique()),
        },
        'nDomains': {
            'comment': 'how many domains are there?',
            'value': int(usfhn.views.get_taxonomization()['Umbrella'].nunique()),
        },
        'nYears': {
            'comment': 'how many years does the dataset cover?',
            'value': len(usfhn.views.get_years()),
        },
        'firstYear': {
            'comment': 'what is the first year of the dataset?',
            'value': min(usfhn.views.get_years()),
            'add_commas': False,
        },
        'lastYear': {
            'comment': 'what is the last year of the dataset?',
            'value': max(usfhn.views.get_years()),
            'add_commas': False,
        },
        'nFieldsEng': {
            'comment': 'what is the last year of the dataset?',
            'value': fields_in_engineering(),
        },
        'nFieldsMedAndHealth': {
            'comment': 'how many fields are in medicine and health?',
            'value': fields_in_level_and_value(level='Umbrella', value='Medicine and Health'),
        },
        'nFieldsHumanities': {
            'comment': 'how many fields are in Humanities?',
            'value': fields_in_level_and_value(level='Umbrella', value='Humanities'),
        },
        'nNonArtsHumanitiesFields': {
            'comment': 'how many non-arts fields are there in the humanities?',
            'value': non_arts_fields_in_humanities(),
        },
        'pGendered': {
            'comment': 'what percentage of names were we able to gender?',
            'value': gender_stats('Known', get='percent'),
        },
        'nGendered': {
            'comment': 'what number of names were we able to gender?',
            'value': gender_stats('Known', get='number'),
        },
        'pMultiTaxFaculty': {
            'comment': 'what % of faculty had more than one taxonomy annotation?',
            'value': multi_taxonomy_faculty(),

        },
        'nRankedPeople': {
            'comment': 'how many people were ranked?',
            'value': get_n_ranked_people(),
        },
        'nRankedAndGenderedPeople': {
            'comment': 'how many people with a gender were ranked?',
            'value': get_n_ranked_people(gendered=True),
        },
    }


def exclusion_stats_generator():
    general_stats = usfhn.stats.runner.get('pool-reduction/paper-stats/general')

    return {
        # TODO: fix all of these, along with the whole section on exclusion/inclusion
        'nFacultyIncluded': {
            'comment': 'total faculty included',
            'value': general_stats['nIncluded'],
        },
        'pFacultyIncluded': {
            'comment': 'percent of faculty included from maximum set',
            'value': general_stats['pIncluded'],
        },
        'pExcluded': {
            'comment': 'percent of faculty excluded from maximum set',
            'value': general_stats['pExcluded'],
        },
        'nUSDegreePeople': {
            'comment': 'number of people included in the closed calculations',
            'value': general_stats['nClosedIncluded'],
        },
        'pUSDegreePeople': {
            'comment': 'percent of people included in the closed calculations',
            'value': general_stats['pClosedIncluded'],
        },
        'nPeopleRemovedFromOpenToClosed': {
            'comment': 'number of people excluded in US calculations',
            'value': general_stats['nClosedExcluded'],
        },
        'pPeopleRemovedFromOpenToClosed': {
            'comment': 'percent of people excluded in US calculations',
            'value': general_stats['pClosedExcluded'],
        },
        'nFacultyInPool': {
            'comment': 'number of faculty in the the maximal set',
            'value': general_stats['nPool'],
        },
        'nDomainExcludedPeople': {
            'comment': 'number of faculty excluded from domain level analysis',
            'value': general_stats['nDomainExcludedPeople'],
        },
        'pDomainExcludedPeople': {
            'comment': 'perct of faculty excluded from domain level analysis',
            'value': general_stats['pDomainExcludedPeople'],
        },
        'nFieldExcludedPeople': {
            'comment': 'number of faculty excluded from field level analysis',
            'value': general_stats['nFieldExcludedPeople'],
        },
        'pFieldExcludedPeople': {
            'comment': 'perct of faculty excluded from field level analysis',
            'value': general_stats['pFieldExcludedPeople'],
        },
        'nJournalismExcludedPeople': {
            'comment': 'number of faculty excluded for being in journalism',
            'value': general_stats['nComExcludedPeople'],
        },
        'nPublicPolicyExcludedPeople': {
            'comment': 'number of faculty excluded for being in public policy',
            'value': general_stats['nPolicyExcludedPeople'],
        },
        'nNotInAllYearsExcludedPeople': {
            'comment': 'number of faculty excluded for being in departments not in all years',
            'value': general_stats['nNotInAllYearsExcludedPeople'],
        },
        'nNoPrimaryAppointmentExcludedPeople': {
            'comment': 'number of faculty excluded for not having any primary appointments',
            'value': general_stats['nNoPrimaryAppointmentExcludedPeople'],
        },
        'nNoDegreeExcludedPeople': {
            'comment': 'number of faculty excluded for lacking a degree',
            'value': general_stats['nNoDegreeExcludedPeople'],
        },
        'nNonEmployingInstitutionExcludedPeople': {
            'comment': 'number of faculty excluded for working at a non-phd producing institutions',
            'value': general_stats['nNonEmployingInstitutionExcludedPeople'],
        },
    }


def production_stats_generator():
    return {
        'pEightyPOfFaculty': {
            'comment': 'what percent of institutions did 80% of faculty come from?',
            'value': eighty_percent_of_faculty(),
        },
        'pFromTopFiveInstitutions': {
            'comment': 'what percent of faculty were produced by the top 5 institutions?',
            'value': percent_produced_by_top_five_institutions(),
        },
        'topFiveProducingInstitutions': {
            'comment': 'what are the top 5 most producingest institutions?',
            'value': top_five_most_producing_institutions(),
        },
        'nInstitutionsProducingFirstTwentyPercent': {
            'comment': 'how many institutions produce the "first" twenty percent of faculty?',
            'value': n_institutions_producing_twenty_percent_of_faculty(twenty_percent=0),
        },
        'nInstitutionsProducingLastTwentyPercent': {
            'comment': 'how many institutions produce the "last" twenty percent of faculty?',
            'value': n_institutions_producing_twenty_percent_of_faculty(twenty_percent=4),
        },
        'pRangeOfEightyPOfFacultyDomainLevel': {
            'comment': 'what is the range of the percent of institutions producing 80% of faculty?',
            'value': range_of_percents_of_institutions_producing_eighty_percent_of_faculty_at_domain_level(),
        },
        'correlationBWInstSizeAndProdAcademia': {
            'comment': 'what is the correlation between institution size and production?',
            'value': correlation_between_institution_size_and_production()['correlation'],
        },
        'pValOfCorrelationBWInstSizeAndProdAcademia': {
            'comment': 'what is the p value of the correlation between institution size and production?',
            'value': correlation_between_institution_size_and_production()['p'],
        },
        'nFieldsProdAndEmpDifferent': {
            'comment': 'how many units are significantly different in production and employment?',
            'value': fields_production_and_employment_distribution_stat(different=True),
        },
        'nFieldsProdAndEmpDifferentField': {
            'comment': 'how many units are significantly different in production and employment for fields?',
            'value': fields_production_and_employment_distribution_stat(level='Field', different=True),
        },
        'pFieldsProdAndEmpDifferentField': {
            'comment': 'what percentage of units are significantly different in production and employment for fields?',
            'value': fields_production_and_employment_distribution_stat(level='Field', get='percent', different=True),
        },
        'nFieldsProdAndEmpDifferentDomain': {
            'comment': 'how many units are significantly different in production and employment for domains?',
            'value': fields_production_and_employment_distribution_stat(level='Umbrella', different=True),
        },
        'nFieldsProdAndEmpSame': {
            'comment': 'how many fields have distributions that arent different?',
            'value': fields_production_and_employment_distribution_stat(level='Field'),
        },
        'pFieldsProdAndEmpSame': {
            'comment': 'what percent of fields have distributions that arent different?',
            'value': fields_production_and_employment_distribution_stat(level='Field', get='percent'),
        },
        'nDomainProdAndEmpSame': {
            'comment': 'how many domain have distributions that arent different?',
            'value': fields_production_and_employment_distribution_stat(level='Umbrella'),
        },
        'nProdAndEmpSame': {
            'comment': 'how many units have distributions that arent different?',
            'value': fields_production_and_employment_distribution_stat(level=None),
        },
        'nProdAndEmpSameNoSH': {
            'comment': 'how many units have distributions that arent different without self hires?',
            'value': fields_production_and_employment_distribution_stat(level=None, exclude_self_hires=True),
        },
        'nProdAndEmpSameNoSHField': {
            'comment': 'how many units have distributions that arent different without self hires at the field level?',
            'value': fields_production_and_employment_distribution_stat(level='Field', exclude_self_hires=True),
        },
        'pProdAndEmpSameNoSHField': {
            'comment': 'what percent of units have distributions that arent different without self hires at the field level?',
            'value': fields_production_and_employment_distribution_stat(level='Field', exclude_self_hires=True, get='percent'),
        },
    }

def gini_stats_generator():
    return {
        'giniAcademia': {
            'comment': 'what is the gini value at the academic level?',
            'value': gini_at_level_and_value('Academia', 'Academia'),
        },
        'giniEducation': {
            'comment': 'what is the gini value of Education?',
            'value': gini_at_level_and_value('Umbrella', 'Education'),
        },
        'giniHumanities': {
            'comment': 'what is the gini value of Humanitites?',
            'value': gini_at_level_and_value('Umbrella', 'Humanities'),
        },
        'giniMedAndHlth': {
            'comment': 'what is the gini value of MedicineAndHealth?',
            'value': gini_at_level_and_value('Umbrella', 'Medicine and Health'),
        },
        'giniEng': {
            'comment': 'what is the gini value of Humanitites?',
            'value': gini_at_level_and_value('Umbrella', 'Engineering'),
        },
        'minGiniInEngFields': {
            'comment': 'what is the min gini of fields in engineering?',
            'value': gini_stats_engineering_fields()['min'],
        },
        'maxGiniInEngFields': {
            'comment': 'what is the max gini of fields in engineering?',
            'value': gini_stats_engineering_fields()['max'],
        },
        'nFieldsWhereGiniIsLessThanUmbrella': {
            'comment': 'how many fields have less unequal production than their umbrellas?',
            'value': number_of_fields_less_unequal_than_their_umbrella(),
        },
        'pFieldsWhereGiniIsLessThanUmbrella': {
            'comment': 'what percentage of fields have less unequal production than their umbrellas?',
            'value': number_of_fields_less_unequal_than_their_umbrella(get='percent'),
        },
        'nUmbrellasWhereGiniIsLessThanAcademia': {
            'comment': 'how many umbrellas have less unequal production than academia?',
            'value': number_of_umbrellas_less_unequal_than_academia(),
        },
        'pUmbrellasWhereGiniIsLessThanAcademia': {
            'comment': 'what percentage of umbrellas have less unequal production than academia?',
            'value': number_of_umbrellas_less_unequal_than_academia(get='percent'),
        },
        'nNewHireGiniLowerThanExistingHireGini': {
            'comment': 'in how many fields/domains/academia is the new hire gini < existing hire gini?',
            'value': number_of_taxonomies_where_new_hire_gini_less_than_existing(),
        },
        'nGiniChangeSlopes': {
            'comment': 'how many of the Gini OLS models were there?',
            'value': get_gini_ols_models_stat(stat='total'),
        },
        'nGiniChangeSlopesSignificant': {
            'comment': 'how many of the Gini OLS models were significant?',
            'value': get_gini_ols_models_stat(stat='significant'),
        },
        'pGiniChangeSlopesSignificant': {
            'comment': 'what percent of the Gini OLS models were significant?',
            'value': get_gini_ols_models_stat(stat='significant', get='percent'),
        },
        'nGiniChangeSlopesNegative': {
            'comment': 'how many of the significant Gini OLS models had downward slopes?',
            'value': get_gini_ols_models_stat(stat='downward'),
        },
        'pGiniChangeSlopesNegative': {
            'comment': 'how many of the significant Gini OLS models had downward slopes?',
            'value': get_gini_ols_models_stat(stat='downward', get='percent'),
        },
        'giniChangeNegativeSlopesRange': {
            'comment': 'what was the range of negative slopes for gini coefficients?',
            'value': get_gini_ols_models_stat(stat='downward', get='range'),
        },
        'giniChangeYearsToExplainNewHiresRange': {
            'comment': 'what was the range of years to explain new hires/existing hires?',
            'value': get_gini_ols_models_stat(stat='years-to-explain-differences', get='range')
        },
        
        'giniChangeYearsToExplainNewHiresMin': {
            'comment': 'what was the minimum number of years to explain new hires/existing hires?',
            'value': get_gini_ols_models_stat(stat='years-to-explain-differences', get='number', which=min)
        },
        'nFieldsProductionToAttritionSig': {
            'comment': 'in how many fields is the logistic regression of production to attrition significant?',
            'value': get_production_to_attrition_significant_stat(),
        },
        'pFieldsProductionToAttritionSig': {
            'comment': 'in what percent of fields is the logistic regression of production to attrition significant?',
            'value': get_production_to_attrition_significant_stat(get='percent'),
        },
        'nFieldsProductionToAttritionSigAcademiaRank': {
            'comment': 'in how many fields is the logistic regression of production to attrition significant, using academia ranks?',
            'value': get_production_to_attrition_significant_stat(academia_rank_only=True),
        },
        'pFieldsProductionToAttritionSigAcademiaRank': {
            'comment': 'in what percent of fields is the logistic regression of production to attrition significant, using academia ranks?',
            'value': get_production_to_attrition_significant_stat(get='percent', academia_rank_only=True),
        },
        'minYearsToGoFromExistingHireGiniToNewHireGini': {
            'comment': 'how long would it take to go from the existing hire gini to the new hire gini at the observed rate of change?',
            'value': years_to_go_from_existing_hire_gini_to_new_hire_gini(),
        },
    }


def gender_stats_generator():
    return {
        'nMen': {
            'comment': 'how many men are there?',
            'value': count_of_gender('Male'),
        },
        'pMen': {
            'comment': 'what percent of faculty are men?',
            'value': hnelib.utils.fraction_to_percent(fraction_of_gender('Male')),
        },
        'nWomen': {
            'comment': 'how many women are there?',
            'value': count_of_gender('Female'),
        },
        'pWomen': {
            'comment': 'what percent of faculty are women?',
            'value': hnelib.utils.fraction_to_percent(fraction_of_gender('Female')),
        },
        'pGenderAnnotated': {
            'comment': 'what percent of faculty had annotations?',
            'value': fraction_of_gender_annotated(),
        },
        'pGenderUnknown': {
            'comment': 'what percent of faculty had an unknown gender?',
            'value': fraction_of_gender_not_known(),
        },
        'pGenderKnown': {
            'comment': 'what percent of faculty had a known gender?',
            'value': fraction_of_gender_known(),
        },
        'pMenInEng': {
            'comment': 'what percent of engineering faculty are men?',
            'value': percent_of_gender_in_domain('Male', 'Engineering'),
        },
        'pMenInEd': {
            'comment': 'what percent of education faculty are men?',
            'value': percent_of_gender_in_domain('Male', 'Education'),
        },
        'pWomenInEng': {
            'comment': 'what percent of engineering faculty are women?',
            'value': percent_of_gender_in_domain('Female', 'Engineering'),
        },
        'pWomenInEd': {
            'comment': 'what percent of education faculty are women?',
            'value': percent_of_gender_in_domain('Female', 'Education'),
        },
        'rangeOfFieldsOfWomenInEd': {
            'comment': 'what is the range of percents of women in education fields?',
            'value': range_of_percents_of_gender_in_fields_within_domain('Female', 'Education'),
        },
        'rangeOfFieldsOfWomenInEng': {
            'comment': 'what is the range of percents of women in Engineering fields?',
            'value': range_of_percents_of_gender_in_fields_within_domain('Female', 'Engineering'),
        },
        'rangeOfFieldsOfWomenInHum': {
            'comment': 'what is the range of percents of women in Humanities fields?',
            'value': range_of_percents_of_gender_in_fields_within_domain('Female', 'Humanities'),
        },
        'pFemaleSTEM': {
            'comment': 'what percentage of STEM faculty are female?',
            'value': percent_female_in_stem(),
        },
        'nMoreFemaleJuniorFacultyThanSenior': {
            'comment': 'how many fields have more junior women than senior women?',
            'value': get_number_of_fields_with_more_junior_women_than_senior_women(),
        },
        'pMoreFemaleJuniorFacultyThanSenior': {
            'comment': 'what percentage of fields have more junior women than senior women?',
            'value': get_number_of_fields_with_more_junior_women_than_senior_women(get='percent'),
        },
        'fieldsWhereLessWomenThanAtStart': {
            'comment': 'what fields is there a lower % of women than at the start?',
            'value': places_where_representation_of_women_is_increasing(get='string', level='Field', increasing=False),
        },
        'nFieldsWhereMoreWomenThanAtStart': {
            'comment': 'in how many fields is there a greater percentage of women than at the start?',
            'value': places_where_representation_of_women_is_increasing(level='Field'),
        },
        'nFieldsWhereFewerWomenThanAtStart': {
            'comment': 'in how many fields is there a smaller percentage of women than at the start?',
            'value': places_where_representation_of_women_is_increasing(level='Field'),
        },
        'pFieldsWhereMoreWomenThanAtStart': {
            'comment': 'in what percent of fields is there a greater percentage of women than at the start?',
            'value': places_where_representation_of_women_is_increasing(get='percent', level='Field'),
        },
        'pWomenNewHiresAcademia': {
            'comment': 'what is the % women among new hires in academia?',
            'value': percent_female_new_hires('Academia', value='Academia'),
        },
        'minDomainMinorityWomenNewHires': {
            'comment': 'what is the smallest minority women new hires in a domain?',
            'value': percent_female_new_hires('Umbrella', which=min),
        },
        'minDomainMinorityWomenNewHiresString': {
            'comment': 'what domain has the smallest minority women new hires?',
            'value': percent_female_new_hires('Umbrella', which=min, get='string'),
        },
        'secondMinDomainMinorityWomenNewHires': {
            'comment': 'what is the second smallest minority women new hires in a domain?',
            'value': percent_female_new_hires('Umbrella', which=min, iloc=1),
        },
        'secondMinDomainMinorityWomenNewHiresString': {
            'comment': 'what domain has the second smallest minority women new hires?',
            'value': percent_female_new_hires('Umbrella', which=min, get='string', iloc=1),
        },
        'nPlacesMinorityWomenNewHires': {
            'comment': 'how many places are minority women new hires?',
            'value': minority_women_places(),
        },
        'pPlacesMinorityWomenNewHires': {
            'comment': 'what % of places are minority women new hires?',
            'value': minority_women_places(get='percent'),
        },
        'nDomainsMinorityWomenNewHires': {
            'comment': 'how many domains are minority women new hires?',
            'value': minority_women_places(level='Umbrella'),
        },
        'pDomainsMinorityWomenNewHires': {
            'comment': 'what % of domains are minority women new hires?',
            'value': minority_women_places(get='percent', level='Umbrella'),
        },
        'nFieldsMinorityWomenNewHires': {
            'comment': 'how many fields are minority women new hires?',
            'value': minority_women_places(level='Field'),
        },
        'pFieldsMinorityWomenNewHires': {
            'comment': 'what % of fields are minority women new hires?',
            'value': minority_women_places(get='percent', level='Field'),
        },
        'nGenderChangeSlopes': {
            'comment': 'how many of the Gender OLS models were there?',
            'value': get_gender_ols_models_stat(stat='total'),
        },
        'nGenderChangeSlopesSignificant': {
            'comment': 'how many of the gender OLS models were significant?',
            'value': get_gender_ols_models_stat(stat='significant'),
        },
        'pGenderChangeSlopesSignificant': {
            'comment': 'what percent of the gender OLS models were significant?',
            'value': get_gender_ols_models_stat(stat='significant', get='percent'),
        },
        'nGenderChangeSlopesNegative': {
            'comment': 'how many of the significant gender OLS models had downward slopes?',
            'value': get_gender_ols_models_stat(stat='downward'),
        },
        'pGenderChangeSlopesNegative': {
            'comment': 'how many of the significant gender OLS models had downward slopes?',
            'value': get_gender_ols_models_stat(stat='downward', get='percent'),
        },
        'genderChangeNegativeSlopesRange': {
            'comment': 'what was the range of downward slopes?',
            'value': get_gender_ols_models_stat(stat='downward', get='range'),
        },
        'genderChangeYearsToExplainNewHiresAcademia': {
            'comment': 'what was the minimum number of years to explain new hires/existing hires?',
            'value': get_gender_ols_models_stat(stat='years-to-explain-differences', get='number', which=min, level='Academia')
        },
        'genderChangeYearsToExplainNewHiresRangeDomain': {
            'comment': 'what was the range of years to explain new hires/existing hires?',
            'value': get_gender_ols_models_stat(stat='years-to-explain-differences', get='range', level='Umbrella'),
        },
        'genderChangeYearsToExplainNewHiresDomainMin': {
            'comment': 'what was the minimum number of years to explain new hires/existing hires?',
            'value': get_gender_ols_models_stat(stat='years-to-explain-differences', get='number', which=min, level='Umbrella')
        },
        'genderChangeYearsToExplainNewHiresDomainMinString': {
            'comment': 'what was the minimum number of years to explain new hires/existing hires?',
            'value': get_gender_ols_models_stat(stat='years-to-explain-differences', get='string', which=min, level='Umbrella')
        },
        'genderChangeYearsToExplainNewHiresDomainMax': {
            'comment': 'what was the max number of years to explain new hires/existing hires?',
            'value': get_gender_ols_models_stat(stat='years-to-explain-differences', get='number', which=max, level='Umbrella')
        },
        'genderChangeYearsToExplainNewHiresDomainMSxtring': {
            'comment': 'what was the minimum number of years to explain new hires/existing hires?',
            'value': get_gender_ols_models_stat(stat='years-to-explain-differences', get='string', which=max, level='Umbrella')
        },
        'genderNewHiresSlopesRange': {
            'comment': 'what was the range of slopes for new hires gender stuff?',
            'value': get_gender_ols_models_stat(stat='slopes', get='range', by_new_hire=True)
        },
        'nGenderNewHiresChangeSlopesSignificant': {
            'comment': 'how many of the gender OLS models were significant for new hires?',
            'value': get_gender_ols_models_stat(stat='significant', by_new_hire=True),
        },
        'pGenderNewHiresChangeSlopesSignificant': {
            'comment': 'what percent of the gender OLS models were significant for new hires?',
            'value': get_gender_ols_models_stat(stat='significant', get='percent', by_new_hire=True),
        },
        'nGenderNewHiresChangeSlopesSignificantField': {
            'comment': 'how many of the gender OLS models were significant for new hires (field level)?',
            'value': get_gender_ols_models_stat(stat='significant', by_new_hire=True, level='Field'),
        },
        'pGenderNewHiresChangeSlopesSignificantField': {
            'comment': 'what percent of the gender OLS models were significant for new hires? (field level)',
            'value': get_gender_ols_models_stat(stat='significant', get='percent', by_new_hire=True, level='Field'),
        },
        'nGenderNewHiresChangeSlopesNotSignificantField': {
            'comment': 'how many of the gender OLS models were not significant for new hires (field level)?',
            'value': get_gender_ols_models_stat(stat='significant', by_new_hire=True, level='Field', significant=False),
        },
        'pGenderNewHiresChangeSlopesNotSignificantField': {
            'comment': 'what percent of the gender OLS models were not significant for new hires? (field level)',
            'value': get_gender_ols_models_stat(stat='significant', get='percent', by_new_hire=True, level='Field', significant=False),
        },
        'nGenderChangeSlopesField': {
            'comment': 'how many of the Gender OLS models were there (field level)?',
            'value': get_gender_ols_models_stat(stat='total', level='Field'),
        },

        'pEducDomainGenderStart': {
            'comment': 'what % women does education start at?',
            'value': get_start_or_end_gender_stat('Start', level='Umbrella', value="Education"),
        },
        'pEducDomainGenderEnd': {
            'comment': 'what % women does education end at?',
            'value': get_start_or_end_gender_stat('End', level='Umbrella', value="Education"),
        },

        'pEngDomainGenderStart': {
            'comment': 'what % women does engineering start at?',
            'value': get_start_or_end_gender_stat('Start', level='Umbrella', value="Engineering"),
        },
        'pEngDomainGenderEnd': {
            'comment': 'what % women does Engineering end at?',
            'value': get_start_or_end_gender_stat('End', level='Umbrella', value="Engineering"),
        },
        'nDomainsStartEndGenderDiffSig': {
            'comment': 'how many domains have significant gender differences overall?',
            'value': places_where_gender_parity_is_increasing(level='Umbrella'),
        },
        'nFieldsStartEndGenderDiffSig': {
            'comment': 'how many fields have significant gender differences overall?',
            'value': places_where_gender_parity_is_increasing(level='Field'),
        },
        'pFieldsStartEndGenderDiffSig': {
            'comment': 'what percent of fields have significant gender differences overall?',
            'value': places_where_gender_parity_is_increasing(level='Field', get='percent'),
        },
        'nFieldsGenderParityIncSig': {
            'comment': 'how many fields have significant and increasing gender parity?',
            'value': places_where_gender_parity_is_increasing(level='Field', parity_direction='increasing'),
        },
        'pFieldsGenderParityIncSig': {
            'comment': 'what percent of fields have significant and increasing gender parity?',
            'value': places_where_gender_parity_is_increasing(level='Field', parity_direction='increasing', get='percent'),
        },
        'nFieldsGenderParityDecSig': {
            'comment': 'how many fields have significant and decreasing gender parity?',
            'value': places_where_gender_parity_is_increasing(level='Field', parity_direction='decreasing'),
        },
        'pFieldsGenderParityDecSig': {
            'comment': 'what percent of fields have significant and decreasing gender parity?',
            'value': places_where_gender_parity_is_increasing(level='Field', parity_direction='decreasing', get='percent'),
        },
    }


def closedness_stats_generator():
    return {
        'pFacWhoHoldAPhD': {
            'comment': 'what percent of faculty hold a PhD?',
            'value': percent_of_faculty_who_hold_a_phd(),
        },
        'pMaxNonPhDsDomain': {
            'comment': 'what is the maximum % of people in a domain without a PhD?',
            'value': percent_faculty_without_a_phd(get='percent', which=max),
        },
        'sMaxNonPhDsDomain': {
            'comment': 'which domain has the greatest % of people without a PhD?',
            'value': percent_faculty_without_a_phd(get='string', which=max),
        },
        'pMinNonPhDsDomain': {
            'comment': 'what is the min % of people in a domain without a PhD?',
            'value': percent_faculty_without_a_phd(get='percent', which=min),
        },
        'sMinNonPhDsDomain': {
            'comment': 'which domain has the smallest  % of people without a PhD?',
            'value': percent_faculty_without_a_phd(get='string', which=min),
        },
        'pFacFromOutsideTheUS': {
            'comment': 'what percent of faculty come from outside the US?',
            'value': faculty_from_outside_the_us(get='percent'),
        },
        'nFacFromOutsideTheUS': {
            'comment': 'how many faculty come from outside the US?',
            'value': faculty_from_outside_the_us(),
        },
        'pInFieldFacAtFieldLvl': {
            'comment': 'what percent of faculty are hired in-field, at the field level?',
            'value': percent_of_in_field_faculty_at_level('Field'),
        },
        'pOutOfFieldFacAtFieldLvl': {
            'comment': 'what percent of faculty are hired out-of-field, at the field level?',
            'value': percent_of_out_of_field_faculty_at_level('Field'),
        },
        'pInFieldFacAtDomainLvl': {
            'comment': 'what percent of faculty are hired in-field, at the domain level?',
            'value': percent_of_in_field_faculty_at_level('Umbrella'),
        },
        'pOutOfFieldFacAtDomainLvl': {
            'comment': 'what percent of faculty are hired out-of-field, at the domain level?',
            'value': percent_of_out_of_field_faculty_at_level('Umbrella'),
        },
        'inFieldToOutOfFieldOddsAtFieldLvl': {
            'comment': 'how much more likely are you to be hired in-field than out of field, at the field level?',
            'value': in_field_to_out_of_field_odds('Field'),
        },
        'inFieldToOutOfFieldOddsAtDomainLvl': {
            'comment': 'how much more likely are you to be hired in-field than out of field, at the domain level?',
            'value': in_field_to_out_of_field_odds('Umbrella'),
        },
        'pForeignPhDsEd': {
            'comment': 'how many PhDs are from outside the US in education?',
            'value': percent_non_us_at_level_and_value('Umbrella', 'Education')
        },
        'pForeignPhDsNatSci': {
            'comment': 'how many PhDs are from outside the US in natural sciences?',
            'value': percent_non_us_at_level_and_value('Umbrella', 'Natural Sciences')
        },
        'pMaxNonPhDsInNonArtsHumFields': {
            'comment': 'what is the max percent out of field for non-arts Humanities fields?',
            'value': max_percent_non_us_in_non_arts_humanities_fields(),
        },
        'pOutOfFieldPhDsFormSci': {
            'comment': 'how many PhDs are from out of field in Mathematics and Computing?',
            'value': percent_out_of_field_at_level_and_value('Umbrella', 'Mathematics and Computing'),
        },
        'pOutOfFieldPhDsEng': {
            'comment': 'how many PhDs are from out of field in engineering?',
            'value': percent_out_of_field_at_level_and_value('Umbrella', 'Engineering'),
        },
        'pOutOfFieldPhDsEd': {
            'comment': 'how many PhDs are from out of field in Education?',
            'value': percent_out_of_field_at_level_and_value('Umbrella', 'Education'),
        },
        'pValOfProportionalityBWOutOfFieldAndNonUSAtDomain': {
            'comment': 'what is the p value of the difference in proportions between out of field and non US PhD production?',
            'value': p_value_of_proportionality_between_out_of_field_and_non_us_at_level('Umbrella'),
        },
        'pPeopleOutOfSample': {
            'comment': 'what percentage of people were at U.S. institutions but not in sample?',
            'value': people_out_of_us_sample(),
        },
        'nPeopleOutOfSample': {
            'comment': 'how many people were at U.S. institutions but not in sample?',
            'value': people_out_of_us_sample(get='number'),
        },
        'pInstitutionsInSample': {
            'comment': 'what percentage of employing institutions had production records?',
            'value': institutions_out_of_sample(get='percent', to_sample='employing'),
        },
        'pEmployingInstitutionsOutOfSample': {
            'comment': 'what percentage of employing institutions were out of sample?',
            'value': institutions_out_of_sample(get='percent', to_sample='employing not producing'),
        },
        'nEmployingInstitutionsOutOfSample': {
            'comment': 'how many employing institutions were out of sample?',
            'value': institutions_out_of_sample(get='number', to_sample='employing not producing'),
        },
        'pProducingInstitutionsOutOfSample': {
            'comment': 'what percentage of producing institutions lacked employment records?',
            'value': institutions_out_of_sample(get='percent', to_sample='producing not employing'),
        },
        'nProducingInstitutionsOutOfSample': {
            'comment': 'how many employing institutions lacked employment records?',
            'value': institutions_out_of_sample(get='number', to_sample='producing not employing'),
        },
    }


def get_non_doctorate_stat(level=None, value=None, get='percent'):
    df = usfhn.stats.runner.get('non-doctorate/df')
    df = usfhn.views.filter_by_taxonomy(df, level=level, value=value)

    row = df.iloc[0]

    return get_number_or_percent(
        get=get,
        numerator=row['NonDoctorates'],
        denominator=row['Faculty'],
    )



def non_doctorate_stats_generator():
    return {
        'pNonDoctoratesFieldEnglish': {
            'comment': "what % of faculty in field English don't have a doctorate?",
            'value': get_non_doctorate_stat(level='Field', value='English Language and Literature'),
        },
        'pNonDoctoratesFieldMusic': {
            'comment': "what % of faculty in field Music don't have a doctorate?",
            'value': get_non_doctorate_stat(level='Field', value='Music'),
        },
        'pNonDoctoratesFieldArtHist': {
            'comment': "what % of faculty in field Art History don't have a doctorate?",
            'value': get_non_doctorate_stat(level='Field', value='Art History and Criticism'),
        },
        'pNonDoctoratesFieldTheatre': {
            'comment': "what % of faculty in field Theatre don't have a doctorate?",
            'value': get_non_doctorate_stat(level='Field', value='Theatre Literature, History and Criticism'),
        },
    }


def non_us_stats_generator():
    return {
        'nFieldsHigherNonUSAttrition': {
            'comment': 'in how many fields is the attrition risk higher for non us?',
            'value': non_us_attrition_risk_stat(),
        },
        'pFieldsHigherNonUSAttrition': {
            'comment': 'in what percent of fields is the attrition risk higher for non us?',
            'value': non_us_attrition_risk_stat(get='percent'),
        },
        'nFieldsHigherNonUSAttritionNonCanNonUK': {
            'comment': 'in how many fields is the attrition risk higher for non us (excluding canada/uk)?',
            'value': highly_productive_attrition_risk_stat(),
        },
        'pFieldsHigherNonUSAttritionNonCanNonUK': {
            'comment': 'in what percent of fields is the attrition risk higher for non us (excluding canada/uk)?',
            'value': highly_productive_attrition_risk_stat(get='percent'),
        },
        'pNonUSNubbins': {
            'comment': 'what % of non us faculty came from non-US/UK/canada?',
            'value': non_us_nubbins_stat(),
        },
        'pEnglishNubbins': {
            'comment': 'what % of non us faculty came from UK/Canada america/NA-non-canada?',
            'value': english_nubbins_stat(),
        },
        'nNonUSNubbins': {
            'comment': 'what # of non us faculty came from non-US/UK/canada?',
            'value': non_us_nubbins_stat(get='number'),
        },
        'nEnglishNubbins': {
            'comment': 'what # of non us faculty came from UK/Canada america/NA-non-canada?',
            'value': english_nubbins_stat(get='number'),
        },
    }

def prestige_stats_generator():
    return {
        'pSmallestViolations': {
            'comment': 'what is the smallest percent of violations across any level?',
            'value': min_violations_percent(),
        },
        'pLargestViolations': {
            'comment': 'what is the largest percent of violations across any level?',
            'value': max_violations_percent(),
        },
        'pViolationsHum': {
            'comment': 'how steep are the Humanities?',
            'value': percent_steepness_at_level_and_value('Umbrella', 'Humanities'),
        },
        'pViolationsFormSci': {
            'comment': 'how steep are the Mathematics and Computing?',
            'value': percent_steepness_at_level_and_value('Umbrella', 'Mathematics and Computing'),
        },
        'pViolationsMedAndHlth': {
            'comment': 'how steep is Medicine and Health?',
            'value': percent_steepness_at_level_and_value('Umbrella', 'Medicine and Health'),
        },
        'pMeanMovementOnHierarchy': {
            'comment': 'where on the hierarchy are compared to your advisor?',
            'value': mean_movement_on_hierarchy(),
        },
        'medianMovementOnHierarchy': {
            'comment': 'where on the hierarchy are compared to your advisor?',
            'value': median_movement_on_hierarchy(),
        },
        'meanProdDiffBWYouAndAdvisor': {
            'comment': 'how many fewer faculty are you expected to advise than your advisor?',
            'value': mean_production_difference_from_advisor(),
        },
        'pLargestMovementDownward': {
            'comment': 'how far is the largest mean movement downward?',
            'value': mean_movement_in_direction('down', 'max'),
        },
        'largestMovementDownwardField': {
            'comment': 'what field do people move the most downward?',
            'value': mean_movement_in_direction('down', 'max', get='field'),
        },
        'pSmallestMovementDownward': {
            'comment': 'how far is the smallest mean movement downward?',
            'value': mean_movement_in_direction('down', 'min')
        },
        'smallestMovementDownwardField': {
            'comment': 'what field do people move the most downward?',
            'value': mean_movement_in_direction('down', 'min', get='field'),
        },
        'pLargestMovementUpward': {
            'comment': 'how far is the largest mean movement upward?',
            'value': mean_movement_in_direction('up', 'max')
        },
        'largestMovementUpwardField': {
            'comment': 'what field do people move the most upward?',
            'value': mean_movement_in_direction('up', 'max', get='field'),
        },
        'pSmallestMovementUpward': {
            'comment': 'how far is the smallest mean movement upward?',
            'value': mean_movement_in_direction('up', 'min')
        },
        'smallestMovementUpwardField': {
            'comment': 'what field do people move the most upward?',
            'value': mean_movement_in_direction('up', 'min', get='field'),
        },
        'nFieldsWithSignificantDifferencesInMobility': {
            'comment': 'how many fields have significant differences between men and women for movement up/down?',
            'value': get_mobility_stat(significant=True),
        },
        'nFieldsWithoutSignificantDifferencesInMobility': {
            'comment': 'how many fields did not have significant differences between men and women for movement up/down?',
            'value': get_mobility_stat(significant=False),
        },
        'nFieldsWithSignificantDifferencesInMedAndHlth': {
            'comment': 'how many fields have significant differences between men and women for movement up/down, in medicine and health?',
            'value': get_mobility_stat(significant=True, level='Field', domain='Medicine and Health'),
        },
        'pViolationsAcademia': {
            'comment': 'what percent of people move up the hierarchy at the academia level?',
            'value': p_movement_at_level_and_value(direction='up'),
        },
        'pSelfHiresAcademia': {
            'comment': 'what percent of people work at their own institution at the academia level?',
            'value': p_movement_at_level_and_value(direction='self-hires'),
        },
        'pDownwardAcademia': {
            'comment': 'what percent of people move down the hierarchy at the academia level?',
            'value': p_movement_at_level_and_value(direction='down'),
        },
    }

def null_model_stats_generator():
    return {
        'nCMAlwaysSig': {
            'comment': 'how many hierarchies are always more hierarchical than the null model?',
            'value': get_null_models_stat(max_violations=0),
        },
        'pCMAlwaysSig': {
            'comment': 'what percent of hierarchies are always more hierarchical than the null model?',
            'value': get_null_models_stat(max_violations=0, get='percent'),
        },
        'nCMAlwaysSigField': {
            'comment': 'how many field-level hierarchies are always more hierarchical than the null model?',
            'value': get_null_models_stat(level='Field', max_violations=0),
        },
        'pCMAlwaysSigField': {
            'comment': 'what % of field-level hierarchies are always more hierarchical than the null model?',
            'value': get_null_models_stat(level='Field', max_violations=0, get='percent'),
        },
        'nCMSigPPointZeroOneField': {
            'comment': 'how many hierarchies are more hierarchical than 99% of null models?',
            'value': get_null_models_stat(level='Field', max_violations=10),
        },
        'pCMSigPPointZeroOneField': {
            'comment': 'what percent of hierarchies are always more hierarchical than 99% of null models?',
            'value': get_null_models_stat(level='Field', max_violations=10, get='percent'),
        },
        'nCMSigPPointZeroFiveField': {
            'comment': 'how many field-level hierarchies were significantly more hierarchical at p=.05?',
            'value': get_null_models_stat(level='Field', significant=True),
        },
        'pCMSigPPointZeroFiveField': {
            'comment': 'what percent of field-level hierarchies were significantly more hierarchical at p=.05?',
            'value': get_null_models_stat(level='Field', significant=True, get='percent'),
        },
        'maxPCMFieldString': {
            'comment': 'what field had the most violations?',
            'value': get_null_models_stat(level='Field', get='string'),
        },
        'maxPCMFieldP': {
            'comment': 'what was the p value of the field that had the most violations?',
            'value': get_null_models_stat(level='Field', get='p'),
        },
        'secondMaxPCMFieldString': {
            'comment': 'what field had the 2nd most violations?',
            'value': get_null_models_stat(level='Field', iloc=1, get='string'),
        },
        'secondMaxPCMFieldP': {
            'comment': 'what was the p value of the field that had the 2nd most violations?',
            'value': get_null_models_stat(level='Field', iloc=1, get='p'),
        },
        'thirdMaxPCMFieldString': {
            'comment': 'what field had the 3jd most violations?',
            'value': get_null_models_stat(level='Field', iloc=2, get='string'),
        },
        'thirdMaxPCMFieldP': {
            'comment': 'what was the p value of the field that had the 3rd most violations?',
            'value': get_null_models_stat(level='Field', iloc=2, get='p'),
        },
    }


def field_correlations_stats_generator():
    return {
        ################################################################################
        # Prestige
        ################################################################################
        'nFToFCorrelations': {
            'comment': 'how many field to field correlations are there?',
            'value': get_field_to_field_correlations_stat(threshold_type=None),
        },
        'nPositiveFToFCorrelations': {
            'comment': 'how many positive field to field correlations are there?',
            'value': get_field_to_field_correlations_stat(),
        },
        'nNegativeFToFCorrelations': {
            'comment': 'how many negative field to field correlations are there?',
            'value': get_field_to_field_correlations_stat(threshold=0, threshold_type='below'),
        },
        'pPositiveFToFCorrelations': {
            'comment': 'what percent of field to field correlations are positive?',
            'value': get_field_to_field_correlations_stat(get='percent', round_to=1),
        },
        'nOverSeventyFToFCorrelations': {
            'comment': 'how many positive field to field correlations are there?',
            'value': get_field_to_field_correlations_stat(threshold=.7),
        },
        'pOverSeventyFToFCorrelations': {
            'comment': 'what percent of field to field correlations are positive?',
            'value': get_field_to_field_correlations_stat(threshold=.7, get='percent', round_to=1),
        },
        'nNegativeFToFCorrelations': {
            'comment': 'how many field to field correlations are negative?',
            'value': get_field_to_field_correlations_stat(threshold_type='below'),
        },
        'meanPathologyCorrelation': {
            'comment': 'what is the mean correlation between fields and Pathology?',
            'value': mean_correlation_of_field_to_field('Pathology'),
        },
        ################################################################################
        # Production
        ################################################################################
        'nPositiveFToFCorrelationsProd': {
            'comment': 'how many positive field to field correlations are there (production)?',
            'value': get_field_to_field_correlations_stat(rank_type='production'),
        },
        'pPositiveFToFCorrelationsProd': {
            'comment': 'what percent of field to field correlations are positive (production)?',
            'value': get_field_to_field_correlations_stat(rank_type='production', get='percent', round_to=1),
        },
        'pOverFiftyFToFCorrelationsProd': {
            'comment': 'what percent of field to field correlations are above .5 (production)?',
            'value': get_field_to_field_correlations_stat(rank_type='production', threshold=.7, get='percent', round_to=1),
        },
    }


def monopolization_stats_generator():
    return {
        'namesOfTopFiveInstitutionalProducersOfTopTenDepts': {
            'comment': 'what are the names of the top five institutional producers of top-10 departments?',
            'value': get_names_of_top_five_institutional_producers_of_top_ten_departments(),
        },
        'nTopTenDepts': {
            'comment': 'how many top 10 departments are there?',
            'value': number_of_top_10_departments(),
        },
        'nZeroTopTenDepts': {
            'comment': 'how many institutions have zero top 10 departments?',
            'value': number_of_institutions_without_top_10_departments(),
        },
        'pZeroTopTenDepts': {
            'comment': 'what percent of institutions have zero top 10 departments?',
            'value': percent_of_institutions_without_top_10_departments(),
        },
        'pFiveInstitutions': {
            'comment': 'what percent of all institutions is 5?',
            'value': percent_of_institutions(5),
        },
        'nTopTenDeptsAtTopFiveInsts': {
            'comment': 'how many top-10 departments are at top 5 institutions?',
            'value': top_10_departments_at_top_five_insts(),
        },
        'pTopTenDeptsAtTopFiveInsts': {
            'comment': 'what percent of top-10 departments are at top 5 institutions?',
            'value': top_10_departments_at_top_five_insts(get='percent'),
        },
        'nInstsWithOneTopTenDept': {
            'comment': 'how many institutions have 1 top 10 department?',
            'value': number_of_institutions_with_1_top_10_department(),
        },
        'pInstsWithOneTopTenDept': {
            'comment': 'what percent of institutions have 1 top 10 department?',
            'value': percent_of_institutions_with_1_top_10_department(),
        },
        'nTopTenDeptsFromTopFiveInsts': {
            'comment': 'how many top 10 departments are from top 5 schools?',
            'value': number_of_top_10_departments_from_top_five_schools(),
        },
        'pTopTenDeptsFromTopFiveInsts': {
            'comment': 'what percent of top 10 departments belong to the top 5 schools?',
            'value': percent_of_top_10_departments_from_top_five_schools(),
        },
        'pTopTenDepartmentsNotFromTopFive': {
            'comment': 'what percent of top 10 departments come from schools other than the top 5?',
            'value': percent_of_top_10_departments_not_from_top_five_schools(),
        },
        'nInstsWithTopTenDeptsButNotTopFive': {
            'comment': 'how many institutions have a top ten department, excluding the top 5?',
            'value': number_of_institutions_with_a_top_10_department_but_not_top_5(),
        },
        'pInstsWithTopTenDeptsButNotTopFive': {
            'comment': 'what percentage of institutions have a top ten department, excluding the top 5?',
            'value': percent_of_institutions_with_a_top_10_department_but_not_top_5(),
        },
        'nInstsWithMoreThanOneTopTenDeptButNotTopFive': {
            'comment': 'how many institutions have more than one top ten department, excluding the top 5?',
            'value': number_of_institutions_with_more_than_one_top_10_department_but_not_top_5(),
        },
        'pInstsWithMoreThanOneTopTenDeptButNotTopFive': {
            'comment': 'what percent of institutions have more than one top ten department, excluding the top 5?',
            'value': percent_of_institutions_with_more_than_one_top_10_department_but_not_top_5(),
        },
        'pInstsWithTopTenDeptsAcademia': {
            'comment': 'what is the percent of institutions have a top 10 department?',
            'value': percent_of_institutions_with_a_top_10_department_academia(),
        },
        'nInstsWithTopTenDeptsAcademia': {
            'comment': 'what is the number of institutions have a top 10 department?',
            'value': number_of_institutions_with_a_top_10_department_academia(),
        },
        'maxPInstsWithTopTenDeptsDomainLevel': {
            'comment': 'what is the largest percent of institutions in a domain that make all the top 10 departments?',
            'value': percent_of_institutions_with_top_10_domain_level(function='max'),
        },
        'domainWithMaxPInstsWithTopTenDepts': {
            'comment': 'what domain has the largest percent of institutions in a domain that make all the top 10 departments?',
            'value': percent_of_institutions_with_top_10_domain_level(function='max', get='name'),
        },
        'maxNInstsWithTopTenDepts': {
            'comment': 'what is the largest number of institutiosn in a domain that have a top 10 departments?',
            'value': percent_of_institutions_with_top_10_domain_level(function='max', get='number'),
        },
        'minPInstsWithTopTenDeptsDomainLevel': {
            'comment': 'what is the smallest percent of institutions in a domain that make all the top 10 departments?',
            'value': percent_of_institutions_with_top_10_domain_level(function='min'),
        },
        'minNInstsWithTopTenDepts': {
            'comment': 'what is the smallest number of institutions in a domain that have a top 10 departments?',
            'value': percent_of_institutions_with_top_10_domain_level(function='min', get='number'),
        },
        'domainWithMinPInstsWithTopTenDepts': {
            'comment': 'what domain has the smallest percent of institutions in a domain that make all the top 10 departments?',
            'value': percent_of_institutions_with_top_10_domain_level(function='max', get='name'),
        },
    }

def self_hiring_stats_generator():
    return {
        'nSelfHires': {
            'comment': 'how many people are self hires?',
            'value': self_hire_rate_at_level_and_value_and_gender(get='number'),
        },
        'pSelfHires': {
            'comment': 'what is the overall self hire rate?',
            'value': self_hire_rate_at_level_and_value_and_gender(),
        },
        'nNonSelfHires': {
            'comment': 'how many people arent self hires?',
            'value': self_hire_rate_at_level_and_value_and_gender(get='number', column='NonSelfHires'),
        },
        'pSelfHiresSocSci': {
            'comment': 'what is the self hire rate in the social sciences?',
            'value': self_hire_rate_at_level_and_value_and_gender(level='Umbrella', value='Social Sciences'),
        },
        'pSelfHiresHum': {
            'comment': 'what is the self hire rate in the Humanities?',
            'value': self_hire_rate_at_level_and_value_and_gender(level='Umbrella', value='Humanities'),
        },
        'pSelfHiresMedAndHlth': {
            'comment': 'what is the self hire rate in Medicine and Health?',
            'value': self_hire_rate_at_level_and_value_and_gender(level='Umbrella', value='Medicine and Health'),
        },
        'nFieldsSelfHiringLessThanTopFive': {
            'comment': 'how many fields self hired less than their top 5 departments?',
            'value': number_of_fields_self_hiring_less_than_top_5_departments(),
        },
        'pSelfHiresWomen': {
            'comment': 'what percent of women are self hires?',
            'value': self_hire_rate_at_level_and_value_and_gender(gender='Female'),
        },
        'pSelfHireMen': {
            'comment': 'what percent of men are self hires?',
            'value': self_hire_rate_at_level_and_value_and_gender(gender='Male'),
        },
        'nFieldsSignDiffAtSH': {
            'comment': 'how many fields self hire significantly differently by gender?',
            'value': get_self_hiring_differs_by_gender_stat(),
        },
        'pFieldsSignDiffAtSH': {
            'comment': 'what percent of fields self hire significantly differently by gender?',
            'value': get_self_hiring_differs_by_gender_stat(get='percent'),
        },
        'pFieldsSignDiffAtSH': {
            'comment': 'what percent of fields self hire significantly differently by gender?',
            'value': get_self_hiring_differs_by_gender_stat(get='percent'),
        },
        'maxSHPAtFieldLevel': {
            'comment': 'what is the maximum p value for self hiring rates at the field level?',
            'value': get_self_hiring_differs_by_gender_stat(get='p'),
        },
        'nFieldsSignDiffAtSHWomen': {
            'comment': 'how many fields self hire significantly differently by gender and hire more women?',
            'value':  get_self_hiring_differs_by_gender_stat(women_more_than_men=True),
        },
        'nFieldsSignDiffAtSHWomenMedAndHealth': {
            'comment': 'how many fields in medicine and health self hire significantly differently by gender and hire more women?',
            'value': get_self_hiring_differs_by_gender_stat(
                women_more_than_men=True,
                domain='Medicine and Health',
            ),
        },
        'minActualOverExpectedSH': {
            'comment': 'what is the minimum value for the ratio of actual/expected self hires?',
            'value': self_hiring_actual_over_expected(function='min'),
        },
        'minActualOverExpectedSHField': {
            'comment': 'what is the field with the minimum value for the ratio of actual/expected self hires?',
            'value': self_hiring_actual_over_expected(function='min', value='name'),
        },
        'maxActualOverExpectedSH': {
            'comment': 'what is the max value for the ratio of actual/expected self hires?',
            'value': self_hiring_actual_over_expected(function='max'),
        },
        'maxActualOverExpectedSHField': {
            'comment': 'what is the field with the max value for the ratio of actual/expected self hires?',
            'value': self_hiring_actual_over_expected(function='max', value='name'),
        },
        'SHCorrelationWithPrestigePearson': {
            'comment': 'how correlated is self-hiring with prestige?',
            'value': self_hiring_correlation(level='Academia', value='Academia', get='pearson'),
        },
        'SHCorrelationWithPrestigeP': {
            'comment': 'what is the p value of the correlation between self-hiring and prestige?',
            'value': self_hiring_correlation(level='Academia', value='Academia', get='p'),
        },
        'nFieldsWherePrestigiousInstsSelfHireMoreThanOthers': {
            'comment': 'in how many fields do the top institutions self-hire more than the rest?',
            'value': fields_where_prestigeous_institutions_self_hire_more(),
        },
        'pFieldsWherePrestigiousInstsSelfHireMoreThanOthers': {
            'comment': 'in what percentage of fields do the top institutions self-hire more than the rest?',
            'value': fields_where_prestigeous_institutions_self_hire_more(get='percent'),
        },
    }

def time_stats_generator():
    return {
        'professorsStart': {
            'comment': 'how many professors were there at the start?',
            'value': get_professor_count_at_point(when='start'),
        },
        'professorsEnd': {
            'comment': 'how many professors were there at the end?',
            'value': get_professor_count_at_point(when='end'),
        },
        'maxProfessors': {
            'comment': 'what is the max number of professors over time?',
            'value': get_professor_count(kind='max', rank='Professor'),
        },
        'minProfessors': {
            'comment': 'what is the min number of professors over time?',
            'value': get_professor_count(kind='min', rank='Professor'),
        },
        'maxAssistantProfessors': {
            'comment': 'what is the max number of assistant professors over time?',
            'value': get_professor_count(kind='max', rank='Assistant Professor'),
        },
        'minAssistantProfessors': {
            'comment': 'what is the min number of assistant professors over time?',
            'value': get_professor_count(kind='min', rank='Assistant Professor'),
        },
        'maxAssociateProfessors': {
            'comment': 'what is the max number of associate professors over time?',
            'value': get_professor_count(kind='max', rank='Associate Professor'),
        },
        'minAssociateProfessors': {
            'comment': 'what is the min number of associate professors over time?',
            'value': get_professor_count(kind='min', rank='Associate Professor'),
        },
        'pMaxProfessors': {
            'comment': 'what is the max percent of professors over time?',
            'value': get_professor_count(kind='max', rank='Professor', get='percent'),
        },
        'pMinProfessors': {
            'comment': 'what is the min percent of professors over time?',
            'value': get_professor_count(kind='min', rank='Professor', get='percent'),
        },
        'pMaxAssistantProfessors': {
            'comment': 'what is the max percent of assistant professors over time?',
            'value': get_professor_count(kind='max', rank='Assistant Professor', get='percent'),
        },
        'pMinAssistantProfessors': {
            'comment': 'what is the min percent of assistant professors over time?',
            'value': get_professor_count(kind='min', rank='Assistant Professor', get='percent'),
        },
        'pMaxAssociateProfessors': {
            'comment': 'what is the max percent of associate professors over time?',
            'value': get_professor_count(kind='max', rank='Associate Professor', get='percent'),
        },
        'pMinAssociateProfessors': {
            'comment': 'what is the min percent of associate professors over time?',
            'value': get_professor_count(kind='min', rank='Associate Professor', get='percent'),
        },
    }


def mcm_stats_generator():
    return {
        'nMCMSelfHiresSignificant': {
            'comment': 'in how many disciplines did mcms significantly alter the self-hiring rates',
            'value': get_mcm_significance(get='number', movement_type='Self-Hire'),
        },
        'nMCMViolationsSignificant': {
            'comment': 'in how many disciplines did mcms significantly alter the violation rates',
            'value': get_mcm_significance(get='number', movement_type='Upward'),
        },
        'pMCMViolationsFirstJobSecondJob': {
            'comment': 'how much more likely to move upwards are things post-mcm than pre?',
            'value': get_mcm_difference(level='Academia', value='Academia', movement_type='Upward'),
        },
        'MCMViolationsFirstJobSecondJobP': {
            'comment': 'what is the p value for academia upwards mcm change?',
            'value': get_mcm_significance(level='Academia', value='Academia', get='p', movement_type='Upward'),
        },
    }


def cleaning_stats_generator():
    degree_stats = json.loads(usfhn.constants.AA_2022_DEGREE_FILTERED_STATS_PATH.read_text())
    multi_inst_stats = json.loads(usfhn.constants.AA_2022_MULTI_INSTITUTION_FILTERED_STATS_PATH.read_text())
    imputed_stats = json.loads(usfhn.constants.AA_2022_PEOPLE_IMPUTED_STATS_PATH.read_text())
    primary_apt_stats = json.loads(usfhn.constants.AA_2022_PRIMARY_APPOINTED_STATS_PATH.read_text())
    mcm_stats = json.loads(usfhn.constants.AA_2022_MULTI_CAREER_MOVE_STATS_PATH.read_text())
    dept_threshold_stats = usfhn.pool_reduction.department_year_threshold_stats()
    dept_all_years_stats = usfhn.pool_reduction.get_departments_in_all_years_filtering_stats()

    n_mid_career_movers = len(usfhn.careers.get_mid_career_movers())
    n_people = int(usfhn.datasets.get_dataset_df('closedness_data')['PersonId'].nunique())
    p_mid_career_movers = n_mid_career_movers / n_people

    return {
        'nEmployingInstsAtStart': {
            'comment': 'how many institutions are there before filtering?',
            'value': len(usfhn.institutions.get_us_phd_granting_institutions()),
        },
        'nProducingInstsCleaned': {
            'comment': 'how many institutions are there before filtering?',
            'value': len(usfhn.institutions.get_us_phd_granting_institutions()),
        },
        'pDegreeInfoKnown': {
            'comment': 'what % of people do we have degrees for?',
            'value': hnelib.utils.fraction_to_percent(degree_stats['pPeopleWithDegrees'], 1),
        },
        'pKnownDegreeNonDoctorate': {
            'comment': "what % of people whose degree we know don't have a doctorate?",
            'value': hnelib.utils.fraction_to_percent(degree_stats['pNonDoctorates'], 1),
        },
        'pDegreeInfoKnownMasters': {
            'comment': "what % of people whose degree we know have a master's?",
            'value': hnelib.utils.fraction_to_percent(degree_stats['pMasters'], 1),
        },
        'pDegreeInfoKnownBachelors': {
            'comment': "what % of people whose degree we know have a bachelor's?",
            'value': hnelib.utils.fraction_to_percent(degree_stats['pBachelors'], 1),
        },
        'pMultiInstitutionRowsRemoved': {
            'comment': "what % of rows were removed for multi inst reasons?",
            'value': hnelib.utils.fraction_to_percent(multi_inst_stats['pMultiInstitutionRowsRemoved'], 2),
        },
        'pImputationRows': {
            'comment': 'what percentage of rows does imputation add?',
            'value': hnelib.utils.fraction_to_percent(imputed_stats['pImputationRows'], 1),
        },
        'pImputationPeople': {
            'comment': 'what percentage of people does imputation effect?',
            'value': hnelib.utils.fraction_to_percent(imputed_stats['pImputationPeople'], 1),
        },
        'pNonPrimaryApptRowsRemoved': {
            'comment': 'what percentage of rows does primary appointment filtering remove?',
            'value': hnelib.utils.fraction_to_percent(primary_apt_stats['pNonPrimaryApptRowsRemoved'], 1),
        },
        'pNonPrimaryApptPeopleExcluded': {
            'comment': 'what percentage of people does primary appointment filtering exclude?',
            'value': hnelib.utils.fraction_to_percent(primary_apt_stats['pNonPrimaryApptPeopleExcluded'], 1),
        },
        'pMidCareerMoveRowsRemoved': {
            'comment': 'what percentage of rows does mcm filtering remove?',
            'value': hnelib.utils.fraction_to_percent(mcm_stats['pMidCareerMoveRowsRemoved'], 1),
        },
        'nPeopleRemovedDeptThresholdYears': {
            'comment': 'how many people are removed by the depts threshold years filter?',
            'value': dept_threshold_stats['nPeopleRemoved'],
        },
        'pPeopleRemovedDeptThresholdYears': {
            'comment': 'what percent of people are removed by the depts threshold years filter?',
            'value': hnelib.utils.fraction_to_percent(dept_threshold_stats['pPeopleRemoved'], 1),
        },
        'nDeptsRemovedDeptThresholdYears': {
            'comment': 'how many departments are removed by the depts threshold years filter?',
            'value': dept_threshold_stats['nDepartmentsRemoved'],
        },
        'pDeptsRemovedDeptThresholdYears': {
            'comment': 'what percent of departments are removed by the depts threshold years filter?',
            'value': hnelib.utils.fraction_to_percent(dept_threshold_stats['pDepartmentsRemoved'], 1),
        },
        'nEmployingInstsRemovedDeptThresholdYears': {
            'comment': 'how many employing institutions are removed by the depts threshold years filter?',
            'value': dept_threshold_stats['nEmployingInstitutionsRemoved'],
        },
        'pEmployingInstsRemovedDeptThresholdYears': {
            'comment': 'what percent of employing institutions are removed by the depts threshold years filter?',
            'value': hnelib.utils.fraction_to_percent(dept_threshold_stats['pEmployingInstitutionsRemoved'], 1),
        },
        'nRowsRemovedDeptThresholdYears': {
            'comment': 'how many rows are removed by the depts threshold years filter?',
            'value': dept_threshold_stats['nRowsRemoved'],
        },
        'pRowsRemovedDeptThresholdYears': {
            'comment': 'what percent of rows are removed by the depts threshold years filter?',
            'value': hnelib.utils.fraction_to_percent(dept_threshold_stats['pRowsRemoved'], 1),
        },
        'nPeopleRemovedDeptsAllYears': {
            'comment': 'how many people are removed by the depts all years filter?',
            'value': dept_all_years_stats['nPeopleRemoved'],
        },
        'pPeopleRemovedDeptsAllYears': {
            'comment': 'what percent of people are removed by the depts all years filter?',
            'value': hnelib.utils.fraction_to_percent(dept_all_years_stats['pPeopleRemoved'], 1),
        },
        'nDeptsRemovedDeptsAllYears': {
            'comment': 'how many departments are removed by the depts all years filter?',
            'value': dept_all_years_stats['nDepartmentsRemoved'],
        },
        'pDeptsRemovedDeptsAllYears': {
            'comment': 'what percent of departments are removed by the depts all years filter?',
            'value': hnelib.utils.fraction_to_percent(dept_all_years_stats['pDepartmentsRemoved'], 1),
        },
        'nEmployingInstsRemovedDeptsAllYears': {
            'comment': 'how many employing institutions are removed by the depts all years filter?',
            'value': dept_all_years_stats['nEmployingInstitutionsRemoved'],
        },
        'pEmployingInstsRemovedDeptsAllYears': {
            'comment': 'what percent of employing institutions are removed by the depts all years filter?',
            'value': hnelib.utils.fraction_to_percent(dept_all_years_stats['pEmployingInstitutionsRemoved'], 1),
        },
        'nRowsRemovedDeptsAllYears': {
            'comment': 'how many rows are removed by the depts all years filter?',
            'value': dept_all_years_stats['nRowsRemoved'],
        },
        'pRowsRemovedDeptsAllYears': {
            'comment': 'what percent of rows are removed by the depts all years filter?',
            'value': hnelib.utils.fraction_to_percent(dept_all_years_stats['pRowsRemoved'], 1),
        },
        'pMidCareerMovers': {
            'comment': "what percent of people make a mid-career move?",
            'value': hnelib.utils.fraction_to_percent(p_mid_career_movers, 2)
        },
    }


def new_faculty_stats_generator():
    df = usfhn.stats.runner.get('new-hire/df')

    n_new_hires = df['PersonId'].nunique()
    n_people = usfhn.datasets.get_dataset_df('closedness_data')['PersonId'].nunique()

    return {
        'nNewFaculty': {
            'comment': 'how many new faculy are there?',
            'value': n_new_hires,
        },
        'nExistingFaculty': {
            'comment': 'how many non-new faculy are there?',
            'value': n_people - n_new_hires,
        },
        'pNewFaculty': {
            'comment': 'what % of faculty are new hires?',
            'value': hnelib.utils.fraction_to_percent(n_new_hires / n_people, 1),
        },
        'nYearsFromDegreeToJobForNewFaculty': {
            'comment': 'how many years are allowed between PhD and employment for someone to be a new faculty?',
            'value': usfhn.constants.MAX_YEARS_BETWEEN_DEGREE_AND_CAREER_START,
        },

    }


def attrition_stats_generator():
    overall_attrition = usfhn.views.filter_by_taxonomy(
        usfhn.stats.runner.get('attrition/risk/taxonomy'),
        'Academia',
    ).iloc[0]

    women_attrition = usfhn.views.filter_by_taxonomy(
        usfhn.stats.runner.get('attrition/risk/gender'),
        'Academia',
    )

    women_attrition = women_attrition[
        women_attrition['Gender'] == 'Female'
    ].iloc[0]

    attrition_age_df = usfhn.stats.runner.get('attrition/by-gender/age-of-attrition') 
    attrition_age_df = usfhn.views.filter_by_taxonomy(attrition_age_df, 'Academia')

    mean_attrition_age = attrition_age_df[
        attrition_age_df['Gender'] == usfhn.constants.GENDER_AGNOSTIC
    ].iloc[0]['MeanCareerAgeOfAttrition']

    mean_male_attrition_age = attrition_age_df[
        attrition_age_df['Gender'] == 'Male'
    ].iloc[0]['MeanCareerAgeOfAttrition']

    mean_female_attrition_age = attrition_age_df[
        attrition_age_df['Gender'] == 'Female'
    ].iloc[0]['MeanCareerAgeOfAttrition']

    return {
        'nAttritions': {
            'comment': 'how many people attrition total?',
            'value': overall_attrition['AttritionEvents'],
        },
        'nAttritionsWomen': {
            'comment': 'how many women attrition total?',
            'value': women_attrition['AttritionEvents'],
        },
        'meanAgeOfAttrition': {
            'comment': 'what is the mean age of attrition overall?',
            'value': round(mean_attrition_age),
        },
        'meanMaleAgeOfAttrition': {
            'comment': 'what is the mean age of attrition for men?',
            'value': round(mean_male_attrition_age),
        },
        'meanFemaleAgeOfAttrition': {
            'comment': 'what is the mean age of attrition for women?',
            'value': round(mean_female_attrition_age),
        },
        'nFieldsWhereSelfHiresAttritionMore': {
            'comment': 'in how many fields do self-hires attrition more?',
            'value': self_hire_attrition_stat(get='number'),
        },
        'pFieldsWhereSelfHiresAttritionMore': {
            'comment': 'in what percent of fields do self-hires attrition more?',
            'value': self_hire_attrition_stat(get='percent'),
        },
        'maxRelativeSelfHireAttritionRateMultiplier': {
            'comment': 'what is the largest ratio of self-hire/non-self-hire attrition?',
            'value': self_hire_attrition_stat(stat='ratio'),
        },
        'maxRelativeSelfHireAttritionRateField': {
            'comment': 'what field has the largest ratio of self-hire/non-self-hire attrition?',
            'value': self_hire_attrition_stat(stat='ratio', get='string'),
        },
        'secondMaxRelativeSelfHireAttritionRateMultiplier': {
            'comment': 'what is the 2nd largest ratio of self-hire/non-self-hire attrition?',
            'value': self_hire_attrition_stat(stat='ratio', iloc=1),
        },
        'secondMaxRelativeSelfHireAttritionRateField': {
            'comment': 'what field has the 2nd largest ratio of self-hire/non-self-hire attrition?',
            'value': self_hire_attrition_stat(stat='ratio', iloc=1, get='string'),
        },
        'minRelativeSelfHireAttritionRateMultiplier': {
            'comment': 'what is the smallest ratio of self-hire/non-self-hire attrition?',
            'value': self_hire_attrition_stat(stat='ratio', ascending=True),
        },
        'minRelativeSelfHireAttritionRateField': {
            'comment': 'what field has the smallest ratio of self-hire/non-self-hire attrition?',
            'value': self_hire_attrition_stat(stat='ratio', ascending=True, get='string'),
        },
        'selfHireVNonSelfHireAttritionRiskAcademia': {
            'comment': 'what is the self-hire/non-self-hire attrition risk for academia?',
            'value': self_hire_attrition_stat(stat='ratio', level='Academia'),
        },
    }


def market_share_stats_generator():
    return {

        'pMarketGrowthAcademia': {
            'comment': 'how much did academia grow?',
            'value': get_growth_stat(level='Academia', value='Academia', stat='faculty-count', change_type='growth'),
        },
        'pMarketChangeEng': {
            'comment': 'how much did the market share for Engineering change?',
            'value': get_growth_stat(level='Umbrella', value='Engineering', change_type='growth', round_to=1),
        },
        'pMarketChangeMedAndHealth': {
            'comment': 'how much did the market share for Medicine and Health change?',
            'value': get_growth_stat(level='Umbrella', value='Medicine and Health', change_type='growth', round_to=1),
        },
        'pMarketChangeHumanities': {
            'comment': 'how much did the market share for Humanities change?',
            'value': get_growth_stat(level='Umbrella', value='Humanities', change_type='loss', round_to=1),
        },
        'nFieldsShrinkingHumanities': {
            'comment': 'how many fields in humanities shrunk?',
            'value': get_growth_stat(level='Field', umbrella='Humanities', get='number', change_type='loss', stat='field-count'),
        },
        'nFieldsGrowingEng': {
            'comment': 'how many fields in Engineering grew?',
            'value': get_growth_stat(level='Field', umbrella='Engineering', get='number', change_type='growth', stat='field-count'),
        },
        'nFieldsGrowingMedAndHealth': {
            'comment': 'how many fields in Medicine and Health grew?',
            'value': get_growth_stat(level='Field', umbrella='Medicine and Health', get='number', change_type='growth', stat='field-count'),
        },
        'pMaxLossMarketChangeHumanitiesField': {
            'comment': 'what was the largest loss of a field in humanities?',
            'value': get_growth_stat(level='Field', umbrella='Humanities', change_type='loss', stat='value-change'),
        },
        'maxLossMarketChangeHumanitiesFieldString': {
            'comment': 'what field in humanities had the largest loss?',
            'value': get_growth_stat(level='Field', umbrella='Humanities', change_type='loss', stat='value-change', get='string'),
        },
        'pSecondMaxLossMarketChangeHumanitiesField': {
            'comment': 'what was the 2nd largest loss of a field in humanities?',
            'value': get_growth_stat(level='Field', umbrella='Humanities', change_type='loss', stat='value-change', iloc=1),
        },
        'secondMaxLossMarketChangeHumanitiesFieldString': {
            'comment': 'what field in humanities had the largest loss?',
            'value': get_growth_stat(level='Field', umbrella='Humanities', change_type='loss', stat='value-change', get='string', iloc=1),
        },
        'pThirdMaxLossMarketChangeHumanitiesField': {
            'comment': 'what was the 3nd largest loss of a field in humanities?',
            'value': get_growth_stat(level='Field', umbrella='Humanities', change_type='loss', stat='value-change', iloc=2),
        },
        'thirdMaxLossMarketChangeHumanitiesFieldString': {
            'comment': 'what field in humanities had 3rd the largest loss?',
            'value': get_growth_stat(level='Field', umbrella='Humanities', change_type='loss', stat='value-change', get='string', iloc=2),
        },
        'pMinLossMarketChangeHumanitiesField': {
            'comment': 'what was the smallest loss in humanities?',
            'value': get_growth_stat(level='Field', umbrella='Humanities', change_type='loss', stat='value-change', ascending=True),
        },
        'minLossMarketChangeHumanitiesFieldString': {
            'comment': 'what field had the smallest loss in humanities?',
            'value': get_growth_stat(level='Field', umbrella='Humanities', change_type='loss', stat='value-change', get='string', ascending=True),
        },
        'pSecondMinLossMarketChangeHumanitiesField': {
            'comment': 'what was the second smallest loss in humanities?',
            'value': get_growth_stat(level='Field', umbrella='Humanities', change_type='loss', stat='value-change', iloc=1, ascending=True),
        },
        'secondMinLossMarketChangeHumanitiesFieldString': {
            'comment': 'what field had the second smallest loss in humanities?',
            'value': get_growth_stat(level='Field', umbrella='Humanities', change_type='loss', stat='value-change', get='string', iloc=1, ascending=True),
        },
        'maxGainMarketChangeEngFieldP': {
            'comment': 'what was the largest growth in Engineering?',
            'value': get_growth_stat(level='Field', umbrella='Engineering', change_type='growth', stat='value-change'),
        },
        'maxGainMarketChangeEngFieldString': {
            'comment': 'what field had the largest growth in Engineering?',
            'value': get_growth_stat(level='Field', umbrella='Engineering', change_type='growth', stat='value-change', get='string'),
        },
        'secondMaxGainMarketChangeEngFieldP': {
            'comment': 'what was the second largest growth in Engineering?',
            'value': get_growth_stat(level='Field', umbrella='Engineering', change_type='growth', stat='value-change', iloc=1),
        },
        'secondMaxGainMarketChangeEngFieldString': {
            'comment': 'what field had the second largest growth in Engineering?',
            'value': get_growth_stat(level='Field', umbrella='Engineering', change_type='growth', stat='value-change', get='string', iloc=1),
        },
        'pMaxGainMarketChangeField': {
            'comment': 'what was the largest growing field?',
            'value': get_growth_stat(level='Field', change_type='growth', stat='value-change'),
        },
        'maxGainMarketChangeFieldString': {
            'comment': 'what field had the largest growing field?',
            'value': get_growth_stat(level='Field', change_type='growth', stat='value-change', get='string'),
        },
        'pSecondMaxGainMarketChangeField': {
            'comment': 'what was the second largest growing field?',
            'value': get_growth_stat(level='Field', change_type='growth', stat='value-change', iloc=1),
        },
        'secondMaxGainMarketChangeFieldString': {
            'comment': 'what field had the second largest growing field?',
            'value': get_growth_stat(level='Field', change_type='growth', stat='value-change', get='string', iloc=1),
        },
        'pThirdMaxGainMarketChangeField': {
            'comment': 'what was the third largest growing field?',
            'value': get_growth_stat(level='Field', change_type='growth', stat='value-change', iloc=2),
        },
        'thirdMaxGainMarketChangeFieldString': {
            'comment': 'what field had the third largest growing field?',
            'value': get_growth_stat(level='Field', change_type='growth', stat='value-change', get='string', iloc=2),
        },
        'fourthMaxGainMarketChangeFieldP': {
            'comment': 'what was the fourth largest growing field?',
            'value': get_growth_stat(level='Field', change_type='growth', stat='value-change', iloc=3),
        },
        'fourthMaxGainMarketChangeFieldString': {
            'comment': 'what field had the fourth largest growing field?',
            'value': get_growth_stat(level='Field', change_type='growth', stat='value-change', get='string', iloc=3),
        },
        'fifthMaxGainMarketChangeFieldP': {
            'comment': 'what was the fifth largest growing field?',
            'value': get_growth_stat(level='Field', change_type='growth', stat='value-change', iloc=4),
        },
        'fifthMaxGainMarketChangeFieldString': {
            'comment': 'what field had the fifth largest growing field?',
            'value': get_growth_stat(level='Field', change_type='growth', stat='value-change', get='string', iloc=4),
        },
    }


def get_career_length_stat(
    level=None,
    value=None,
    stat='difference', # difference | year
    get='number',
    when=None, # start|end
    ascending=False,
    iloc=0,
    age_direction='older',
):
    df = usfhn.stats.runner.get('careers/length/by-year/taxonomy')

    df = usfhn.views.filter_by_taxonomy(df, level=level, value=value)

    start = min(df['Year'])
    end = max(df['Year'])

    df = df[
        df['Year'].isin([start, end])
    ]

    df = hnelib.pandas.aggregate_df_over_column(
        df,
        agg_col='Year',
        join_cols=['TaxonomyLevel', 'TaxonomyValue'],
        value_cols=['MeanCareerLength'],
        agg_value_to_label={
            start: 'Start',
            end: 'End',
        },
    )

    df['Difference'] = df['EndMeanCareerLength'] - df['StartMeanCareerLength']

    if stat == 'difference':
        df = df.sort_values(by='Difference', ascending=ascending)

        row = df.iloc[iloc]
        if get == 'number':
            return round(row['Difference'], 1)
        elif get == 'percent':
            return hnelib.utils.fraction_to_percent(
                row['EndMeanCareerLength'] / row['StartMeanCareerLength'] - 1,
                1,
            )
        elif get == 'string':
            return usfhn.plot_utils.main_text_taxonomy_string(row['TaxonomyValue'])
    elif stat == 'length':
        row = df.iloc[0]

        if when == 'start': 
            value = row['StartMeanCareerLength']
        elif when == 'end':
            value = row['EndMeanCareerLength']

        return round(value, 1)
    elif stat == 'delta':
        denominator = len(df)
        if age_direction == 'older':
            df = df[
                df['Difference'] > 0
            ]
        elif age_direction == 'younger':
            df = df[
                df['Difference'] < 0
            ]

        return get_number_or_percent(get, len(df), denominator)


def career_lengths_stats_generator():
    return {
        'nYearsAgeChange': {
            'comment': 'what is the difference between avg career length in 2020 and 2011?',
            'value': get_career_length_stat(level='Academia', value='Academia'),
        },
        'pYearsAgeChange': {
            'comment': 'what is the % difference between avg career length in 2020 and 2011?',
            'value': get_career_length_stat(level='Academia', value='Academia', get='percent'),
        },
        'nDomainsThatGotOlder': {
            'comment': 'how many domains got older?',
            'value': get_career_length_stat(level='Umbrella', stat='delta'),
        },
        'nFieldsThatGodOlder': {
            'comment': 'how many fields got older?',
            'value': get_career_length_stat(level='Field', stat='delta'),
        },
        'pFieldsThatGodOlder': {
            'comment': 'how many fields got older?',
            'value': get_career_length_stat(level='Field', stat='delta', get='percent'),
        },
        'careerLengthStart': {
            'comment': 'how long is a career in the first year?',
            'value': get_career_length_stat(level='Academia', value='Academia', stat='length', when='start'),
        },
        'careerLengthEnd': {
            'comment': 'how long is a career in the last year?',
            'value': get_career_length_stat(level='Academia', value='Academia', stat='length', when='end'),
        },
        'maxYearsAgeChangeDomain': {
            'comment': 'what is the biggest change in career length at the domain level?',
            'value': get_career_length_stat(level='Umbrella'),
        },
        'maxYearsAgeChangeDomainString': {
            'comment': 'which domain had the biggest change in career length?',
            'value': get_career_length_stat(level='Umbrella', get='string'),
        },
        'secondMaxYearsAgeChangeDomain': {
            'comment': 'what is the 2nd biggest change in career length at the domain level?',
            'value': get_career_length_stat(level='Umbrella', iloc=1),
        },
        'secondMaxYearsAgeChangeDomainString': {
            'comment': 'which domain had the 2nd biggest change in career length?',
            'value': get_career_length_stat(level='Umbrella', get='string', iloc=1),
        },
        'minYearsAgeChangeDomain': {
            'comment': 'what is the smallest change in career length at the domain level?',
            'value': get_career_length_stat(level='Umbrella', ascending=True),
        },
        'minYearsAgeChangeDomainString': {
            'comment': 'which domain had the smallest change in career length?',
            'value': get_career_length_stat(level='Umbrella', ascending=True, get='string'),
        },
        'secondMinYearsAgeChangeDomain': {
            'comment': 'what is the 2nd smallest change in career length at the domain level?',
            'value': get_career_length_stat(level='Umbrella', ascending=True, iloc=1),
        },
        'secondMinYearsAgeChangeDomainString': {
            'comment': 'which domain had the 2nd smallest change in career length?',
            'value': get_career_length_stat(level='Umbrella', ascending=True, get='string', iloc=1),
        },
        'maxYearsAgeChangeField': {
            'comment': 'what is the biggest change in career length at the field level?',
            'value': get_career_length_stat(level='Field'),
        },
        'maxYearsAgeChangeFieldString': {
            'comment': 'which field had the biggest change in career length?',
            'value': get_career_length_stat(level='Field', get='string'),
        },
        'secondMaxYearsAgeChangeField': {
            'comment': 'what is the 2nd biggest change in career length at the field level?',
            'value': get_career_length_stat(level='Field', iloc=1),
        },
        'secondMaxYearsAgeChangeFieldString': {
            'comment': 'which field had the 2nd biggest change in career length?',
            'value': get_career_length_stat(level='Field', get='string', iloc=1),
        },
        'minYearsAgeChangeField': {
            'comment': 'what is the smallest change in career length at the field level?',
            'value': get_career_length_stat(level='Field', ascending=True),
        },
        'minYearsAgeChangeFieldString': {
            'comment': 'which field had the smallest change in career length?',
            'value': get_career_length_stat(level='Field', ascending=True, get='string'),
        },
        'secondMinYearsAgeChangeField': {
            'comment': 'what is the 2nd smallest change in career length at the field level?',
            'value': get_career_length_stat(level='Field', ascending=True, iloc=1),
        },
        'secondMinYearsAgeChangeFieldString': {
            'comment': 'which field had the 2nd smallest change in career length?',
            'value': get_career_length_stat(level='Field', ascending=True, get='string', iloc=1),
        },
    }


def get_stats(): 
    stats_generators = {
        'General stats': general_stats_generator,
        'Exclusions': exclusion_stats_generator,
        'Production': production_stats_generator,
        'Ginis': gini_stats_generator,
        'Gender': gender_stats_generator,
        'Closedness': closedness_stats_generator,
        'Non doctorate': non_doctorate_stats_generator,
        'Non US': non_us_stats_generator,
        'Prestige': prestige_stats_generator,
        'Null models': null_model_stats_generator,
        'Field correlations': field_correlations_stats_generator,
        'Monopolization of top slots': monopolization_stats_generator,
        'Self-hiring': self_hiring_stats_generator,
        'MCMs': mcm_stats_generator,
        'Time': time_stats_generator,
        'Cleaning': cleaning_stats_generator,
        'New Hires': new_faculty_stats_generator,
        'Attrition': attrition_stats_generator,
        'Market Share': market_share_stats_generator,
        'Career Lengths': career_lengths_stats_generator,
    }

    print('generating stats')

    stats_count = 0
    stats = {}
    for key, generator in stats_generators.items():
        start = time.time()
        stats[key] = generator()
        stats_count += len(stats[key])
        end = time.time()
        print(f'\t{key}: {round(end-start)}')

    print(f"{stats_count} stats. yay!")
    return stats


def make_header(string):
    lines = [
        80 * '%',
        f'% {string}',
        80 * '%',
    ]

    return "\n".join(lines)



def generate_stats():
    lines = []
    for title, values in get_stats().items():
        lines.append(make_header(title))

        for name, subvalue in values.items():
            lines.append(f"% {subvalue['comment']}")

            if 'value' in subvalue:
                output = subvalue['value']

                if subvalue.get('add_commas', True):
                    try:
                        output = "{:,}".format(output)
                    except:
                        pass

                line = "\\newcommand{\\" + name + "}{" + str(output)

                if subvalue.get('add_percent_sign') == None and name.startswith('p'):
                    line += '\\%'
                elif subvalue.get('add_percent_sign'):
                    line += '\\%'

                if subvalue.get('add_xspace', True):
                    line += '\\xspace'

                line += "}"

                line = line.replace('&', '\&')
                lines.append(line)

            lines.append("")

    return "\n".join(lines)


def write_stats(path=usfhn.constants.PAPER_STATS_PATH):
    stats = generate_stats()
    path.write_text(stats)


def report_unused_stats():
    stats_lines = usfhn.constants.PAPER_STATS_PATH.read_text().split('\n')
    stats_lines = [l for l in stats_lines if l.startswith('\\newcommand')]
    defined_stats = [] 
    for stat_line in stats_lines:
        stat_line = stat_line.split('{\\', 1)[1]
        stat = stat_line.split('}', 1)[0]
        defined_stats.append(stat)

    paper_latex = usfhn.constants.PAPER_TEX_PATH.read_text()
    unused_stats = [s for s in defined_stats if s not in paper_latex]

    for stat in unused_stats:
        print(stat)

    print(len(unused_stats))
