import argparse
import itertools
import pandas as pd
import statsmodels.api as sm

import hnelib.runner
import hnelib.pandas

import usfhn.datasets
import usfhn.constants
import usfhn.measurements
import usfhn.views
import usfhn.non_us
import usfhn.fieldwork
import usfhn.attrition
import usfhn.careers
import usfhn.interdisciplinarity
import usfhn.new_hires
import usfhn.self_hiring
import usfhn.taxonomy
import usfhn.rank
import usfhn.gender
import usfhn.pool_reduction
import usfhn.demographics
import usfhn.closedness
import usfhn.webs
import usfhn.non_doctorate
import usfhn.stats_utils


################################################################################
#
#
# Utilities for standard names
#
#
################################################################################
VARIABLE_TO_COLUMN = {
    'ginis': 'GiniCoefficient',
    'gender': 'FractionFemale',
    'self-hires': 'SelfHiresFraction',
    'self-hire': 'SelfHire',
    'non-us': 'NonUS',
    'faculty-rank': 'Rank',
    'seniority': 'Senior',
    'new-hire': 'NewHire',
    'year': 'Year',
    'career-age': 'CareerAge',
}

# higher priority = happens earlier in the path
FLAG_PARAMETER_PRIORITY = {
    'by_year': 0,
}

def is_flag_parameter(parameter):
    """
    so I get in something like `by_faculty_rank`

    return false if doesn't start with `by_`

    replace '_' with '-'

    return true if is in VARIABLE_TO_COLUMN
    """
    if not parameter.startswith('by_'):
        return False

    parameter = parameter.replace('by_', '')
    parameter = parameter.replace('_', '-')

    return parameter in VARIABLE_TO_COLUMN


def get_flag_parameters_from_kwargs(**kwargs):
    """
    we're going to return the flag parameters in a specific order because
    sometimes there are multiple qualifiers
    """
    parameters = [k for k, v in kwargs.items() if v and is_flag_parameter(k)]

    max_priority = max(FLAG_PARAMETER_PRIORITY.values()) + 1
    return sorted(parameters, key=lambda p: FLAG_PARAMETER_PRIORITY.get(p, max_priority), reverse=True)


def get_variable_from_flag_parameter(string):
    string = string.replace('by_', '')
    string = string.replace('_', '-')
    return string


def is_variable_qualifier(string):
    if not string.startswith('by-'):
        return False

    string = string.replace('by-', '')

    return string in VARIABLE_TO_COLUMN


def get_groupby_cols_from_kwargs(exclude=[], **kwargs):
    parameters = get_flag_parameters_from_kwargs(**kwargs)
    variables = [get_variable_from_flag_parameter(p) for p in parameters]
    groupby_cols = [VARIABLE_TO_COLUMN[v] for v in variables]
    groupby_cols = [c for c in groupby_cols if c not in exclude]
    return groupby_cols


def get_variable_dataframe(variable, suffix=None, item_kwargs={}, **kwargs):
    flag_parameters = get_flag_parameters_from_kwargs(**kwargs)
    qualifiers = [p.replace('_', '-') for p in flag_parameters]

    parts = [variable] + qualifiers + [suffix]
    path = "/".join([p for p in parts if p])
    df = runner.get(path, **item_kwargs)

    return df


def add_groupby_annotations_to_df(
    df,
    by_gender=False,
    by_new_hire=False,
    by_non_us=False,
    by_faculty_rank=False,
    by_seniority=False,
    by_career_age=False,
    by_self_hire=False,
    explode_gender=False,
    **kwargs,
):
    cols = list(df.columns)

    groupby_cols = []

    if by_gender:
        df = usfhn.gender.annotate_gender(df, explode_gender=explode_gender)
        groupby_cols.append('Gender')

    if by_career_age:
        df = usfhn.careers.annotate_career_age(df)
        groupby_cols.append('CareerAge')

    if by_faculty_rank:
        df = usfhn.rank.annotate_faculty_rank(df)
        groupby_cols.append('Rank')

    if by_seniority:
        df = usfhn.rank.annotate_seniority(df)
        groupby_cols.append('Senior')

    if by_non_us:
        df = usfhn.non_us.annotate_non_us(df)
        groupby_cols.append('NonUS')

    if by_new_hire:
        df = usfhn.new_hires.annotate_new_hires(df)
        groupby_cols.append('NewHire')

    if by_self_hire:
        df = usfhn.self_hiring.annotate_self_hires(df)
        groupby_cols.append('SelfHire')

    df = df[
        cols + groupby_cols
    ].drop_duplicates()

    return df, groupby_cols


################################################################################
#
#
# Expanders and stats collections
#
#
################################################################################
RANKS_EXPANDER = {'rank_type': ['production', 'prestige']}
ALL_RANKS_EXPANDER = {'rank_type': ['production', 'doctoral-institution', 'employing-institution']}

def get_attrition_risk_stats_functions():
    return {
        'taxonomy': usfhn.attrition.get_taxonomy_attrition_risk,
        'gender': usfhn.attrition.get_gender_attrition_risk,
        'self-hires': usfhn.attrition.get_self_hire_attrition_risk,
        'institution': {
            'do': usfhn.attrition.get_institution_attrition_risk,
            'prefix_expansions': {'institution_column': ['DegreeInstitutionId', 'InstitutionId']},
        },
        'us': usfhn.attrition.get_non_us_attrition_risk,
        'non-us-by-is-english': usfhn.attrition.get_non_us_by_is_english_attrition_risk,
        'rank-change': {
            'do': usfhn.attrition.get_rank_change_attrition_risk,
            'directory_expansions': RANKS_EXPANDER,
        },
    }

def get_rank_stats_functions():
    return {
        'directory_expansions': RANKS_EXPANDER,
        'df': usfhn.rank.get_ranks_df_for_stats,
        'change': usfhn.rank.get_rank_change_df,
        'hierarchy-stats': usfhn.rank.get_hierarchy_stats,
        'mean-change': usfhn.rank.get_mean_rank_change,
        'movement-distance': usfhn.rank.get_gendered_movement_distance_significance,
        'gender': {
            'kwargs': {'by_gender': True},
            'hierarchy-stats': usfhn.rank.get_hierarchy_stats,
            'mean-change': usfhn.rank.get_mean_rank_change,
            'placements': usfhn.rank.get_placements,
        },
        'seniority': {
            'kwargs': {'by_seniority': True},
            'hierarchy-stats': usfhn.rank.get_hierarchy_stats,
            'mean-change': usfhn.rank.get_mean_rank_change,
        },
    }


################################################################################
#
# 
# basics
#
#
################################################################################
def get_faculty_hiring_network(by_year=False, use_closedness_data=False, by_gender=False):
    """
    Columns:
    - DegreeInstitutionId: producing institution id
    - InstitutionId: employing institution id
    - Count: faculty at InstitutionId with PhDs from DegreeInstitutionId
    - OutDegree: total faculty with PhDs from DegreeInstitutionId
    - InDegree: total faculty employed by InstitutionId
    - TotalCount: total faculty

    we include zeroes in when the level is academia because we know that these
    are phd-granting institutions, so they are producers.
    
    we do not know whether a particular phd-granting institution is phd granting
    in any particular field/area/umbrella, however, so we exclude them to arrive
    at a lower bound.
    """
    groupby_cols = [
        'TaxonomyLevel',
        'TaxonomyValue',
    ]

    df_name = 'closedness_data' if use_closedness_data else 'data'

    if by_year:
        groupby_cols.append('Year')

    df = usfhn.datasets.get_dataset_df(df_name, by_year=by_year)

    df = usfhn.fieldwork.linearize_df_taxonomy_hierarchy(df)

    df = df[
        groupby_cols + [
            'InstitutionId',
            'DegreeInstitutionId',
            'PersonId',
        ]
    ].drop_duplicates()

    df, extra_groupby_cols = add_groupby_annotations_to_df(df, by_gender=by_gender)
    groupby_cols += extra_groupby_cols

    df['Count'] = df.groupby(
        ['InstitutionId', 'DegreeInstitutionId'] + groupby_cols
    )['PersonId'].transform('nunique').fillna(0)

    df['OutDegree'] = df.groupby(
        ['DegreeInstitutionId'] + groupby_cols
    )['PersonId'].transform('nunique').fillna(0)

    df['InDegree'] = df.groupby(
        ['InstitutionId'] + groupby_cols
    )['PersonId'].transform('nunique').fillna(0)

    df['TotalCount'] = df.groupby(
        groupby_cols
    )['PersonId'].transform('nunique').fillna(0)

    df = df.drop(columns=['PersonId']).drop_duplicates()

    df = df[
        (df['Count'] > 0)
        | 
        (df['TaxonomyLevel'] == 'Academia')
    ]

    return df

def get_faculty_production(by_year=False, new_hires=False, non_new_hires=False, non_us=False):
    """
    We include zeroes in when the level is academia because we KNOW that these
    are PhD-granting institutions, so they are Producers.
    
    We do not know whether a particular PhD-granting institution is PhD granting
    in any particular field/area/umbrella, however, so we exclude them to arrive
    at a lower bound.
    """
    groupby_cols = [
        'TaxonomyLevel',
        'TaxonomyValue',
    ]

    dataset_name = 'closedness_data' if non_us else 'data'

    if by_year:
        groupby_cols.append('Year')

    df = usfhn.datasets.get_dataset_df(dataset_name, by_year=by_year)
    df = usfhn.fieldwork.linearize_df_taxonomy_hierarchy(df)

    df = df[
        groupby_cols + [
            'PersonId',
            'DegreeInstitutionId',
            'InstitutionId',
        ]
    ].drop_duplicates()

    if new_hires or non_new_hires:
        if by_year:
            new_hires_df = runner.get('new-hire/by-year/df')
        else:
            new_hires_df = runner.get('new-hire/df')

        new_hires_df['NewHire'] = True

        df = df.merge(
            new_hires_df,
            on=list(df.columns),
            how='left',
        )
        
        df['NewHire'] = df['NewHire'].fillna(False)

        if non_new_hires:
            df = df[~df['NewHire']]
        else:
            df = df[df['NewHire']]

        df = df.drop(columns=['NewHire'])

    df['ProductionCount'] = df.groupby(
        groupby_cols + ['DegreeInstitutionId']
    )['PersonId'].transform('nunique').fillna(0)

    df = df[
        (df['ProductionCount'] > 0)
        | 
        (df['TaxonomyLevel'] == 'Academia')
    ]

    df['TotalProduction'] = df.groupby(groupby_cols)['PersonId'].transform('nunique')

    df = df.drop(columns=['PersonId']).drop_duplicates()

    df['ProductionFraction'] = df['ProductionCount'] / df['TotalProduction']
    df['ProductionPercent'] = 100 * df['ProductionFraction']

    return df

def get_faculty_hiring(by_year=False):
    """
    We include zeroes in when the level is academia because we KNOW that these
    are PhD-granting institutions, so they are Producers.
    
    We do not know whether a particular PhD-granting institution is PhD granting
    in any particular field/area/umbrella, however, so we exclude them to arrive
    at a lower bound.
    """
    groupby_cols = [
        'TaxonomyLevel',
        'TaxonomyValue',
    ]

    if by_year:
        groupby_cols.append('Year')

    df = usfhn.datasets.get_dataset_df('data', by_year=by_year)

    df = usfhn.fieldwork.linearize_df_taxonomy_hierarchy(df)

    df = df[
        groupby_cols + [
            'DegreeInstitutionId',
            'InstitutionId',
            'PersonId',
        ]
    ].drop_duplicates()

    df['HiredFaculty'] = df.groupby(
        groupby_cols + ['InstitutionId', 'DegreeInstitutionId',]
    )['PersonId'].transform('nunique').fillna(0)

    df['TotalHiredFaculty'] = df.groupby(groupby_cols)['PersonId'].transform('nunique')

    df = df.drop(columns=['PersonId']).drop_duplicates()

    df = df[
        (df['HiredFaculty'] > 0)
        | 
        (df['TaxonomyLevel'] == 'Academia')
    ]

    df['HiredFacultyFraction'] = df['HiredFaculty'] / df['TotalHiredFaculty']

    return df


def get_lorenz_curves(by_year=False, new_hires=False, non_new_hires=False):
    groupby_cols = [
        'TaxonomyLevel',
        'TaxonomyValue',
    ]

    if by_year:
        groupby_cols.append('Year')

        if new_hires:
            df = runner.get('new-hire/by-year/production')
        else:
            df = runner.get('basics/by-year/production')
    else:
        if new_hires:
            df = runner.get('new-hire/production')
        elif non_new_hires:
            df = runner.get('non-new-hire/production')
        else:
            df = runner.get('basics/production')

    curves = []
    for _, rows in df.groupby(groupby_cols):
        rows = rows.drop_duplicates(subset=['DegreeInstitutionId'])
        curve = usfhn.measurements.lorenz_curve(rows['ProductionFraction'])
        for col in groupby_cols:
            curve[col] = rows.iloc[0][col]

        curves.append(curve)

    return pd.concat(curves)

################################################################################
#
# 
# Changes over time
#
#
################################################################################
def run_ols_and_get_slopes(variable, p_threshold=.05, **kwargs):
    df = get_variable_dataframe(variable, suffix='df', by_year=True, **kwargs)

    groupby_cols = ['TaxonomyLevel', 'TaxonomyValue']
    groupby_cols += get_groupby_cols_from_kwargs(**kwargs)

    value_col = VARIABLE_TO_COLUMN[variable]

    result_rows = []
    for _, rows in df.groupby(groupby_cols):
        groupby_info = {c: rows.iloc[0][c] for c in groupby_cols}

        X = sm.add_constant(rows['Year'] - (min(rows['Year']) - 1))
        Y = rows[value_col]
        model = sm.OLS(Y, X)
        result = model.fit()

        result_rows.append({
            **groupby_info,
            'Slope': result.params['Year'],
            'R^2': result.rsquared,
            'P': result.pvalues['Year'],
            'Significant': p_threshold > result.pvalues['Year'],
        })

    df = pd.DataFrame(result_rows)

    df = usfhn.stats_utils.correct_multiple_hypotheses(df)
    df['Significant'] = df['PCorrected'] < p_threshold

    return pd.DataFrame(result_rows)


def degree_years_by_taxonomy():
    df = usfhn.datasets.CURRENT_DATASET.closedness_data[
        [
            'PersonId',
            'DegreeYear',
            'Taxonomy',
        ]
    ].drop_duplicates()

    df = usfhn.fieldwork.linearize_df_taxonomy_hierarchy(df)

    return df


def get_years_until_like_new_hires(variable, significant=True):
    value_col = VARIABLE_TO_COLUMN[variable]

    df = get_variable_dataframe(variable, suffix='df', by_new_hire=True)
    df = hnelib.pandas.aggregate_df_over_column(
        df,
        agg_col='NewHire',
        join_cols=['TaxonomyLevel', 'TaxonomyValue'],
        value_cols=[value_col],
        agg_value_to_label={
            True: 'NewHire',
            False: 'ExistingHire',
        },
    )

    df['Difference'] = df[f'NewHire{value_col}'] - df[f'ExistingHire{value_col}']
    df = df[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'Difference',
        ]
    ].drop_duplicates()

    ols_df = get_variable_dataframe(variable, suffix='slopes')

    if significant:
        ols_df = ols_df[
            ols_df['Significant']
        ]

    ols_df = ols_df[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'Slope',
            'Significant',
            'P',
        ]
    ].drop_duplicates()

    df = df.merge(
        ols_df,
        on=[
            'TaxonomyLevel',
            'TaxonomyValue',
        ]
    )

    df['YearsToProduceChange'] = df['Difference'] / df['Slope']

    df = df[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'YearsToProduceChange',
        ]
    ].drop_duplicates()

    df = df[
        df['YearsToProduceChange'] > 0
    ]

    return df




################################################################################
#
# 
# Stats collection
#
#
################################################################################
STATS_COLLECTION = {
    # to replace with stats calls:
    # - get_steepness_by_taxonomy
    'basics': {
        'faculty-hiring-network': get_faculty_hiring_network,
        'closedness-faculty-hiring-network': {
            'kwargs': {'use_closedness_data': True},
            'df': get_faculty_hiring_network,
            'by-gender': {
                'kwargs': {'by_gender': True},
                'do': get_faculty_hiring_network,
            },
        },
        'production': get_faculty_production,
        'hiring': get_faculty_hiring,
        'lorenz': get_lorenz_curves,
    },
    'demographics': {
        'degree-years': degree_years_by_taxonomy,
        'non-doctorates-by-taxonomy': usfhn.demographics.get_non_doctorate_fraction_by_taxonomy,
    },
    'closedness': {
        'df': usfhn.closedness._get_closednesses,
    },
    'taxonomy': {
        'df': usfhn.taxonomy.get_taxonomy_size,
        'institutions': {
            'kwargs': {'count_col': 'InstitutionId'},
            'do': usfhn.taxonomy.get_taxonomy_size,
        },
        'by-year': {
            'kwargs': {'by_year': True},
            'df': usfhn.taxonomy.get_taxonomy_size,
            'institutions': {
                'kwargs': {'count_col': 'InstitutionId'},
                'do': usfhn.taxonomy.get_taxonomy_size,
            },
        },
    },
    'ginis': {
        'kwargs': {'variable': 'ginis'},
        'df': usfhn.gini.get_gini_coefficients,
        'slopes': run_ols_and_get_slopes,
        'stasis': get_years_until_like_new_hires,
        'by-year': {
            'kwargs': {'by_year': True},
            'df': usfhn.gini.get_gini_coefficients,
        },
        'by-faculty-rank': {
            'kwargs': {'by_faculty_rank': True},
            'df': usfhn.gini.get_gini_coefficients,
            'slopes': run_ols_and_get_slopes,
            'by-year': {
                'kwargs': {'by_year': True},
                'df': usfhn.gini.get_gini_coefficients,
            },
        },
        'by-seniority': {
            'kwargs': {'by_seniority': True},
            'df': usfhn.gini.get_gini_coefficients,
            'slopes': run_ols_and_get_slopes,
            'by-year': {
                'kwargs': {'by_year': True},
                'df': usfhn.gini.get_gini_coefficients,
            },
        },
        'by-new-hire': {
            'kwargs': {'by_new_hire': True},
            'df': usfhn.gini.get_gini_coefficients,
            'slopes': run_ols_and_get_slopes,
            'by-year': {
                'kwargs': {'by_year': True},
                'df': usfhn.gini.get_gini_coefficients,
            },
        },
    },
    'gender': {
        'kwargs': {'variable': 'gender'},
        'df': usfhn.gender.get_taxonomy_gender_fraction,
        'slopes': run_ols_and_get_slopes,
        'stasis': get_years_until_like_new_hires,
        'by-year': {
            'kwargs': {'by_year': True},
            'df': usfhn.gender.get_taxonomy_gender_fraction,
        },
        'by-faculty-rank': {
            'kwargs': {'by_faculty_rank': True},
            'df': usfhn.gender.get_taxonomy_gender_fraction,
            'slopes': run_ols_and_get_slopes,
            'by-year': {
                'kwargs': {'by_year': True},
                'df': usfhn.gender.get_taxonomy_gender_fraction,
            },
        },
        'by-seniority': {
            'kwargs': {'by_seniority': True},
            'df': usfhn.gender.get_taxonomy_gender_fraction,
            'slopes': run_ols_and_get_slopes,
            'by-year': {
                'kwargs': {'by_year': True},
                'df': usfhn.gender.get_taxonomy_gender_fraction,
            },
        },
        'by-new-hire': {
            'kwargs': {'by_new_hire': True},
            'df': usfhn.gender.get_taxonomy_gender_fraction,
            'slopes': run_ols_and_get_slopes,
            'by-year': {
                'kwargs': {'by_year': True},
                'df': usfhn.gender.get_taxonomy_gender_fraction,
            },
            'by-rank': {
                'directory_expansions': RANKS_EXPANDER,
                'logits': usfhn.gender.rank_vs_gender_logit,
                'existing-hire-logits': {
                    'kwargs': {'existing_hires': True},
                    'do': usfhn.gender.rank_vs_gender_logit,
                },
            },
        },
        'by-non-us': {
            'kwargs': {'by_non_us': True},
            'df': usfhn.gender.get_taxonomy_gender_fraction,
        },
        'by-career-age': {
            'kwargs': {'by_career_age': True},
            'df': usfhn.gender.get_taxonomy_gender_fraction,
        },
        'by-rank': {
            'directory_expansions': RANKS_EXPANDER,
            'logits': usfhn.gender.rank_vs_gender_logit,
        },
    },
    'self-hires': {
        'kwargs': {'variable': 'self-hires'},
        'df': usfhn.self_hiring.get_taxonomy_self_hiring,
        'slopes': run_ols_and_get_slopes,
        'stasis': get_years_until_like_new_hires,
        'by-year': {
            'kwargs': {'by_year': True},
            'df': usfhn.self_hiring.get_taxonomy_self_hiring,
        },
        'by-faculty-rank': {
            'kwargs': {'by_faculty_rank': True},
            'df': usfhn.self_hiring.get_taxonomy_self_hiring,
            'slopes': run_ols_and_get_slopes,
            'by-year': {
                'kwargs': {'by_year': True},
                'df': usfhn.self_hiring.get_taxonomy_self_hiring,
            },
        },
        'by-gender': {
            'kwargs': {'by_gender': True},
            'df': usfhn.self_hiring.get_taxonomy_self_hiring,
        },
        'by-seniority': {
            'kwargs': {'by_seniority': True},
            'df': usfhn.self_hiring.get_taxonomy_self_hiring,
            'slopes': run_ols_and_get_slopes,
            'by-year': {
                'kwargs': {'by_year': True},
                'df': usfhn.self_hiring.get_taxonomy_self_hiring,
            },
        },
        'by-new-hire': {
            'kwargs': {'by_new_hire': True},
            'df': usfhn.self_hiring.get_taxonomy_self_hiring,
            'slopes': run_ols_and_get_slopes,
            'by-year': {
                'kwargs': {'by_year': True},
                'df': usfhn.self_hiring.get_taxonomy_self_hiring,
            },
            'by-rank': {
                'directory_expansions': RANKS_EXPANDER,
                'logits': usfhn.self_hiring.rank_vs_self_hires_logit,
                'existing-hire-logits': {
                    'kwargs': {'existing_hires': True},
                    'do': usfhn.self_hiring.rank_vs_self_hires_logit,
                },
            },
        },
        'by-career-age': {
            'kwargs': {'by_career_age': True},
            'df': usfhn.self_hiring.get_taxonomy_self_hiring,
        },
        'by-institution': {
            'kwargs': {'by_institution': True},
            'df': usfhn.self_hiring.get_taxonomy_self_hiring,
        },
        'by-rank': {
            'logits': {
                'directory_expansions': RANKS_EXPANDER,
                'do': usfhn.self_hiring.rank_vs_self_hires_logit,
            },
            'binned-logits': {
                'directory_expansions': ALL_RANKS_EXPANDER,
                'do': usfhn.self_hiring.self_hiring_by_career_age_logistics,
            },
        },
    },
    'non-us': {
        'df': usfhn.non_us.get_fraction_non_us,
        'by-year': {
            'kwargs': {'by_year': True},
            'df': usfhn.non_us.get_fraction_non_us,
        },
        'by-institution': {
            'df': usfhn.non_us.get_fraction_non_us_by_institution,
            'by-rank': {
                'directory_expansions': RANKS_EXPANDER,
                'df': usfhn.non_us.get_fraction_non_us_by_institution_with_rank,
                'logits': usfhn.non_us.rank_vs_non_us_logit,
            },
        },
        'by-new-hire': {
            'kwargs': {'by_new_hire': True},
            'by-rank': {
                'directory_expansions': RANKS_EXPANDER,
                'logits': usfhn.non_us.rank_vs_non_us_logit,
                'existing-hire-logits': {
                    'kwargs': {'existing_hires': True},
                    'do': usfhn.non_us.rank_vs_non_us_logit,
                },
            },
        },
        'by-career-age': {
            'kwargs': {'by_career_age': True},
            'df': usfhn.non_us.get_fraction_non_us,
            'by-continent': {
                'kwargs': {'by_continent': True},
                'df': usfhn.non_us.get_fraction_non_us_by_continent_and_career_age,
            },
        },
        'by-country': usfhn.non_us.non_us_countries_by_production,
        'by-continent': usfhn.non_us.non_us_continents_by_production,
        'production': {
            'kwargs': {'non_us': True},
            'do': get_faculty_production,
        },
    },
    'ranks': {
        # we run a lot of the same stats for ranks in different things, so put
        # them into a function. functions:
        # - df
        # - change
        # - hierarchy-stats
        # - mean-change
        **get_rank_stats_functions(),
        'institution-rank-correlations': {
            'directory_expansions': RANKS_EXPANDER,
            'do': usfhn.rank.get_field_to_field_rank_correlations,
        },
        'by-year': {
            'kwargs': {'by_year': True},
            **get_rank_stats_functions(),
        },
        'placements': usfhn.rank.get_placements,
        'institution-fields': usfhn.rank.get_institution_fields,
        'institution-placement-stats': usfhn.rank.get_institution_placement_stats,
    },
    'interdisciplinarity': {
        'df': usfhn.interdisciplinarity.get_employment_interdisciplinarity_df,
        'taxonomy': usfhn.interdisciplinarity.get_taxonomy_interdisciplinarity,
        'institution': usfhn.interdisciplinarity.get_institution_interdisciplinarity,
        'degree-institution': usfhn.interdisciplinarity.get_degree_institution_interdisciplinarity,
        'by-year': {
            'kwargs': {'by_year': True},
            'df': usfhn.interdisciplinarity.get_employment_interdisciplinarity_df,
            'taxonomy': usfhn.interdisciplinarity.get_taxonomy_interdisciplinarity,
        },
        'institution-with-ranks': {
            'prefixes': RANKS_EXPANDER,
            'do': usfhn.interdisciplinarity.institutions_with_ranks,
        },
        'by-rank': {
            'directory_expansions': RANKS_EXPANDER,
            'logits': usfhn.interdisciplinarity.rank_vs_interdisciplinarity_logit,
        },
    },
    'careers': {
        'df': usfhn.careers.get_career_moves_df,
        'gender': usfhn.careers.get_move_risk_by_gender, # gender-overall
        'by-year': {
            'gender': usfhn.careers.get_move_risk_by_gender_and_year, # gender-by-year
        },
        'institution-risk': usfhn.careers.get_institution_move_risk,
        'taxonomy-hires-before-and-after-mcms': usfhn.careers.get_taxonomy_hires_before_and_after_mcms,
        'hierarchy-changes-from-mcms': usfhn.careers.get_mcm_stats,
        'hierarchy-changes-from-mcms-compare-to-normal': usfhn.careers.get_mcm_stats_comparison,
        'length': {
            'df': usfhn.careers.get_career_lengths_df,
            'taxonomy': usfhn.careers.career_length_by_taxonomy,
            'gender': usfhn.careers.career_length_by_gender,
            'gini': usfhn.careers.get_gini_coefficients_by_career_length,
            'by-year': {
                'kwargs': {'by_year': True},
                'df': usfhn.careers.get_career_lengths_df,
                'gender': usfhn.careers.career_length_by_gender,
                'taxonomy': usfhn.careers.career_length_by_taxonomy,
                'gini': usfhn.careers.get_gini_coefficients_by_career_length,
            },
            'by-rank': {
                'directory_expansions': RANKS_EXPANDER,
                'logits': usfhn.careers.rank_vs_career_length_logit,
            },
        },
    },
    'new-hire': {
        'df': usfhn.new_hires.get_new_hires_df,
        'self-hires': usfhn.new_hires.get_self_hires,
        'non-us': usfhn.non_us.get_new_hire_fraction,
        'interdisciplinarity': usfhn.interdisciplinarity.get_new_hire_fraction,
        'steepness': usfhn.new_hires.get_steepness,
        'production': {
            'kwargs': {'new_hires': True},
            'do': get_faculty_production,
        },
        'lorenz': {
            'kwargs': {'new_hires': True},
            'do': get_lorenz_curves,
        },
        'by-year': {
            'kwargs': {'by_year': True},
            'df': usfhn.new_hires.get_new_hires_df,
            'self-hires': usfhn.new_hires.get_self_hires,
            'non-us': usfhn.non_us.get_new_hire_fraction,
            'interdisciplinarity': usfhn.interdisciplinarity.get_new_hire_fraction,
            'steepness': usfhn.new_hires.get_steepness,
        },
    },
    'non-new-hire': {
        'production': {
            'kwargs': {'non_new_hires': True},
            'do': get_faculty_production,
        },
        'lorenz': {
            'kwargs': {'non_new_hires': True},
            'do': get_lorenz_curves,
        },
    },
    'non-doctorate': {
        'df': usfhn.non_doctorate.get_non_doctorates_by_taxonomy,
        'by-year': {
            'kwargs': {'by_year': True},
            'df': usfhn.non_doctorate.get_non_doctorates_by_taxonomy,
        },
    },
    'attrition': {
        'df': usfhn.attrition.get_attritions_df,
        'by-gender': {
            'age-of-attrition': usfhn.attrition.mean_age_of_attrition,
        },
        'by-self-hire': {
            'risk-ratio': usfhn.attrition.self_hire_to_non_self_hire_risk_ratio,
            'prestige-deciles': {
                'directory_expansions': ALL_RANKS_EXPANDER,
                'do': usfhn.self_hiring.self_hiring_binned_by_prestige_attrition_risk,
            },
        },
        'risk': {
            # for risk, we run a LOT of different risk combinations, so I put
            # them in a function. The keys are:
            # - taxonomy
            # - gender
            # - self-hires
            # - institution, which is expanded over:
            #   - institution_column: ['DegreeInstitutionId', 'InstitutionId']
            **get_attrition_risk_stats_functions(),
            'by-career-stage': {
                'kwargs': {'by_career_stage': True},
                **get_attrition_risk_stats_functions(),
            },
            'by-time': {
                'kwargs': {'by_year': True},
                **get_attrition_risk_stats_functions(),
                'by-career-stage': {
                    'kwargs': {'by_career_stage': True},
                    **get_attrition_risk_stats_functions(),
                },
                'by-degree-year': {
                    'kwargs': {'by_degree_year': True},
                    **get_attrition_risk_stats_functions(),
                },
            },
            'by-degree-year': {
                'kwargs': {'by_degree_year': True},
                **get_attrition_risk_stats_functions(),
            },
            'gender-balanced-hiring': usfhn.attrition.get_gender_balance_given_attrition_and_balanced_hiring,
        },
        'by-rank': {
            'directory_expansions': ALL_RANKS_EXPANDER,
            'df': usfhn.attrition.attritions_df_with_ranks,
            'institution': {
                'risk': usfhn.attrition.institutions_with_ranks,
                'logits': usfhn.attrition.rank_vs_attrition_risk_logit,
                'gendered-logits': {
                    'kwargs': {'by_gender': True},
                    'do': usfhn.attrition.rank_vs_attrition_risk_logit,
                },
                'by-self-hire': {
                    'kwargs': {'by_self_hire': True},
                    'logits': usfhn.attrition.rank_vs_attrition_risk_logit,
                },
            },
            'academia-ranks': {
                'kwargs': {'academia_rank_only': True},
                'df': usfhn.attrition.attritions_df_with_ranks,
                'institution': {
                    'logits': usfhn.attrition.rank_vs_attrition_risk_logit,
                },
            }
        },
    },
    'pool-reduction': {
        'umbrella-stats': usfhn.pool_reduction.umbrella_exclusion_stats,
        'excluded-taxonomy-sizes': usfhn.pool_reduction.excluded_taxonomy_sizes,
        'excluded-at-field-level': {
            'do': usfhn.pool_reduction.people_excluded_at_field_level_by_umbrella,
            'prefix_expansions': {'dataset': ['unfiltered-census-pool', 'alternate-fields']},
        },
        'excluded-at-umbrella-level': {
            'do': usfhn.pool_reduction.people_excluded_at_domain_level_by_umbrella,
            'prefix_expansions': {'dataset': ['unfiltered-census-pool', 'alternate-fields']},
        },
        'paper-stats': {
            'expansion_type': hnelib.runner.JSONExpansion,
            'general': usfhn.pool_reduction.get_stats,
        },
    },
    'geodesics': {
        'df': usfhn.webs.get_geodesics_df,
    },
}


runner = hnelib.runner.DataFrameRunner(
    collection=STATS_COLLECTION,
    directory=usfhn.constants.STATS_PATH,
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='stat runner')
    parser.add_argument('--which', '-w', type=str, help='which thing to run')
    parser.add_argument('--all', default=False, action='store_true')
    parser.add_argument('--run_expansions', '-e', default=False, action='store_true', help='run combinations')
    parser.add_argument('--clean', default=False, action='store_true', help='clean old stats')
    args = parser.parse_args()

    if args.clean:
        runner.clean()

    to_run = []
    if args.which:
        to_run.append(args.which)
    elif args.all:
        to_run = list(runner.collection.keys())

    for item_name in to_run:
        runner.run(item_name, run_expansions=args.run_expansions)
