import numpy as np
import pandas as pd
import itertools
from functools import lru_cache
from statsmodels.stats.proportion import proportions_ztest

import usfhn.datasets
import usfhn.views
import usfhn.fieldwork

def get_closedness(df): 
    """
    Closedness will be calculated only for people who have degrees.
    (People with null DegreeInstitutionIds will be excluded)
    """
    df = df[df['DegreeInstitutionId'].notnull()]
    denominator = df['PersonId'].nunique()

    numerator = df[
        df['DegreeInstitutionId'].isin(df['InstitutionId'].unique())
    ]['PersonId'].nunique()

    closedness = numerator / denominator if denominator else 0

    return closedness

def lower_level_to_higher_level_closedness(df, lower_level, higher_level):
    """
    Return the closedness of one level of the taxonomization hierarchy within another.

    The way this works is we add dummy rows of `[higher_level], InstitutionId`
    to inflate the institutions in the df we calculate closedness with
    """
    levels_df = df[
        [
            lower_level,
            higher_level,
        ]
    ].drop_duplicates()

    higher_df = df[
        [
            higher_level,
            'InstitutionId',
        ]
    ].drop_duplicates().merge(
        levels_df,
        on=higher_level
    )

    higher_df['PersonId'] = np.nan
    higher_df['DegreeInstitutionId'] = -1

    lower_df = df[
        ['PersonId', 'DegreeInstitutionId', 'InstitutionId', lower_level, higher_level]
    ].drop_duplicates()

    closedness_calculation_df = pd.concat([lower_df, higher_df])

    closednesses = []
    for (lower, higher), rows in closedness_calculation_df.groupby([lower_level, higher_level]):
        closednesses.append({
            'Closedness': get_closedness(rows),
            lower_level: lower,
            higher_level: higher,
        })

    closednesses = pd.DataFrame(closednesses)

    return closednesses


def get_closedness_subtype(
    df,
    denominator_filters={},
    numerator_filters={},
):
    """
    Each row should have the following columns:
    - PersonId
    - InstitutionId
    - DegreeInstitutionId

    numerator_filters/denominator_filters is a dict of:
    - key: Column to filter
    - value: a tuple of two:
        - a filter type:
            - 'Equals' (to required a value)
            - 'IsIn'
            - 'NotIsIn'
        - the value for the filter.
            - If 'Equals': a literal value
            - If 'IsIn': a column to require the `Column` to be in
            - If 'NotIsIn': a column to require the `Column` to not be in

    Closedness will be calculated only for people who have degrees.
    (People with null DegreeInstitutionIds will be excluded)
    """
    for column, (filter_type, filter_value) in denominator_filters.items():
        df = apply_df_filter(df, column, filter_type, filter_value)

    denominator = df['PersonId'].nunique()

    numerator_df = df
    for column, (filter_type, filter_value) in numerator_filters.items():
        numerator_df = apply_df_filter(numerator_df, column, filter_type, filter_value)

    numerator = numerator_df['PersonId'].nunique()

    return numerator


def get_closedness_subtypes_dict(df):
    df = df.copy()

    df['PhD'] = df['DegreeInstitutionId'].notnull()
    df['US'] = df['CountryName'] == 'United States'

    phd = get_closedness_subtype(df, numerator_filters={'PhD': ('Equals', True)})
    no_phd = get_closedness_subtype(df, numerator_filters={'PhD': ('Equals', False)})

    us = get_closedness_subtype(df, numerator_filters={'US': ('Equals', True)})
    non_us_phd = phd - us

    us_phd_in_field = get_closedness_subtype(
        df, 
        numerator_filters={
            'PhD': ('Equals', True),
            'US': ('Equals', True),
            'DegreeInstitutionId': ('IsIn', 'InstitutionId'),
        }
    )

    us_phd_out_of_field = get_closedness_subtype(
        df, 
        numerator_filters={
            'PhD': ('Equals', True),
            'US': ('Equals', True),
            'DegreeInstitutionId': ('NotIsIn', 'InstitutionId'),
        }
    )

    faculty_count = phd + no_phd

    counts = {
        'PhD': phd,
        'NonPhD': no_phd,
        'US': us,
        'NonUSPhD': non_us_phd,
        'USPhDInField': us_phd_in_field,
        'USPhDOutOfField': us_phd_out_of_field,
    }

    fractions = {f"{k}Fraction": v / faculty_count for k, v in counts.items()}

    return {
        'Closedness': get_closedness(df),
        'FacultyCount': faculty_count,
        **counts,
        **fractions,
    }


def apply_df_filter(df, column, filter_type, filter_value):
    if filter_type == 'IsIn':
        df_filter = df[column].isin(df[filter_value].unique())
    elif filter_type == 'NotIsIn':
        df_filter = ~df[column].isin(df[filter_value].unique())
    elif filter_type == 'Equals':
        df_filter = df[column] == filter_value

    return df[df_filter]


@lru_cache(maxsize=128)
def get_closednesses():
    """
    We _include_ non-us/employing institutions at this level, because it is the lower bound.

    we have the following attributes for "closedness":
    - PhD: boolean
    - In-Field: boolean
    - US: boolean

    so we have the boolean expansion of types of closedness:
    1. PhD
    2. NoPhD
    3. US
    4. NonUS
    -
    7. US+PhD+InField
    8. US+PhD+OutOfField
    9. NonUS+PhD
    -
    9. Closedness: {PhD+InField}/{everyone}
    """
    return _get_closednesses()


def _get_closednesses():
    df = usfhn.datasets.get_dataset_df('closedness_data')
    df = df[
        ['DegreeInstitutionId', 'InstitutionId', 'Taxonomy', 'PersonId']
    ].drop_duplicates()

    df = df.merge(
        usfhn.datasets.CURRENT_DATASET.institution_countries,
        on='DegreeInstitutionId',
        how='left',
    )

    df = usfhn.fieldwork.linearize_df_taxonomy_hierarchy(df)

    closednesses = []
    for (level, value), rows in df.groupby(['TaxonomyLevel', 'TaxonomyValue']):
        closednesses.append({
            **get_closedness_subtypes_dict(rows),
            'TaxonomyLevel': level,
            'TaxonomyValue': value,
        })

    closednesses = pd.DataFrame(closednesses).sort_values(by=[
        'TaxonomyLevel',
        'TaxonomyValue',
        'Closedness',
    ])

    return closednesses

@lru_cache(maxsize=128)
def get_closedness_across_levels():
    """
    Like the above function, but ignores TaxonomyValue
    """
    df = get_closednesses()

    closednesses = []
    for level, rows in df.groupby(['TaxonomyLevel']):
        faculty_count = rows['FacultyCount'].sum()
        phd = rows['PhD'].sum()
        no_phd = rows['NonPhD'].sum()
        us = rows['US'].sum()
        non_us_phd = rows['NonUSPhD'].sum()
        us_phd_in_field = rows['USPhDInField'].sum()
        us_phd_out_of_field = rows['USPhDOutOfField'].sum()

        closednesses.append({
            'FacultyCount': faculty_count,
            'PhD': phd,
            'NonPhD': no_phd,
            'US': us,
            'NonUSPhD': non_us_phd,
            'USPhDInField': us_phd_in_field,
            'USPhDOutOfField': us_phd_out_of_field,
            'TaxonomyLevel': level,
        })

    closednesses = pd.DataFrame(closednesses).sort_values(by=[
        'TaxonomyLevel',
    ])

    return closednesses


@lru_cache(maxsize=128)
def get_closedness_across_levels_at_level(taxonomy_level):
    df = usfhn.views.filter_exploded_df(get_closedness_across_levels())
    return df[
        df['TaxonomyLevel'] == taxonomy_level
    ].iloc[0]


@lru_cache(maxsize=128)
def get_closedness_at_level_and_value(taxonomy_level, taxonomy_value):
    df = usfhn.views.filter_exploded_df(get_closednesses())
    return df[
        (df['TaxonomyLevel'] == taxonomy_level)
        &
        (df['TaxonomyValue'] == taxonomy_value)
    ].iloc[0]



def get_level_to_level_proportionality_tests(threshold=1.96):
    """
    we're testing the proportionality of Field-Area and Field-Umbrella
    rates of out of field hiring
    """
    closednesses = get_closedness_across_levels()
    df = []
    for upper_level in ['Area', 'Umbrella', 'Academia']:
        field_row = closednesses[
            closednesses['TaxonomyLevel'] == 'Field'
        ].iloc[0]

        upper_row = closednesses[
            closednesses['TaxonomyLevel'] == upper_level
        ].iloc[0]

        field_n = field_row['FacultyCount']
        field_out_of_field = field_row['USPhDOutOfField']

        upper_n = upper_row['FacultyCount']
        upper_out_of_field = upper_row['USPhDOutOfField']

        z_value, p_value = proportions_ztest(
            [field_out_of_field, upper_out_of_field],
            [field_n, upper_n],
        )
        df.append({
            'LowerTaxonomyLevel': 'Field',
            'UpperTaxonomyLevel': upper_level,
            'Z': z_value,
            'P': p_value,
            'RejectNull': z_value > threshold,
        })

    return pd.DataFrame(df)


def get_proportionality_tests_for_non_us_vs_out_of_field_at_level(taxonomy_level, threshold=1.96):
    df = usfhn.views.filter_exploded_df(get_closednesses())
    df = df[
        df['TaxonomyLevel'] == taxonomy_level
    ]

    new_df = []
    for i, row in df.iterrows():
        z_value, p_value = proportions_ztest(
            [
                row['NonUSPhD'],
                row['USPhDOutOfField'],
            ],
            [
                row['FacultyCount'],
                row['FacultyCount'],
            ],
        )

        z_value = abs(z_value)

        new_df.append({
            'TaxonomyLevel': taxonomy_level,
            'TaxonomyValue': row['TaxonomyValue'],
            'P': p_value,
            'Z': z_value,
            'RejectNull': z_value > threshold,
        })

    return pd.DataFrame(new_df)


def get_level_value_to_level_value_proportionality_tests(threshold=1.96):
    """
    we're testing the proportionality of Field-Area and Field-Umbrella
    rates of out of field hiring
    """
    closednesses = usfhn.views.filter_exploded_df(get_closednesses())[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'FacultyCount',
            'USPhDOutOfField',
        ]
    ].drop_duplicates()

    levels = ['Field', 'Area', 'Umbrella', 'Academia']

    closednesses = closednesses[
        closednesses['TaxonomyLevel'].isin(levels)
    ]

    taxonomy = usfhn.views.get_taxonomization()[
        levels
    ].drop_duplicates()

    df = []
    for i, row in closednesses.iterrows():
        level = row['TaxonomyLevel']
        value = row['TaxonomyValue']

        level_index = levels.index(level)
        levels_above_current_level = levels[level_index + 1:]

        taxonomy_row = taxonomy[taxonomy[level] == value].iloc[0]

        for upper_level in levels_above_current_level:
            upper_value = taxonomy_row[upper_level]

            if not upper_value:
                continue

            upper_row = closednesses[
                (closednesses['TaxonomyLevel'] == upper_level)
                &
                (closednesses['TaxonomyValue'] == upper_value)
            ].iloc[0]

            z_value, p_value = proportions_ztest(
                [
                    row['USPhDOutOfField'],
                    upper_row['USPhDOutOfField'],
                ],
                [
                    row['FacultyCount'],
                    upper_row['FacultyCount'],
                ],
            )
            df.append({
                'LowerTaxonomyLevel': level,
                'LowerTaxonomyValue': value,
                'UpperTaxonomyLevel': upper_level,
                'UpperTaxonomyValue': upper_value,
                'Z': z_value,
                'P': p_value,
                'RejectNull': z_value > threshold,
            })

    return pd.DataFrame(df)
