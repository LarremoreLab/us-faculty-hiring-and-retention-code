import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import PerfectSeparationError

import hnelib.model

import usfhn.datasets
import usfhn.fieldwork
import usfhn.views


def get_employment_interdisciplinarity_df(by_year=False):
    """
    NOTE: this is NOT based on closedness data, so it is the fraction of
    within-sample U.S. PhD-holding hires in a field that are from out of field.

    returns:
    - PersonId
    - InstitutionId
    - DegreeInstitutionId
    - TaxonomyLevel
    - TaxonomyValue
    - InField: boolean. True if the DegreeInstitutionId is in-field

    if `by_year` == True, adds `Year` column
    """
    columns = [
        'PersonId',
        'InstitutionId',
        'DegreeInstitutionId',
        'TaxonomyLevel',
        'TaxonomyValue',
    ]

    if by_year:
        columns.append('Year')


    df = usfhn.datasets.get_dataset_df('data', by_year=by_year)

    df = usfhn.fieldwork.linearize_df_taxonomy_hierarchy(df)

    df = df[columns].drop_duplicates()

    degree_institution_taxonomies = df[
        [
            'InstitutionId',
            'TaxonomyLevel',
            'TaxonomyValue',
        ]
    ].drop_duplicates().rename(columns={
        'InstitutionId': 'DegreeInstitutionId',
    })

    degree_institution_taxonomies['InField'] = True

    df = df.merge(
        degree_institution_taxonomies,
        on=[
            'TaxonomyLevel',
            'TaxonomyValue',
            'DegreeInstitutionId',
        ],
        how='left',
    )

    df['InField'] = df['InField'].fillna(False)

    return df


def get_taxonomy_interdisciplinarity(by_year=False, by_institution=False, by_degree_institution=False):
    import usfhn.stats

    groupby_cols = [
        'TaxonomyLevel',
        'TaxonomyValue',
    ]

    if by_year:
        groupby_cols.append('Year')
        df = usfhn.stats.runner.get(f'interdisciplinarity/by-year/df')
    else:
        df = usfhn.stats.runner.get(f'interdisciplinarity/df')

    if by_institution:
        groupby_cols.append('InstitutionId')

    if by_degree_institution:
        groupby_cols.append('DegreeInstitutionId')

    df = df[
        groupby_cols +
        [
            'PersonId',
            'InField'
        ]
    ].drop_duplicates()

    df['Faculty'] = df.groupby(groupby_cols)['PersonId'].transform('nunique')
    df = df[
        ~df['InField']
    ].drop(columns=['InField'])

    df['OutOfFieldFaculty'] = df.groupby(groupby_cols)['PersonId'].transform('nunique')
    df = df[
        groupby_cols +
        [
            'Faculty',
            'OutOfFieldFaculty',
        ]
    ].drop_duplicates()

    df['OutOfFieldFraction'] = df['OutOfFieldFaculty'] / df['Faculty']

    return df


def get_institution_interdisciplinarity(by_year=False):
    return get_taxonomy_interdisciplinarity(by_year=by_year, by_institution=True)


def get_degree_institution_interdisciplinarity(by_year=False):
    return get_taxonomy_interdisciplinarity(by_year=by_year, by_degree_institution=True)

def get_new_hire_fraction(by_year=False):
    import usfhn.stats

    groupby_cols = [
        'TaxonomyLevel',
        'TaxonomyValue',
    ]

    if by_year:
        groupby_cols.append('Year')
        df = usfhn.stats.runner.get('interdisciplinarity/by-year/df')
        new_hire_df = usfhn.stats.runner.get('new-hire/by-year/df')
    else:
        df = usfhn.stats.runner.get('interdisciplinarity/df')
        new_hire_df = usfhn.stats.runner.get('new-hire/df')

    new_hire_df = new_hire_df[
        ['PersonId'] + groupby_cols
    ]

    df = df[
        groupby_cols +
        [
            'PersonId',
            'InField'
        ]
    ].drop_duplicates()

    df = df.merge(
        new_hire_df,
        on=['PersonId'] + groupby_cols
    )

    df['Faculty'] = df.groupby(groupby_cols)['PersonId'].transform('nunique')
    df = df[
        ~df['InField']
    ].drop(columns=['InField'])

    df['OutOfFieldFaculty'] = df.groupby(groupby_cols)['PersonId'].transform('nunique')
    df = df[
        groupby_cols +
        [
            'Faculty',
            'OutOfFieldFaculty',
        ]
    ].drop_duplicates()

    df['OutOfFieldFraction'] = df['OutOfFieldFaculty'] / df['Faculty']

    return df


def institutions_with_ranks(rank_type='production'):
    """
    rank_types:
    - production
    - doctoral-institution
    - employing-institution

    """
    import usfhn.stats

    ranks = usfhn.stats.runner.get('ranks/df', rank_type=rank_type)

    if rank_type == 'production':
        institution_column = 'DegreeInstitutionId'
        ranks = ranks.rename(columns={
            'InstitutionId': 'DegreeInstitutionId',
        })
    else:
        institution_column = 'InstitutionId'

    if institution_column == 'InstitutionId':
        df = usfhn.stats.runner.get('interdisciplinarity/institution')
    else:
        df = usfhn.stats.runner.get('interdisciplinarity/degree-institution')

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


def rank_vs_interdisciplinarity_logit(rank_type='production'):
    import usfhn.stats

    df = usfhn.stats.runner.get('interdisciplinarity/df', rank_type=rank_type)

    df['OutOfField'] = ~df['InField']
    df['OutOfField'] = df['OutOfField']

    ranks = usfhn.stats.runner.get('ranks/df', rank_type=rank_type)

    if rank_type == 'production':
        institution_column = 'DegreeInstitutionId'
        ranks = ranks.rename(columns={
            'InstitutionId': 'DegreeInstitutionId',
        })
    else:
        institution_column = 'InstitutionId'

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

    # don't drop duplicates, because we want them by year and PersonId
    df = df[
        [
            'OutOfField',
            'TaxonomyLevel',
            'TaxonomyValue',
            'Percentile',
        ]
    ]

    return hnelib.model.get_logits(
        df,
        endog='OutOfField',
        exog='Percentile',
        groupby_cols=[
            'TaxonomyLevel',
            'TaxonomyValue',
        ],
    )
