import numpy as np
import pandas as pd

import usfhn.fieldwork
import usfhn.datasets
import usfhn.constants
import usfhn.institutions


def get_gini_coefficients(
    by_year=False,
    by_new_hire=False,
    by_faculty_rank=False,
    by_seniority=False,
):
    from usfhn.stats import add_groupby_annotations_to_df

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
            'PersonId',
            'DegreeInstitutionId',
        ]
    ].drop_duplicates()
    
    df, extra_groupby_cols = add_groupby_annotations_to_df(
        df,
        by_faculty_rank=by_faculty_rank,
        by_new_hire=by_new_hire,
        by_seniority=by_seniority,
    )

    groupby_cols += extra_groupby_cols

    return get_gini_coefficients_for_df(df, groupby_cols)


def gini_coefficient(population):
    """
    args:
    - population: a list of integer counts

    returns:
    - the gini coefficient for the list


    calculated by way of the simplified version at:
    https://en.wikipedia.org/wiki/Gini_coefficient#Calculation
    """
    numerator = 0
    denominator = 0
    n = len(population)
    for i, y_i in enumerate(sorted(population)):
        numerator += ((2 * i + 1) - n - 1) * y_i
        denominator += n * y_i

    if denominator == 0:
        return 0

    return round(numerator / denominator, 3)


def get_gini_coefficients_for_df(df, groupby_cols):
    """
    expects df to have the following columns:
    - PersonId
    - DegreeInstitutionId
    - `groupby_cols`
    """
    df = df.copy()

    # add in zero values for academia
    institutions = set(usfhn.institutions.get_us_phd_granting_institutions())
    degree_institutions = set(df['DegreeInstitutionId'].unique())
    institutions_to_add_to_academia = institutions - degree_institutions

    df['TotalProduction'] = df.groupby(groupby_cols)['PersonId'].transform('nunique')

    df['InstitutionProduction'] = df.groupby(
        ['DegreeInstitutionId'] + groupby_cols
    )['PersonId'].transform('nunique')

    df['ProductionFraction'] = df['InstitutionProduction'] / df['TotalProduction']
    df = df[
        [
            'ProductionFraction',
            'DegreeInstitutionId',
        ]
        +
        groupby_cols
    ].drop_duplicates()

    academia_rows = []
    for institution in institutions_to_add_to_academia:
        academia_rows.append({
            'DegreeInstitutionId': institution,
            'TaxonomyLevel': 'Academia',
            'TaxonomyValue': 'Academia',
            'ProductionFraction': 0,
        })

    academia_rows = pd.DataFrame(academia_rows)

    for column in [c for c in groupby_cols if c not in ['TaxonomyLevel', 'TaxonomyValue']]:
        extra_academia_rows = []
        for column_value in list(df[column].unique()):
            new_academia_rows = academia_rows.copy()
            new_academia_rows[column] = column_value
            extra_academia_rows.append(new_academia_rows)

        academia_rows = pd.concat(extra_academia_rows)

    df = pd.concat([df, academia_rows])

    gini_coefficients = []
    for _, rows in df.groupby(groupby_cols):
        rows = rows.drop_duplicates(subset=['DegreeInstitutionId'])

        result = {c: rows.iloc[0][c] for c in groupby_cols}
        result['GiniCoefficient'] = gini_coefficient(rows['ProductionFraction'])

        gini_coefficients.append(result)
        # if result['GiniCoefficient']:
        #     gini_coefficients.append(result)

    return pd.DataFrame(gini_coefficients)
