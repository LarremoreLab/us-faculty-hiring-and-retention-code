import pandas as pd

import usfhn.datasets
import usfhn.fieldwork

def get_non_doctorate_fraction_by_taxonomy(by_year=False, count_col='PersonId'):
    """
    returns df:
    - TaxonomyLevel
    - TaxonomyValue
    - Year (if 'by_year')
    - TotalCount: faculty/institutions in taxonomy
    - Count: non-doctorate faculty/institutions in taxonomy
    - Fraction: faculty/institutions / TotalCount
    """
    groupby_cols = [
        'TaxonomyLevel',
        'TaxonomyValue',
    ]

    if by_year:
        groupby_cols.append('Year')

    df = usfhn.datasets.get_dataset_df('closedness_data', by_year=by_year)

    df = usfhn.fieldwork.linearize_df_taxonomy_hierarchy(df)

    df = df[
        groupby_cols + [
            count_col,
            'DegreeInstitutionId',
        ]
    ].drop_duplicates()

    df['TotalCount'] = df.groupby(groupby_cols)[count_col].transform('nunique')

    df = df[
        df['DegreeInstitutionId'].isna()
    ]

    df['Count'] = df.groupby(groupby_cols)[count_col].transform('nunique')

    df = df[
        groupby_cols + [
            'TotalCount',
            'Count',
        ]
    ].drop_duplicates()

    df['Fraction'] = df['Count'] / df['TotalCount']

    return df

