import pandas as pd

import usfhn.constants
import usfhn.datasets
import usfhn.fieldwork


def get_taxonomy_size(by_year=False, count_col='PersonId'):
    """
    returns df:
    - TaxonomyLevel
    - TaxonomyValue
    - Year (if 'by_year')
    - TotalCount: total faculty/institutions in academia
    - Count: faculty/institutions in taxonomy
    - Fraction: faculty/institutions / TotalCount
    """
    groupby_cols = [
        'TaxonomyLevel',
        'TaxonomyValue',
    ]

    overall_cols = []

    if by_year:
        groupby_cols.append('Year')
        overall_cols.append('Year')

    df = usfhn.datasets.get_dataset_df('closedness_data', by_year=by_year)

    df = usfhn.fieldwork.linearize_df_taxonomy_hierarchy(df)

    df = df[
        groupby_cols + [
            count_col
        ]
    ].drop_duplicates()

    if overall_cols:
        df['TotalCount'] = df.groupby(overall_cols)[count_col].transform('nunique')
    else:
        df['TotalCount'] = df[count_col].nunique()

    df['Count'] = df.groupby(groupby_cols)[count_col].transform('nunique')

    df = df[
        groupby_cols + [
            'TotalCount',
            'Count',
        ]
    ].drop_duplicates()

    df['Fraction'] = df['Count'] / df['TotalCount']

    return df
