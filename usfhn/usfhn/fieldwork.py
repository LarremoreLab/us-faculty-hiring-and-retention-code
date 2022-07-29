import pandas as pd
import numpy as np
import json
from functools import lru_cache
import itertools
from collections import defaultdict
import scipy.stats
from sklearn.linear_model import LinearRegression
from dataclasses import dataclass

import usfhn.closedness
import usfhn.datasets


def get_linear_taxonomy_from_df(df):
    columns = ['Taxonomy', 'Field', 'Area', 'Umbrella', 'Academia']
    columns = [c for c in columns if c in df.columns]
    dfs = []
    for column in columns:
        new_df = df.copy()
        new_df['TaxonomyLevel'] = column
        new_df['TaxonomyValue'] = new_df[column]
        new_df = new_df.drop(columns=columns)
        dfs.append(new_df)

    return pd.concat(dfs).drop_duplicates()


def linearize_df_taxonomy_hierarchy(df, taxonomy_level_exclusions=['Taxonomy', 'Area']):
    taxonomy_cols = ['Taxonomy', 'Field', 'Area', 'Umbrella', 'Academia']

    if all([t in df.columns for t in taxonomy_cols]):
        taxonomization = df.copy()
    else:
        taxonomization = usfhn.datasets.get_dataset_df('data')

    taxonomization = taxonomization[taxonomy_cols].drop_duplicates()

    linearized_taxonomization_df = []
    for i, row in taxonomization.iterrows():
        for taxonomy_level in taxonomization.columns:
            linearized_taxonomization_df.append({
                'TaxonomyLevel': taxonomy_level,
                'TaxonomyValue': row[taxonomy_level],
                'Taxonomy': row['Taxonomy'],
            })

    linearized_taxonomization_df = pd.DataFrame(linearized_taxonomization_df)
    linearized_taxonomization_df = linearized_taxonomization_df[
        linearized_taxonomization_df['TaxonomyValue'].notnull()
    ].drop_duplicates()

    df = df.merge(
        linearized_taxonomization_df,
        on='Taxonomy'
    )

    for taxonomization_level in taxonomization.columns:
        if taxonomization_level in df.columns:
            df = df.drop(columns=[taxonomization_level])

    df['TaxonomyValue'] = df['TaxonomyValue'].astype(str)

    if taxonomy_level_exclusions:
        df = df[
            ~df['TaxonomyLevel'].isin(taxonomy_level_exclusions)
        ]

    return df


def get_analysis_data_with_one_area_for_cross_listed_taxonomies(df):
    """
    For each taxonomy that is in multiple areas, use the area that taxonomy is
    more closed in. 

    Note: we use the closedness for the union on years
    """
    closednesses = usfhn.closedness.lower_level_to_higher_level_closedness(
        df,
        'Taxonomy',
        'Area',
    )

    closednesses['MaxClosedness'] = closednesses.groupby(
        'Taxonomy'
    )['Closedness'].transform('max')

    closednesses = closednesses[
        closednesses['Closedness'] == closednesses['MaxClosedness']
    ].drop(columns=['MaxClosedness', 'Closedness'])

    # add in taxonomies without areas
    original_taxonomies = set(list(df['Taxonomy'].unique()))
    closedness_taxonomies = set(list(closednesses['Taxonomy'].unique()))
    arealess_taxonomies = original_taxonomies - closedness_taxonomies

    arealess_df = df[
        df['Taxonomy'].isin(arealess_taxonomies)
    ]

    df = df.merge(
        closednesses,
        on=['Taxonomy', 'Area']
    )

    df = pd.concat([df, arealess_df])

    return df
