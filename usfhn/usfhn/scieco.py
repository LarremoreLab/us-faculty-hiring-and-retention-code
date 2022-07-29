import os
import sys
import pandas as pd
from pathlib import Path
from datetime import date

import usfhn.constants
import usfhn.datasets
import usfhn.views


RANK_COLUMNS = [
    'Rank',
    'NormalizedRank',
    'OrdinalRank',
    'Percentile',
    'TaxonomyLevel',
    'TaxonomyValue',
    'InstitutionId',
]

RANK_COLUMN_RENAMES = {
    'NormalizedRank': 'ZeroToOneRank',
    'Percentile': 'PercentileRank',
}


def get_date_string():
    return date.today().strftime('%Y%m%d')


def get_scieco_dataset_path(dataset_type):
    date = get_date_string()
    return usfhn.constants.SCIECO_DATA_AA_PATH.joinpath(dataset_type, f'{date}.gz')


def save_data():
    df = usfhn.datasets.get_dataset_df('verbose_data', usfhn.constants.SCIECO_DATASET)
    df.to_csv(
        get_scieco_dataset_path('data'),
        index=False,
    )

def save_degree_institution_countries():
    df = pd.read_csv(usfhn.constants.AA_2022_DEGREE_INSTITUTION_REMAP_INSTITUTION_COUNTRIES_PATH)
    df.to_csv(
        get_scieco_dataset_path('degree-institution-countries'),
        index=False,
    )

def save_ranks():
    format_and_save_ranks(
        usfhn.datasets.get_dataset_df('ranks', usfhn.constants.SCIECO_DATASET),
        get_scieco_dataset_path('ranks'),
        RANK_COLUMNS,
    )

def save_ranks_by_year():
    format_and_save_ranks(
        usfhn.datasets.get_dataset_df('ranks_by_year', usfhn.constants.SCIECO_DATASET),
        get_scieco_dataset_path('ranks-by-year'),
        RANK_COLUMNS + ['Year'],
    )

def format_and_save_ranks(df, path, columns):
    df = df[columns].drop_duplicates()
    df = df.rename(columns=RANK_COLUMN_RENAMES)

    df.to_csv(
        path,
        index=False,
    )

def push_changes():
    cwd = Path.cwd()
    os.chdir(usfhn.constants.SCIECO_DATA_PATH)
    os.system('git add .')
    os.system(f'git commit -m "updating aa data {get_date_string()}"')
    os.system("git push")
    os.chdir(cwd)


def update_scieco():
    if not usfhn.constants.SCIECO_DATA_PATH.exists():
        return

    save_data()
    save_ranks()
    save_ranks_by_year()
    save_degree_institution_countries()
    push_changes()


if __name__ == '__main__':
    update_scieco()
