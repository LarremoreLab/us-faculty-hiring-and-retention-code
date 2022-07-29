from functools import lru_cache
import pandas as pd
import functools

import usfhn.datasets
import usfhn.constants


HIGHLY_PRODUCTIVE_NON_US_COUNTRIES = ['Canada', 'United Kingdom']


@lru_cache(maxsize=8)
def get_institution_id_to_name_map(employing=True, producing=True, canonical_name=True):
    id_to_name_map = {}

    if producing:
        df = pd.read_csv(usfhn.constants.AA_2022_DEGREE_INSTITUTION_REMAP_DEGREE_INSTITUTIONS_PATH)
        id_to_name_map.update(df.set_index('DegreeInstitutionId')['DegreeInstitutionName'].to_dict())

    if employing:
        if canonical_name:
            df = pd.read_csv(
                usfhn.constants.AA_2022_DEGREE_INSTITUTION_REMAP_CANONICAL_INSTITUTION_NAMES
            ).drop(columns=[
                'InstitutionName',
            ]).rename(columns={
                'CanonicalName': 'InstitutionName',
            })
        else:
            df = pd.read_csv(usfhn.constants.AA_2022_DEGREE_INSTITUTION_REMAP_INSTITUTIONS_PATH)

        id_to_name_map.update(df.set_index('InstitutionId')['InstitutionName'].to_dict())

    id_to_name_map = {k: v.strip() for k, v in id_to_name_map.items()}

    return id_to_name_map


def annotate_institution_name(
    df,
    id_col='InstitutionId',
    name_col='InstitutionName',
    canonical_name=True,
):
    df = df.copy()

    df[name_col] = df[id_col].apply(get_institution_id_to_name_map(canonical_name=canonical_name).get)

    return df


def annotate_country(df, institution_column='DegreeInstitutionId'):
    if 'CountryName' not in df.columns:
        countries = usfhn.datasets.CURRENT_DATASET.institution_countries[
            [
                'DegreeInstitutionId',
                'CountryName',
            ]
        ].drop_duplicates().rename(columns={
            'DegreeInstitutionId': institution_column,
        })

        df = df.copy().merge(
            countries,
            on=institution_column,
            how='left',
        )

    return df

def annotate_continent(df, institution_column='DegreeInstitutionId'):
    columns = list(df.columns)
    df = annotate_country(df, institution_column=institution_column)

    df = df.merge(
        pd.read_csv(usfhn.constants.INSTITUTION_LOCATION_COUNTRY_CONTINENTS_PATH),
        on='CountryName',
        how='left',
    )

    df = df[columns + ['Continent']]

    return df


def annotate_country_continent(df):
    df = df.merge(
        pd.read_csv(usfhn.constants.INSTITUTION_LOCATION_COUNTRY_CONTINENTS_PATH),
        on='CountryName',
        how='left',
    )

    return df

def annotate_us(df, institution_column='DegreeInstitutionId', drop_country_name=True):
    df = df.copy()
    df = annotate_country(df, institution_column=institution_column)

    df['US'] = df['CountryName'] == 'United States'

    if drop_country_name:
        df = df.drop(columns=['CountryName'])

    return df


def filter_to_non_us(df):
    columns = list(df.columns)
    df = annotate_country(df)

    df = df[
        df['CountryName'].notna()
    ]

    df = annotate_us(df)

    df = df[
        ~df['US']
    ]

    df = df[columns]
    return df


def annotate_highly_productive_non_us_countries(df):
    """
    see HIGHLY_PRODUCTIVE_NON_US_COUNTRIES for list
    """
    columns = list(df.columns)

    df = annotate_country(df)
    df['IsHighlyProductiveNonUSCountry'] = df['CountryName'].isin(HIGHLY_PRODUCTIVE_NON_US_COUNTRIES)

    df = df[
        columns + ['IsHighlyProductiveNonUSCountry']
    ]

    return df


@functools.lru_cache
def get_us_phd_granting_institutions(dataset_name=None, require_employing=True):
    """
    we are taking the original set of employing institutions PhD granting
    """
    df = pd.read_csv(usfhn.constants.AA_2022_DEGREE_INSTITUTION_REMAP_INSTITUTIONS_PATH)
    return list(df['InstitutionId'].unique())


@functools.lru_cache
def get_unfiltered_producting_insts():
    """
    we are taking the original set of employing institutions PhD granting
    """
    df = pd.read_csv(usfhn.constants.AA_2022_DEGREE_INSTITUTION_REMAP_DEGREE_INSTITUTIONS_PATH)
    return list(df['DegreeInstitutionId'].unique())
