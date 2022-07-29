import pandas as pd
import json
from functools import lru_cache

import hnelib.plot

import usfhn.constants as constants
import usfhn.datasets
import usfhn.fieldwork
import usfhn

from usfhn.plot_utils import PLOT_VARS


################################################################################
#
#
# Query writing help
#
#
################################################################################
def explode_gender(df):
    gender_known = df[df['Gender'].isin(['Male', 'Female'])].copy(deep=True)
    gender_known['Gender'] = constants.GENDER_KNOWN
    gender_agnostic = df.copy(deep=True)
    gender_agnostic['Gender'] = constants.GENDER_AGNOSTIC
    return pd.concat([df, gender_known, gender_agnostic])


def filter_gender(df, gender=constants.GENDER_AGNOSTIC):
    if 'Gender' in df.columns:
        df = df[
            df['Gender'] == gender
        ].drop(columns=['Gender'])

    return df

def filter_generalized_df(df, taxonomy_level='Area', gender=constants.GENDER_AGNOSTIC):
    df = df[
        (df['Gender'] == gender)
        &
        (df['TaxonomyLevel'] == taxonomy_level)
    ].copy()
    df[taxonomy_level] = df['TaxonomyValue']
    df = df.drop(columns=['Gender', 'TaxonomyLevel'])
    return df

def filter_exploded_df(
    df,
    year=constants.YEAR_UNION,
    gender=constants.GENDER_AGNOSTIC,
    drop_columns=True,
):
    columns_to_drop = []

    if year != None and 'Year' in df.columns:
        df = df[
            df['Year'] == year
        ]
        columns_to_drop.append('Year')

    if gender != None and 'Gender' in df.columns:
        df = df[
            df['Gender'] == gender
        ]
        columns_to_drop.append('Gender')

    if drop_columns:
        df = df.drop(columns=columns_to_drop)

    return df


def filter_by_taxonomy(df, level=None, value=None):
    df = df.copy()
    if level:
        df = df[
            df['TaxonomyLevel'] == level
        ]

    if value:
        df = df[
            df['TaxonomyValue'] == value
        ]

    return df


def filter_to_academia_and_domain(df):
    df = df[
        df['TaxonomyLevel'].isin(['Academia', 'Umbrella'])
    ]

    return df


################################################################################
#
#
# Taxonomization
#
#
################################################################################
@lru_cache(maxsize=1)
def get_taxonomization():
    return usfhn.datasets.CURRENT_DATASET.data[
        ['Taxonomy', 'Field', 'Area', 'Umbrella', 'Academia']
    ].drop_duplicates()

@lru_cache(maxsize=1)
def get_linear_taxonomization():
    return usfhn.fieldwork.linearize_df_taxonomy_hierarchy(get_taxonomization())

################################################################################
#
#
# Umbrella and coloring
#
#
################################################################################
def annotate_umbrella(df, taxonomization_level, taxonomization_column='TaxonomyValue'):
    df = df.copy()

    if taxonomization_level == 'Academia':
        df['Umbrella'] = 'Academia'

    if taxonomization_level not in df.columns:
        df[taxonomization_level] = df[taxonomization_column]

    if 'Umbrella' not in df.columns:
        taxonomy_to_umbrella = get_taxonomization().set_index(taxonomization_level)['Umbrella'].to_dict()
        df['Umbrella'] = df[taxonomization_level].apply(taxonomy_to_umbrella.get)

    return df


def annotate_umbrella_color(df, taxonomization_level='Field', taxonomization_column='TaxonomyValue'):
    df = df.copy()
    if 'Umbrella' not in df.columns:
        df = annotate_umbrella(df, taxonomization_level, taxonomization_column)

    df['UmbrellaColor'] = df['Umbrella'].apply(PLOT_VARS['colors']['umbrellas'].get)

    df['FadedUmbrellaColor'] = df['UmbrellaColor'].apply(hnelib.plot.set_alpha_on_colors)

    return df


def get_gender_df_and_rename_non_join_columns(df, gender):
    df = df.copy()[
        df['Gender'] == gender
    ].drop(columns=['Gender'])

    join_columns = ['InstitutionId', 'Year', 'TaxonomyLevel', 'TaxonomyValue']

    df = df.rename(columns={col: f"{gender}{col}" for col in df.columns if col not in join_columns})
    return df


def get_df_with_male_and_female_stats(df, stat_column):
    possible_join_columns = ['InstitutionId', 'Year', 'TaxonomyLevel', 'TaxonomyValue']
    join_columns = [c for c in possible_join_columns if c in df.columns]

    female_df = get_df_with_stat_column_for_gender(df, 'Female', stat_column)[
        join_columns + [f"Female{stat_column}"]
    ].drop_duplicates()
    male_df = get_df_with_stat_column_for_gender(df, 'Male', stat_column)[
        join_columns + [f"Male{stat_column}"]
    ].drop_duplicates()

    return female_df.merge(male_df, on=join_columns)


def get_df_with_stat_column_for_gender(df, gender, stat_column):
    df = df.copy()
    df = df[
        df['Gender'] == gender
    ]
    df = df.rename(columns={
        stat_column: f"{gender}{stat_column}",
    })

    return df


@lru_cache(maxsize=1)
def get_fields():
    return sorted(list(set(f for f in get_taxonomization()['Field'].unique() if pd.notnull(f))))


@lru_cache(maxsize=1)
def _get_umbrellas():
    df = get_taxonomization()
    df = df[
        df['Umbrella'].notna()
    ]
    return sorted(list(set(df['Umbrella'].unique())))


def get_umbrellas():
    return [u for u in _get_umbrellas()]


@lru_cache(maxsize=1)
def employing_department_ids():
    employment = get_employment()
    return set(employment.DepartmentId.unique())


@lru_cache(maxsize=1)
def degree_department_ids():
    degree_departments = get_degree_departments()
    return set(degree_departments.DegreeDepartmentId.unique())


@lru_cache(maxsize=1)
def get_years():
    return sorted(list(usfhn.datasets.get_dataset_df('data')['Year'].unique()))

################################################################################
#
#
# Departments
#
#
################################################################################
@lru_cache(maxsize=1)
def department_id_to_name():
    return usfhn.dataframes.departments.set_index('DepartmentId')['DepartmentName'].to_dict()


################################################################################
#
#
# Institutions
#
#
################################################################################
@lru_cache(maxsize=1)
def institution_id_to_name():
    return usfhn.dataframes.institutions.set_index('InstitutionId')['InstitutionName'].to_dict()


################################################################################
#
#
# People
#
#
################################################################################
@lru_cache(maxsize=1)
def person_id_to_degree_year():
    return usfhn.dataframes.degrees.set_index('PersonId')['DegreeYear'].to_dict()


################################################################################
#
#
# Etcetera
#
#
################################################################################
@lru_cache(maxsize=1)
def get_appointments():
    # add degree information
    appointments = get_employment().merge(
        degrees = usfhn.dataframes.degrees,
        on=['PersonId'],
    ).drop_duplicates()

    return appointments


@lru_cache(maxsize=1)
def get_degree_departments():
    df = pd.read_csv(constants.PHD_DEPARTMENTS_PATH)
    # A couple of departments have been filtered out at this point (112
    # departments associated with 922 people, 335 of whom have other phd
    # department rows)
    df = df[df.DepartmentId.isin(employing_department_ids())]
    df['DepartmentName'] = df['DepartmentId'].apply(department_id_to_name().get)
    df['DegreeYear'] = df['PersonId'].apply(person_id_to_degree_year().get)

    df = usfhn.institutions.annotate_institution_name(df)

    df = df.rename(columns={
        'DepartmentId': 'DegreeDepartmentId',
        'DepartmentName': 'DegreeDepartmentName',
        'InstitutionId': 'DegreeInstitutionId',
        'InstitutionName': 'DegreeInstitutionName',
    })

    return df


@lru_cache(maxsize=1)
def get_department_faculty_hiring():
    return get_employment().merge(
        get_degree_departments(),
        on='PersonId'
    )
