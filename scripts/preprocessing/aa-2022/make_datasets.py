import argparse
import shutil
import pandas as pd
import numpy as np
import functools

import usfhn.constants
import usfhn.fieldwork
import usfhn.datasets
import usfhn.rank
import usfhn.scieco
import usfhn.stats
import usfhn.institutions
import usfhn.utils


DATASETS_TO_RANK = [
    'default',
    'scieco-data',
]


DATASETS_TO_CLEAR_STATS_FOR = {'default', 'dynamics', 'unfiltered-census-pool'}


def filter_dataset(df, conditions, for_closedness=False):
    """
    All datasets are prefiltered to be tenure-track professors and PhD granting
    institutions only.

    flags: 
    - DepartmentsInAllYears: Boolean. If True, require departments to be in all years of
      the dataset
    - DepartmentsInThresholdYears: Boolean. If True, require departments to be
      in usfhn.constants.YEAR_REQUIREMENT years.
    - LatestInstitutionEmployment: Boolean. If true, exclude employment rows at
      institutions that predate the person's most recent institution (only
      relevant for people who make mid-career moves)
    - MinimumTaxonomySize: Int. Remove taxonomies smaller than this number.
    - USPhDGrantingDegreeInstitutions: Boolean. If True, filter degree
      granting institutions to U.S. PhD granting institutions
    - YearUnion: Boolean. If true, set year = True and drop duplicates.
    - Years: List of ints. Only consider employment records within these years.
    - UmbrellaExclusions: List of strings. Exclude these umbrellas from the
      dataset.
    - AlternateTaxonomy: if true, use the alternate field taxonomy
    - ImputeMissingYearsIntoAcademia: defaults to True. If true, add
      'academia' appointments in years where people have gaps in employment
    - PrimaryAppointments: if true, exclude non-primary appointments

    If the `for_closedness` flag is true, we won't filter degree institution ids at all
    """
    df = df.copy(deep=True)

    columns = list(df.columns)

    filters = []

    if conditions.get('Years'):
        df['IncludeYear'] = df['Year'].isin(conditions.get('Years'))
        filters.append('IncludeYear')

    if conditions.get('DepartmentsInAllYears'):
        df['DepartmentYearCount'] = df.groupby('DepartmentId')['Year'].transform('nunique')
        df['DepartmentInAllYears'] = df['DepartmentYearCount'] == df['Year'].nunique()
        filters.append('DepartmentInAllYears')

    if conditions.get('DepartmentsInThresholdYears'):
        df['DepartmentYearCount'] = df.groupby('DepartmentId')['Year'].transform('nunique')
        df['DepartmentMeetsYearThreshold'] = df['DepartmentYearCount'] >= usfhn.constants.YEAR_REQUIREMENT
        filters.append('DepartmentMeetsYearThreshold')

    if conditions.get('USPhDGrantingDegreeInstitutions') and not for_closedness:
        # We don't restrict degrees for closedness

        df['DegreeInstitutionIsUSPhDGranting'] = df['DegreeInstitutionId'].isin(
            usfhn.institutions.get_us_phd_granting_institutions()
        )
        filters.append('DegreeInstitutionIsUSPhDGranting')

    if conditions.get('LatestInstitutionEmployment'):
        latest_institution_employment_df = pd.read_csv(
            usfhn.constants.AA_2022_LAST_EMPLOYING_INSTITUTION_ANNOTATIONS_PATH
        )

        latest_institution_employment_df['LatestEmployment'] = True
        df = df.merge(
            latest_institution_employment_df,
            on=['PersonId', 'InstitutionId'],
            how='left',
        )

        df['LatestEmployment'] = df['LatestEmployment'].fillna(False)
        filters.append('LatestEmployment')

    # apply all filters except (minimum taxonomy size) simultaneously
    for column in filters:
        df = df[
            df[column]
        ]

    unfiltered_df = df.copy()

    if conditions.get('PrimaryAppointments'):
        df = df[
            df['PrimaryAppointment']
        ]

    # we apply this last to make sure that rankings are fed only departments
    # with at least 3 people.
    if conditions.get('MinimumTaxonomySize'):
        df['InstitutionTaxonomyPeople'] = df.groupby(
            ['InstitutionId', 'Taxonomy']
        )['PersonId'].transform('nunique')

        df['MeetsMinTaxonomySize'] = df['InstitutionTaxonomyPeople'] >= conditions.get('MinimumTaxonomySize') 
        df = df[
            df['MeetsMinTaxonomySize']
        ]

    df = add_missing_appointments(df, unfiltered_df)

    # drop extra columns
    df = df[columns]

    df = annotate_taxonomization(
        df,
        use_alternate_taxonomy=conditions.get('AlternateTaxonomy'),
        umbrellas_to_exclude=conditions.get('UmbrellaExclusions'),
    )

    if conditions.get('YearUnion'):
        df['Year'] = usfhn.constants.YEAR_UNION
        df = df.drop_duplicates()

    df['Academia'] = 'Academia'

    return df


def annotate_taxonomization(
    df,
    use_alternate_taxonomy=False,
    umbrellas_to_exclude=[],
):
    if use_alternate_taxonomy:
        taxonomy_path = usfhn.constants.AA_2022_FIELD_DEFINED_ALTERNATE_TAXONOMIZATION_PATH
    else: 
        taxonomy_path = usfhn.constants.AA_2022_FIELD_DEFINED_TAXONOMIZATION_PATH

    taxonomy = pd.read_csv(taxonomy_path)[
        [
            'Taxonomy',
            'Field',
            'Area',
            'Umbrella',
        ]
    ].drop_duplicates()

    if umbrellas_to_exclude:
        taxonomy = nullify_umbrellas(taxonomy, umbrellas_to_exclude)

    df = df.merge(
        taxonomy,
        on='Taxonomy',
        how='left',
    )

    return df


def nullify_umbrellas(taxonomy, umbrellas_to_exclude=[]):
    """
    We still want to include this at the academia level, so we're going to
    just remove umbrellas/areas/fields beneath excluded umbrellas
    """
    included_taxonomy = taxonomy.copy()[
        ~taxonomy['Umbrella'].isin(umbrellas_to_exclude)
    ]

    excluded_taxonomy = taxonomy.copy()[
        taxonomy['Umbrella'].isin(umbrellas_to_exclude)
    ]

    taxonomy_cols = ['Field', 'Area', 'Umbrella']

    for col in taxonomy_cols:
        excluded_taxonomy[col] = np.nan

    return pd.concat([included_taxonomy, excluded_taxonomy])


def add_missing_appointments(df, unfiltered_df):
    """
    we have the following cases:
    1. someone appears in the unfiltered df in a year, but not in the filtered df
        --> create an `Academia` primary appointment in that year
        (this covers taxonomy exclusions and primary appointment exclusions)
    2. someone is missing an appointment in a year
        --> create an `Academia` primary appointment in that year
        (this covers data integrity issues)

    After step #1, everyone has a primary appointment in every year that they
    appear; we've dealt with primary appoint exclusions, and only have to deal
    with data integrity issues (#2)
    """
    appointments_removed_by_filtering = get_appointments_removed_by_filtering(df, unfiltered_df)

    df = pd.concat([df, appointments_removed_by_filtering])

    appointments_in_missing_years = get_appointments_in_missing_years(df)

    df = pd.concat([df, appointments_in_missing_years])

    return df


def get_appointments_removed_by_filtering(filtered_df, unfiltered_df):
    cols = ['PersonId', 'Year']

    df = unfiltered_df.copy()[cols].drop_duplicates()

    filtered_df = filtered_df.copy()[
        filtered_df['PrimaryAppointment']
    ][cols].drop_duplicates()

    filtered_df['WasRemoved'] = False

    df = df.merge(
        filtered_df,
        on=cols,
        how='left',
    )

    df['WasRemoved'] = df['WasRemoved'].fillna(True)

    df = df[
        df['WasRemoved']
    ].drop(columns=['WasRemoved']).rename(columns={
        'Year': 'YearToAdd',
    })

    return create_appointments(
        source_df=unfiltered_df,
        appointments_to_create=df,
    )


def get_appointments_in_missing_years(current_df):
    df = current_df.copy()

    df = df[
        [
            'PersonId',
            'Year',
            'PrimaryAppointment',
        ]
    ].drop_duplicates()

    years_set = set(df['Year'].unique())

    df['FirstYear'] = df.groupby('PersonId')['Year'].transform('min')
    df['LastYear'] = df.groupby('PersonId')['Year'].transform('max')

    df = df[
        df['PrimaryAppointment']
    ].drop(columns=['PrimaryAppointment']).drop_duplicates()

    df['Years'] = df.groupby('PersonId')['Year'].transform('nunique')
    df['YearRange'] = 1 + df['LastYear'] - df['FirstYear']
    df['MissingYears'] = df['Years'] != df['YearRange']

    df = df[
        df['MissingYears']
    ]

    missing_appointment_rows = []
    for person_id, rows in df.groupby('PersonId'):
        start = rows.iloc[0]['FirstYear']
        end = rows.iloc[0]['LastYear']

        person_years = set(rows['Year'].unique())
        missing_years = {y for y in years_set if start < y < end} - person_years

        for year in missing_years:
            missing_appointment_rows.append({
                'PersonId': person_id,
                'YearToAdd': year,
            })

    return create_appointments(
        source_df=current_df,
        appointments_to_create=pd.DataFrame(missing_appointment_rows),
    )


def create_appointments(source_df, appointments_to_create):
    """
    created appointments
    - are primary appointments
    - are at the most recent previously observed institution
    - have the most recent previously observed rank
    - have no taxonomy
    - have no department
    """
    df = source_df.copy().merge(
        appointments_to_create,
        on='PersonId',
    )

    df = df[
        df['Year'] <= df['YearToAdd']
    ]

    df['MaxYear'] = df.groupby(['PersonId', 'YearToAdd'])['Year'].transform('max')

    df = df[
        df['Year'] == df['MaxYear']
    ].drop(columns=[
        'DepartmentId',
        'DepartmentName',
        'PrimaryAppointment',
        'Taxonomy',
        'MaxYear',
        'Year',
    ]).rename(columns={
        'YearToAdd': 'Year',
    })

    df['PrimaryAppointment'] = True

    return df


# def impute_missing_years_into_academia(df):
#     """
#     The premise of this is the following: people who appear, leave, and the
#     reappear did not really attrite, but we don't know what taxonomy they should
#     appear in during their absence, so we only impute employments for them at
#     the Academia level.
#     """
#     impute_df = df.copy()[
#         [
#             'PersonId',
#             'PersonName',
#             'Gender',
#             'InstitutionId',
#             'InstitutionName',
#             'DegreeYear',
#             'DegreeInstitutionId',
#             'DegreeInstitutionName',
#             'Rank',
#             'Year',
#         ]
#     ].drop_duplicates()

#     impute_df['FirstYear'] = impute_df.groupby('PersonId')['Year'].transform('min')
#     impute_df['LastYear'] = impute_df.groupby('PersonId')['Year'].transform('max')
#     impute_df['Years'] = impute_df.groupby('PersonId')['Year'].transform('nunique')
#     impute_df['YearRange'] = 1 + impute_df['LastYear'] - impute_df['FirstYear']
#     impute_df['MissingYears'] = impute_df['Years'] != impute_df['YearRange']

#     impute_df = impute_df[
#         impute_df['MissingYears']
#     ]

#     years_set = set(df['Year'].unique())

#     # now we need to fill in ranks and institutions from earlier years
#     imputations = []
#     for person_id, rows in impute_df.groupby('PersonId'):
#         start = rows.iloc[0]['FirstYear']
#         end = rows.iloc[0]['LastYear']

#         person_years = set(rows['Year'].unique())
#         missing_years = {y for y in years_set if start < y < end} - person_years
#         for year in missing_years:
#             nearest_existing_year = max([y for y in person_years if y < year])

#             nearest_existing_year = rows[
#                 rows['Year'] == nearest_existing_year
#             ].iloc[0]

#             imputations.append({
#                 'PersonId': person_id,
#                 'PersonName': nearest_existing_year['PersonName'],
#                 'Gender': nearest_existing_year['Gender'],
#                 'InstitutionId': nearest_existing_year['InstitutionId'],
#                 'InstitutionName': nearest_existing_year['InstitutionName'],
#                 'DegreeYear': nearest_existing_year['DegreeYear'],
#                 'DegreeInstitutionId': nearest_existing_year['DegreeInstitutionId'],
#                 'DegreeInstitutionName': nearest_existing_year['DegreeInstitutionName'],
#                 'PrimaryAppointment': True,
#                 'Rank': nearest_existing_year['Rank'],
#                 'Year': year,
#                 'Academia': 'Academia',
#             })

#     return pd.concat([df, pd.DataFrame(imputations)])


@functools.lru_cache
def get_raw_dataframe():
    """
    there's only one filter here, and that's whether we want to restrict by
    taxonomy (field size/etc) or not.
    """
    employment = pd.read_csv(usfhn.constants.AA_2022_PEOPLE_IMPUTED_EMPLOYMENT_PATH)[
        ['PersonId', 'DepartmentId', 'InstitutionId', 'Year']
    ].drop_duplicates()

    institutions = pd.read_csv(usfhn.constants.AA_2022_DEGREE_INSTITUTION_REMAP_INSTITUTIONS_PATH)[
        ['InstitutionName', 'InstitutionId']
    ].drop_duplicates()

    departments = pd.read_csv(
        usfhn.constants.AA_2022_TAXONOMY_CLEANED_DEPARTMENTS_PATH,
        low_memory=False,
    )[
        ['DepartmentId', 'DepartmentName', 'Taxonomy']
    ].drop_duplicates()

    degrees = pd.read_csv(usfhn.constants.AA_2022_DEGREE_FILTERED_DEGREES_PATH)[
        ['PersonId', 'DegreeInstitutionId', 'DegreeInstitutionName', 'DegreeYear']
    ].drop_duplicates()

    primary_appointments = pd.read_csv(usfhn.constants.AA_2022_PRIMARY_APPOINTED_EMPLOYMENT_PATH)[
        ['PersonId', 'Year', 'DepartmentId', 'InstitutionId', 'PrimaryAppointment', 'Rank']
    ].drop_duplicates()

    names = pd.read_csv(usfhn.constants.AA_2022_PEOPLE_IMPUTED_EMPLOYMENT_PATH)[
        ['PersonId', 'PersonName']
    ].drop_duplicates()

    gender = pd.read_csv(usfhn.constants.AA_2022_PEOPLE_GENDERED_PEOPLE_PATH)[
        ['PersonId', 'Gender']
    ].drop_duplicates()

    ################################################################################
    # MERGING
    ################################################################################
    df = employment.merge(
        institutions,
        on='InstitutionId',
    ).merge(
        departments,
        on='DepartmentId',
    ).merge(
        degrees,
        on='PersonId',
        how='left',
    ).merge(
        primary_appointments,
        on=['PersonId', 'Year', 'DepartmentId', 'InstitutionId'],
    ).merge(
        names,
        on='PersonId',
    ).merge(
        gender,
        on='PersonId',
        how='left',
    )

    df['PrimaryAppointment'] = df['PrimaryAppointment'].fillna(False)

    df = df[
        [
            "PersonId",
            "PersonName",
            "Gender",
            "DepartmentId",
            "DepartmentName",
            "InstitutionId",
            "InstitutionName",
            "Year",
            "Rank",
            "DegreeYear",
            "DegreeInstitutionId",
            'DegreeInstitutionName',
            "PrimaryAppointment",
            "Taxonomy",
        ]
    ]

    return df


def save_dataset(name):
    print(f'making "{name}":')
    filters = usfhn.datasets.DATASETS[name]

    dataset_object = usfhn.datasets.DataSet(name, filters)

    df = get_raw_dataframe().copy(deep=True)

    verbose_dataset_df = filter_dataset(df, filters)

    short_columns = [
        "PersonId",
        "Gender",
        "DepartmentId",
        "InstitutionId",
        "Year",
        "Rank",
        "DegreeYear",
        "DegreeInstitutionId",
        "Taxonomy",
        "Field",
        "Area",
        "Umbrella",
        "Academia",
        'PrimaryAppointment',
    ]

    dataset_df = verbose_dataset_df[
        short_columns
    ]

    closedness_dataset_df = filter_dataset(df, filters, for_closedness=True)[
        short_columns
    ]

    print(f"\trows: {len(dataset_df)}")

    dataset_object.clean()
    usfhn.datasets.set_dataset(name)

    verbose_dataset_df.to_csv(dataset_object.verbose_data_path, index=False)
    dataset_df.to_csv(dataset_object.data_path, index=False)
    closedness_dataset_df.to_csv(dataset_object.closedness_data_path, index=False)

    if name in DATASETS_TO_RANK:
        print(f"\tregenerating ranks")

        usfhn.rank.get_ranks(dataset=name).to_csv(
            dataset_object.ranks_path,
            index=False,
        )

        usfhn.rank.get_ranks(dataset=name, by_year=True).to_csv(
            dataset_object.ranks_by_year_path,
            index=False,
        )

    if name == 'scieco-data':
        print('updating scieco-data')
        usfhn.scieco.update_scieco()


if __name__ == '__main__':
    """
    This defines the data used in almost all other analyses.

    Conditions:
    - Employing Institutions:
        - PhD Granting
    - People:
        - Tenure Track
        - have Degrees from a PhD granting US Institution
    - Departments:
        - are present in all years
            - we drop the 'Law' Taxonomy in this case because AA stops tracking it in 2017

    For groupby convenience, we also augment the dataframe with:
    - a Year-agnostic, deduped set
    - a dummy column 'Academia' for aggregating everything
    """
    usfhn.utils.print_cleaning_step_start("making datasets")
    parser = argparse.ArgumentParser(description='make datasets for usfhn')
    parser.add_argument('--datasets', '-d', nargs='+', type=str,
                        default=list(usfhn.datasets.DATASETS.keys()),
                        help="datasets to run. defaults to all.")
    parser.add_argument('--exclude_scieco', action='store_true', default=False)

    args = parser.parse_args()

    datasets = args.datasets

    if args.exclude_scieco:
        datasets = [d for d in datasets if d != 'scieco-data']

    for name in datasets:
        save_dataset(name)

    if set(args.datasets) & DATASETS_TO_CLEAR_STATS_FOR:
        shutil.rmtree(usfhn.stats.runner.directory)
