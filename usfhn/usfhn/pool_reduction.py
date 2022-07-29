import functools
import pandas as pd

import hnelib.utils

import usfhn.datasets
import usfhn.constants
import usfhn.institutions



def umbrella_exclusion_stats(dataset='unfiltered-census-pool'):
    df = usfhn.datasets.get_dataset_df('closedness_data', dataset)[
        [
            'PersonId',
            'Field',
            'Taxonomy',
            'Umbrella',
        ]
    ].drop_duplicates()

    df['ExcludedTaxonomy'] = df['Field'].isna()
    df['ExcludedTaxonomy'] |= df['Umbrella'].isin(usfhn.datasets.CRITERIA['UmbrellaExclusions']) 

    df['Faculty'] = df.groupby('Umbrella')['PersonId'].transform('nunique')
    
    df = df[
        df['ExcludedTaxonomy']
    ]

    df['ExcludedFaculty'] = df.groupby('Umbrella')['PersonId'].transform('nunique')

    df['FractionExcluded'] = df['ExcludedFaculty'] / df['Faculty']

    df = df[
        [
            'Umbrella',
            'Faculty',
            'ExcludedFaculty',
            'FractionExcluded',
        ]
    ].drop_duplicates()

    return df


def excluded_taxonomy_sizes(dataset='unfiltered-census-pool'):
    """
    returns df:
    - Taxonomy
    - Umbrella
    - Faculty
    """
    df = usfhn.datasets.get_dataset_df('closedness_data', dataset)[
        [
            'PersonId',
            'Field',
            'Taxonomy',
            'Umbrella',
        ]
    ].drop_duplicates()

    df['ExcludedTaxonomy'] = df['Field'].isna()
    df['ExcludedTaxonomy'] |= df['Umbrella'].isin(usfhn.datasets.CRITERIA['UmbrellaExclusions']) 

    df = df[
        df['ExcludedTaxonomy']
    ]

    df['Faculty'] = df.groupby('Taxonomy')['PersonId'].transform('nunique')

    df = df[
        [
            'Taxonomy',
            'Umbrella',
            'Faculty',
        ]
    ].drop_duplicates()

    return df


def people_excluded_at_field_level_by_umbrella(dataset='unfiltered-census-pool'):
    df = usfhn.datasets.get_dataset_df('closedness_data', dataset)[
        [
            'PersonId',
            'Taxonomy',
            'Field',
            'Umbrella',
        ]
    ].drop_duplicates()

    df = df[
        ~df['Umbrella'].isin(usfhn.datasets.CRITERIA['UmbrellaExclusions']) 
    ]

    df['Faculty'] = df.groupby('Umbrella')['PersonId'].transform('nunique')
    df['IsInAField'] = df['Field'].notna()
    df['IsInAField'] = df['IsInAField'].apply(int)

    df['PersonFieldCount'] = df.groupby('PersonId')['IsInAField'].transform('sum')
    df['PersonIsInAField'] = df['PersonFieldCount'].apply(bool)

    df = df[
        ~df['PersonIsInAField']
    ]

    df = df[
        df['Umbrella'].notna()
    ]

    df['ExcludedFaculty'] = df.groupby('Umbrella')['PersonId'].transform('nunique')

    df = df[
        [
            'Umbrella',
            'Faculty',
            'ExcludedFaculty',
        ]
    ].drop_duplicates()

    df['FractionExcluded'] = df['ExcludedFaculty'] / df['Faculty']

    return df


def people_excluded_at_domain_level_by_umbrella(dataset='unfiltered-census-pool'):
    df = usfhn.datasets.get_dataset_df('closedness_data', dataset)[
        [
            'PersonId',
            'Umbrella',
        ]
    ].drop_duplicates()

    df['Faculty'] = df['PersonId'].nunique()

    df['IsInAnUmbrella'] = ~df['Umbrella'].isin(usfhn.datasets.CRITERIA['UmbrellaExclusions']) 
    df['IsInAnUmbrella'] = df['IsInAnUmbrella'].apply(int)

    df['PersonUmbrellaCount'] = df.groupby('PersonId')['IsInAnUmbrella'].transform('sum')
    df['PersonIsInAnUmbrella'] = df['PersonUmbrellaCount'].apply(bool)

    df = df[
        ~df['PersonIsInAnUmbrella']
    ]

    df['ExcludedFaculty'] = df.groupby('Umbrella')['PersonId'].transform('nunique')

    df = df[
        [
            'Umbrella',
            'Faculty',
            'ExcludedFaculty',
        ]
    ].drop_duplicates()

    df['FractionExcluded'] = df['ExcludedFaculty'] / df['Faculty']

    return df


def get_stats():
    df = usfhn.datasets.get_dataset_df('closedness_data')
    df = usfhn.institutions.annotate_us(df)

    primary_people = df['PersonId'].unique()

    pool_df = usfhn.datasets.get_dataset_df('closedness_data', 'unfiltered-census-pool')
    pool_people = pool_df['PersonId'].unique()

    ########################################
    # data to closedness exclusions
    ########################################
    n_field_excluded_people = df[
        df['Field'].isna()
    ]['PersonId'].nunique()

    n_no_degree_people = df[
        df['DegreeInstitutionId'].isnull()
    ]['PersonId'].nunique()

    n_non_us = df[
        df['US']
    ]['PersonId'].nunique()

    ########################################
    # field/domain level analysis excluded
    ########################################
    overlap_df = pool_df[
        pool_df['PersonId'].isin(primary_people)
    ]
    n_umbrella_excluded = overlap_df[
        overlap_df['Umbrella'].isin(usfhn.datasets.CRITERIA['UmbrellaExclusions'])
    ]['PersonId'].nunique()

    n_communications_excluded = overlap_df[
        overlap_df['Umbrella'] == 'Journalism, Media, Communication'
    ]['PersonId'].nunique()

    n_policy_excluded = overlap_df[
        overlap_df['Umbrella'] == 'Public Administration and Policy'
    ]['PersonId'].nunique()


    ########################################
    # true exclusions
    ########################################
    pool_df = pool_df[
        ~pool_df['PersonId'].isin(primary_people)
    ]

    primary_appointment_dropped = pd.read_csv(
        usfhn.constants.AA_2022_PRIMARY_APPOINTED_PEOPLE_WHO_WHERE_DROPPED,
    )

    n_primary_appointment_dropped = primary_appointment_dropped[
        ~primary_appointment_dropped['PersonId'].isin(primary_people)
    ]['PersonId'].nunique()

    pool_df['YearsCount'] = pool_df.groupby('DepartmentId')['Year'].transform('nunique')

    n_not_in_all_years_people = pool_df[
        pool_df['YearsCount'] < usfhn.constants.YEAR_REQUIREMENT
    ]['PersonId'].nunique()

    ########################################
    # open to closed
    ########################################
    closed_df = usfhn.datasets.get_dataset_df('data')
    open_people_excluded_from_closed = df[
        ~df['PersonId'].isin(closed_df['PersonId'].unique())
    ]['PersonId'].nunique()

    non_employing_degree_granting_institution_people = df[
        (df['DegreeInstitutionId'].notnull())
        &
        (df['US'])
        &
        (~df['DegreeInstitutionId'].isin(closed_df['InstitutionId'].unique()))
    ]['PersonId'].nunique()


    included_people = len(primary_people)
    people = len(pool_people)

    closed_people = included_people - open_people_excluded_from_closed

    fraction_included = included_people / people
    fraction_excluded = 1 - fraction_included

    return {
        'nIncluded': included_people,
        'pIncluded': hnelib.utils.fraction_to_percent(fraction_included, 1),
        'pExcluded': hnelib.utils.fraction_to_percent(fraction_excluded, 1),
        'nPool': people,
        'nClosedExcluded': open_people_excluded_from_closed,
        'pClosedExcluded': hnelib.utils.fraction_to_percent(
            open_people_excluded_from_closed / included_people,
            1,
        ),
        'nClosedIncluded': closed_people,
        'pClosedIncluded': hnelib.utils.fraction_to_percent(
            closed_people / included_people,
            1,
        ),
        'nDomainExcludedPeople': n_umbrella_excluded,
        'pDomainExcludedPeople': hnelib.utils.fraction_to_percent(n_umbrella_excluded / included_people, 1),
        'nFieldExcludedPeople': n_field_excluded_people,
        'pFieldExcludedPeople': hnelib.utils.fraction_to_percent(n_field_excluded_people / included_people, 1),
        'nPolicyExcludedPeople': n_policy_excluded,
        'nComExcludedPeople': n_communications_excluded,
        'nNotInAllYearsExcludedPeople': n_not_in_all_years_people,
        'nNoPrimaryAppointmentExcludedPeople': n_primary_appointment_dropped,
        'nNoDegreeExcludedPeople': n_no_degree_people,
        'nNonEmployingInstitutionExcludedPeople': non_employing_degree_granting_institution_people,
    }


def get_department_filtering_stats(df, pool_df):
    n_people = df['PersonId'].nunique()
    n_departments = df['DepartmentId'].nunique()
    n_employing_institutions = df['InstitutionId'].nunique()
    n_rows = len(df)

    n_pool_people = pool_df['PersonId'].nunique()
    n_pool_departments = pool_df['DepartmentId'].nunique()
    n_pool_employing_institutions = pool_df['InstitutionId'].nunique()
    n_pool_rows = len(pool_df)

    n_people_removed = n_pool_people - n_people
    n_departments_removed = n_pool_departments - n_departments
    n_employing_institutions_removed = n_pool_employing_institutions - n_employing_institutions
    n_rows_removed = n_pool_rows - n_rows

    p_people_removed = n_people_removed / n_pool_people
    p_departments_removed = n_departments_removed / n_pool_departments
    p_employing_institutions_removed = n_employing_institutions_removed / n_pool_employing_institutions
    p_rows_removed = n_rows_removed / n_pool_rows

    return {
        'nPeopleRemoved': n_people_removed,
        'pPeopleRemoved': p_people_removed,
        'nDepartmentsRemoved': n_departments_removed,
        'pDepartmentsRemoved': p_departments_removed,
        'nEmployingInstitutionsRemoved': n_employing_institutions_removed,
        'pEmployingInstitutionsRemoved': p_employing_institutions_removed,
        'nRowsRemoved': n_rows_removed,
        'pRowsRemoved': p_rows_removed,
    }


@functools.lru_cache(maxsize=1)
def department_year_threshold_stats():
    df = usfhn.datasets.get_dataset_df('closedness_data')
    pool_df = usfhn.datasets.get_dataset_df('closedness_data', 'unfiltered-census-pool')
    return get_department_filtering_stats(df, pool_df)

def get_departments_in_all_years_filtering_stats():
    df = usfhn.datasets.get_dataset_df('closedness_data', by_year=True)
    pool_df = usfhn.datasets.get_dataset_df('closedness_data')
    return get_department_filtering_stats(df, pool_df)
