from functools import lru_cache
import pandas as pd
import numpy as np

import usfhn.constants
import usfhn.closedness
import usfhn.fieldwork
import usfhn.utils


def get_taxonomization_with_fields(threshold_cols=[]):
    """
    We consider a Taxonomy large enough to analyze as a field if EITHER:
    - the fraction of Institutions that have at least one Department in the
      Taxonomy is >= {constants.FIELD_CRITERIA['InstitutionThreshold']}
    - or both:
        - the number of People employed by Departments in the Taxonomy
          is >= {constants.FIELD_CRITERIA['PersonThreshold']}
        - the Taxonomy's closedness
          is >= {constants.FIELD_CRITERIA['ClosednessThreshold']}
    
    If {constants.FIELD_CRITERIA['ExcludeVarious']} is True, Taxonomies that end
    in "various" (e.g., "Engineering, Various") are excluded from consideration as a field.

    DataFrame column requirements (all other columns will be ignored):
    - PersonId
    - DepartmentId
    - InstitutionId
    - Taxonomy
    - DegreeInstitutionId

    The appointments should be primary and tenure track, likely
    """
    df = get_employment_df_with_degrees().copy()
    starting_taxonomy_count = df['Taxonomy'].nunique()

    df = annotate_closedness_threshold(df)
    df = annotate_person_threshold(df)
    df = annotate_institution_threshold(df)

    df['MeetsPopulationThreshold'] = df['MeetsPersonThreshold'] & df['MeetsClosednessThreshold']
    df['MeetsClosedPeopleThreshold'] = df['PersonCount'] * df['Closedness'] > 500

    df = df[
        [
            'Taxonomy',
            'MeetsClosednessThreshold',
            'MeetsPersonThreshold',
            'MeetsInstitutionThreshold',
            'MeetsPopulationThreshold',
            'MeetsClosedPeopleThreshold',
        ]
    ].drop_duplicates()

    df['Field'] = False
    for col in threshold_cols:
        df['Field'] |= df[col]

    if usfhn.constants.FIELD_CRITERIA['ExcludeVarious']:
        df = exclude_various_fields(df)

    df = exclude_specific_taxonomies(df)

    df['Field'] = df.apply(lambda r: r['Taxonomy'] if r['Field'] else np.nan, axis=1)

    df = df.merge(
        pd.read_csv(usfhn.constants.CLEAN_FTAS_TAXONOMY_HIERARCHY_PATH),
        on='Taxonomy'
    )

    df = clean_areas_with_null_fields(df)

    df = get_taxonomy_without_duplicate_areas(df)

    df = df[
        [
            'Taxonomy',
            'Field',
            'Area',
            'Umbrella',
        ]
    ].drop_duplicates()

    ending_taxonomy_count = df['Taxonomy'].nunique()
    assert(starting_taxonomy_count == ending_taxonomy_count)

    field_count = df[df['Field'].notna()]['Field'].nunique()

    print(f"\t{field_count} of {ending_taxonomy_count} taxonomies are fields")
    return df


def annotate_closedness_threshold(df):
    closednesses = {}
    for (taxonomy), rows in df.groupby('Taxonomy'):
        closednesses[taxonomy] = usfhn.closedness.get_closedness(rows)

    df['Closedness'] = df['Taxonomy'].apply(closednesses.get)

    df['MeetsClosednessThreshold'] = df['Closedness'] >= usfhn.constants.FIELD_CRITERIA['ClosednessThreshold']

    return df


def annotate_person_threshold(df):
    df['PersonCount'] = df.groupby('Taxonomy')['PersonId'].transform('nunique')
    df['MeetsPersonThreshold'] = df['PersonCount'] >= usfhn.constants.FIELD_CRITERIA['PersonThreshold']

    return df


def annotate_institution_threshold(df):
    institutions = df['InstitutionId'].nunique()
    threshold = institutions * usfhn.constants.FIELD_CRITERIA['InstitutionThreshold']

    df['InstitutionCount'] = df.groupby('Taxonomy')['InstitutionId'].transform('nunique')

    df['MeetsInstitutionThreshold'] = df['InstitutionCount'] >= threshold

    return df


def exclude_various_fields(df):
    """
    We never want to include fields like `Biology, various` as a field on its own.
    (we do want to include it in, say, the `Natural Sciences` domain)
    """
    columns = list(df.columns)

    df['IsVarious'] = df['Taxonomy'].apply(lambda t: t.endswith('various'))
    df['Field'] &= ~df['IsVarious']

    return df[columns]

def exclude_specific_taxonomies(df):
    columns = list(df.columns)

    df['Exclude'] = df['Taxonomy'].isin(usfhn.constants.FIELD_CRITERIA['TaxonomyExclusions'])
    df['Field'] &= ~df['Exclude']

    return df[columns]




def clean_areas_with_null_fields(df):
    """
    if the field is null and there's only one field in the area, null the area
    """
    columns = list(df.columns)

    df['FieldsPerArea'] = df.groupby('Area')['Field'].transform('nunique').fillna(0)
    df['FieldIsNull'] = df['Field'].isnull()
    df['NullifyArea'] = df['FieldIsNull'] & (df['FieldsPerArea'] < 2)

    df['Area'] = df.apply(lambda r: np.nan if r['NullifyArea'] else r['Area'], axis=1)

    return df[columns]


def get_taxonomy_without_duplicate_areas(taxonomization):
    df = get_employment_df_with_degrees().merge(
        taxonomization,
        on='Taxonomy',
    ).drop(columns=['DepartmentId']).drop_duplicates()

    df = usfhn.fieldwork.get_analysis_data_with_one_area_for_cross_listed_taxonomies(df)

    df = df[
        [
            'Taxonomy',
            'Field',
            'Area',
            'Umbrella',
        ]
    ].drop_duplicates()

    return df


@lru_cache(maxsize=1)
def get_employment_df():
    return df


def get_employment_df_with_degrees():
    """
    NOTE that degrees are LEFT joined; this is important for closedness.
    """
    employment_df = pd.read_csv(usfhn.constants.AA_2022_PRIMARY_APPOINTED_EMPLOYMENT_PATH)[
        [
            'PersonId',
            'DepartmentId',
            'InstitutionId',
        ]
    ].drop_duplicates()

    departments_df = pd.read_csv(usfhn.constants.AA_2022_TAXONOMY_CLEANED_DEPARTMENTS_PATH)[
        [
            'DepartmentId',
            'Taxonomy',
        ]
    ].drop_duplicates()

    degrees_df = pd.read_csv(usfhn.constants.AA_2022_DEGREE_FILTERED_DEGREES_PATH)[
        [
            'PersonId',
            'DegreeInstitutionId',
        ]
    ].drop_duplicates()

    df = employment_df.merge(
        departments_df,
        on='DepartmentId'
    ).merge(
        degrees_df,
        on='PersonId',
        how='left',
    )

    return df

    
if __name__ == '__main__':
    usfhn.utils.print_cleaning_step_start("Annotating Fields")

    print('with standard field requirements (institution|(person&closedness)):')
    get_taxonomization_with_fields(threshold_cols=[
        'MeetsInstitutionThreshold',
        'MeetsClosedPeopleThreshold',
        # 'MeetsPopulationThreshold',
    ]).to_csv(
        usfhn.constants.AA_2022_FIELD_DEFINED_TAXONOMIZATION_PATH,
        index=False
    )

    print()
    print('with alternate field requirements (institution|person):')
    get_taxonomization_with_fields(threshold_cols=[
        'MeetsInstitutionThreshold',
        'MeetsPersonThreshold',
    ]).to_csv(
        usfhn.constants.AA_2022_FIELD_DEFINED_ALTERNATE_TAXONOMIZATION_PATH,
        index=False
    )
