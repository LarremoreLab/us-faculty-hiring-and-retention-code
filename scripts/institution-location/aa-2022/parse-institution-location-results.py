import pandas as pd

import usfhn.constants
import usfhn.datasets
import usfhn.utils


def get_previously_annotated_data():
    df = pd.read_csv(usfhn.constants.AA_2022_DEGREE_FILTERED_DEGREES_PATH)[
        [
            'DegreeInstitutionId',
            'DegreeInstitutionName',
        ]
    ].drop_duplicates()

    df['CleanDegreeInstitutionName'] = df['DegreeInstitutionName'].apply(
        usfhn.utils.clean_institution_name_for_permissive_joining
    )

    df = df[
        df['DegreeInstitutionName'].notnull()
    ]

    old_institutions_df = pd.concat([
        pd.read_csv(usfhn.constants.AA_V1_INSTITUTION_LOCATION_RESULTS_PATH),
        pd.read_csv(usfhn.constants.AA_V1_INSTITUTION_LOCATION_HAND_ANNOTATED_PATH),
    ])[
        [
            'DegreeInstitutionName',
            'CountryName',
        ]
    ].drop_duplicates()

    old_institutions_df['CleanDegreeInstitutionName'] = old_institutions_df['DegreeInstitutionName'].apply(
        usfhn.utils.clean_institution_name_for_permissive_joining
    )

    old_institutions_df = old_institutions_df.drop(columns=['DegreeInstitutionName'])

    df = df.merge(
        old_institutions_df,
        on='CleanDegreeInstitutionName',
        how='left',
    )

    df = df[
        df['CountryName'].notna()
    ]

    df = df.drop(columns=['CleanDegreeInstitutionName'])

    df = df[
        [
            'DegreeInstitutionName',
            'DegreeInstitutionId',
            'CountryName',
        ]
    ].drop_duplicates()

    return df


def get_employing_institution_locations():
    df = pd.read_csv(usfhn.constants.AA_2022_BASIC_CLEANING_EMPLOYMENT_PATH)[
        [
            'InstitutionId',
            'InstitutionName',
        ]
    ].drop_duplicates()

    df = df.rename(columns={
        'InstitutionId': 'DegreeInstitutionId',
        'InstitutionName': 'DegreeInstitutionName',
    })

    df['CountryName'] = 'United States'

    return df


if __name__ == '__main__':
    df = pd.read_csv(usfhn.constants.AA_2022_INSTITUTION_LOCATION_QUERIES_RESULTS_PATH)[
        [
            'HITId',
            'WorkerId',
            'Input.DegreeInstitutionId',
            'Input.DegreeInstitutionName',
            'Input.DuplicateNumber',
            'Input.SearchLink',
            'Answer.country.label',
        ]
    ].rename(columns={
        'Input.DegreeInstitutionId': 'DegreeInstitutionId',
        'Input.DegreeInstitutionName': 'DegreeInstitutionName',
        'Input.DuplicateNumber': 'DuplicateNumber',
        'Input.SearchLink': 'SearchLink',
        'Answer.country.label': 'CountryName',
    })

    df = df[
        df['DegreeInstitutionId'].notnull()
    ]

    all_degree_institution_ids = set(df['DegreeInstitutionId'].unique())

    clean_df = df[~df['CountryName'].isin(['Other', 'Could not determine'])]

    df['CountryNamePerId'] = df.groupby('DegreeInstitutionId')['CountryName'].transform('nunique')

    clean_df = df[
        df['CountryNamePerId'] == 1
    ]

    dirty_df = df[
        df['CountryNamePerId'] == 2
    ]

    if usfhn.constants.AA_2022_INSTITUTION_LOCATION_HAND_ANNOTATED_PATH.exists():
        hand_annotations = pd.read_csv(usfhn.constants.AA_2022_INSTITUTION_LOCATION_HAND_ANNOTATED_PATH)
        dirty_df = dirty_df[
            ~dirty_df['DegreeInstitutionId'].isin(hand_annotations['DegreeInstitutionId'])
        ]

        clean_df = pd.concat([
            clean_df,
            hand_annotations
        ])

    if len(dirty_df):
        dirty_df = dirty_df.sort_values(by=['DegreeInstitutionName'])[
            [
                'CountryName',
                'DegreeInstitutionName',
                'SearchLink',
                'DegreeInstitutionId',
                'WorkerId',
            ]
        ]

        dirty_df.to_csv(
            usfhn.constants.AA_2022_INSTITUTION_LOCATION_DIRTY_IN_PROGRESS_PATH,
            index=False
        )
    else:
        cleaned_degree_institution_ids = set(clean_df['DegreeInstitutionId'].unique())

        assert(all_degree_institution_ids == cleaned_degree_institution_ids)

        clean_df = clean_df[
            [
                'DegreeInstitutionName',
                'DegreeInstitutionId',
                'CountryName',
            ]
        ].drop_duplicates()

        clean_df = pd.concat([
            clean_df,
            get_employing_institution_locations(),
            get_previously_annotated_data(),
        ]).merge(
            pd.read_csv(usfhn.constants.INSTITUTION_LOCATION_COUNTRIES_PATH),
            on='CountryName',
        )
        
        clean_df.to_csv(
            usfhn.constants.AA_2022_INSTITUTION_LOCATION_RESULTS_PATH,
            index=False,
        )
