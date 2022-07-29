import numpy as np
import pandas as pd
import hnelib.utils

import usfhn.constants
import usfhn.utils
import usfhn.institutions


def print_count_update_with_fraction(count, denominator, verb_string, object_string):
    percent = hnelib.utils.fraction_to_percent(count / denominator, 2)
    print(f"\t{verb_string} {count} {object_string} ({percent}%)")


def filter_and_print_effect(df, condition, filter_string, drop_condition_column=True):
    n_start = len(df)

    if isinstance(condition, str):
        df[condition] = df[condition].fillna(False)
        df = df[
            df[condition]
        ]
    else:
        df = df[condition]

    n_end = len(df)

    n_removed = n_start - n_end
    p_removed = hnelib.utils.fraction_to_percent(n_removed / n_start, 2)
    
    print(f'\tremoving {filter_string}: {n_removed} ({p_removed}%)')

    if drop_condition_column and isinstance(condition, str):
        df = df.drop(columns=[condition])

    return df


def format_annotations(df):
    df = df.rename(columns={
        'Country': 'CountryName',
        'Institution': 'DegreeInstitutionName',
        'Updated Institution': 'NewDegreeInstitutionName',
        'Updated Country': 'NewCountryName',
    })

    original_institutions = pd.read_csv(usfhn.constants.AA_2022_BASIC_CLEANING_DEGREE_INSTITUTIONS_PATH)

    institution_countries = pd.read_csv(usfhn.constants.AA_2022_INSTITUTION_LOCATION_RESULTS_PATH)[
        [
            'DegreeInstitutionName',
            'CountryName',
        ]
    ].drop_duplicates()

    original_institutions = original_institutions.merge(
        institution_countries,
        on='DegreeInstitutionName',
        how='left',
    )

    df = original_institutions.merge(
        df,
        on=['DegreeInstitutionName', 'CountryName'],
        how='left',
    )

    df['Remove'] = df['Remove'].fillna(False)
    df['Keep'] = ~df['Remove']

    df = df.drop(columns=['Remove'])

    df = df.rename(columns={
        'DegreeInstitutionId': 'InstitutionId',
        'DegreeInstitutionName': 'InstitutionName',
        'NewDegreeInstitutionName': 'NewInstitutionName',
    })

    df['NewInstitutionName'] = df['NewInstitutionName'].fillna(df['InstitutionName'])

    return df


def validate_annotated_countries(df):
    countries_df = pd.read_csv(usfhn.constants.INSTITUTION_LOCATION_COUNTRIES_PATH)
    countries = set(countries_df['CountryName'].unique())

    annotation_countries = set(df[df['NewCountryName'].notna()]['NewCountryName'].unique())

    assert(len(annotation_countries - countries) == 0)


def apply_country_annotations(df):
    n_start = len(df)
    print('applying country annotations')

    n_country_added = len(df[df['CountryName'].isna()])

    df['NewCountryName'] = df['NewCountryName'].fillna(df['CountryName'])

    n_country_changed = len(df[df['CountryName'] != df['NewCountryName']]) - n_country_added

    df['CountryName'] = df['NewCountryName']

    df = df.drop(columns=['NewCountryName'])

    print_count_update_with_fraction(n_country_added, n_start, 'added countries to', 'rows')
    print_count_update_with_fraction(n_country_changed, n_start, 'updated countries of', 'rows')

    return df


def get_existing_institution_name_id_map():
    employing_institutions = pd.read_csv(usfhn.constants.AA_2022_BASIC_CLEANING_INSTITUTIONS_PATH)

    degree_institutions = pd.read_csv(usfhn.constants.AA_2022_BASIC_CLEANING_DEGREE_INSTITUTIONS_PATH)
    degree_institutions = degree_institutions.rename(columns={
        'DegreeInstitutionId': 'InstitutionId',
        'DegreeInstitutionName': 'InstitutionName',
    })

    institutions = pd.concat([
        degree_institutions,
        employing_institutions,
    ])

    institutions = institutions[
        institutions['InstitutionId'].notna()
    ]

    institutions['InstitutionId'] = institutions['InstitutionId'].astype(int)

    return institutions.set_index('InstitutionName')['InstitutionId'].to_dict()


def assign_ids_to_institutions(df, institution_name_id_map, institution_name_col, institution_id_col):
    unused_id = max(institution_name_id_map.values()) + 1
    for institution in df[institution_name_col]:
        if institution not in institution_name_id_map:
            institution_name_id_map[institution] = unused_id
            unused_id += 1

    df[institution_id_col] = df[institution_name_col].apply(institution_name_id_map.get)

    return df, institution_name_id_map


def apply_institution_annotations(df):
    institution_name_id_map = get_existing_institution_name_id_map()

    n_start = len(df)

    n_null_ids = len(df[df['InstitutionId'].isna()])

    print_count_update_with_fraction(n_null_ids, n_start, 'added ids to', 'institutions')

    df, institution_name_id_map = assign_ids_to_institutions(
        df,
        institution_name_id_map,
        'NewInstitutionName',
        'NewInstitutionId',
    )

    n_names_remapped = len(df[df['NewInstitutionId'] != df['InstitutionId']])
    print_count_update_with_fraction(n_names_remapped, n_start, 'remapped', 'to existing institutions')

    institutions = set(df['InstitutionId'].unique())
    new_institutions = set(df['NewInstitutionId'].unique()) - institutions
    all_institutions = institutions | new_institutions

    n_new_institutions = len(new_institutions)
    print_count_update_with_fraction(n_new_institutions, len(all_institutions), 'created', 'new institutions')

    assert(not len(df[df['NewInstitutionId'].isna()]))
    assert(not len(df[df['NewInstitutionName'].isna()]))

    return df


def get_country_df_from_remaps(remaps_df):
    countries_df = pd.read_csv(usfhn.constants.INSTITUTION_LOCATION_COUNTRIES_PATH)

    remaps_df = remaps_df.copy().rename(columns={
        'InstitutionId': 'DegreeInstitutionId',
        'InstitutionName': 'DegreeInstitutionName',
        'NewInstitutionId': 'NewDegreeInstitutionId',
        'NewInstitutionName': 'NewDegreeInstitutionName',
    })

    remaps_df = remaps_df.merge(
        countries_df,
        on="CountryName",
    ).rename(columns={
        'CountryId': 'NewCountryId',
        'CountryName': 'NewCountryName',
    })

    cols = [
        'DegreeInstitutionId',
        'DegreeInstitutionName',
        'CountryName',
        'CountryId',
    ]

    df = pd.read_csv(usfhn.constants.AA_2022_INSTITUTION_LOCATION_RESULTS_PATH)[
        cols
    ].drop_duplicates()

    n_institutions_start = len(df)

    df = df.merge(
        remaps_df,
        on=[
            'DegreeInstitutionId',
            'DegreeInstitutionName',
        ],
        how='outer',
    )

    for col in cols:
        df[f"New{col}"] = df[f"New{col}"].fillna(df[col])

    df = df.drop(columns=cols)
    df = df.rename(columns={f"New{col}": col for col in cols})

    print("institution countries:")
    print(f"\tstarted with {n_institutions_start} rows")
    df = filter_and_print_effect(df, 'Keep', "unidentifiable institutions") 

    df = df[
        [
            'DegreeInstitutionId',
            'DegreeInstitutionName',
            'CountryName',
            'CountryId',
        ]
    ].drop_duplicates()

    print(f"\tended with {len(df)} rows")

    return df



def apply_remaps(df, remaps_df, remap_degree_institutions=False, is_employment=False):
    df = df.copy()
    remaps_df = remaps_df.copy()

    id_col = 'InstitutionId'
    name_col = 'InstitutionName'

    if remap_degree_institutions:
        id_col = 'DegreeInstitutionId'
        name_col = 'DegreeInstitutionName'

        remaps_df = remaps_df.rename(columns={
            'InstitutionId': id_col,
            'InstitutionName': name_col,
        })

    df = df.merge(
        remaps_df,
        on=[id_col, name_col],
        how='left',
    )

    df['NewInstitutionId'] = df['NewInstitutionId'].fillna(df[id_col])
    df['NewInstitutionName'] = df['NewInstitutionName'].fillna(df[name_col])


    df[id_col] = df['NewInstitutionId']
    df[name_col] = df['NewInstitutionName']

    df = df.drop(columns=[
        'NewInstitutionId',
        'NewInstitutionName',
    ])

    if is_employment:
        df['Keep'] = df['Keep'].fillna(False)
        institutions = df.copy()[
            [
                id_col,
                name_col,
                'Keep',
            ]
        ].drop_duplicates()

        institutions[f"New{id_col}"] = institutions[id_col]
        institutions[f"New{name_col}"] = institutions[name_col]

        bad_institutions = institutions.copy()[
            ~institutions['Keep']
        ]

        bad_institutions[f"New{id_col}"] = np.nan
        bad_institutions[f"New{name_col}"] = np.nan

        good_institutions = institutions.copy()[
            institutions['Keep']
        ]

        institutions = pd.concat([good_institutions, bad_institutions]).drop(columns='Keep')
        
        df = df.merge(
            institutions,
            on=[id_col, name_col]
        )

        df[id_col] = df[f"New{id_col}"]
        df[name_col] = df[f"New{name_col}"]

        df = df.drop(columns=[f"New{id_col}", f"New{name_col}", 'Keep'])
    else:
        df = filter_and_print_effect(df, 'Keep', "unidentifiable institutions") 

    df = df.drop_duplicates()

    return df


if __name__ == '__main__':
    usfhn.utils.print_cleaning_step_start("Degree Institution Cleaning")
    annotations = pd.read_csv(usfhn.constants.AA_2022_DEGREE_INSTITUTION_REMAP_ANNOTATED_PATH)

    df = format_annotations(annotations)

    print(f"starting with {len(df)} degree institutions")

    validate_annotated_countries(df)

    df = apply_country_annotations(df)

    remaps_df = apply_institution_annotations(df)

    # create a remapped institution countries file
    institution_countries = get_country_df_from_remaps(remaps_df)
    institution_countries.to_csv(
        usfhn.constants.AA_2022_DEGREE_INSTITUTION_REMAP_INSTITUTION_COUNTRIES_PATH,
        index=False,
    )

    remaps_df = remaps_df.drop(columns=['CountryName'])

    # generate a csv of employing institutions, based off of the one produced by
    # basic cleaning
    institutions = pd.read_csv(usfhn.constants.AA_2022_BASIC_CLEANING_INSTITUTIONS_PATH)
    n_institutions_start = len(institutions)
    institutions = apply_remaps(institutions, remaps_df)
    
    print(f"institutions:")
    print(f"\tstarted with: {n_institutions_start}")
    print(f"\tended with: {len(institutions)}")

    institutions.to_csv(
        usfhn.constants.AA_2022_DEGREE_INSTITUTION_REMAP_INSTITUTIONS_PATH,
        index=False,
    )

    # do the same thing for degree institutions
    degree_institutions = pd.read_csv(usfhn.constants.AA_2022_BASIC_CLEANING_DEGREE_INSTITUTIONS_PATH)
    n_degree_institutions_start = len(degree_institutions)

    print(f"degree institutions:")
    print(f"\tstarted with: {n_degree_institutions_start}")
    degree_institutions = apply_remaps(degree_institutions, remaps_df, remap_degree_institutions=True)
    print(f"\tended with: {len(degree_institutions)}")
    
    degree_institutions.to_csv(
        usfhn.constants.AA_2022_DEGREE_INSTITUTION_REMAP_DEGREE_INSTITUTIONS_PATH,
        index=False,
    )

    # do the same thing for employment
    employment = pd.read_csv(usfhn.constants.AA_2022_TENURE_TRACK_FILTERED_EMPLOYMENT_PATH)

    print(f"employment:")
    print(f"\tstarted with: {len(employment)}")

    employment = apply_remaps(employment, remaps_df)
    employment = apply_remaps(employment, remaps_df, remap_degree_institutions=True, is_employment=True)

    print(f"\tended with: {len(employment)}")

    employment.to_csv(
        usfhn.constants.AA_2022_DEGREE_INSTITUTION_REMAP_EMPLOYMENT_PATH,
        index=False,
    )
