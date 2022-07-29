import pandas as pd

import usfhn.constants
import usfhn.datasets
import usfhn.utils


def get_google_search_string(institution_name):
    institution_name = "+".join(institution_name.split())
    return f"https://www.google.com/search?q={institution_name}"


if __name__ == '__main__':
    print('must remove institution names we already know')

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

    institution_count = len(df)
    print(f"{institution_count} institutions in the new dataset")

    # filter out institutions we've already determined the country of
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

    df = df.drop(columns=['CleanDegreeInstitutionName'])

    df = df[
        df['CountryName'].isna()
    ]

    unknown_country_institution_count = len(df)
    known_country_institution_count = institution_count - unknown_country_institution_count
    known_country_institution_percent = round(100 * (known_country_institution_count / institution_count), 2)
    unknown_country_institution_percent = round(100 * (unknown_country_institution_count / institution_count), 2)
    print(f"we already know the country of {known_country_institution_count} of those ({known_country_institution_percent}%)")
    print(f"we need to query for the country of {unknown_country_institution_count} institutions ({unknown_country_institution_percent}%)")
    
    df = df.drop(columns=['CountryName'])

    df['SearchLink'] = df['DegreeInstitutionName'].astype(str).apply(get_google_search_string)
    df['DuplicateNumber'] = 1

    df2 = df.copy(deep=True)
    df2['DuplicateNumber'] = 2

    df = pd.concat([df, df2])

    df.to_csv(
        usfhn.constants.AA_2022_INSTITUTION_LOCATION_QUERIES_PATH,
        index=False,
    )
