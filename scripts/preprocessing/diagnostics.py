import pandas as pd

import usfhn.constants


def check_ids_and_names(df, id_column, name_column):
    df = df.copy()

    df = df[
        [
            id_column,
            name_column,
        ]
    ].drop_duplicates()

    df['NamesPerId'] = df.groupby(id_column)[name_column].transform('nunique')
    df['IdsPerName'] = df.groupby(name_column)[id_column].transform('nunique')

    id_df = df[
        [
            id_column,
            'NamesPerId',
        ]
    ].drop_duplicates().copy()

    n_ids = df[id_column].nunique()
    n_ids_with_more_than_1_name = df[
        df['NamesPerId'] > 1
    ][id_column].nunique()

    percent_ids_with_more_than_one_name = round(100 * n_ids_with_more_than_1_name / n_ids, 2)

    name_df = df[
        [
            name_column,
            'IdsPerName',
        ]
    ].drop_duplicates().copy()

    n_names = df[name_column].nunique()
    n_names_with_more_than_1_id = df[
        df['IdsPerName'] > 1
    ][name_column].nunique()

    percent_names_with_more_than_one_id = round(100 * n_names_with_more_than_1_id / n_names, 2)

    print(
        f"\tids with more than one name: {percent_ids_with_more_than_one_name}% "
        +
        f"({n_ids_with_more_than_1_name} of {n_ids})"
    )

    print(
        f"\tnames with more than one id: {percent_names_with_more_than_one_id}% "
        +
        f"({n_names_with_more_than_1_id} of {n_names})"
    )


def validate_institutions_in_employment_and_degrees(employment_df, degrees_df):
    """
    check that InstitutionIds and DegreeInstitutionIds match when joined on:
    - InstitutionName
    - DegreeInstitutionName

    check that InstitutionNames and DegreeInstitutionNames match when joined on:
    - InstitutionId
    - DegreeInstitutionId
    """
    employment_df = employment_df.copy()[
        [
            'InstitutionId',
            'InstitutionName',
        ]
    ].drop_duplicates()

    degrees_df = degrees_df.copy()[
        [
            'DegreeInstitutionId',
            'DegreeInstitutionName',
        ]
    ].drop_duplicates()

    degree_inst_id_name_dict = degrees_df.set_index('DegreeInstitutionId')['DegreeInstitutionName'].to_dict()
    employment_df['DegreeInstitutionName'] = employment_df['InstitutionId'].apply(degree_inst_id_name_dict.get)

    id_match_name_mismatch_in_employment = employment_df[
        (employment_df['DegreeInstitutionName'] != employment_df['InstitutionName'])
        &
        (employment_df['DegreeInstitutionName'].notna())
    ]

    assert(not len(id_match_name_mismatch_in_employment))

    inst_id_name_dict = employment_df.set_index('InstitutionId')['InstitutionName'].to_dict()
    degrees_df['InstitutionName'] = degrees_df['DegreeInstitutionId'].apply(inst_id_name_dict.get)

    id_match_name_mismatch_in_degrees = degrees_df[
        (degrees_df['DegreeInstitutionName'] != degrees_df['InstitutionName'])
        &
        (degrees_df['InstitutionName'].notna())
    ]

    assert(not len(id_match_name_mismatch_in_degrees))

    degree_inst_name_id_dict = degrees_df.set_index('DegreeInstitutionName')['DegreeInstitutionId'].to_dict()
    employment_df['DegreeInstitutionId'] = employment_df['InstitutionName'].apply(degree_inst_name_id_dict.get)

    name_match_id_mismatch_in_employment = employment_df[
        (employment_df['DegreeInstitutionId'] != employment_df['InstitutionId'])
        &
        (employment_df['DegreeInstitutionId'].notna())
    ]

    assert(not len(name_match_id_mismatch_in_employment))

    inst_name_id_dict = employment_df.set_index('InstitutionName')['InstitutionId'].to_dict()
    degrees_df['InstitutionId'] = degrees_df['DegreeInstitutionName'].apply(inst_name_id_dict.get)

    name_match_id_mismatch_in_degrees = degrees_df[
        (degrees_df['DegreeInstitutionId'] != degrees_df['InstitutionId'])
        &
        (degrees_df['InstitutionId'].notna())
    ]

    assert(not len(name_match_id_mismatch_in_degrees))


def validate_people_at_institutions_in_a_year(df):
    df = df.copy()[
        [
            'PersonId',
            'InstitutionId',
            'Year',
        ]
    ].drop_duplicates()

    df['InstitutionsPerYear'] = df.groupby([
        'PersonId',
        'Year',
    ])['InstitutionId'].transform('nunique')

    multi = df[
        df['InstitutionsPerYear'] > 1
    ]

    print(f"{multi['PersonId'].nunique()} are listed as working at more than one institution in the same year")

def validate_people_ranks_within_a_year(df):
    df = df.copy()[
        [
            'PersonId',
            'Rank',
            'Year'
        ]
    ]

    df['RanksPerYear'] = df.groupby([
        'PersonId',
        'Year',
    ])['Rank'].transform('nunique')

    multi = df[
        df['RanksPerYear'] > 1
    ]

    print(f"{multi['PersonId'].nunique()} are listed as having more than one rank in the same year")


def validate_people_degrees(df):
    df['Degrees'] = df.groupby([
        'PersonId',
    ])['DegreeInstitutionId'].transform('nunique')

    multi = df[
        df['Degrees'] > 1
    ]

    print(f"{multi['PersonId'].nunique()} are listed as degrees from more than one place")

    df['DegreeNames'] = df.groupby([
        'PersonId',
    ])['DegreeTypeName'].transform('nunique')

    multi = df[
        df['DegreeNames'] > 1
    ]

    print(f"{multi['PersonId'].nunique()} are listed as having more than one type of degree")


def validate_same_name_same_degree_different_id_people(employment_df, degrees_df):
    print(degrees_df.columns)
    df = employment_df.copy()[
        [
            'PersonId',
            'PersonName',
        ]
    ].drop_duplicates().merge(
        degrees_df[
            [
                'PersonId',
                'DegreeInstitutionId',
                'DegreeName',
                'DegreeYear',
            ]
        ].drop_duplicates().copy(),
        on='PersonId',
    )

    df['SameNameSameDegreePersonIdCount'] = df.groupby([
        'PersonName',
        'DegreeInstitutionId',
        'DegreeName',
        'DegreeYear',
    ])['PersonId'].transform('nunique')

    multi_df = df[
        df['SameNameSameDegreePersonIdCount'] > 1
    ]

    multi_people = multi_people['PersonId'].nunique()
    total_people = employment_df['PersonId'].nunique()
    multi_percent = round(100 * (multi_people / total_people), 2)
    
    print(f"{multi_people} people with the same name and degree but different PersonIds ({multi_people}\% of people)")


if __name__ == '__main__':
    """
    Things to look for:
    1. institutions:
        - ids per name?
        - names per id?
        - institutions in all years?
    2. departments:
        - ids per name?
        - names per id?
        - departments per year?
        - institutions per department?
    3. degrees:
        - institutions in degrees:
            - ids per name?
            - names per id?
        - people in degrees:
            - ids per name?
            - names per id?
            - degrees per id?
            - intersection of people with degrees with employed people?
    4. people:
        - ids per name?
        - names per id?
        - institutions per id within a year?
        - ranks per year?
    """
    employment_df = pd.read_csv(usfhn.constants.AA_2022_DEGREE_FILTERED_EMPLOYMENT_PATH)
    degrees_df = pd.read_csv(usfhn.constants.AA_2022_DEGREE_FILTERED_DEGREES_PATH)

    print('Institutions:')
    check_ids_and_names(employment_df, 'InstitutionId', 'InstitutionName')

    print('Degree Institutions:')
    check_ids_and_names(degrees_df, 'DegreeInstitutionId', 'DegreeInstitutionName')

    validate_institutions_in_employment_and_degrees(employment_df, degrees_df)

    check_ids_and_names(employment_df, 'DepartmentId', 'DepartmentName')

    check_ids_and_names(employment_df, 'PersonId', 'PersonName')

    validate_people_at_institutions_in_a_year(employment_df)
    validate_people_ranks_within_a_year(employment_df)

    validate_same_name_same_degree_different_id_people(employment_df, degrees_df)
    # print(df.columns)
    # print(df.nunique())
