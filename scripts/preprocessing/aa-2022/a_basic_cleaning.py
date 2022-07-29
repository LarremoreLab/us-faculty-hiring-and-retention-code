import pandas as pd

import usfhn.constants
import usfhn.utils


def validate_degree_type_name_and_degree_type_name_1(df):
    """
    validate that it's safe to drop the `DegreeTypeName.1` column. conditions:
    - it agrees with `DegreeTypeName` when it is not null
    """
    df = df.copy()
    df = df[
        df['DegreeTypeName.1'].notnull()
    ]

    different_degree_rows = df[
        df['DegreeTypeName'] != df['DegreeTypeName.1']
    ]

    assert(different_degree_rows.empty)


def get_institutions(df, degree_institutions=False):
    id_col = 'DegreeInstitutionId' if degree_institutions else 'InstitutionId'
    name_col = 'DegreeInstitutionName' if degree_institutions else 'InstitutionName'

    df = df[
        [
            id_col,
            name_col,
        ]
    ].drop_duplicates()

    df = df[
        df[name_col].notna()
    ]

    return df


if __name__ == '__main__':
    """
    1. Validate that the `DegreeTypeName.1` column is useless and drop it.
    2. Map `RankTypeId` to `Rank`
    3. Drop `TenureStatusTypeId`
    4. Drop `TaxonomyLevel01Id`
    5. Rename `DegreeInstitutionID` to `DegreeInstitutionId`
    6. Rename `AADYear` to `Year`
    7. Rename `TaxonomyLevel01Name` to `Taxonomy`
    8. Rename `UnitId` to `DepartmentId`
    9. Rename `UnitName` to `DepartmentName`
    10. Rename `DegreeTypeName` to `DegreeName`
    """
    usfhn.utils.print_cleaning_step_start("Basic Cleaning")
    df = pd.read_csv(usfhn.constants.AA_2022_REFRESH_FACULTY_PATH)

    validate_degree_type_name_and_degree_type_name_1(df)
    df = df.drop(columns=['DegreeTypeName.1'])

    df['Rank'] = df['RankTypeId'].apply(usfhn.constants.RANK_TYPE_ID_TO_RANK.get)
    df = df.drop(columns=['RankTypeId'])

    df = df.drop(columns=['TaxonomyLevel01Id'])

    df = df.drop(columns=['TenureStatusTypeId'])

    df = df.rename(columns={'DegreeInstitutionID': 'DegreeInstitutionId'})

    df = df.rename(columns={'AADYear': 'Year'})

    df = df.rename(columns={'TaxonomyLevel01Name': 'Taxonomy'})

    df = df.rename(columns={'UnitId': 'DepartmentId'})

    df = df.rename(columns={'UnitName': 'DepartmentName'})

    df = df.rename(columns={'DegreeTypeName': 'DegreeName'})
    
    df.to_csv(
        usfhn.constants.AA_2022_BASIC_CLEANING_EMPLOYMENT_PATH,
        index=False,
    )

    get_institutions(df).to_csv(
        usfhn.constants.AA_2022_BASIC_CLEANING_INSTITUTIONS_PATH,
        index=False,
    )

    get_institutions(df, degree_institutions=True).to_csv(
        usfhn.constants.AA_2022_BASIC_CLEANING_DEGREE_INSTITUTIONS_PATH,
        index=False,
    )
