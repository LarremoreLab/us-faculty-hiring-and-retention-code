import pandas as pd

import usfhn.constants


def annotate_institution_names(df):
    inst_df = pd.read_csv(usfhn.constants.AA_2022_BASIC_CLEANING_EMPLOYMENT_PATH)[
        [
            'InstitutionId',
            'InstitutionName',
        ]
    ].drop_duplicates()

    return df.merge(
        inst_df,
        on='InstitutionId',
    )


def annotate_department_taxonomies(df):
    tax_df = pd.read_csv(usfhn.constants.AA_2022_TAXONOMY_CLEANED_DEPARTMENTS_PATH)[
        [
            'InstitutionId',
            'DepartmentId',
            'DepartmentName',
            'Taxonomy',
        ]
    ].drop_duplicates()

    return df.merge(
        tax_df,
        on=[
            'InstitutionId',
            'DepartmentId',
        ],
    )


def get_df_for_spreadsheet(df):
    df = df.copy()
    df = df[
        [
            'DepartmentId',
            'InstitutionId',
            'Year',
        ]
    ].drop_duplicates()

    df = annotate_institution_names(df)
    df = annotate_department_taxonomies(df)

    min_year = df['Year'].min()
    max_year = df['Year'].max()

    df['DeptStart'] = df.groupby('DepartmentId')['Year'].transform('min')
    df['DeptEnd'] = df.groupby('DepartmentId')['Year'].transform('max')
    df['DeptYears'] = df.groupby('DepartmentId')['Year'].transform('nunique')

    df['DiedEarly'] = df['DeptEnd'] != max_year
    df['BornLate'] = df['DeptStart'] != min_year

    df['Discontinuous'] = (df['DeptEnd'] - df['DeptStart']) + 1 != df['DeptYears']
    df['NMissingYears'] = df['DeptYears'] - (df['DeptEnd'] - df['DeptStart']) + 1

    # df = df[
    #     (df['InstitutionName'] == "University of Louisiana at Monroe, The")
    #     &
    #     (df['DepartmentName'] == "Agribusiness, Department of")
    # ]

    # print(df.head())

    for year in df['Year'].unique():
        year_df = df[
            df['Year'] == year
        ][
            [
                'DepartmentId',
                'Year',
            ]
        ].drop_duplicates().copy()

        year_df[year] = True
        year_df = year_df.drop(columns=['Year'])

        df = df.merge(
            year_df,
            on='DepartmentId',
            how='left',
        )

        df[year] = df[year].fillna(False)
        
    df['WouldBeDropped'] = (
        df['DeptYears'] < 6
    )

#     df['WouldBeDropped'] = (
#         (df['DiedEarly'])
#         |
#         (df['BornLate'])
#         |
#         (df['Discontinuous'])
#     )

    years = sorted(list(df['Year'].unique()))

    df = df[
        [
            'Taxonomy',
            'InstitutionName',
            'DepartmentId',
            'DepartmentName',
        ]
        +
        years
        +
        [
            'WouldBeDropped',
            'Discontinuous',
            'DiedEarly',
            'BornLate',
            'NMissingYears',
        ]
    ].drop_duplicates()

    df = df.sort_values(by=[
        'WouldBeDropped',
        'Taxonomy',
        'InstitutionName',
        'DepartmentName',
    ])

    return df


if __name__ == '__main__':
    # we're using AA_2022_DEPARTMENT_CLEANED_EMPLOYMENT_PATH
    # because this is the minimum filtered set of departments. In terms of
    # finding out if a department is in all years, we don't care if there were
    # non-primary appointment apts/etc.

    df = pd.read_csv(usfhn.constants.AA_2022_DEPARTMENT_CLEANED_EMPLOYMENT_PATH)

    department_count = df['DepartmentId'].nunique()
    institution_count = df['InstitutionId'].nunique()

    df = get_df_for_spreadsheet(df)

    df.to_csv(
        usfhn.constants.DEPARTMENT_DROPPING_DIAGNOSTIC_CSV_PATH,
        index=False,
    )

    df = df.drop(columns='Taxonomy').drop_duplicates()

    department_count = df['DepartmentId'].nunique()
    insts = set(df['InstitutionName'].unique())
    institution_count = df['InstitutionName'].nunique()

    would_be_dropped = df[df['WouldBeDropped']]
    would_be_dropped_count = len(would_be_dropped)
    would_be_dropped_for_1_year = would_be_dropped[
        would_be_dropped['NMissingYears'] == 1
    ]

    print(len(would_be_dropped_for_1_year))

    would_be_dropped_department_count = would_be_dropped['DepartmentId'].nunique()
    
    not_dropped = df[~df['WouldBeDropped']]
    not_dropped_insts = set(not_dropped['InstitutionName'].unique())
    would_be_dropped_institution_count = institution_count - len(not_dropped_insts)

    print('total:')
    print(f"\tints: {institution_count}")
    print(f"\tdepts: {department_count}")
    print('would be dropped:')
    print(f"\tints: {would_be_dropped_institution_count}")
    print(f"\tdepts: {would_be_dropped_department_count}")

    discontinuous_count = len(df[df['Discontinuous']])

    df['DiscontinuousButStartAndEndAtStartAndEnd'] = (
        (~df['DiedEarly'])
        &
        (~df['BornLate'])
        &
        (df['Discontinuous'])
    )

    discontinuous_but_not_at_ends_count = len(df[df['DiscontinuousButStartAndEndAtStartAndEnd']])

    # print(f"{would_be_dropped_count} depts will be dropped for not appearing in all years")
    # print(f"{discontinuous_count} discontinuous departments")
    # print(f"{discontinuous_but_not_at_ends_count} discontinuous but not at ends departments")
