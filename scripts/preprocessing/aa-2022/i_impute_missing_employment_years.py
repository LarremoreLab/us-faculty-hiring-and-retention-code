from functools import lru_cache
import json
import pandas as pd
import hnelib.utils

import usfhn.constants
import usfhn.utils


@lru_cache(maxsize=1)
def get_employment():
    return pd.read_csv(usfhn.constants.AA_2022_MULTI_INSTITUTION_FILTERED_EMPLOYMENT_PATH)


def get_person_df():
    df = get_employment().copy()[
        [
            'PersonId',
            'PersonName',
            'FirstName',
            'MidName',
            'LastName',
        ]
    ].drop_duplicates()

    return df


def get_department_df():
    df = get_employment().copy()[
        [
            'DepartmentId',
            'DepartmentName',
        ]
    ].drop_duplicates()

    return df


def get_institution_df():
    df = get_employment().copy()[
        [
            'InstitutionId',
            'InstitutionName',
        ]
    ].drop_duplicates()

    return df


@lru_cache(maxsize=1)
def get_years():
    return sorted(list(get_employment()['Year'].unique()))


def get_departments_in_years_df():
    df = get_employment().copy()[
        [
            'DepartmentId',
            'Year',
        ]
    ].drop_duplicates()

    return df

@lru_cache(maxsize=1)
def get_original_appointments_df():
    df = get_employment().copy()[
        [
            'PersonId',
            'DepartmentId',
            'InstitutionId',
            'Year',
            'Rank',
            'IsPrimaryAppointment',
        ]
    ].drop_duplicates()
    return df


def get_appointments_to_impute(df=get_original_appointments_df()):
    df = df.copy()
    columns = list(df.columns)

    groupby_cols = ['PersonId', 'DepartmentId']

    df['DepartmentStart'] = df.groupby(groupby_cols)['Year'].transform('min')
    df['DepartmentEnd'] = df.groupby(groupby_cols)['Year'].transform('max')
    df['DepartmentYears'] = df.groupby(groupby_cols)['Year'].transform('nunique')
    df['DepartmentYearRange'] = 1 + df['DepartmentEnd'] - df['DepartmentStart']

    appointments_to_impute = df.copy()[
        df['DepartmentYears'] < df['DepartmentYearRange']
    ].drop_duplicates()

    n_people = df['PersonId'].nunique()
    n_people_to_impute = appointments_to_impute['PersonId'].nunique()
    p_people_to_impute = hnelib.utils.fraction_to_percent(n_people_to_impute / n_people, round_to=2)

    print(f"{n_people_to_impute} people to impute rows for ({p_people_to_impute}%)")

    return appointments_to_impute


def impute_missing_appointments(appointments_to_impute):
    """
    `to_impute_from` columns:
    - PersonId
    - DepartmentId
    - Year
    - Rank
    - DepartmentStart: first year PersonId is in DepartmentId
    - DepartmentEnd: last year PersonId is in DepartmentId
    """
    years = get_years()
    years_set = set(years)

    appointments_to_impute = appointments_to_impute.copy()[
        [
            'PersonId',
            'DepartmentId',
            'InstitutionId',
            'Year',
            'Rank',
            'DepartmentStart',
            'DepartmentEnd',
        ]
    ].drop_duplicates()

    imputations = []
    for (person_id, department_id), rows in appointments_to_impute.groupby(['PersonId', 'DepartmentId']):
        start = rows.iloc[0]['DepartmentStart']
        end = rows.iloc[0]['DepartmentEnd']
        institution_id = rows.iloc[0]['InstitutionId']

        person_years = set(rows['Year'].unique())
        missing_years = {y for y in years_set if start < y < end} - person_years
        for year in missing_years:
            nearest_existing_year = max([y for y in person_years if y < year])

            nearest_existing_year_rank = rows[
                rows['Year'] == nearest_existing_year
            ].iloc[0]['Rank']

            imputations.append({
                'PersonId': person_id,
                'DepartmentId': department_id,
                'InstitutionId': institution_id,
                'Year': year,
                'Rank': nearest_existing_year_rank,
            })

    return pd.DataFrame(imputations)


def exclude_imputations_when_other_appointments_exist(df, n_imputable_rows):
    """
    If we see a person who is:
    1. in department A in year 1
    2. in department B in year 2
    3. in department A in year 3

    We're not going to impute department A in year 2.
    """
    # every row in `imputations` is a row `DepartmentId`-`Year` pair that does
    # not exist in the original appointments, so if we left join on that, all
    # the rows that show up indicate the person has other existing appointments.
    person_years_df = get_original_appointments_df().copy()[
        [
            'PersonId',
            'Year',
        ]
    ].drop_duplicates()

    person_years_df['HasAppointmentsInYear'] = True

    df = df.merge(
        person_years_df,
        on=[
            'PersonId',
            'Year',
        ],
        how='left',
    )

    df['HasAppointmentsInYear'] = df['HasAppointmentsInYear'].fillna(False)

    df = df[
        ~df['HasAppointmentsInYear']
    ]

    n_rows_to_impute = len(df)
    n_rows_not_to_impute = n_imputable_rows - n_rows_to_impute

    p_rows_not_to_impute = hnelib.utils.fraction_to_percent(n_rows_not_to_impute / n_imputable_rows, 2)

    print(f"\tnot imputing {n_rows_not_to_impute} ({p_rows_not_to_impute}%) because the people had other appointments in the year that would otherwise be imputed")

    df = df[
        [
            'PersonId',
            'DepartmentId',
            'InstitutionId',
            'Year',
            'Rank',
        ]
    ].drop_duplicates()

    return df


def exclude_imputations_into_departments_absent_in_year(df, n_imputable_rows_original):
    department_years_df = get_departments_in_years_df()

    department_years_df['DepartmentExistsInYear'] = True

    df = df.merge(
        department_years_df,
        on=[
            'DepartmentId',
            'Year',
        ],
        how='left',
    )

    df['DepartmentExistsInYear'] = df['DepartmentExistsInYear'].fillna(False)

    n_imputable_rows = len(df)

    df = df[
        df['DepartmentExistsInYear']
    ]

    n_rows_to_impute = len(df)
    n_rows_not_to_impute = n_imputable_rows - n_rows_to_impute

    p_rows_not_to_impute = hnelib.utils.fraction_to_percent(n_rows_not_to_impute / n_imputable_rows_original, 2)

    print(f"\tnot imputing {n_rows_not_to_impute} ({p_rows_not_to_impute}%) because the departments did not exist in the year to be imputed")

    df = df[
        [
            'PersonId',
            'DepartmentId',
            'InstitutionId',
            'Year',
            'Rank',
        ]
    ].drop_duplicates()

    return df

def add_imputations_to_employment(imputed_appointments):
    original_employment = get_employment()
    original_columns = list(original_employment.columns)

    imputed_appointments['IsPrimaryAppointment'] = False

    all_appointments = pd.concat([imputed_appointments, get_original_appointments_df()])

    n_imputed_people = imputed_appointments['PersonId'].nunique()
    n_people = original_employment['PersonId'].nunique()
    p_imputed_people = hnelib.utils.fraction_to_percent(n_imputed_people / n_people, 2)
    p_imputed_rows = hnelib.utils.fraction_to_percent(len(imputed_appointments) / len(all_appointments), 2)

    print(f"% of people: {p_imputed_people}")
    print(f"% of rows: {p_imputed_rows}")

    df = all_appointments.merge(
        get_institution_df(),
        on='InstitutionId',
    ).merge(
        get_department_df(),
        on='DepartmentId'
    ).merge(
        get_person_df(),
        on='PersonId',
    )

    assert(set(original_columns) == set(list(df.columns)))
    assert(len(original_employment) + len(imputed_appointments) == len(df))
    assert(set(original_employment['PersonId'].unique()) == set(df['PersonId'].unique()))

    df = df[original_columns]

    return df


if __name__ == '__main__':
    """
    We're going to people with appointments like:
    - year 1: department A
    - year 2: nothing
    - year 3: department A

    We're going to impute appointments for these people. In the above case,
    we'd impute the following appointment:
    - year 2: department A

    We're going to set the rank of an imputed row to the rank of the most
    recent (previous) observed rank.

    So if we see a person with appointments like:
    - year 1: department A, Assistant Professor
    - year 2: nothing
    - year 3: department A, Associate Professor

    We'll impute the following row:
    - year 2: department A, Assistant Professor

    If we see someone with appointments in another department during an
    imputable gap, we will NOT impute over that gap. 

    As an example of a situation where appointments will NOT be imputed:
    - year 1: department A
    - year 2: department B
    - year 3: department A

    We won't impute when an imputation would add rows to a department that
    disappears in a given year.
    """
    usfhn.utils.print_cleaning_step_start("Imputing Appointments")

    appointments_to_impute = get_appointments_to_impute()

    imputed_appointments = impute_missing_appointments(appointments_to_impute)

    n_imputable_rows = len(imputed_appointments)

    print(f"{n_imputable_rows} rows could be imputed")

    imputed_appointments = exclude_imputations_when_other_appointments_exist(
        imputed_appointments,
        n_imputable_rows,
    )
    imputed_appointments = exclude_imputations_into_departments_absent_in_year(
        imputed_appointments,
        n_imputable_rows,
    )

    n_rows_to_impute = len(imputed_appointments)
    n_people_to_impute_rows_for = imputed_appointments['PersonId'].nunique()

    print(f"imputing {n_rows_to_impute} rows ({n_people_to_impute_rows_for} people)")

    df = add_imputations_to_employment(imputed_appointments)

    df.to_csv(
        usfhn.constants.AA_2022_PEOPLE_IMPUTED_EMPLOYMENT_PATH,
        index=False,
    )

    df_start = get_employment()
    n_rows_start = len(df_start)
    n_people_start = df_start['PersonId'].nunique()

    n_rows_end = len(df)
    n_people_end = df['PersonId'].nunique()

    n_rows_imputed = n_rows_end - n_rows_start

    stats = {
        'pImputationRows': n_rows_imputed / n_rows_start,
        'pImputationPeople': n_people_to_impute_rows_for / n_people_start,
    }

    usfhn.constants.AA_2022_PEOPLE_IMPUTED_STATS_PATH.write_text(json.dumps(stats, indent=4, sort_keys=True))
