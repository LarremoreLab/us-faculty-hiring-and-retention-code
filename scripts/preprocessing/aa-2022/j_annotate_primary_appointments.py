from functools import lru_cache
import pandas as pd
import json

import usfhn.constants
import usfhn.utils


RANKS = [
    'Unknown',
    'Other',
    'Instructor',
    'Lecturer',
    'Assistant Professor',
    'Associate Professor',
    'Professor',
]

RANK_TO_INDEX = {rank: i for i, rank in enumerate(RANKS)}
INDEX_TO_RANK = {i: rank for i, rank in enumerate(RANKS)}


def get_sanity_check_df(df, when):
    sanity_df = df.copy()[
        [
            'PersonId',
            'Year',
            'InstitutionId',
        ]
    ].drop_duplicates()

    sanity_df[when] = True

    return sanity_df


def get_pre_filter_sanity_df(df):
    return get_sanity_check_df(df, when='Pre')

def get_post_filter_sanity_df(df):
    return get_sanity_check_df(df, when='Post')

def check_sanity(pre_df, post_df):
    """
    this is to be used when a filter should not be removing people
    """
    df = pre_df.merge(
        post_df,
        on=[
            'PersonId',
            'Year',
            'InstitutionId',
        ],
    )

    assert(df[df['Pre'].isna()].empty)
    assert(df[df['Post'].isna()].empty)


def get_employment(print_results=True):
    df = pd.read_csv(usfhn.constants.AA_2022_PEOPLE_IMPUTED_EMPLOYMENT_PATH)[
        [
            'PersonId',
            'Year',
            'Rank',
            'DepartmentId',
            'InstitutionId',
            'IsPrimaryAppointment',
        ]
    ].drop_duplicates()

    taxonomy_df = pd.read_csv(usfhn.constants.AA_2022_TAXONOMY_CLEANED_DEPARTMENTS_PATH)[
        [
            'Taxonomy',
            'DepartmentId',
        ]
    ].drop_duplicates()

    df = df.merge(
        taxonomy_df,
        on='DepartmentId'
    )

    pre_df = get_pre_filter_sanity_df(df)

    df['IsPrimaryAppointment'] = df['IsPrimaryAppointment'].apply(bool)

    degrees = pd.read_csv(usfhn.constants.AA_2022_DEGREE_FILTERED_DEGREES_PATH)[
        [
            'PersonId',
            'DegreeYear',
        ]
    ].drop_duplicates()

    df = df.merge(
        degrees,
        on='PersonId',
        how='left'
    )
    df['DegreeYear'] = df['DegreeYear'].fillna(0)

    df = df[
        df['DegreeYear'] <= df['Year']
    ].drop(columns=['DegreeYear']).drop_duplicates()

    if print_results:
        print(f"{len(df)} rows")
        print(f"\t{df.PersonId.nunique()} people")

    check_sanity(pre_df, get_post_filter_sanity_df(df))

    return df


def filter_graduate_employments(df):
    degree_years = pd.read_csv(usfhn.constants.AA_2022_DEGREE_FILTERED_DEGREES_PATH)[
        [
            'PersonId',
            'DegreeYear',
        ]
    ].drop_duplicates()

    pre_df = get_pre_filter_sanity_df(df)

    df = df.merge(
        degree_years,
        on='PersonId',
        how='left',
    )

    df['DegreeYear'] = df['DegreeYear'].fillna(0)

    pre_len = len(df)

    df = df[
        df['Year'] >= df['DegreeYear']
    ]
    post_len = len(df)
    print(f"dropping {pre_len - post_len} rows with DegreeYear > Year")
    df = df.drop(columns=['DegreeYear'])

    check_sanity(pre_df, get_post_filter_sanity_df(df))

    return df


@lru_cache()
def get_clean_employment():
    """
    filters graduate employments and removes rank disagreements
    """
    df = get_employment()
    df = filter_graduate_employments(df.copy())
    df = remove_rank_disagreeing_appointments(df.copy(deep=True))
    return df


def print_filtering_results(clean_df, df, pass_df, fail_df, pass_string, fail_string, indent="\t"):
    rows = len(df) + len(clean_df)
    pass_count = len(pass_df)
    fail_count = len(fail_df)
    pass_percent = round(100 * (pass_count / rows), 2)
    fail_percent = round(100 * (fail_count / rows), 2)

    print(f"{indent}{pass_string}: {len(pass_df)} ({pass_percent}%)")
    print(f"{indent}{fail_string}: {len(fail_df)} ({fail_percent}%)")


def sanitize_df(df, columns):
    return df[list(columns)]


def sanitize_and_update_cleaned_df(clean_df, just_cleaned_df, to_clean_df):
    just_cleaned_df = sanitize_df(just_cleaned_df, clean_df.columns)
    to_clean_df = sanitize_df(to_clean_df, clean_df.columns)

    clean_df = pd.concat([clean_df, just_cleaned_df])

    return clean_df, to_clean_df


def expand_cleaned_departments(df, pass_column):
    """
    if you ever have a primary appointment in a department, you always have a
    primary appointment in that department
    """
    primary_appointments_df = df[
        df[pass_column]
    ][
        [
            'PersonId',
            'DepartmentId',
        ]
    ].drop_duplicates()

    primary_appointments_df['IsOnlyDepartmentAtSomePoint'] = True

    df = df.merge(
        primary_appointments_df,
        on=['PersonId', 'DepartmentId'],
        how='left',
    )

    df['IsOnlyDepartmentAtSomePoint'] = df['IsOnlyDepartmentAtSomePoint'].fillna(False)

    df[pass_column] = df['IsOnlyDepartmentAtSomePoint']
    return df

def expand_cleaned_taxonomies(df, pass_column):
    """
    if you ever have a primary appointment in a taxonomy, you always have a
    primary appointment in that taxonomy whenever it appears
    """
    primary_appointments_df = df[
        df[pass_column]
    ][
        [
            'PersonId',
            'Taxonomy',
        ]
    ].drop_duplicates()

    primary_appointments_df['IsOnlyTaxonomyAtSomePoint'] = True

    df = df.merge(
        primary_appointments_df,
        on=['PersonId', 'Taxonomy'],
        how='left',
    )

    df['IsOnlyTaxonomyAtSomePoint'] = df['IsOnlyTaxonomyAtSomePoint'].fillna(False)

    df[pass_column] = df['IsOnlyTaxonomyAtSomePoint']
    return df

def clean_people_in_only_one_dept_per_inst(clean_df, df):
    df['DepartmentCountPerInstitution'] = df.groupby(
        ['PersonId', 'InstitutionId']
    )['DepartmentId'].transform('nunique')

    df['InOneDepartmentPerInst'] = df['DepartmentCountPerInstitution'] == 1

    df = expand_cleaned_departments(df, 'InOneDepartmentPerInst')

    pass_df = df.copy()[
        df['InOneDepartmentPerInst']
    ]

    fail_df = df.copy()[
        ~df['InOneDepartmentPerInst']
    ]

    print(f"marking people only in one department per institution as clean:")
    print_filtering_results(
        clean_df,
        df,
        pass_df,
        fail_df,
        pass_string='employment rows of people only ever in one department per institution',
        fail_string='employment rows of people in more than one department per institution',
    )

    return sanitize_and_update_cleaned_df(clean_df, pass_df, fail_df)


def clean_people_always_in_only_one_dept(clean_df, df):
    columns = df.columns
    df['CareerLength'] = df.groupby('PersonId')['Year'].transform('nunique')
    df['DepartmentCareerLength'] = df.groupby(['PersonId', 'DepartmentId'])['Year'].transform('nunique')

    df['InDepartmentEveryYear'] = df['CareerLength'] == df['DepartmentCareerLength']

    df['DepartmentsPerYear'] = df.groupby(['PersonId', 'Year'])['DepartmentId'].transform('nunique')

    df['AlwaysInOnlyOneDepartment'] = df['DepartmentsPerYear'] == 1
    df['AlwaysInOnlyOneDepartment'] &= df['InDepartmentEveryYear']

    pass_df = df.copy()[
        df['AlwaysInOnlyOneDepartment']
    ]

    fail_df = df.copy()[
        ~df['AlwaysInOnlyOneDepartment']
    ]

    print(f"marking people who are always in one department as clean:")
    print_filtering_results(
        clean_df,
        df,
        pass_df,
        fail_df,
        pass_string='employment rows of people always in one department',
        fail_string='employment rows of people not always in one department',
    )

    return sanitize_and_update_cleaned_df(clean_df, pass_df, fail_df)


def clean_people_always_in_only_one_dept_per_inst(clean_df, df):
    """
    look for
    1. employments at departments that are the same length as employments at the institution

    years where those were the person's only employment are `clean`.

    mark all employments in those departments as primary
    """

    df['InstitutionYears'] = df.groupby(['PersonId', 'InstitutionId'])['Year'].transform('nunique')
    df['FirstInstitutionYear'] = df.groupby(['PersonId', 'InstitutionId'])['Year'].transform('min')

    df['DepartmentYears'] = df.groupby(['PersonId', 'DepartmentId'])['Year'].transform('nunique')
    df['FirstDepartmentYear'] = df.groupby(['PersonId', 'DepartmentId'])['Year'].transform('min')

    df['DeptStart == InstStart'] = df['FirstDepartmentYear'] == df['FirstInstitutionYear']
    df['DeptYears == InstYears'] = df['DepartmentYears'] == df['InstitutionYears']

    df['AlwaysInDeptAtInst'] = df['DeptStart == InstStart'] & df['DeptYears == InstYears']

    df['DepartmentsPerYear'] = df.groupby(['PersonId', 'Year'])['DepartmentId'].transform('nunique')

    df['AlwaysInOneDeptPerInst'] = df['AlwaysInDeptAtInst']
    df['AlwaysInOneDeptPerInst'] &= df['DepartmentsPerYear'] == 1

    df = expand_cleaned_departments(df, 'AlwaysInOneDeptPerInst')

    pass_df = df.copy()[
        df['AlwaysInOneDeptPerInst']
    ]

    fail_df = df.copy()[
        ~df['AlwaysInOneDeptPerInst']
    ]

    print(f"marking people who are always in one department per institution as clean:")
    print_filtering_results(
        clean_df,
        df,
        pass_df,
        fail_df,
        pass_string='employment rows of people always in one department per institution',
        fail_string='employment rows of people not always in one department per institution',
    )

    return sanitize_and_update_cleaned_df(clean_df, pass_df, fail_df)


def clean_people_in_one_department_multiple_times(clean_df, df):
    df['OnlyDepartmentInYear'] = ~df['MultipleDepartmentsInYear']

    df = expand_cleaned_departments(df, 'OnlyDepartmentInYear')

    pass_df = df.copy()[
        df['OnlyDepartmentInYear']
    ]

    fail_df = df.copy()[
        ~df['OnlyDepartmentInYear']
    ]

    print(f"marking employments in departments that are ever a person's only employment as clean:")
    print_filtering_results(
        clean_df,
        df,
        pass_df,
        fail_df,
        pass_string='employment rows marked clean',
        fail_string="employment rows of two or more departments that are never a person's only employment",
    )

    return sanitize_and_update_cleaned_df(clean_df, pass_df, fail_df)


def clean_people_in_one_taxonomy_multiple_times(clean_df, df):
    df['OnlyTaxonomyInYear'] = ~df['MultipleTaxonomiesInYear']

    df = expand_cleaned_taxonomies(df, 'OnlyTaxonomyInYear')
    df = expand_cleaned_departments(df, 'OnlyTaxonomyInYear')

    pass_df = df.copy()[
        df['OnlyTaxonomyInYear']
    ]

    fail_df = df.copy()[
        ~df['OnlyTaxonomyInYear']
    ]

    print(f"marking employments in departments with taxonomies that are ever a person's only taxonomy as clean:")
    print_filtering_results(
        clean_df,
        df,
        pass_df,
        fail_df,
        pass_string='employment rows marked clean',
        fail_string="employment rows of two or more taxonomies that are never a person's only taxonomy",
    )

    return sanitize_and_update_cleaned_df(clean_df, pass_df, fail_df)


def get_aa_primary_listed(employment, clean_df, unclean_df):
    # This bit here is just to (accurately) lower the count of rows we're adding
    # when we add back in AA's annotations
    clean_df = clean_df.copy()
    clean_df['Primary'] = True
    employment = employment.copy().merge(
        clean_df,
        on=['PersonId', 'DepartmentId', 'InstitutionId', 'Year'],
        how='left'
    )
    employment = employment[
        employment['Primary'].isnull()
    ].drop(columns=['Primary'])

    # Remove some of the columns
    unclean_df = unclean_df[employment.columns]

    aa_primary = employment[
        employment['IsPrimaryAppointment']
    ][
        ['PersonId', 'DepartmentId', 'InstitutionId']
    ].drop_duplicates()

    passes_condition_df = employment.merge(
        aa_primary,
        on=['PersonId', 'DepartmentId', 'InstitutionId']
    )[
        ['PersonId', 'DepartmentId', 'InstitutionId', 'Year']
    ]

    passes_condition_df['Primary'] = True
        
    fails_condition_df = unclean_df.merge(
        passes_condition_df,
        on=['PersonId', 'DepartmentId', 'InstitutionId', 'Year'],
        how='left'
    )

    fails_condition_df = fails_condition_df[
        fails_condition_df['Primary'].isnull()
    ].drop(columns=['Primary'])

    passes_condition_df = passes_condition_df.drop(columns=['Primary'])

    print(f"{len(passes_condition_df)} rows added back in because AA marked them as primary")
    print(f"\t{passes_condition_df.PersonId.nunique()} people")

    print(f"{len(fails_condition_df)} rows remain")
    print(f"\t{fails_condition_df.PersonId.nunique()} people")
    return fails_condition_df, passes_condition_df


def remove_rank_disagreeing_appointments(df):
    """
    1. assign each Person-Department-Year to the highest rank
    2. remove appointment rows that disagree on rank
    """
    pre_df = get_pre_filter_sanity_df(df)
    columns = df.columns

    df['RankIndex'] = df['Rank'].apply(RANK_TO_INDEX.get)
    df['MaxDepartmentRankIndexInYear'] = df.groupby([
        'PersonId', 'DepartmentId', 'Year'
    ])['RankIndex'].transform('max')

    df['Rank'] = df['MaxDepartmentRankIndexInYear'].apply(INDEX_TO_RANK.get)
    df['RankIndex'] = df['Rank'].apply(RANK_TO_INDEX.get)

    rows_at_start = len(df)

    df = df.drop(columns=[
        'MaxDepartmentRankIndexInYear'
    ]).drop_duplicates()

    row_count = len(df)
    within_department_rank_disagreements_rows = rows_at_start - row_count

    print(f"rows before dropping within-department rank conflicts: {rows_at_start}")
    print(f"\trows with within-department rank conflicts: {within_department_rank_disagreements_rows}")
    print(f"\trows remaining: {row_count}")

    df['RanksInYear'] = df.groupby(['PersonId', 'Year'])['Rank'].transform('nunique')
    df['MaxRankIndexInYear'] = df.groupby(['PersonId', 'Year'])['RankIndex'].transform('max')

    # identify rank disagreeing rows
    df['ToRemoveDueToRankDisagreement'] = (
        df['RanksInYear'] > 1
    ) & (
        df['RankIndex'] != df['MaxRankIndexInYear']
    )

    to_remove = df[
        df['ToRemoveDueToRankDisagreement']
    ]

    df = df[
        ~df['ToRemoveDueToRankDisagreement']
    ]

    rank_conflict_rows = len(to_remove)
    rows_at_end = len(df)

    print(f"rows before dropping rank conflicts: {row_count}")
    print(f"\trows with rank conflicts: {rank_conflict_rows}")
    print(f"\trows remaining: {rows_at_end}")

    df = df[
        columns
    ]

    check_sanity(pre_df, get_post_filter_sanity_df(df))

    return sanitize_df(df, columns)


def assign_ranks_to_primary_appointments(primary_appointments, raw_employment):
    ranks = raw_employment[
        ['PersonId', 'DepartmentId', 'InstitutionId', 'Year', 'Rank']
    ].drop_duplicates()

    primary_appointments = primary_appointments.merge(
        ranks,
        on=['PersonId', 'DepartmentId', 'InstitutionId', 'Year'],
    )

    # Set the rank to the max rank in the year
    primary_appointments['RankIndex'] = primary_appointments['Rank'].apply(RANK_TO_INDEX.get)
    primary_appointments['MaxRankIndexInYear'] = primary_appointments.groupby([
        'PersonId', 'Year'
    ])['RankIndex'].transform('max')
    primary_appointments['Rank'] = primary_appointments['MaxRankIndexInYear'].apply(INDEX_TO_RANK.get)

    primary_appointments = primary_appointments.drop(columns=[
        'RankIndex', 'MaxRankIndexInYear'
    ]).drop_duplicates()
    return primary_appointments


def print_out_unclean_stats(to_clean, cleaned):
    """
    We're going to say a person who has a primary appointment for every year 

    """
    to_clean = to_clean[cleaned.columns]

    partially_unclean_cleaned = cleaned[
        cleaned['PersonId'].isin(to_clean['PersonId'].unique())
    ][
        ['PersonId', 'Year']
    ].copy().drop_duplicates()

    partially_unclean_cleaned['HasPrimaryAppointmentForPersonYear'] = True

    to_clean = to_clean.merge(
        partially_unclean_cleaned,
        on=['PersonId', 'Year'],
        how='left'
    )

    print(f"there are {len(to_clean)} rows that we haven't ruled out from being primary appointments")
    to_clean = to_clean[
        to_clean['HasPrimaryAppointmentForPersonYear'].isnull()
    ].drop(columns=['HasPrimaryAppointmentForPersonYear'])

    print(f"but only {len(to_clean)} rows that are in a Person-Year we don't have any Primary Appointment for")
    print(to_clean.nunique())

    entirely_dropped_people = set(to_clean['PersonId']) - set(cleaned['PersonId'])
    entirely_dropped_to_clean = to_clean[
        to_clean.PersonId.isin(entirely_dropped_people)
    ].copy()
    print(f"{len(entirely_dropped_people)} people have no primary appointment (are dropped)")
    print(f"{len(entirely_dropped_to_clean)} employment rows")
    print(entirely_dropped_to_clean.nunique())

    entirely_dropped_to_clean.to_csv(
        usfhn.constants.AA_2022_PRIMARY_APPOINTED_PEOPLE_WHO_WHERE_DROPPED,
        index=False,
    )

def annotate_multi_department_and_multi_taxonomy_employments(df):
    columns = list(df.columns)
    df['DepartmentsInYear'] = df.groupby([
        'PersonId',
        'Year',
    ])['DepartmentId'].transform('nunique')

    df['MultipleDepartmentsInYear'] = df['DepartmentsInYear'] > 1
    columns.append('MultipleDepartmentsInYear')

    df['TaxonomiesInYear'] = df.groupby([
        'PersonId',
        'Year',
    ])['Taxonomy'].transform('nunique')

    df['MultipleTaxonomiesInYear'] = df['TaxonomiesInYear'] > 1
    columns.append('MultipleTaxonomiesInYear')

    return sanitize_df(df, columns)


def run():
    employment = get_clean_employment()

    to_clean_df = annotate_multi_department_and_multi_taxonomy_employments(employment.copy(deep=True))

    clean_df = pd.DataFrame(columns=to_clean_df.columns)

    # filtering steps
    clean_df, to_clean_df = clean_people_in_only_one_dept_per_inst(clean_df, to_clean_df)
    clean_df, to_clean_df = clean_people_always_in_only_one_dept_per_inst(clean_df, to_clean_df)
    clean_df, to_clean_df = clean_people_in_one_department_multiple_times(clean_df, to_clean_df)
    clean_df, to_clean_df = clean_people_in_one_taxonomy_multiple_times(clean_df, to_clean_df)

    clean_df = clean_df[
        [
            'PersonId',
            'DepartmentId',
            'InstitutionId',
            'Year',
        ]
    ].drop_duplicates()

    to_clean_df, aa_primary_listed = get_aa_primary_listed(employment.copy(deep=True), clean_df, to_clean_df)
    primary_appointments = pd.concat([clean_df, aa_primary_listed])
    primary_appointments = assign_ranks_to_primary_appointments(primary_appointments, employment)

    print('people/year @start:')
    print(employment.groupby(['Year'])['PersonId'].nunique())
    print('people/year @end:')
    print(primary_appointments.groupby(['Year'])['PersonId'].nunique())

    # how many people are we dropping total?
    # how many people-years are we dropping?
    print_out_unclean_stats(to_clean_df, primary_appointments)
    write_primary_appointments(primary_appointments)
    write_stats()


def check_person_year_ranks_are_unique(df):
    df = df.copy()[
        [
            'PersonId',
            'Year',
            'Rank',
        ]
    ].drop_duplicates()

    df['PersonYearRanks'] = df.groupby(['PersonId', 'Year'])['Rank'].transform('nunique')

    people_with_multiple_ranks_in_a_year = len(df[df['PersonYearRanks'] > 1])
    assert(not people_with_multiple_ranks_in_a_year)


def write_primary_appointments(primary_appointments):
    employment = get_clean_employment()[
        [
            'PersonId',
            'DepartmentId',
            'InstitutionId',
            'Year',
            'Rank',
        ]
    ].drop_duplicates()

    primary_appointments = primary_appointments[
        [
            'PersonId',
            'DepartmentId',
            'InstitutionId',
            'Year',
        ]
    ].drop_duplicates()

    primary_appointments['PrimaryAppointment'] = True

    df = employment.merge(
        primary_appointments,
        on=[
            'PersonId',
            'DepartmentId',
            'InstitutionId',
            'Year',
        ],
        how='left',
    )

    df['PrimaryAppointment'] = df['PrimaryAppointment'].fillna(False)

    check_person_year_ranks_are_unique(df)

    df.to_csv(
        usfhn.constants.AA_2022_PRIMARY_APPOINTED_EMPLOYMENT_PATH,
        index=False
    )


def write_stats():
    df_start = pd.read_csv(usfhn.constants.AA_2022_PRIMARY_APPOINTED_EMPLOYMENT_PATH)

    df_end = df_start[
        df_start['PrimaryAppointment']
    ]

    n_rows_start = len(df_start)
    n_rows_end = len(df_end)
    n_rows_removed = n_rows_start - n_rows_end
    p_rows_removed = n_rows_removed / n_rows_start

    n_people_start = df_start['PersonId'].nunique()
    n_people_end = df_end['PersonId'].nunique()
    n_people_removed = n_people_start - n_people_end
    p_people_removed = n_people_removed / n_people_start

    stats = {
        'pNonPrimaryApptRowsRemoved': p_rows_removed,
        'pNonPrimaryApptPeopleExcluded': p_people_removed,
    }

    usfhn.constants.AA_2022_PRIMARY_APPOINTED_STATS_PATH.write_text(json.dumps(stats, indent=4, sort_keys=True))


if __name__ == '__main__':
    """
    In order to do almost any analysis where we use the department information
    for faculty, we have to identify their primary appointment.

    AA provided us a dataset that contains (some of) these primary appointments
    in 2017, using voluntary responses from universities. We're not going to use
    this except to validate our heuristics.

    This leaves us the data from 2009-2016 to backfill, as well as the rest of
    the 2017 data that was not supplied by universities.

    Before we start doing any of this, we remove appointments that precede a person's degree year.

    We make the following assumptions:
    1. A single appointment is a primary appointment -- always. In other words,
       a person with the following appointments (departments A, B, C):
        - employment
            - 2009: A
            - 2010: A, B
            - 2011: A, B, C
            - 2012: C
        - appointments:
            - 2009: A
            - 2010: A
            - 2011: A, C
            - 2012: C
    2. A person does not hold a primary appointment in a department that fails to update their rank.

    The first thing we do is remove employment rows that fall under the category of assumption #2.

    We then identify cleaned people in steps, because it helps us be sure we're
    being reasonble, and also allows us to filter out additional rows along the way:
    1. people only ever associated with a single department
        - example:
            - employment:
                - 2009: A
                - 2010: A
            - appointments:
                - 2009: A
                - 2010: A
        - consider cleaned: all PersonIds in this set
    2. people only ever associated with a single department per institution
        - example:
            - employment:
                - 2009: A (@1)
                - 2010: B (@2)
                - 2011: C, D (@3)
                - 2012: E, F (@3)
            - appointments:
                - 2009: A
                - 2010: B
            - to clean:
                - 2011: C, D (@3)
                - 2012: E, F (@3)
        - consider cleaned: all PersonId-InstitutionId pairs in this set
          To be explicit: we will still try to clean @3.
    3. people always in only one department
        - example:
            - employment:
                - 2009: A
                - 2010: A, B
                - 2011: A
            - appointments:
                - 2009: A
                - 2010: A
                - 2011: A
        - consider cleaned: all PersonIds in this set
    4. people always in only one department per institution
        - example:
            - employment:
                - 2009: A (@1)
                - 2010: A, B (@1)
                - 2011: C, D (@2)
                - 2012: C (@2)
            - appointments:
                - 2009: A
                - 2010: A
                - 2011: C
                - 2012: C
        - consider cleaned: all PersonId-InstitutionId pairs in this set
    5. people in one department multiple times
        - example:
            - employment:
                - 2009: A
                - 2010: A, B
                - 2011: B
            - appointments:
                - 2009: A
                - 2010: A, B
                - 2011: B
        - consider cleaned: None. We don't remove rows after this step.
    6. We add in appointments that AA has marked as primary
        (and propogate them backwards, along assumtion #1)
    """
    usfhn.utils.print_cleaning_step_start("Annotating Primary Appointments")
    run()
