import numpy as np
import pandas as pd
import json

import usfhn.constants
import usfhn.utils


def remove_spurious_multi_institution_employments(df):
    """
    expects a dataframe with the following columns:
    - PersonId
    - InstitutionId
    - Year
    """
    employment_columns = list(df.columns)
    starting_row_count = len(df)
    starting_people = set(df['PersonId'].unique())
    person_count = df['PersonId'].nunique()

    df['JobStart'] = df.groupby(['PersonId', 'InstitutionId'])['Year'].transform('min')

    df['JobCountByYear'] = df.groupby([
        'PersonId',
        'Year',
    ])['InstitutionId'].transform('nunique')

    multi_df = df[
        df['JobCountByYear'] > 1
    ].copy()

    df = df[
        df['JobCountByYear'] == 1
    ]

    multi_job_row_count = len(multi_df)
    multi_job_rows_percent = round(100 * (multi_job_row_count / starting_row_count), 2)
    multi_job_person_count = multi_df['PersonId'].nunique()
    multi_job_person_percent = round(100 * (multi_job_person_count / person_count), 2)

    print(f"{multi_job_row_count} multiple-institution employment rows ({multi_job_rows_percent}% of rows)")
    print(
        f"{multi_job_person_count} people have multiple-institutions employments in a given year"
        +
        f" ({multi_job_person_percent}% of people)"
    )

    multi_df['MostRecentJobStartByYear'] = multi_df.groupby([
        'PersonId',
        'Year',
    ])['JobStart'].transform('max')

    multi_df = multi_df[
        multi_df['JobStart'] == multi_df['MostRecentJobStartByYear']
    ]

    removed_row_count = multi_job_row_count - len(multi_df)
    removed_row_percent = round(100 * (removed_row_count / starting_row_count), 2)

    print()
    print(f"removed {removed_row_count} spurious employment rows ({removed_row_percent}% of rows)")

    df = pd.concat([df, multi_df])

    ending_row_count = len(df)
    ending_row_percent = round(100 * (ending_row_count / starting_row_count), 2)
    print(f"{ending_row_count} rows remain ({ending_row_percent}% of rows at start)")

    ending_people = set(df['PersonId'].unique())

    assert(not len(starting_people ^ ending_people))

    df = df[employment_columns]

    return df


if __name__ == '__main__':
    """
    Sometimes people are listed in two institutions at the same time, but then
    stop being listed in one.

    This happens when, for example, someone moves from one institution to
    another, and the first institution doesn't remove them from the roster.

    So, imagine we have the following employment information for someone (x
    means they show up as employed at the institution, ):

    Years:             2011 | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020
    - @institution 1:     x |    x |    x |    x |    x |      |      |      |      |     
    - @institution 2:       |      |      |    x |    x |    x |    x |    x |      |     
    - @institution 3:       |      |      |      |      |      |      |    x |    x |    x

    But not all of these are legitimate; let's use y to denote an illegitimate
    employment (i.e., an employment that should be filtered out because it's not real)

    Years:             2011 | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020
    - @institution 1:     x |    x |    x |    y |    y |      |      |      |      |     
    - @institution 2:       |      |      |    x |    x |    x |    x |    y |      |     
    - @institution 3:       |      |      |      |      |      |      |    x |    x |    x

    We want to turn the above information into this:
    
    Years:             2011 | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020
    - @institution 1:     x |    x |    x |      |      |      |      |      |      |     
    - @institution 2:       |      |      |    x |    x |    x |    x |      |      |     
    - @institution 3:       |      |      |      |      |      |      |    x |    x |    x
    
    And that's what we'll do here
    """
    usfhn.utils.print_cleaning_step_start("Removing Spurious Multi-Institution Employments")

    df = pd.read_csv(usfhn.constants.AA_2022_TAXONOMY_CLEANED_EMPLOYMENT_PATH)

    n_start = len(df)

    df = remove_spurious_multi_institution_employments(df)

    df.to_csv(
        usfhn.constants.AA_2022_MULTI_INSTITUTION_FILTERED_EMPLOYMENT_PATH,
        index=False
    )

    n_end = n_start - len(df)

    stats = {
        'pMultiInstitutionRowsRemoved': n_end / n_start
    }

    usfhn.constants.AA_2022_MULTI_INSTITUTION_FILTERED_STATS_PATH.write_text(
        json.dumps(stats, indent=4, sort_keys=True)
    )
