import pandas as pd
import json

import usfhn.constants
import usfhn.utils


if __name__ == '__main__':
    usfhn.utils.print_cleaning_step_start("Annotating Mid-career Moves")
    df = pd.read_csv(usfhn.constants.AA_2022_PRIMARY_APPOINTED_EMPLOYMENT_PATH)

    n_start = len(df)

    df = df[
        [
            'PersonId',
            'InstitutionId',
            'Year',
        ]
    ].drop_duplicates()

    df['StartingYearAtInstitution'] = df.groupby([
        'PersonId',
        'InstitutionId',
    ])['Year'].transform('min')

    df['LatestStart'] = df.groupby('PersonId')['StartingYearAtInstitution'].transform('max')

    df = df[
        df['StartingYearAtInstitution'] == df['LatestStart']
    ]

    df = df[
        [
            'PersonId',
            'InstitutionId',
        ]
    ].drop_duplicates()

    df.to_csv(
        usfhn.constants.AA_2022_LAST_EMPLOYING_INSTITUTION_ANNOTATIONS_PATH,
        index=False,
    )

    n_end = len(df)

    stats = {
        'pMidCareerMoveRowsRemoved': n_end / n_start
    }

    usfhn.constants.AA_2022_MULTI_CAREER_MOVE_STATS_PATH.write_text(
        json.dumps(stats, indent=4, sort_keys=True)
    )
