import pandas as pd

import usfhn.utils
import usfhn.constants


if __name__ == '__main__':
    """
    We're going to filter out non-tenure track people here
    """
    usfhn.utils.print_cleaning_step_start("Tenure Track Filter")
    df = pd.read_csv(usfhn.constants.AA_2022_BASIC_CLEANING_EMPLOYMENT_PATH)

    starting_count = df['PersonId'].nunique()
    print(f"starting with {starting_count} people")

    df = df[
        df['Rank'].isin(usfhn.constants.RANKS)
    ]

    ending_count = df['PersonId'].nunique()
    print(f"\tremoving {starting_count - ending_count} non-tenure track people")
    print(f"ending with {ending_count} people")

    df = df.to_csv(
        usfhn.constants.AA_2022_TENURE_TRACK_FILTERED_EMPLOYMENT_PATH,
        index=False,
    )
