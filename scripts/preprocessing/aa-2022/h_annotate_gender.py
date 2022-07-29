import argparse
import numpy as np
import pandas as pd

import usfhn.constants
import usfhn.gender
import usfhn.utils


def split_name(name):
    first = None
    last = name

    if ',' in name:
        last, first = name.split(', ', maxsplit=1)

        if ' ' in first:
            first, _ = first.split(' ', maxsplit=1)

    return pd.Series((first, last))


def load_people():
    if not usfhn.constants.AA_2022_PEOPLE_GENDERED_INTERIM_PEOPLE_PATH.exists():
        df = pd.read_csv(usfhn.constants.AA_2022_MULTI_INSTITUTION_FILTERED_EMPLOYMENT_PATH)[
            [
                'PersonId',
                'PersonName',
                'FirstName',
                'MidName',
                'LastName',
            ]
        ].drop_duplicates().merge(
            pd.read_csv(usfhn.constants.AA_2022_PEOPLE_GENDERED_OLD_EMPLOYMENT_PEOPLE_PATH)[
                [
                    'PersonName',
                    'Gender',
                ]
            ],
            on='PersonName',
            how='left'
        )

        df['CheckedOffline'] = df.apply(lambda r: True if pd.notnull(r['Gender']) else False, axis=1)
        df['CheckedGenni'] = df.apply(lambda r: True if pd.notnull(r['Gender']) else False, axis=1)

        df.to_csv(
            usfhn.constants.AA_2022_PEOPLE_GENDERED_INTERIM_PEOPLE_PATH,
            index=False,
        )

    df = pd.read_csv(usfhn.constants.AA_2022_PEOPLE_GENDERED_INTERIM_PEOPLE_PATH).drop_duplicates()
    df["FirstName"] = df["FirstName"].fillna("")
    df["LastName"] = df["LastName"].fillna("")
    return df

def batch_gender_querying(df, gender_function, checked_column, batchsize=10000):
    person_id_to_gender = {}
    person_id_to_checked = {}
    for person_id, checked, gender in zip(df['PersonId'], df[checked_column], df['Gender']):
        person_id_to_checked[person_id] = checked
        person_id_to_gender[person_id] = gender

    print(f'df len: {len(df)}')
    print(f"unchecked ungendered len: {len(df[~df[checked_column] & df['Gender'].isnull()])}")
    count = 1
    while len(df[~df[checked_column] & df['Gender'].isnull()]):
        batch = df[
            (~df[checked_column])
            &
            (df['Gender'].isnull())
        ][
            ['PersonId', 'FirstName', 'LastName']
        ]

        batch = batch.head(batchsize).copy()

        batch['Gender'] = batch.apply(gender_function, axis=1)

        for person_id, gender in zip(batch['PersonId'], batch['Gender']):
            person_id_to_checked[person_id] = True
            person_id_to_gender[person_id] = gender

        df['Gender'] = df['PersonId'].apply(person_id_to_gender.get)
        df[checked_column] = df['PersonId'].apply(person_id_to_checked.get)
        df.to_csv(
            usfhn.constants.AA_2022_PEOPLE_GENDERED_INTERIM_PEOPLE_PATH,
            index=False
        )
        print(f'batch {count}:')
        print(f"\tunchecked ungendered len: {len(df[~df[checked_column] & df['Gender'].isnull()])}")
        count += 1


def add_gender_annotations(inferred_df):
    annotated_df = pd.read_csv(usfhn.constants.AA_FACULTY_GENDER_2022_DROP_PATH)[
        [
            'PersonId',
            'Gender',
            'URLPrimary'
        ]
    ].drop_duplicates()

    annotated_df = annotated_df[
        annotated_df['URLPrimary'].notna()
    ].drop(columns=['URLPrimary'])

    gender_remap = {
        'U': np.nan,
        np.nan: np.nan,
        'M': 'Male',
        'F': 'Female',
    }
    annotated_df['Gender'] = annotated_df['Gender'].apply(gender_remap.get)

    annotated_df = annotated_df[
        annotated_df['Gender'].notna()
    ]

    annotated_df['Source'] = 'Annotation'
    
    inferred_df['Source'] = 'Inferred'

    inferred_df = inferred_df[
        ~inferred_df['PersonId'].isin(annotated_df['PersonId'].unique())
    ]

    df = pd.concat([annotated_df, inferred_df])
    df['Gender'] = df['Gender'].fillna('Unknown')

    return df


if __name__ == '__main__':
    usfhn.utils.print_cleaning_step_start("Annotating Gender")

    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", "-r", default=False, action='store_true', help="reset the gendering.")
    
    args = parser.parse_args()

    if args.reset and usfhn.constants.AA_2022_PEOPLE_GENDERED_INTERIM_PEOPLE_PATH.exists():
        print('resetting')
        usfhn.constants.AA_2022_PEOPLE_GENDERED_INTERIM_PEOPLE_PATH.unlink()
        load_people()
        print('loaded people')

    df = load_people()
    batch_gender_querying(
        df,
        gender_function=usfhn.gender.get_offline_gender,
        checked_column='CheckedOffline',
        batchsize=10000
    )

    for g, rows in df.groupby('Gender', dropna=False):
        print(f"{g}: {len(rows)}")

    batch_gender_querying(
        df,
        gender_function=usfhn.gender.query_ethnea,
        checked_column='CheckedGenni',
        batchsize=1000
    )

    df = pd.read_csv(
        usfhn.constants.AA_2022_PEOPLE_GENDERED_INTERIM_PEOPLE_PATH,
    )

    gender_remap = {
        '-': 'Unknown',
        np.nan: 'Unknown',
        'M': 'Male',
        'F': 'Female',
    }
    df['Gender'] = df['Gender'].apply(gender_remap.get)

    df = df[
        ['PersonId', 'Gender']
    ]

    df = add_gender_annotations(df)

    df.to_csv(
        usfhn.constants.AA_2022_PEOPLE_GENDERED_PEOPLE_PATH,
        index=False
    )
