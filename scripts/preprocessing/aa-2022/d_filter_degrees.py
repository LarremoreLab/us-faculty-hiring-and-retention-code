from functools import lru_cache
import json
import numpy as np
import pandas as pd

import hnelib.utils

import usfhn.utils
import usfhn.constants


MAX_DEGREE_YEAR_TO_EMPLOYMENT_YEAR = 100


DEGREE_TYPES_TO_REJECT = [
    'Bachelor',
    'Master',
]

DEGREE_SUBSTRING_TO_DEGREE_TYPE = {
    'Bachelor': "Bachelor",
    'Master': "Master",
    'Doctor': "Doctorate",
}

DEGREE_NAMES_TO_REJECT = [
    '(Unknown)',
    'Agregation of Mathematics',
    'Artist Diploma',
    'Candidata Magisterii',
    'Dipl.Tzt.',
    'Diploma',
    'Diploma di perfezionamento',
    'Diploma in Architecture',
    'Diploma of the State Conservatorium of Music',
    'Educational Specialist',
    'Engineers Degree',
    'Graduate',
    'Graduate Performance Diploma',
    'Graduate Unspecified',
    'Licenciatura en Veterinaria',
    'Member of the Royal College of Veterinary Surgeons',
    'Nurse Practitioner',
    'Other',
    'Philosophy Diploma',
    'Post Graduate Diploma',
    'Registered Nurse',
]

DEGREE_SUBSTRINGS_TO_REJECT = [
    'Associate',
    'Candidate',
    'Candidature',
    'Certificate',
]

DEGREES_TO_RENAME = {
    'Doktor-Ingenieur': "Doctor of Engineering",
    'Doctoral Degree in Speech-Language Pathology': 'Doctor of Speech-Language Pathology',
    'Doctorate in Speech-Language Pathology': 'Doctor of Speech-Language Pathology',
    'Doctor of Nurse Anesthesia Practice': 'Doctor of Nursing Practice',
    'Doctor of Nursing': 'Doctor of Nursing Practice',
    'Doctorate of Sacred Scripture': 'Doctor of Sacred Scripture',
    'Doctor of Podiatric Medicine': 'Doctor of Medicine',
    'Doctor of Osteopathy': 'Doctor of Osteopathic Medicine',
    'Doctor of Music': 'Doctor of Musical Arts',
    'Doctor of Sciences': 'Doctor of Science',
    'Doctor of Rehabilitation': 'Doctor of Medical Science',
    'Doctor of Engineering Science': 'Doctor of Engineering',
    'Doctor of the Performing Art': 'Doctor of Performing Arts',
    'Doctor of Civil Law': 'Doctor of Jurisprudence',
    'Doctor of Law': 'Doctor of Jurisprudence',
}

PHD_EQUIVALENTS = [
    'Doctoral Degree',
    'Doctor Rerum Politicarum',
    'Doctor rerum naturalium',
    'Doctor of Natural Science',
    'Doctor of Astrophysics',
    'Doctor of Applied Science',
    'Doctor of Chemistry',
    'Doctor of Computer Science',
    'Doctor of Economics',
    'Doctor of Education',
    'Doctor of Forestry',
    'Doctor of Literature and Philosophy',
    'Doctor of Modern Languages',
    'Doctor of Electrical Engineering',
    'Doctor of Science',
    'Doctor of Family Economics',
    'Doctor of Industrial Technology',
    'Doctor of Environmental Science and Engineering',
    'Doctor of Library Science',
    'Doctor of Philosophy, Habilitation',
    'Doctor of Habilitation',
    'Doctor of Medical Humanities',
    'Doctor of Health and Safety',
    'Doctor of Professional Studies',
    'Doctor of Recreation',
    'Doctor of Social Science',
    'Doctor of Sport Management',
    # we're going to consider 'Doctor of Science X' PhD equivalent
    'Doctor of Science in Dentistry',
    'Doctor of Science in Physical Therapy',
    'Doctor of Science in Dentistry',
    'Doctor of Science in Veterinary Medicine',
    'Doctor of Technical Sciences',
    'Doctor of Management',
    'Doctor of Commerce',
    'Doctor of Business Administration', # literally considered equivalent
    'Doctor of Criminology',
    'Doctor of Science and Technology',
    'Doctor of Environmental Design',
    'Doctor of Juridical Science'
]

# DEGREES_TO_REJECT = [
#     'Doctor of Chiropractic',
#     'Doctor of Manual Therapy',
#     'Doctor of Occupational Therapy',
#     'Doctor of Humane Letters',
#     'Doctor of Strategic Leadership',
#     'Doctor of Technology',
# ]

DEGREE_COLUMNS = [
    'DegreeYear',
    'DegreeName',
    'DegreeInstitutionId',
    'DegreeInstitutionName',
]


def filter_degrees_and_print_update(
    df,
    condition_col,
    filter_type_string,
    unfiltered_count=None,
    stat_to_print='removed',
):
    n_start = len(df)

    df = df[
        df[condition_col]
    ]

    n_end = len(df)

    if stat_to_print == 'removed':
        n_to_print = n_start - n_end
    elif stat_to_print == 'remaining':
        n_to_print = n_end

    string = f"{filter_type_string}: {n_to_print}"

    if unfiltered_count:
        p_removed = hnelib.utils.fraction_to_percent(n_to_print / unfiltered_count, 2)
        string += f" ({p_removed}%)"
    
    if stat_to_print == 'removed':
        string = "\tremoving " + string

    # string = f"\t{descriptor} {filter_type_string}: {n_to_print}"

    print(string)

    df = df.drop(columns=[condition_col])

    return df


def filter_future_degrees(df):
    """
    remove degrees awarded after the employment date
    """
    df['DegreeYear'] = df['DegreeYear'].fillna(0)
    df['DegreeYear'] = df['DegreeYear'].astype(int)

    df['DegreeNotFromTheFuture'] = df['DegreeYear'] < df['Year']
    df = filter_degrees_and_print_update(
        df,
        'DegreeNotFromTheFuture',
        'rows where `DegreeYear` >= `Year`',
        unfiltered_count=len(df),
    )

    df['DegreeYear'] = df['DegreeYear'].apply(lambda d: d if d else np.nan)

    return df


def rename_degrees(df):
    df['DegreeName'] = df['DegreeName'].apply(lambda d: DEGREES_TO_RENAME.get(d, d))

    df['DegreeName'] = df['DegreeName'].apply(
        lambda d: 'Doctor of Philosophy' if d in PHD_EQUIVALENTS else d
    )

    return df


def not_candidate_or_certificate(degree_name):
    return all([c not in degree_name for c in DEGREE_SUBSTRINGS_TO_REJECT])


def annotate_degree_type(df):
    typed_degrees = []
    for degree_type_substring, degree_type in DEGREE_SUBSTRING_TO_DEGREE_TYPE.items():
        degrees_of_type = df[
            df['DegreeName'].str.contains(degree_type_substring)
        ].copy()

        degrees_of_type['DegreeType'] = degree_type
        typed_degrees.append(degrees_of_type)

    df = df.merge(
        pd.concat(typed_degrees),
        on=list(df.columns),
    )

    return df


def filter_to_doctorates(df, n_start):
    df = df.copy()
    df['IsDoctorate'] = df['DegreeType'] == 'Doctorate'
    df = filter_degrees_and_print_update(
        df,
        'IsDoctorate',
        'doctoral degrees',
        n_start,
        stat_to_print='remaining',
    )

    return df


def filter_to_non_doctorates(df, n_start):
    df = df.copy()

    df['NonDoctorate'] = df['DegreeType'] != 'Doctorate'
    df = filter_degrees_and_print_update(
        df,
        'NonDoctorate',
        'non-doctoral degrees',
        n_start,
        stat_to_print='remaining',
    )

    for degree_type in sorted(df['DegreeType'].unique()):
        df['IsOfType'] = df['DegreeType'] == degree_type
        filter_degrees_and_print_update(
            df,
            'IsOfType',
            f"\t{degree_type}'s degrees",
            n_start,
            stat_to_print='remaining',
        )

    return df

def clean_degree_years(employment):
    employment = employment.drop_duplicates().copy()
    columns = list(employment.columns)

    n_start = len(employment)

    df = employment.copy()[
        [
            'PersonId',
            'Year',
            'DegreeYear',
        ]
    ].drop_duplicates()

    n_people = df['PersonId'].nunique()

    n_null_start = df[
        df['DegreeYear'].isna()
    ]['PersonId'].nunique()

    df = df[
        df['DegreeYear'].notna()
    ]

    df['NullifyDegreeYear'] = df['Year'] - df['DegreeYear'] > MAX_DEGREE_YEAR_TO_EMPLOYMENT_YEAR

    df = df[
        df['NullifyDegreeYear']
    ].drop(columns=['Year', 'DegreeYear']).drop_duplicates()

    n_nullified = df['PersonId'].nunique()

    n_null_end = n_null_start + n_nullified

    n_reasonable = n_people - n_null_end

    nullified_employment = employment.copy()[
        employment['PersonId'].isin(df['PersonId'].unique())
    ]

    nullified_employment['DegreeYear'] = np.nan

    employment = employment[
        ~employment['PersonId'].isin(df['PersonId'].unique())
    ]

    employment = pd.concat([employment, nullified_employment])

    employment = employment[columns].drop_duplicates()

    n_end = len(employment)
    assert(n_start == n_end)

    p_null_start = hnelib.utils.fraction_to_percent(n_null_start / n_people, 1)
    p_nullified = hnelib.utils.fraction_to_percent(n_nullified / n_people, 1)
    p_null_end = hnelib.utils.fraction_to_percent(n_null_end / n_people, 1)
    p_reasonable = hnelib.utils.fraction_to_percent(n_reasonable / n_people, 1)

    print('cleaning degree years:')
    print(f'\tthere are {n_people} people')
    print(f'\t{n_reasonable} ({p_reasonable}%) got a degree in a reasonable year')
    print(f"\t{n_null_end} ({p_null_end}%) got a degree in an unknown year (`DegreeYear` == null)")
    print(f"\t\t{n_null_start} ({p_null_start}%) of these got a degree in an unknown year (`DegreeYear` == null)")
    print(f"\t\t{n_nullified} ({p_nullified}%) of these got a degree in an unreasonable year, and we nullified their DegreeYear.")

    return employment


def clean_degrees(df):
    """
    We're going to remove degrees:
    - without a DegreeName
    - without a DegreeInstitutionId
    - without a DegreeInstitutionName
    - with a DegreeName that includes one of the following:
        - Associate
        - Bachelor
        - Master

    The rest of the degrees either contain `Doctor` or are trash
    """
    n_start = len(df)

    df['DegreeNameIsValid'] = df['DegreeName'].notnull()
    df = filter_degrees_and_print_update(df, 'DegreeNameIsValid', 'unnamed degrees', n_start)

    df['IdIsValid'] = df['DegreeInstitutionId'].notnull()
    df = filter_degrees_and_print_update(df, 'IdIsValid', 'empty institution id degrees', n_start)

    df['InstitutionNameIsValid'] = df['DegreeInstitutionName'].notnull()
    df = filter_degrees_and_print_update(df, 'InstitutionNameIsValid', 'empty institution name degrees', n_start)

    df = rename_degrees(df)

    df['AcceptableDegreeName'] = ~df['DegreeName'].isin(DEGREE_NAMES_TO_REJECT)
    df = filter_degrees_and_print_update(df, 'AcceptableDegreeName', 'bad degree name degrees', n_start)

    df['BA+'] = df['DegreeName'].apply(not_candidate_or_certificate)
    df = filter_degrees_and_print_update(df, 'BA+', 'candidate/certificate/associate degrees', n_start)

    df = annotate_degree_type(df)

    df['KnownDegreeType'] = df['DegreeType'].notnull()
    df = filter_degrees_and_print_update(df, 'KnownDegreeType', 'unknown degree type degrees')

    return df
    

def print_cleaning_stats(degrees_df, n_start):
    n_end = len(degrees_df)

    n_removed = n_start - n_end
    p_remaining = hnelib.utils.fraction_to_percent(n_end / n_start, 2)
    
    print(f'after cleaning there are {n_end} degrees ({p_remaining}% of original set)')

def print_final_stats(doctorates, n_start):
    n_end = len(doctorates)

    n_removed = n_start - n_end
    p_remaining = hnelib.utils.fraction_to_percent(n_end / n_start, 2)
    
    print(f'ending with {n_end} doctoral degrees ({p_remaining}% of original set)')


if __name__ == '__main__':
    """
    We're going to separate degrees and filter them to only PhD degrees here, so
    that we can do some nice things with them later
    """
    usfhn.utils.print_cleaning_step_start("Degree Cleaning")

    df = pd.read_csv(usfhn.constants.AA_2022_DEGREE_INSTITUTION_REMAP_EMPLOYMENT_PATH)

    print(f'starting with {len(df)} employment rows')

    df = filter_future_degrees(df)

    df = clean_degree_years(df)

    employment_df = df.copy().drop(columns=DEGREE_COLUMNS)

    employment_df.to_csv(
        usfhn.constants.AA_2022_DEGREE_FILTERED_EMPLOYMENT_PATH,
        index=False,
    )

    raw_degrees = df[
        ['PersonId'] + DEGREE_COLUMNS
    ].drop_duplicates()

    n_start = len(raw_degrees)

    print(f'starting with {n_start} degrees')

    degrees_df = clean_degrees(raw_degrees)

    print_cleaning_stats(degrees_df, n_start)

    non_doctorates = filter_to_non_doctorates(degrees_df, n_start)

    non_doctorates.to_csv(
        usfhn.constants.AA_2022_DEGREE_FILTERED_NON_DOCTORAL_DEGREES_PATH,
        index=False,
    )

    doctorates = filter_to_doctorates(degrees_df, n_start)

    print_final_stats(doctorates, n_start)

    doctorates.to_csv(
        usfhn.constants.AA_2022_DEGREE_FILTERED_DEGREES_PATH,
        index=False,
    )

    ################################################################################
    # stats saving
    ################################################################################
    n_people = employment_df['PersonId'].nunique()
    n_valid_degrees = len(degrees_df)
    n_doctorates = len(doctorates)
    
    p_doctorates = n_doctorates / n_valid_degrees

    n_non_doctorates = n_valid_degrees - n_doctorates
    p_non_doctorates = n_non_doctorates / n_valid_degrees

    n_masters = len(degrees_df[degrees_df['DegreeType'] == 'Master'])
    p_masters = n_masters / n_valid_degrees

    n_bachelors = len(degrees_df[degrees_df['DegreeType'] == 'Bachelor'])
    p_bachelors = n_bachelors / n_valid_degrees

    stats = {
        'nPeople': n_people,
        'nDegrees': n_valid_degrees,
        'nDoctorates': n_doctorates,
        'nNonDoctorates': n_non_doctorates,
        'nMasters': n_masters,
        'nBachelors': n_bachelors,
        'pPeopleWithDegrees': n_valid_degrees / n_people,
        'pDoctorates': p_doctorates,
        'pNonDoctorates': p_non_doctorates,
        'pMasters': p_masters,
        'pBachelors': p_bachelors,
    }

    usfhn.constants.AA_2022_DEGREE_FILTERED_STATS_PATH.write_text(json.dumps(stats, indent=4, sort_keys=True))

