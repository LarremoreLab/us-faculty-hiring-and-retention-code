from functools import lru_cache
import json
import time
import requests
import pandas as pd
from functools import partial

from cachetools import cached, Cache
from cachetools.keys import hashkey

import hnelib.model

import usfhn.constants
import usfhn.fieldwork
import usfhn.views


################################################################################
# gender stats functions
################################################################################
def annotate_gender(df, explode_gender=False):
    if 'Gender' not in df.columns:
        gender_df = usfhn.datasets.get_dataset_df('closedness_data')[
            [
                'PersonId',
                'Gender',
            ]
        ].drop_duplicates()

        df = df.merge(
            gender_df,
            on='PersonId',
        )

    if explode_gender:
        df = usfhn.views.explode_gender(df)
    else:
        df = df[
            df['Gender'].isin(['Male', 'Female'])
        ]

    return df


def get_taxonomy_gender_fraction(
    by_year=False,
    by_new_hire=False,
    by_non_us=False,
    by_faculty_rank=False,
    by_seniority=False,
    by_career_age=False,
):
    """
    return columns:
    - TaxonomyLevel
    - TaxonomyValue
    - Year (if `by_year`)
    - Rank (if `by_faculty_rank`)

    and the following stats columns (which vary by the above columns):
    - Faculty
    - FemaleFaculty
    - FractionFemale
    - MaleFaculty
    - FractionMale
    """
    from usfhn.stats import add_groupby_annotations_to_df

    groupby_cols = [
        'TaxonomyLevel',
        'TaxonomyValue',
    ]

    if by_year:
        groupby_cols.append('Year')

    df = usfhn.datasets.get_dataset_df('closedness_data', by_year=by_year)

    df = usfhn.fieldwork.linearize_df_taxonomy_hierarchy(df)

    df = df[
        groupby_cols + [
            'PersonId',
            'Gender',
        ]
    ].drop_duplicates()

    df, extra_groupby_cols = add_groupby_annotations_to_df(
        df,
        by_new_hire=by_new_hire,
        by_faculty_rank=by_faculty_rank,
        by_seniority=by_seniority,
        by_non_us=by_non_us,
        by_career_age=by_career_age,
    )

    groupby_cols += extra_groupby_cols

    df = get_gender_ratios(df, groupby_cols)

    df = df[
        df['Gender'] == 'Female'
    ]

    df = df.rename(columns={
        'FacultyCount': 'Faculty',
        'GenderFacultyCount': 'FemaleFaculty',
        'GenderFraction': 'FractionFemale',
    })

    df['MaleFaculty'] = df['Faculty'] - df['FemaleFaculty']
    df['FractionMale'] = 1 - df['FractionFemale']
    
    return df


def get_gender_ratios(df, groupby_cols):
    """
    returns: 
    - groupby_cols
    - Gender
    - FacultyCount
    - GenderFacultyCount
    - GenderFraction
    """
    df = df.copy()
    df = df[
        df['Gender'].isin(['Male', 'Female'])
    ]

    df['FacultyCount'] = df.groupby(groupby_cols)['PersonId'].transform('nunique')
    df['GenderFacultyCount'] = df.groupby(
        ['Gender'] + groupby_cols
    )['PersonId'].transform('nunique')

    df['GenderFraction'] = df['GenderFacultyCount'] / df['FacultyCount']

    df = df[
        groupby_cols + [
            'Gender',
            'FacultyCount',
            'GenderFacultyCount',
            'GenderFraction',
        ]
    ].drop_duplicates()

    return df


@lru_cache()
def get_taxonomy_gender_ratios():
    # NOTE: 20220405 we changed this from `data` to `closedness_data`, because
    # gender should be based on everyone
    df = usfhn.datasets.CURRENT_DATASET.closedness_data
    df = df[
        ['Taxonomy', 'Year', 'PersonId', 'Gender']
    ].drop_duplicates()

    df = usfhn.fieldwork.linearize_df_taxonomy_hierarchy(df)

    groupby_columns = ['TaxonomyLevel', 'TaxonomyValue', 'Year']

    men = df[
        df['Gender'] == 'Male'
    ].copy()
    men['MaleFaculty'] = men.groupby(groupby_columns)['PersonId'].transform('nunique')

    women = df[
        df['Gender'] == 'Female'
    ].copy()
    women['FemaleFaculty'] = women.groupby(groupby_columns)['PersonId'].transform('nunique')

    df = df[
        groupby_columns
    ].merge(
        men[
            groupby_columns + ['MaleFaculty']
        ].drop_duplicates(),
        on=groupby_columns,
    ).merge(
        women[
            groupby_columns + ['FemaleFaculty']
        ].drop_duplicates(),
        on=groupby_columns,
    )

    df['Faculty'] = df['MaleFaculty'] + df['FemaleFaculty']
    df['FractionFemale'] = df['FemaleFaculty'] / df['Faculty']

    df = df[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'Year',
            'FractionFemale',
            'Faculty',
            'MaleFaculty',
            'FemaleFaculty',
        ]
    ].drop_duplicates()

    return df


def rank_vs_gender_logit(rank_type='prestige', by_new_hire=False, existing_hires=False):
    import usfhn.datasets
    df = usfhn.datasets.get_dataset_df('closedness_data')[
        [
            'PersonId',
            'Gender',
            'Taxonomy',
            'InstitutionId',
        ]
    ].drop_duplicates()

    df = usfhn.fieldwork.linearize_df_taxonomy_hierarchy(df)

    df = df[
        df['Gender'].isin(['Male', 'Female'])
    ]

    df['Gender'] = df['Gender'].apply({'Female': 1, 'Male': 0}.get)

    ranks = usfhn.stats.runner.get('ranks/df', rank_type=rank_type)

    df = df.merge(
        ranks,
        on=[
            'TaxonomyLevel',
            'TaxonomyValue',
            'InstitutionId',
        ]
    )

    if by_new_hire:
        import usfhn.new_hires
        df = usfhn.new_hires.annotate_new_hires(df)

        if existing_hires:
            df = df[
                ~df['NewHire']
            ]
        else:
            df = df[
                df['NewHire']
            ]

    df = hnelib.model.get_logits(
        df,
        endog='Gender',
        exog='Percentile',
        groupby_cols=[
            'TaxonomyLevel',
            'TaxonomyValue',
        ],
    )

    df = usfhn.stats_utils.correct_multiple_hypotheses(df, p_col='Percentile-P', corrected_p_col='Percentile-P')

    return df

################################################################################
# functions used to gender names
################################################################################
# 1. load the people
# 2. load the set of people we've tried offline
# 3. for the people who we haven't tried offline, run the offline
#   - save ever 25,000

def split_name(name):
    first = None
    last = name

    if ',' in name:
        last, first = name.split(', ', maxsplit=1)

        if ' ' in first:
            first, _ = first.split(' ', maxsplit=1)

    return pd.Series((first, last))


def result_guesses_are_unanimous(result):
    """if all items in the guesses have either 'M' or 'F', they fully agree"""
    guesses = list(result['guesses'].values())

    if guesses:
        if all(map(lambda g: g == 'M', guesses)):
            return True
        elif all(map(lambda g: g == 'F', guesses)):
            return True
    else:
        return False


def get_result_gender(result):
    """returns the gender of the first guess. should really only be used if
    `result_is_unanimous` is true"""
    return list(result['guesses'].values())[0]


def get_offline_gender(row):
    result = guess_gender(row.FirstName, row.LastName, search_web=False)

    if result_guesses_are_unanimous(result):
        return get_result_gender(result)
    else:
        return None

def guess_gender(first_name, last_name, search_web=True):
    """
    applies all methods to first_name, last_name and combines the results into a simple dictionary

    Parameters:
        first_name
        last_name
        search_web      Boolean. If false, don't send requests to the internetz

    Returns:
        Dictionary.
        {
            'name': "firstname lastname",
            'first_name': "first_name"
            'last_name': "last_name"
            'guesses': {
                'FacCensus'': M|F|-|E,
                'Ethnea': M|F|-|E,
                'genderize.io': M|F|-|E,
                'GoogleSearch': M|F|-|E,
                'namsor': M|F|-|E,
            }
        }
    """

    if not first_name:
        first_name = ''
    if not last_name:
        last_name = ''

    result = {
        'name': first_name + ' ' + last_name,
        'first_name': first_name,
        'last_name': last_name,
        'guesses': {},
    }

    if not first_name and not last_name:
        return result
    elif last_name and not first_name:
        first_name = last_name
    elif first_name and not last_name:
        last_name = first_name

    first_name = first_name.capitalize()
    last_name = last_name.capitalize()

    result['guesses']['FacCensus'] = guess_gender_by_method(
        first_name, last_name, 'FacCensus'
    )
    result['guesses']['gender_guesser'] = guess_gender_by_method(
        first_name, last_name, 'gender_guesser'
    )

    return result


def guess_gender_by_method(first_name, last_name, method='FacCensus'):
    """guesses a gender using a given method. wraps these in a try/except so we always process.
    
    Parameters:
        first_name
        last_name
        method:     FacCensus|Ethnea|genderize.io|GoogleSearch|namsor

    Returns:
        String.
        'M': male
        'F': female
        '-': unknown
        'E': error
    """

    methods = {
        'FacCensus': guessGender_FacultyCensus,
        'gender_guesser': guessGender_genderGuesser,
    }

    try:
        return methods[method](first_name, last_name)
    except:
        return 'E'


@cached(Cache(1), key=partial(hashkey, 'gender_guesser_guesser_guesser'))
def get_gender_guesser_detector():
    detector = gender_guesser_detection.Detector(case_sensitive=False)
    return detector


def guessGender_genderGuesser(first_name, last_name):
    detector = get_gender_guesser_detector()
    guess = detector.get_gender(first_name)

    if guess == 'male':
        return 'M'
    elif guess == 'female':
        return 'F'

    return '-'



@cached(Cache(1), key=partial(hashkey, 'FacCensus'))
def get_fac_census_data():
    return json.loads(constants.HARDCODED_GENDERS_PATH.read_text())



def guessGender_FacultyCensus(firstName, lastName):
    # dictionary-based first-name gender classifier, v1 February 2019
    # written by Aaron Clauset, aaronclauset@gmail.com
    # dict from 20,000 hand-labeled faculty names in Computer Science, History, and Business (all North America)
    #
    #
    # NB 12 February 2019: This classifier is very accurate, and should be the preferred method (see NB below)
    #
    # This function queries the first name against a dictionary of first names, constructed from
    # the 2011-2013 faculty census and its hand-labels of about 20,000 faculty in Computer Science,
    # History, and Business.
    #
    # The dictionary contains M:F frequencies in the census, and the decision rule used here is
    # a simple majority rule for which label is returned. A conservative approach would be to use
    # a two-sided Binomial test against p=0.5 with a Beta prior to decide if H0: M==F can be rejected.
    #
    # NB: the dictionary assumes the first letter of firstName is capitalized, so we force to match
    #
    # Prior to pushing the firstName string against the dictionary, we split it into pieces by whitespace
    # and use a regex to skip over leading parts that are initials. The first non-initial part is how
    # the gender guess is assigned.
    #
    # NOTE: the accuracy rate here is STRONG; on 500 randomly sampled names from the census,
    #       it had a 97.2%% accuracy, and was wrong only 1.4% of the time when it did make a guess.
    #       The remaining 1.4% are labeled as "unsure" and could be MTurked
    #
    #        M   F   - (<-- guess)
    # M: [[413   5   5]
    # F:  [  2  73   2]]
    # accuracy = (413+73)/500 = 0.972

    flag_verb = False
    re_initil = '(?:\A([A-Z]\.?[\-\s]?)+\Z)' # initial, one or more (including hyphens)

    facNames = get_fac_census_data()

    genderFacCensus = '-'                 # unsure (default)
    parts           = firstName.split()   # split firstname into parts, by whitespace
    flag_firstName  = True
    for key in parts:
        if flag_firstName:
            m = re.findall(re_initil,key)   # check for initial only string
            if len(m)==0:                   # proceed only if failed to match initil regex
                keyc = key.capitalize()     # force first letter to be capitalized
                if keyc in facNames:
                    if flag_verb:
                        M = facNames[keyc]['M']
                        F = facNames[keyc]['F']
                        print(f'{firstName} : M:{M} F:{F}')

                    # simple majority rule for decisions
                    if facNames[keyc]['M']>facNames[keyc]['F']:
                        genderFacCensus = 'M'
                    elif facNames[keyc]['M']<facNames[keyc]['F']:
                        genderFacCensus = 'F'
                    if flag_verb: print(f'{firstName} {lastName} : {genderFacCensus}')
                    flag_firstName = False # we only record the first "firstName" we hit
                else:
                    if flag_verb: print(f'{firstName} {lastName} : {genderFacCensus} (name not found)')

    return genderFacCensus



def query_ethnea(row):
    first = row.FirstName.lower()
    last = row.LastName.lower()

    time.sleep(1.)
    url = f'http://abel.ischool.illinois.edu/cgi-bin/ethnea/search.py?Fname={first}&Lname={last}&format=json'

    try:
        resp = requests.get(url)
        # Decode JSON response into a Python dict:
        result = json.loads(resp.text.replace("'", '"'))
        return result['Genni']
    except:
        return None


def record_final_results():
    genders = ['M', 'F']

    gendered_with_genni = pd.read_csv(usfhn.constants.GENNI_GENDER_PATH)[
        ['PersonId', 'Gender']
    ].query(f"Gender in {genders}")

    gendered_offline = pd.read_csv(usfhn.constants.OFFLINE_GENDER_PATH)[
        ['PersonId', 'Gender']
    ].query(f"Gender in {genders}")

    genni_ids = set(gendered_with_genni['PersonId'].unique())
    offline_ids = set(gendered_offline['PersonId'].unique())
    assert(genni_ids & offline_ids == set())

    gendered = pd.concat([gendered_with_genni, gendered_offline])
    gendered.to_csv(usfhn.constants.KNOWN_GENDER_PATH, index=False)

    gender_unknown_people = pd.read_csv(usfhn.constants.AA_CLEAN_PEOPLE_EMPLOYMENT_PATH)['PersonId']
    gender_unknown_people = gender_unknown_people[
        ~gender_unknown_people.isin(genni_ids | offline_ids)
    ].unique()

    gender_unknown_people = pd.Series(gender_unknown_people, name='PersonId')
    gender_unknown_people.to_csv(
        usfhn.constants.UNKNOWN_GENDER_PATH,
        index=False,
        header='PersonId'
    )
