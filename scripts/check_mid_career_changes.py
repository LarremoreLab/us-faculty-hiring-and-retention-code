import pandas as pd
import string

import usfhn.constants as constants

punctuation_remover = str.maketrans(string.punctuation, "".join([' ' for _ in string.punctuation]))

def get_clean_string_words(string):
    string = string.translate(punctuation_remover)
    string = string.lower()
    words = string.split()
    return words
    
def clean_name(person_name):
    words = get_clean_string_words(person_name)
    person_name = " ".join(words)
    return person_name

def clean_first_and_last_name(person_name):
    person_name = person_name.translate(punctuation_remover)
    # nee
    person_name = person_name.replace('-', ' ')
    person_name = person_name.lower()
    words = person_name.split()
    
    # take the first 2 words...
    person_name = " ".join(words[:2])
    return person_name

if __name__ == '__main__':
    degrees = pd.read_csv(constants.AA_CLEAN_PEOPLE_MASTER_DEGREES_PATH)[
        ['PersonId', 'DegreeInstitutionId']
    ].drop_duplicates()

    employment = pd.read_csv(constants.AA_CLEAN_PEOPLE_EMPLOYMENT_PATH)[
        ['PersonId', 'PersonName', 'InstitutionId']
    ].drop_duplicates()

    employment['CleanName'] = employment['PersonName'].apply(clean_name)
    employment['CleanFirstAndLastName'] = employment['PersonName'].apply(clean_first_and_last_name)

    employment = employment.merge(degrees, on='PersonId')

    employment['PersonIdCount'] = employment.groupby([
        'CleanName', 'DegreeInstitutionId'
    ])['PersonId'].transform('nunique')

    employment['InstitutionIdCount'] = employment.groupby([
        'CleanName', 'DegreeInstitutionId'
    ])['InstitutionId'].transform('nunique')

    employment = employment[
        (employment['PersonIdCount'] > 1)
        &
        (employment['InstitutionIdCount'] > 1)
    ]
    print('lowercased names:')
    print(len(employment))
    print(employment.nunique())

    employment['PersonIdCount'] = employment.groupby([
        'CleanFirstAndLastName', 'DegreeInstitutionId'
    ])['PersonId'].transform('nunique')

    employment['InstitutionIdCount'] = employment.groupby([
        'CleanFirstAndLastName', 'DegreeInstitutionId'
    ])['InstitutionId'].transform('nunique')

    employment = employment[
        (employment['PersonIdCount'] > 1)
        &
        (employment['InstitutionIdCount'] > 1)
    ]
    print('first/last lowercased names:')
    print(len(employment))
    print(employment.nunique())
