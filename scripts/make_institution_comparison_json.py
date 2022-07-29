import json
from collections import defaultdict

import usfhn.constants
import usfhn.datasets
import usfhn.measurements
import usfhn.views
import usfhn.institutions
import usfhn.plot_utils
import usfhn.stats


def preclean_field(field):
    return field.replace(', General', '')

def clean_institution_name(string):
    string = string.replace(', The', '')
    # string = string.replace('University', 'U')
    return string

def get_field_data():
    """
    We want:
    - Field
    - CleanField
    - ShortField
    - Domain
    - Color
    - Ordering

    in json: 
    {
        'FIELD1': {
            shortField:
            cleanField:
            domain:
            color:
            ordering:
        },
    }
    """
    df = usfhn.views.get_taxonomization()[
        [
            'Field',
            'Umbrella',
        ]
    ].rename(columns={
        'Umbrella': 'Domain',
    })

    df = df[
        df['Field'].notna()
    ]

    df['CleanField'] = df['Field'].apply(usfhn.plot_utils.clean_taxonomy_string)
    df['ShortField'] = df['Field'].apply(lambda f: usfhn.plot_utils.FIELD_ABBREVIATIONS.get(f, f))

    df['Field'] = df['Field'].apply(preclean_field)
    df['Color'] = df['Domain'].apply(usfhn.plot_utils.PLOT_VARS['colors']['umbrellas'].get)

    df = df.sort_values(by=['Domain', 'Field'])

    df['Ordering'] = [i for i in range(len(df))]

    data = {}
    for i, row in df.iterrows():
        data[row['Field']] = {
            'shortField': row['ShortField'],
            'cleanField': row['CleanField'],
            'domain': row['Domain'],
            'color': row['Color'],
            'ordering': row['Ordering'],
        }

    return data


def jsonify_rank_data(df):
    """
    expects df with columns:
    - InstitutionId
    - TaxonomyValue
    - Percentile

    returns json:
    {
        'INSTITUTION1_NAME': {
            'FIELD1': RANK1,
            ...
        },
        ...
    }
    """
    df = df.rename(columns={
        'TaxonomyValue': 'Field',
        'Percentile': 'Rank',
    })

    df['Field'] = df['Field'].apply(preclean_field)
    df = usfhn.institutions.annotate_institution_name(df)
    df = df.drop(columns='InstitutionId')

    rank_data = {}
    for institution_name, rows in df.groupby('InstitutionName'):
        institution_name = clean_institution_name(institution_name)
        rank_data[institution_name] = {f: r for f, r in zip(rows['Field'], rows['Rank'])}

    return rank_data


def get_prestige_rank_data():
    df = usfhn.stats.runner.get('ranks/df', rank_type='prestige')

    df = usfhn.views.filter_by_taxonomy(df, level='Field')

    df = df[
        [
            'InstitutionId',
            'TaxonomyValue',
            'Percentile',
        ]
    ].drop_duplicates()

    df['Percentile'] *= 100

    return jsonify_rank_data(df)


def get_production_rank_data():
    df = usfhn.stats.runner.get('ranks/df', rank_type='production')
    df = usfhn.views.filter_by_taxonomy(df, level='Field')

    df = df[
        [
            'InstitutionId',
            'TaxonomyValue',
            'Percentile',
        ]
    ].drop_duplicates()

    df['Percentile'] *= 100

    return jsonify_rank_data(df)


def get_institutions(production_ranks, prestige_ranks):
    production_institutions = set(production_ranks.keys())
    prestige_institutions = set(prestige_institutions.keys())

def get_data_to_dump():
    production_ranks = get_production_rank_data()
    prestige_ranks = get_prestige_rank_data()
    return {
        'production': {
            'institutions': sorted(list(production_ranks.keys())),
            'ranks': production_ranks,
        },
        'prestige': {
            'institutions': sorted(list(prestige_ranks.keys())),
            'ranks': prestige_ranks,
        },
        'fields': get_field_data(),
    }


if __name__ == '__main__':
    """
    Other data we want:
    - prestige + production data
    - short field names
    - field to umbrella
    - umbrella color
    - field color
    
    json organization: 
    {
        'production': {
            'institutions': {},
            'ranks': {}
        },
        'prestige': {
            'institutions': {},
            'ranks': {}
        },
        'fields': {
            FIELD1: {
                'domain':
                'color': 
                'shortField':
                'cleanField':
                'ordering':
            },
        },
    }
    """
    json_data = get_data_to_dump()
    usfhn.constants.SCHOOL_COMPARISON_VISUALIZATION_DATA_PATH.write_text(json.dumps(json_data))
