import pandas as pd

import hnelib.utils

import usfhn.datasets
import usfhn.constants
import usfhn.views
import usfhn.institutions
import usfhn.plot_utils
import usfhn.latex_table_helper
import usfhn.gender
import usfhn.null_models
import usfhn.self_hiring
import usfhn.stats
import usfhn.utils


COLUMN_ALIGNMENTS = {
    'production_ranks_academia_1': ['c', 'r', 'l'],
    'production_ranks_academia_2': ['c', 'r', 'l'],
    'prestige_ranks_academia_1': ['c', 'r', 'l'],
    'prestige_ranks_academia_2': ['c', 'r', 'l'],
    # 'rank_change_1': ['l', 'l', 'r', 'r', 'r', 'r', 'r'],
    # 'rank_change_2': ['l', 'l', 'r', 'r', 'r', 'r', 'r'],
    'exclusions_table.tex': ['l', 'r', 'r'],
    'taxonomy_1.tex': ['l', 'l', 'r', 'r', 'r'],
    'taxonomy_2.tex': ['l', 'l', 'r', 'r', 'r'],
}


def get_tables():
    return {
        'taxonomy_1.tex': get_taxonomy_table(part=1),
        'taxonomy_2.tex': get_taxonomy_table(part=2),
        'less_hierarchical_than_null_model.tex': get_hierarchies_less_hierarchical_than_null_table(),
        'rank_change.tex': get_rank_change_table(),
        'production_ranks_academia_1.tex': get_production_ranks_table(part=1),
        'production_ranks_academia_2.tex': get_production_ranks_table(part=2),
        'prestige_ranks_academia.tex': get_prestige_ranks_table(),
        'exclusions_table.tex': get_exclusions_table(),
    }

################################################################################
#
#
#
#
# changes tables
#
#
#
#
################################################################################
def format_changes_dfs_for_table(all_df, asst_df):
    df = all_df.merge(
        asst_df,
        on=['TaxonomyLevel', 'TaxonomyValue']
    )

    umbrella_df = df[
        df['TaxonomyLevel'] == 'Umbrella'
    ].copy().drop(columns=['TaxonomyLevel']).rename(columns={
        'TaxonomyValue': 'Domain'
    })

    umbrella_df['Field'] = ''

    academia_df = df[
        df['TaxonomyLevel'] == 'Academia'
    ].copy().drop(columns=['TaxonomyLevel']).rename(columns={
        'TaxonomyValue': 'Domain'
    })
    academia_df['Field'] = ''

    field_df = df[
        df['TaxonomyLevel'] == 'Field'
    ].copy()
    field_df = usfhn.views.annotate_umbrella(field_df, 'Field')
    field_df = field_df.drop(columns=[
        'TaxonomyValue',
        'TaxonomyLevel',
    ]).rename(columns={
        'Umbrella': 'Domain',
    })

    return academia_df, umbrella_df, field_df

################################################################################
#
#
#
# taxonomy table
#
#
#
################################################################################
def get_taxonomy_table(part=1, include_taxonomy_sizes=True):
    df = usfhn.views.get_taxonomization()

    df = df.drop(columns=['Taxonomy', 'Area', 'Academia'])
    df = df[
        df['Field'].notnull()
    ]

    df = df.sort_values(by=['Umbrella', 'Field'])
    df = df.rename(columns={
        'Umbrella': 'Domain',
    })

    df = df[
        [
            'Domain',
            'Field',
        ]
    ]

    if include_taxonomy_sizes:
        sizes_df = usfhn.stats.runner.get('taxonomy/df')
        umbrellas_df = usfhn.views.filter_by_taxonomy(sizes_df, 'Umbrella')[
            [
                'TaxonomyValue',
                'Fraction',
                'Count',
            ]
        ].rename(columns={
            'TaxonomyValue': 'Domain',
            'Count': 'DomainCount',
            'Fraction': 'DomainFraction',
        })

        fields_df = usfhn.views.filter_by_taxonomy(sizes_df, 'Field')[
            [
                'TaxonomyValue',
                'Count',
            ]
        ].drop_duplicates().rename(columns={
            'Count': 'FieldCount',
            'TaxonomyValue': 'Field'
        })

        df = df.merge(
            umbrellas_df,
            on='Domain',
        ).merge(
            fields_df,
            on='Field'
        )

        df['FieldPercent'] = 100 * df['FieldCount'] / df['DomainCount']
        df['FieldPercent'] = df['FieldPercent'].apply(lambda p: round(p, 1))
        df['FieldPercent'] = df['FieldPercent'].apply(str)

        domains_df = df.copy()[
            [
                'Domain',
                'DomainFraction',
                'DomainCount',
            ]
        ].drop_duplicates().rename(columns={
            'DomainCount': 'Faculty',
            'DomainFraction': 'DomainPercent',
        })

        domains_df['DomainPercent'] *= 100
        domains_df['DomainPercent'] = domains_df['DomainPercent'].apply(lambda p: round(p, 1))
        domains_df['DomainPercent'] = domains_df['DomainPercent'].apply(str)

        df = df.drop(columns=[
            'DomainCount',
            'DomainFraction',
        ]).rename(columns={
            'FieldCount': 'Faculty'
        })

        df = pd.concat([df, domains_df])

        df['Faculty'] = df['Faculty'].apply(add_commas_to_number)

        df['DomainPercent'] = df['DomainPercent'].fillna('')
        df['FieldPercent'] = df['FieldPercent'].fillna('')
        df['Field'] = df['Field'].fillna('')

        df = df.rename(columns={
            'DomainPercent': '% of academia',
            'FieldPercent': '% of domain',
        })

        df = df.sort_values(by=['Domain', 'Field'])

    part_2_domains = [
        'Medicine and Health',
        'Natural Sciences',
        'Social Sciences',
    ]

    if part == 1:
        df = df[
            ~df['Domain'].isin(part_2_domains)
        ]
    else:
        df = df[
            df['Domain'].isin(part_2_domains)
        ]
    
    df['Domain'] = df['Domain'].apply(usfhn.plot_utils.clean_taxonomy_string)
    df['Field'] = df['Field'].apply(usfhn.plot_utils.clean_taxonomy_string)

    df = df.rename(columns={
        'Domain': 'domain',
        'Field': 'field',
        'faculty': 'faculty',
    })

    return df


def get_hierarchies_less_hierarchical_than_null_table():
    df = usfhn.null_models.get_stats()

    df = df[
        df['MoreHierarchicalCount'] != 0
    ]

    df = df[
        [
            'TaxonomyValue',
            'TaxonomyLevel',
            'MoreHierarchicalCount',
        ]
    ].sort_values(by=['MoreHierarchicalCount'], ascending=False).rename(columns={
        'TaxonomyLevel': 'Taxonomy Level',
        'TaxonomyValue': 'Taxonomy Value',
        'MoreHierarchicalCount': '# of null models less hierarchical than empirical (out of 1000)',
    })

    return df

def get_rank_change_table(part=None):
    df = usfhn.self_hiring.get_rank_change_table()
    df = df[
        [
            'Domain',
            'Field',
            'up',
            'down',
            'self-hires',
            'ranks up',
            'ranks down'
        ]
    ]

    part_2_domains = [
        'Natural Sciences',
        'Social Sciences',
    ]

    if part:
        if part == 1:
            df = df[
                ~df['Domain'].isin(part_2_domains)
            ]
        else:
            df = df[
                df['Domain'].isin(part_2_domains)
            ]

    df = df.sort_values(by=[
        'Domain',
        'Field',
    ])

    return df


def get_production_ranks_table(part=1):
    df = usfhn.stats.runner.get('ranks/df', rank_type='production')
    df = usfhn.views.filter_by_taxonomy(df, 'Academia').drop(columns=[
        'TaxonomyLevel',
        'TaxonomyValue',
    ]).drop_duplicates()

    df = usfhn.institutions.annotate_institution_name(df)
    df['Rank'] = [i + 1 for i in range(len(df))]
    df = df.sort_values(by=['Rank'])

    df['ProductionPercent'] = df['ProductionFraction'] * 100
    df['ProductionPercent'] = df['ProductionPercent'].apply(lambda p: round(p, 2))

    # df['ProductionCumSum'] = df['ProductionPercent'].cumsum()
    # df = df[
    #     df['ProductionCumSum'] <= 80
    # ]

    df['ProductionPercent'] = df['ProductionPercent'].astype(str)

    df = df[
        [
            'Rank',
            'InstitutionName',
            'ProductionPercent',
        ]
    ].drop_duplicates().rename(columns={
        'Rank': '#',
        'InstitutionName': 'University',
        'ProductionPercent': '% of faculty produced',
    })

    df['University'] = df['University'].apply(usfhn.latex_table_helper.escape_string)
    df['University'] = df['University'].apply(clean_university_name)

    df = df[
        df['#'] < 101
    ]

    if part == 1:
        df = df[
            df['#'] <= 50
        ]
    else:
        df = df[
            df['#'] > 50
        ]

    return df

def get_prestige_ranks_table():
    df = usfhn.stats.runner.get('ranks/df', rank_type='prestige')
    df = usfhn.views.filter_by_taxonomy(df, 'Academia').drop(columns=[
        'TaxonomyLevel',
        'TaxonomyValue',
    ]).drop_duplicates()[
        [
            'InstitutionId',
            'OrdinalRank',
        ]
    ].drop_duplicates()

    df = df.merge(
        usfhn.stats.runner.get('ranks/institution-fields'),
        on='InstitutionId',
    ).merge(
        usfhn.views.filter_by_taxonomy(
            usfhn.stats.runner.get('ranks/institution-placement-stats'),
            level='Academia',
        ),
        on='InstitutionId',
    )

    df = usfhn.institutions.annotate_institution_name(df)
    df = df.rename(columns={
        'OrdinalRank': '#',
        'InstitutionName': 'University',
        'UpPlacements': 'to ↑',
        'DownPlacements': 'to ↓',
        'UpHires': 'from ↑',
        'DownHires': 'from ↓',
    }).drop(columns=[
        'InstitutionId',
    ])

    df['#'] += 1
    df = df.sort_values(by=['#'])

    df = df[
        df['#'] < 101
    ]

    cols = [
        '#',
        'University',
        'to ↑',
        'to ↓',
        'from ↑',
        'from ↓',
    ]

    df['University'] = df['University'].apply(usfhn.latex_table_helper.escape_string)
    df['University'] = df['University'].apply(clean_university_name)

    df = df[cols
    ].drop_duplicates()

    part_1_df = df[
        df['#'] <= 50
    ].copy()

    part_2_df = df[
        df['#'] > 50
    ].copy()

    part_1_df['I'] = [i for i in range(len(part_1_df))]

    part_2_df['I'] = [i for i in range(len(part_2_df))]

    df = part_1_df.merge(
        part_2_df,
        on='I',
    ).drop(columns=['I'])

    for col in df.columns:
        new_col = None

        for real_col in cols:
            if col.startswith(real_col):
                df = df.rename(columns={col: real_col})

    return df


def add_commas_to_number(number):
    if number > 10000:
        return "{:,}".format(number)
    else:
        return str(number)


def get_exclusions_table(dataset='unfiltered-census-pool'):
    field_df = usfhn.stats.runner.get('pool-reduction/excluded-at-field-level', dataset=dataset)

    umbrella_df = usfhn.stats.runner.get('pool-reduction/excluded-at-umbrella-level', dataset=dataset)

    field_df['ExcludedFaculty'] = field_df['ExcludedFaculty'].apply(add_commas_to_number)
    field_df['%'] = field_df['FractionExcluded'].apply(lambda f: hnelib.utils.fraction_to_percent(f, 2))
    field_df['%'] = field_df['%'].astype(str) + ' (' + field_df['ExcludedFaculty'] + ')'

    field_df = field_df[
        [
            'Umbrella',
            '%',
        ]
    ].drop_duplicates().rename(columns={
        '%': '% of domain excluded',
    })


    umbrella_df['ExcludedFaculty'] = umbrella_df['ExcludedFaculty'].apply(add_commas_to_number)
    umbrella_df['%'] = umbrella_df['FractionExcluded'].apply(lambda f: hnelib.utils.fraction_to_percent(f, 2))
    umbrella_df['%'] = umbrella_df['%'].astype(str) + ' (' + umbrella_df['ExcludedFaculty'] + ')'

    umbrella_df = umbrella_df[
        [
            'Umbrella',
            '%',
        ]
    ].drop_duplicates().rename(columns={
        '%': '% of academia excluded',
    })

    df = umbrella_df.merge(
        field_df,
        on='Umbrella',
        how='outer',
    )

    df['% of domain excluded'] = df['% of domain excluded'].fillna('') 
    df['% of academia excluded'] = df['% of academia excluded'].fillna('') 

    df = df.rename(columns={
        'Umbrella': 'Domain'
    })[
        [
            'Domain',
            '% of domain excluded',
            '% of academia excluded',
        ]
    ]

    df['Domain'] = df['Domain'].apply(usfhn.plot_utils.clean_taxonomy_string)
    df = df.sort_values(by='Domain')
    return df


def write_tables():
    for table_path, table in get_tables().items():
        usfhn.latex_table_helper.style_and_save_df_to_path(
            table,
            usfhn.constants.PAPER_TABLES_PATH.joinpath(table_path),
            column_alignments=COLUMN_ALIGNMENTS.get(table_path, []),
        )

def write_csvs():
    for table_path, df in get_tables().items():
        path = usfhn.datasets.CURRENT_DATASET.table_csvs_path.joinpath(table_path).with_suffix('.csv')
        df.to_csv(
            path,
            index=False,
        )

def write_xlsxs():
    for table_path, df in get_tables().items():
        path = usfhn.datasets.CURRENT_DATASET.table_csvs_path.joinpath(table_path).with_suffix('.xlsx')
        df.to_excel(
            path,
            index=False,
        )

def clean_university_name(string):
    if string.startswith('University of'):
        string = string.replace('University of', 'U')
    
    if string.endswith('University'):
        string = string.replace('University', 'U')

    return string
