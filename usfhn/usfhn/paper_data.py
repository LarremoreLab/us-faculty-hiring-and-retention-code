import pandas as pd
from collections import defaultdict

import hnelib.pandas

import usfhn.constants
import usfhn.datasets
import usfhn.views
import usfhn.closedness
import usfhn.gender
import usfhn.stats
import usfhn.institutions
import usfhn.fieldwork
import usfhn.new_hires
import usfhn.null_models


def get_data_collections():
    return [
    (get_edge_lists, 'edge-lists.csv'),
    (get_institution_stats, 'institution-stats.csv'),
    (get_taxonomy_stats, 'stats.csv'),
    (get_yearly_stats, 'yearly-stats.csv'),
]


def get_edge_lists():
    """
    - Total
    - Women
    - Men
    - DegreeInstitution
    - Institution
    """
    df = usfhn.datasets.get_dataset_df('closedness_data')
    df = usfhn.fieldwork.linearize_df_taxonomy_hierarchy(df)

    df = df[
        [
            'PersonId',
            'Gender',
            'InstitutionId',
            'DegreeInstitutionId',
            'TaxonomyLevel',
            'TaxonomyValue',
        ]
    ].drop_duplicates()

    groupby_cols = [
        'TaxonomyLevel',
        'TaxonomyValue',
        'InstitutionId',
        'DegreeInstitutionId',
    ]

    dfs = [
        (df, 'Total'),
        (df[df['Gender'] == 'Male'], 'Men'),
        (df[df['Gender'] == 'Female'], 'Women'),
    ]

    df = pd.DataFrame()
    for new_df, count_col in dfs:
        new_df = new_df.copy()
        new_df[count_col] = new_df.groupby(groupby_cols)['PersonId'].transform('nunique')
        new_df = new_df.drop(columns=['Gender', 'PersonId']).drop_duplicates()

        if df.empty:
            df = new_df
        else:
            df = df.merge(
                new_df,
                on=groupby_cols,
            )

    df = usfhn.institutions.annotate_institution_name(df)
    df = usfhn.institutions.annotate_institution_name(
        df,
        id_col='DegreeInstitutionId',
        name_col='DegreeInstitutionName',
    )

    df = df[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'InstitutionId',
            'InstitutionName',
            'DegreeInstitutionId',
            'DegreeInstitutionName',
            'Total',
            'Men',
            'Women',
        ]
    ].drop_duplicates()

    return df


def get_institution_stats():
    """
    - TaxonomyLevel
    - TaxonomyValue
    - NonAttritionEvents
    - AttritionEvents
    - ProductionRank
    - PrestigeRank
    - OrdinalPrestigeRank
    """
    # attrition_df = usfhn.stats.runner.get('attrition/risk/taxonomy').rename(columns={
    attrition_df = usfhn.stats.runner.get(
        'attrition/risk/institution',
        institution_column='DegreeInstitutionId',
    ).rename(columns={
        'Events': 'NonAttritionEvents',
        'DegreeInstitutionId': 'InstitutionId',
    }).drop(columns=['AttritionRisk'])

    production_ranks = usfhn.stats.runner.get(
        'ranks/df',
        rank_type='production',
    ).drop(columns=['Percentile'])

    production_ranks = production_ranks.sort_values(
        by=[
            'TaxonomyLevel',
            'TaxonomyValue',
            'ProductionFraction'
        ],
        ascending=False,
    ).drop(columns=['ProductionFraction'])

    production_ranks['ProductionRank'] = production_ranks.groupby([
        'TaxonomyLevel',
        'TaxonomyValue',
    ])['InstitutionId'].transform('cumcount')

    prestige_ranks = usfhn.stats.runner.get(
        'ranks/df',
        rank_type='prestige',
    )[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'InstitutionId',
            'NormalizedRank',
            'OrdinalRank',
        ]
    ].rename(columns={
        'NormalizedRank': 'PrestigeRank',
        'OrdinalRank': 'OrdinalPrestigeRank',
    }).drop_duplicates()

    cols = [
        'TaxonomyLevel',
        'TaxonomyValue',
        'InstitutionId',
    ]

    df = attrition_df.merge(
        production_ranks,
        on=cols,
    ).merge(
        prestige_ranks,
        on=cols,
    )

    df = usfhn.institutions.annotate_institution_name(df)
    df = df[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'NonAttritionEvents',
            'AttritionEvents',
            'ProductionRank',
            'PrestigeRank',
            'OrdinalPrestigeRank',
        ]
    ]

    return df


def get_doctorate_stats():
    df = usfhn.datasets.get_dataset_df('closedness_data')
    df = usfhn.fieldwork.linearize_df_taxonomy_hierarchy(df)
    df = df[
        [
            'PersonId',
            'TaxonomyLevel',
            'TaxonomyValue',
            'DegreeInstitutionId',
        ]
    ].drop_duplicates()

    groupby_cols = [
        'TaxonomyLevel',
        'TaxonomyValue',
    ]

    df = usfhn.institutions.annotate_us(df)
    df = usfhn.institutions.annotate_highly_productive_non_us_countries(df)
    df = usfhn.institutions.annotate_continent(df)

    dfs = [
        (df[df['DegreeInstitutionId'].isna()], 'NoDoctorate'),
        (df[df['US']], 'Doctorate (US)'),
        (df[~df['US']], 'Doctorate (Non-US)'),
        (df[df['IsHighlyProductiveNonUSCountry']], 'Doctorate (Canada/UK)'),
        (df[df['Continent'] == 'Africa'], 'Doctorate (Africa)'),
        (df[df['Continent'] == 'Asia'], 'Doctorate (Asia)'),
        (df[df['Continent'] == 'Europe'], 'Doctorate (Europe)'),
        (df[df['Continent'] == 'North America'], 'Doctorate (North America)'),
        (df[df['Continent'] == 'Oceania'], 'Doctorate (Oceania)'),
        (df[df['Continent'] == 'South America'], 'Doctorate (South America)'),
    ]

    df = pd.DataFrame()
    for new_df, count_col in dfs:
        new_df = new_df.copy()
        new_df[count_col] = new_df.groupby(groupby_cols)['PersonId'].transform('nunique')
        new_df = new_df[
            [
                'TaxonomyLevel',
                'TaxonomyValue',
                count_col,
            ]
        ].drop_duplicates()

        if df.empty:
            df = new_df
        else:
            df = df.merge(
                new_df,
                on=[
                    'TaxonomyLevel',
                    'TaxonomyValue',
                ],
            )

    return df

def get_doctorate_attritions_stats():
    doctorate_attritions_df = usfhn.stats.runner.get('attrition/risk/non-us-by-is-english')[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'US',
            'IsHighlyProductiveNonUSCountry',
            'Events',
            'AttritionEvents',
        ]
    ].drop_duplicates()

    dfs = [
        (doctorate_attritions_df[doctorate_attritions_df['US']], 'US'),
        (doctorate_attritions_df[doctorate_attritions_df['IsHighlyProductiveNonUSCountry']], 'Canada/UK'),
        (
            doctorate_attritions_df[
                ~(doctorate_attritions_df['IsHighlyProductiveNonUSCountry'])
                &
                ~(doctorate_attritions_df['US'])
            ], 'non US/Canada/UK'),
    ]


    df = pd.DataFrame()
    for new_df, annotation in dfs:
        non_attritions_col = f'NonAttritionEvents ({annotation})'
        attritions_col = f'AttritionEvents ({annotation})'

        new_df = new_df.copy().rename(columns={
            'Events': non_attritions_col,
            'AttritionEvents': attritions_col,
        })[
            [
                'TaxonomyLevel',
                'TaxonomyValue',
                non_attritions_col,
                attritions_col,
            ]
        ].drop_duplicates()

        if df.empty:
            df = new_df
        else:
            df = df.merge(
                new_df,
                on=[
                    'TaxonomyLevel',
                    'TaxonomyValue',
                ],
            )

    return df


def get_new_hires_stats_df():
    df = usfhn.datasets.get_dataset_df('closedness_data')
    df = usfhn.fieldwork.linearize_df_taxonomy_hierarchy(df)
    df = df[
        [
            'PersonId',
            'TaxonomyLevel',
            'TaxonomyValue',
        ]
    ].drop_duplicates()

    df = usfhn.new_hires.annotate_new_hires(df)

    dfs = [
        (df[df['NewHire']], 'NewFaculty'),
        (df[~df['NewHire']], 'ExistingFaculty'),
    ]

    groupby_cols = [
        'TaxonomyLevel',
        'TaxonomyValue',
    ]

    df = pd.DataFrame()
    for new_df, count_col in dfs:
        new_df = new_df.copy()
        new_df[count_col] = new_df.groupby(groupby_cols)['PersonId'].transform('nunique')
        new_df = new_df[
            [
                'TaxonomyLevel',
                'TaxonomyValue',
                count_col,
            ]
        ].drop_duplicates()
        if df.empty:
            df = new_df
        else:
            df = df.merge(
                new_df,
                on=[
                    'TaxonomyLevel',
                    'TaxonomyValue',
                ],
            )

    return df


def get_gini_stats():
    df = usfhn.stats.runner.get('ginis/df')
    df_by_hire_type = usfhn.stats.runner.get('ginis/by-new-hire/df')

    new = df_by_hire_type[
        df_by_hire_type['NewHire']
    ].copy().drop(columns=['NewHire']).rename(columns={
        'GiniCoefficient': 'GiniCoefficient (NewFaculty)'
    })

    existing = df_by_hire_type[
        ~df_by_hire_type['NewHire']
    ].copy().drop(columns=['NewHire']).rename(columns={
        'GiniCoefficient': 'GiniCoefficient (ExistingFaculty)'
    })

    df = df.merge(
        new,
        on=[
            'TaxonomyLevel',
            'TaxonomyValue',
        ]
    ).merge(
        existing,
        on=[
            'TaxonomyLevel',
            'TaxonomyValue',
        ]
    )
    return df

def get_gender_stats():
    df = usfhn.stats.runner.get('gender/df')[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'MaleFaculty',
            'FemaleFaculty',
            'FractionFemale',
        ]
    ].drop_duplicates().rename(columns={
        'FractionFemale': 'FractionWomen',
        'FemaleFaculty': 'Women',
        'MaleFaculty': 'Men',
    })

    df_by_hire_type = usfhn.stats.runner.get('gender/by-new-hire/df')[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'NewHire',
            'FractionFemale',
        ]
    ].drop_duplicates()

    new = df_by_hire_type[
        df_by_hire_type['NewHire']
    ].copy().drop(columns=['NewHire']).rename(columns={
        'FractionFemale': 'FractionWomen (NewFaculty)'
    })

    existing = df_by_hire_type[
        ~df_by_hire_type['NewHire']
    ].copy().drop(columns=['NewHire']).rename(columns={
        'FractionFemale': 'FractionWomen (ExistingFaculty)'
    })

    df = df.merge(
        new,
        on=[
            'TaxonomyLevel',
            'TaxonomyValue',
        ]
    ).merge(
        existing,
        on=[
            'TaxonomyLevel',
            'TaxonomyValue',
        ]
    )
    return df


def get_self_hire_stats():
    df = usfhn.stats.runner.get('self-hires/df')[
        [

            'TaxonomyLevel',
            'TaxonomyValue',
            'SelfHires',
        ]
    ]

    gender_df = usfhn.stats.runner.get('self-hires/by-gender/df')[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'Gender',
            'SelfHires',
        ]
    ].drop_duplicates()

    male_df = gender_df.copy()[
        gender_df['Gender'] == 'Male'
    ].drop(columns=['Gender']).rename(columns={
        'SelfHires': 'SelfHires (Men)'
    })

    female_df = gender_df.copy()[
        gender_df['Gender'] == 'Female'
    ].drop(columns=['Gender']).rename(columns={
        'SelfHires': 'SelfHires (Women)'
    })

    df = df.merge(
        male_df,
        on=[
            'TaxonomyLevel',
            'TaxonomyValue',
        ]
    ).merge(
        female_df,
        on=[
            'TaxonomyLevel',
            'TaxonomyValue',
        ]
    )


    attrition_risk_df = usfhn.stats.runner.get('attrition/risk/self-hires')[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'SelfHire',
            'Events',
            'AttritionEvents',
        ]
    ].drop_duplicates()

    self_hires_attrition_risk_df = attrition_risk_df.copy()[
        attrition_risk_df['SelfHire']
    ].drop(columns=['SelfHire']).rename(columns={
        'Events': 'NonAttritionEvents (SelfHires)',
        'AttritionEvents': 'AttritionEvents (SelfHires)',
    })

    non_self_hires_attrition_risk_df = attrition_risk_df.copy()[
        ~attrition_risk_df['SelfHire']
    ].drop(columns=['SelfHire']).rename(columns={
        'Events': 'NonAttritionEvents (non-SelfHires)',
        'AttritionEvents': 'AttritionEvents (non-SelfHires)',
    })

    df = df.merge(
        self_hires_attrition_risk_df,
        on=[
            'TaxonomyLevel',
            'TaxonomyValue',
        ]
    ).merge(
        non_self_hires_attrition_risk_df,
        on=[
            'TaxonomyLevel',
            'TaxonomyValue',
        ]
    )

    return df


def get_hierarchy_stats():
    df = usfhn.null_models.get_stats()[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'NullSteepnessMean',
            'MoreHierarchicalCount',
        ]
    ].drop_duplicates().rename(columns={
        'MoreHierarchicalCount': 'NullModelMoreHierarchicalThanEmpiricalCount',
    })

    steepness_df = usfhn.stats.runner.get('ranks/hierarchy-stats', rank_type='prestige')

    steepness_df = hnelib.pandas.aggregate_df_over_column(
        steepness_df,
        agg_col='MovementType',
        join_cols=[
            'TaxonomyLevel',
            'TaxonomyValue',
        ],
        value_cols=['MovementFraction'],
        agg_value_to_label={
            'Self-Hire': 'SelfHire',
            'Upward': 'Up',
            'Downward': 'Down',
        },
    )

    df = df.merge(
        steepness_df,
        on=[
            'TaxonomyLevel',
            'TaxonomyValue',
        ]
    ).rename(columns={
        'DownMovementFraction': 'FractionDownHierarchyHires',
        'UpMovementFraction': 'FractionUpHierarchyHires',
    })

    df['FractionUpHierarchyHires'] = 1 - df['NullSteepnessMean'] - df['SelfHireMovementFraction']

    df = df[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'FractionUpHierarchyHires',
            'FractionDownHierarchyHires',
            'FractionUpHierarchyHires',
            'NullModelMoreHierarchicalThanEmpiricalCount',
        ]
    ]

    return df


def get_taxonomy_stats():
    """
    - TaxonomyLevel
    - TaxonomyValue

    - NoDoctorate
    - Doctorate (US)
    - Doctorate (Non-US)
    - Doctorate (Canada/UK)
    - Doctorate (Africa)
    - Doctorate (Asia)
    - Doctorate (Europe)
    - Doctorate (North America)
    - Doctorate (Oceania)
    - Doctorate (South America)

    - NonAttritionEvents (US)
    - AttritionEvents (US)
    - NonAttritionEvents (Canada/UK)
    - AttritionEvents (Canada/UK)
    - NonAttritionEvents (non US/Canada/UK)
    - AttritionEvents (non US/Canada/UK)

    - NewFaculty
    - ExistingFaculty

    - GiniCoefficient
    - GiniCoefficient (NewFaculty)
    - GiniCoefficient (ExistingFaculty)

    - Men
    - Women
    - FractionWomen
    - FractionWomen (new faculty)
    - FractionWomen (existing faculty)

    - SelfHires
    - SelfHires (Men)
    - SelfHires (Women)
    - ExpectedSelfHireRate

    - NonAttritionEvents (non-self-hires)
    - AttritionEvents (non-self-hires)
    - NonAttritionEvents (self-hires)
    - AttritionEvents (self-hires)

    - FractionUpHierarchyHires
    - FractionDownHierarchyHires

    - FractionUpHierarchyHires (null model)
    - NullModelMoreHierarchicalThanEmpiricalCount
    """

    df_getters = [
        get_doctorate_stats,
        get_doctorate_attritions_stats,
        get_new_hires_stats_df,
        get_gini_stats,
        get_gender_stats,
        get_self_hire_stats,
        get_hierarchy_stats,
    ]

    df = pd.DataFrame()
    for df_getter in df_getters:
        if df.empty:
            df = df_getter()
        else:
            df = df.merge(
                df_getter(),
                on=[
                    'TaxonomyLevel',
                    'TaxonomyValue',
                ]
            )

    return df


def get_yearly_stats():
    """
    - TaxonomyLevel
    - TaxonomyValue
    - Year
    - GiniCoefficient
    - FractionWomen
    """
    ginis_df = usfhn.stats.runner.get('ginis/by-year/df')
    gender_df = usfhn.stats.runner.get('gender/by-year/df')[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'Year',
            'FractionFemale',
        ]
    ].drop_duplicates()

    df = ginis_df.merge(
        gender_df,
        on=[
            'TaxonomyLevel',
            'TaxonomyValue',
            'Year',
        ]
    )

    return df


def format_df_for_sharing(df):
    """
    renames `TaxonomyLevel=Umbrella` to `TaxonomyLevel=Domain`

    filters `TaxonomyLevel` to: Academia|Domain|Field

    drops duplicates
    """
    df['TaxonomyLevel'] = df['TaxonomyLevel'].apply(lambda l: 'Domain' if l == 'Umbrella' else l)

    df = df[
        df['TaxonomyLevel'].isin(['Academia', 'Domain', 'Field'])
    ]

    df = df.drop_duplicates()

    return df


def write_data(directory=usfhn.constants.DATA_SHARING_ROOT):
    for data_getter, data_file_name in get_data_collections():
        df = data_getter()
        df = format_df_for_sharing(df)

        data_path = directory.joinpath(data_file_name)
        df.to_csv(data_path, index=False)


if __name__ == '__main__':
    write_data()
