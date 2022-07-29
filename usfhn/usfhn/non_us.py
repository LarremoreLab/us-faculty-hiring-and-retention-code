import statsmodels.api as sm
import pandas as pd

import hnelib.model

import usfhn.constants
import usfhn.datasets
import usfhn.fieldwork
import usfhn.views
import usfhn.institutions


def annotate_non_us(df):
    """
    needs at least the following columns:
    - PersonId

    people without degrees are considered to be US
    """
    if 'NonUS' in df.columns:
        return df

    non_us_df = usfhn.datasets.get_dataset_df('closedness_data')[
        [
            'PersonId',
            'DegreeInstitutionId',
        ]
    ].drop_duplicates()

    non_us_df = usfhn.institutions.annotate_us(non_us_df)
    non_us_df['NonUS'] = ~non_us_df['US']

    non_us_df = non_us_df[
        [
            'PersonId',
            'NonUS',
        ]
    ].drop_duplicates()

    df = df.merge(
        non_us_df,
        on='PersonId',
        how='left',
    )

    df['NonUS'] = df['NonUS'].fillna(False)

    return df


def calculate_non_us_fraction_for_df(df, groupby_cols):
    """
    needs at least the following columns:
    - PersonId
    - Taxonomy

    returns df with `groupby_cols` and:
    - FacultyCount: count of faculty in groupby
    - NonUSFacultyCount: count of non U.S. faculty in groupby
    - NonUSFraction: fraction of non U.S. faculty in groupby

    (drops `PersonId`)
    """

    degrees = usfhn.datasets.get_dataset_df('closedness_data')[
        [
            'PersonId',
            'DegreeInstitutionId',
        ]
    ].drop_duplicates()

    df = df.copy().merge(
        degrees,
        on='PersonId',
    )

    df = usfhn.institutions.annotate_us(df)

    df = df[
        [
            'PersonId',
            'US',
            'DegreeInstitutionId',
        ] + groupby_cols
    ].drop_duplicates()

    df['FacultyCount'] = df.groupby(groupby_cols)['PersonId'].transform('nunique')

    df = df[
        (~df['US'])
        &
        (df['DegreeInstitutionId'].notna())
    ].drop(columns=['US'])

    df['NonUSFacultyCount'] = df.groupby(groupby_cols)['PersonId'].transform('nunique')

    df = df.drop(columns=['PersonId']).drop_duplicates()

    df['NonUSFraction'] = df['NonUSFacultyCount'] / df['FacultyCount']

    return df


def get_fraction_non_us(
    by_year=False,
    by_gender=False,
    by_institution=False,
    by_career_age=False,
    by_continent=False,
):
    import usfhn.datasets
    groupby_cols = [
        'TaxonomyLevel',
        'TaxonomyValue',
    ]

    if by_year:
        groupby_cols.append('Year')

    df = usfhn.datasets.get_dataset_df('closedness_data', by_year=by_year)

    df = usfhn.fieldwork.linearize_df_taxonomy_hierarchy(df)

    if by_gender:
        groupby_cols.append('Gender')

    if by_institution:
        groupby_cols.append('InstitutionId')

    if by_career_age:
        import usfhn.careers
        groupby_cols.append('CareerAge')
        df = usfhn.careers.annotate_career_age(df)

    if by_continent:
        import usfhn.institutions
        groupby_cols.append('Continent')
        df = usfhn.institutions.annotate_continent(df)

    df = df[
        groupby_cols + [
            'PersonId',
        ]
    ].drop_duplicates()

    return calculate_non_us_fraction_for_df(
        df, 
        groupby_cols,
    )


def get_fraction_non_us_by_continent_and_career_age():
    import usfhn.datasets
    import usfhn.careers
    import usfhn.institutions

    groupby_cols = [
        'TaxonomyLevel',
        'TaxonomyValue',
        'CareerAge',
    ]

    cols = groupby_cols + ['PersonId']

    df = usfhn.datasets.get_dataset_df('closedness_data')
    df = usfhn.fieldwork.linearize_df_taxonomy_hierarchy(df)

    df = usfhn.careers.annotate_career_age(df)

    df = usfhn.institutions.annotate_continent(df)

    us_df = usfhn.institutions.annotate_us(df)
    us_df = us_df[
        us_df['US']
    ][cols].drop_duplicates()

    results_dfs = []
    for continent, rows in df.groupby('Continent'):
        rows = rows.copy()
        rows = rows[cols].drop_duplicates()

        result_df = calculate_non_us_fraction_for_df(
            pd.concat([rows, us_df]), 
            groupby_cols,
        )

        result_df['Continent'] = continent
        results_dfs.append(result_df)

    return pd.concat(results_dfs)


def get_fraction_non_us_by_gender(by_year=False):
    return get_fraction_non_us(by_year=by_year, by_gender=True)


def get_fraction_non_us_by_institution(by_year=False):
    return get_fraction_non_us(by_year=by_year, by_institution=True)


def get_new_hire_fraction(by_year=False):
    import usfhn.stats
    groupby_cols = [
        'TaxonomyLevel',
        'TaxonomyValue',
    ]

    if by_year:
        groupby_cols.append('Year')
        df = usfhn.stats.runner.get('new-hire/by-year/df')
    else:
        df = usfhn.stats.runner.get('new-hire/df')

    df = df[
        [
            'PersonId',
        ] + groupby_cols
    ].drop_duplicates()

    return calculate_non_us_fraction_for_df(
        df, 
        groupby_cols,
    )


def get_fraction_non_us_by_institution_with_rank(rank_type='production'):
    """
    rank_types:
    - production
    - prestige
    """
    import usfhn.datasets
    import usfhn.stats

    df = usfhn.stats.runner.get('non-us/by-institution/df')
    ranks = usfhn.stats.runner.get('ranks/df', rank_type=rank_type)

    df = df.merge(
        ranks,
        on=[
            'TaxonomyLevel',
            'TaxonomyValue',
            'InstitutionId',
        ]
    )

    return df


def rank_vs_non_us_logit(rank_type='production', by_new_hire=False, existing_hires=False):
    import usfhn.datasets
    df = usfhn.datasets.get_dataset_df('closedness_data')[
        [
            'PersonId',
            'Taxonomy',
            'InstitutionId',
            'DegreeInstitutionId',
        ]
    ].drop_duplicates()

    df = annotate_non_us(df)

    df = usfhn.fieldwork.linearize_df_taxonomy_hierarchy(df)

    ranks = usfhn.stats.runner.get('ranks/df', rank_type=rank_type)

    df = df.merge(
        ranks,
        on=[
            'TaxonomyLevel',
            'TaxonomyValue',
            'InstitutionId',
        ]
    )

    # don't drop duplicates, because we want them by year and PersonId
    df = df[
        [
            'NonUS',
            'TaxonomyLevel',
            'TaxonomyValue',
            'Percentile',
            'PersonId',
        ]
    ]

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
        endog='NonUS',
        exog='Percentile',
        groupby_cols=[
            'TaxonomyLevel',
            'TaxonomyValue',
        ],
    )

    df = usfhn.stats_utils.correct_multiple_hypotheses(df, p_col='Percentile-P', corrected_p_col='Percentile-P')

    return df

def non_us_countries_by_production():
    df = usfhn.datasets.get_dataset_df('closedness_data')[
        [
            'PersonId',
            'Taxonomy',
            'DegreeInstitutionId',
        ]
    ].drop_duplicates()

    df = usfhn.fieldwork.linearize_df_taxonomy_hierarchy(df)

    df = usfhn.institutions.annotate_us(df, drop_country_name=False)

    df = df[
        (df['US'].notna())
        &
        (df['CountryName'].notna())
    ]

    df['CountryFacultyCount'] = df.groupby([
        'TaxonomyLevel',
        'TaxonomyValue',
        'CountryName',
    ])['PersonId'].transform('nunique')

    df['FacultyCount'] = df.groupby([
        'TaxonomyLevel',
        'TaxonomyValue',
    ])['PersonId'].transform('nunique')

    df = df[
        df['US'] == False
    ]

    df['NonUSFacultyCount'] = df.groupby([
        'TaxonomyLevel',
        'TaxonomyValue',
    ])['PersonId'].transform('nunique')

    df['CountryFraction'] = df['CountryFacultyCount'] / df['FacultyCount']
    df['CountryNonUSFraction'] = df['CountryFacultyCount'] / df['NonUSFacultyCount']

    df = df[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'FacultyCount',
            'CountryName',
            'CountryFacultyCount',
            'CountryFraction',
            'CountryNonUSFraction',
            'NonUSFacultyCount',
        ]
    ].drop_duplicates()

    return df


def non_us_continents_by_production():
    df = usfhn.datasets.get_dataset_df('closedness_data')[
        [
            'PersonId',
            'Taxonomy',
            'DegreeInstitutionId',
        ]
    ].drop_duplicates()

    df = usfhn.fieldwork.linearize_df_taxonomy_hierarchy(df)
    df = usfhn.institutions.annotate_continent(df)
    df = usfhn.institutions.annotate_us(df)

    df = df[
        df['Continent'].notna()
    ]

    df = df[
        ~df['US']
    ]

    df = df[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'PersonId',
            'Continent',
        ]
    ].drop_duplicates()

    df['FacultyCount'] = df.groupby([
        'TaxonomyLevel',
        'TaxonomyValue',
    ])['PersonId'].transform('nunique')

    df['ContinentFacultyCount'] = df.groupby([
        'TaxonomyLevel',
        'TaxonomyValue',
        'Continent',
    ])['PersonId'].transform('nunique')

    df['ContinentFraction'] = df['ContinentFacultyCount'] / df['FacultyCount']

    df = df[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'Continent',
            'FacultyCount',
            'ContinentFacultyCount',
            'ContinentFraction',
        ]
    ].drop_duplicates()

    return df
