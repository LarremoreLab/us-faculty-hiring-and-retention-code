import usfhn.datasets
import usfhn.fieldwork
import usfhn.gender
import usfhn.gini
import usfhn.measurements
import usfhn.rank
import usfhn.views


def get_new_hires_df(
    by_year=False,
    max_years_between_degree_and_career_start=usfhn.constants.MAX_YEARS_BETWEEN_DEGREE_AND_CAREER_START,
    exclude_first_year=False,
):
    """
    new hires are those whose first year of employment is within
    `max_years_between_degree_and_career_start` years of their DegreeYear
    """
    df = usfhn.datasets.get_dataset_df('closedness_data', 'careers')

    df = df[
        [
            'PersonId',
            'Year',
            'DegreeInstitutionId',
            'InstitutionId',
            'DegreeYear',
            'Taxonomy',
        ]
    ].drop_duplicates()

    df = usfhn.fieldwork.linearize_df_taxonomy_hierarchy(df)

    df['FirstYear'] = df.groupby('PersonId')['Year'].transform('min')
    df['LatestCareerStartYear'] = df['DegreeYear'] + max_years_between_degree_and_career_start

    df['NewHire'] = df['FirstYear'] <= df['LatestCareerStartYear']

    if exclude_first_year:
        first_year = min(df['Year'])
        df['NewHire'] &= df['FirstYear'] != first_year

    df = df[
        df['NewHire']
    ]

    cols = [
        'PersonId',
        'InstitutionId',
        'DegreeInstitutionId',
        'TaxonomyLevel',
        'TaxonomyValue',
    ]

    if by_year:
        df = df[
            df['FirstYear'] == df['Year']
        ]

        cols.append('Year')

    df = df[cols].drop_duplicates()

    return df


def annotate_new_hires(df):
    import usfhn.stats
    join_columns = [
        'PersonId',
        'InstitutionId',
        'DegreeInstitutionId',
        'TaxonomyLevel',
        'TaxonomyValue',
    ]

    join_cols = ['PersonId']

    if 'Year' in df.columns:
        join_cols.append('Year')
        new_hires_df = usfhn.stats.runner.get('new-hire/by-year/df')
    else:
        new_hires_df = usfhn.stats.runner.get('new-hire/df')

    new_hires_df = new_hires_df[join_cols].drop_duplicates()

    new_hires_df['NewHire'] = True

    df = df.merge(
        new_hires_df,
        on=join_cols,
        how='left',
    )
    
    df['NewHire'] = df['NewHire'].fillna(False)

    return df


def get_self_hires(by_year=False):
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

    df['SelfHire'] = df['InstitutionId'] == df['DegreeInstitutionId']

    df['Faculty'] = df.groupby(groupby_cols)['PersonId'].transform('nunique')
    df['HireTypeFaculty'] = df.groupby(
        ['SelfHire'] + groupby_cols
    )['PersonId'].transform('nunique')

    df['HireTypeFraction'] = df['HireTypeFaculty'] / df['Faculty']

    df = df[
        groupby_cols + [
            'Faculty',
            'SelfHire',
            'HireTypeFaculty',
            'HireTypeFraction',
        ]
    ].drop_duplicates()
    
    return df


def get_steepness(by_year=False):
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

    ranks = usfhn.stats.runner.get('ranks/df', rank_type='prestige')[
        [
            'InstitutionId',
            'TaxonomyLevel',
            'TaxonomyValue',
            'Percentile',
        ]
    ].drop_duplicates()

    df = df.merge(
        ranks,
        on=[
            'InstitutionId',
            'TaxonomyLevel',
            'TaxonomyValue',
        ],
    ).merge(
        ranks.rename(columns={
            'InstitutionId': 'DegreeInstitutionId',
            'Percentile': 'DegreePercentile',
        }),
        on=[
            'DegreeInstitutionId',
            'TaxonomyLevel',
            'TaxonomyValue',
        ],
    )

    df['RankDifference'] = df['Percentile'] - df['DegreePercentile']
    df['RankDifference'] *= -1

    df = df[
        groupby_cols + [
            'PersonId',
            'RankDifference',
        ]
    ].drop_duplicates()

    return usfhn.rank.calculate_hierarchy_stats(
        df,
        'RankDifference',
        groupby_cols,
    )
