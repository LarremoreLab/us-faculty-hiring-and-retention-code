import usfhn.datasets
import usfhn.views
import usfhn.fieldwork


def get_non_doctorates_by_taxonomy(by_year=False):
    """
    returns df with cols:
    - TaxonomyLevel
    - TaxonomyValue
    - Year (if `by_year` == True)
    - Faculty
    - NonDoctorates
    - NonDoctoratesFraction
    """
    groupby_cols = [
        'TaxonomyLevel',
        'TaxonomyValue',
    ]

    if by_year:
        groupby_cols.append('Year')

    df = usfhn.datasets.get_dataset_df('closedness_data', by_year=by_year)
    df = usfhn.fieldwork.linearize_df_taxonomy_hierarchy(df)

    df = df[
        [
            'PersonId',
            'DegreeInstitutionId',
        ] + groupby_cols
    ].drop_duplicates()

    df['Faculty'] = df.groupby(groupby_cols)['PersonId'].transform('nunique')

    df = df[
        df['DegreeInstitutionId'].isna()
    ]

    df['NonDoctorates'] = df.groupby(groupby_cols)['PersonId'].transform('nunique')

    df = df[
        [
            'Faculty',
            'NonDoctorates',
        ] + groupby_cols
    ].drop_duplicates()

    df['NonDoctoratesFraction'] = df['NonDoctorates'] / df['Faculty']

    return df
