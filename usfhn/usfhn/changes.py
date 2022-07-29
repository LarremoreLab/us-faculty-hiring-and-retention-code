import usfhn.datasets
import usfhn.measurements
import usfhn.gender
import usfhn.stats

RANKS_TO_SENIORITY = {
    ('Associate Professor', 'Professor'): True,
    ('Assistant Professor',): False,
}

def get_gini_coefficients_for_rank_subset(
    ranks=(),
    outcolumn='GiniCoefficient',
    drop_other_columns=True,
    by_year=False,
):
    if len(ranks) > 1:
        if by_year:
            df = usfhn.stats.runner.get('ginis/by-seniority/by-year/df')
        else:
            df = usfhn.stats.runner.get('ginis/by-seniority/df')
        df = df[
            df['Senior'] == RANKS_TO_SENIORITY[ranks]
        ]
    else:
        if by_year:
            df = usfhn.stats.runner.get('ginis/by-faculty-rank/by-year/df')
        else:
            df = usfhn.stats.runner.get('ginis/by-faculty-rank/df')
        df = df[
            df['Rank'] == ranks[0]
        ]

    df = df.rename(columns={
        'GiniCoefficient': outcolumn,
    })

    if drop_other_columns:
        df = df[
            [
                'TaxonomyLevel',
                'TaxonomyValue',
                outcolumn,
            ]
        ].drop_duplicates()

    return df

def get_gender_percents_for_rank_subset(ranks=(), outcolumn='PercentFemale'):
    if len(ranks) > 1:
        df = usfhn.stats.runner.get('gender/by-seniority/df')
        df = df[
            df['Senior'] == RANKS_TO_SENIORITY[ranks]
        ]
    else:
        df = usfhn.stats.runner.get('gender/by-faculty-rank/df')
        df = df[
            df['Rank'] == ranks[0]
        ]

    df['PercentFemale'] = df['FractionFemale'] * 100
    df = df.rename(columns={
        'PercentFemale': outcolumn,
    })

    df = df[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            outcolumn,
        ]
    ].drop_duplicates()

    return df

def get_steepness_for_rank_subset(ranks=(), outcolumn='Violations'):
    df = usfhn.stats.runner.get('ranks/seniority/hierarchy-stats', rank_type='prestige')

    df = df[
        (df['Senior'] == RANKS_TO_SENIORITY[ranks])
        &
        (df['MovementType'] == 'Upward')
    ].rename(columns={
        'MovementFraction': outcolumn,
    })[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            outcolumn,
        ]
    ].drop_duplicates()

    return df


def get_self_hire_rate_for_rank_subset(ranks=(), outcolumn='SelfHireRate'):
    if len(ranks) > 1:
        df = usfhn.stats.runner.get('self-hires/by-seniority/df')
        df = df[
            df['Senior'] == RANKS_TO_SENIORITY[ranks]
        ]
    else:
        df = usfhn.stats.runner.get('self-hires/by-faculty-rank/df')
        df = df[
            df['Rank'] == ranks[0]
        ]

    df = df.rename(columns={
        'SelfHiresFraction': outcolumn,
    })[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            outcolumn,
        ]
    ].drop_duplicates()

    return df
