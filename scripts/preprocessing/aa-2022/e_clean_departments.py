import pandas as pd
from functools import lru_cache

import hnelib.utils

import usfhn.constants
import usfhn.utils
import usfhn.views


EMPLOYMENT_OVERLAP_THRESHOLD = .5
MIN_DEPARTMENT_SIZE_FOR_OVERLAP = 5

MINIMAL_DEPARTMENT_REMAPPING_COLUMNS = [
    'PersonId',
    'DepartmentName',
    'Taxonomy',
    'Year',
    'InterimId',
    'FakeDepartmentId',
]


@lru_cache(maxsize=2)
def get_employment(raw=False):
    df = pd.read_csv(usfhn.constants.AA_2022_DEGREE_FILTERED_EMPLOYMENT_PATH)

    if raw:
        return df
    
    df = df[
        [
            'PersonId',
            'InstitutionId',
            'DepartmentId',
            'DepartmentName',
            'Taxonomy',
            'Year'
        ]
    ].drop_duplicates()

    return df


@lru_cache
def get_years():
    return sorted(list(get_employment()['Year'].unique()))


def clean_department_name(name):
    name = name.lower()
    name = name.replace(',', '')
    name = name.replace('the', '')
    name = name.replace('  ', ' ')

    return name


def validate_department_id_to_institution_id_is_unique(df):
    df = df.copy()[
        [
            'DepartmentId',
            'DepartmentName',
            'InstitutionId',
            'Year'
        ]
    ].drop_duplicates()

    df['InstitutionsPerYear'] = df.groupby([
        'DepartmentId',
        'Year',
    ])['InstitutionId'].transform('nunique')

    assert(df[df['InstitutionsPerYear'] > 1].empty)
    print("within a year, department ids are unique to a given institution")

    df['NamesPerYear'] = df.groupby([
        'DepartmentId',
        'Year',
    ])['DepartmentName'].transform('nunique')

    assert(df[df['NamesPerYear'] > 1].empty)
    print("within a year, department names are unique to a given department id")

    df['TotalInstitutions'] = df.groupby([
        'DepartmentId',
    ])['InstitutionId'].transform('nunique')

    n_multi_institution_departments = df[
        df['TotalInstitutions'] > 1
    ]['DepartmentId'].nunique()

    print(f"across years, {n_multi_institution_departments} department ids are associated with more than one institution (in different years)")


def assign_fake_department_ids(df):
    df = df.copy().drop(columns=[
        'DepartmentId'
    ])

    df['FakeDepartmentId'] = df['InstitutionId'].astype(str) + '-' + df['DepartmentName'].astype(str)

    print(f'starting with {df["FakeDepartmentId"].nunique()} "DepartmentIds"')

    return df


def filter_to_good_institutions_for_testing(df, threshold=40):
    institution_departments_df = df.copy()[
        [
            'InstitutionId',
            'FakeDepartmentId',
        ]
    ].drop_duplicates()

    institution_departments_df['InstDepts'] = df.groupby('InstitutionId')['FakeDepartmentId'].transform('nunique')

    institution_departments_df = institution_departments_df[
        institution_departments_df['InstDepts'] > threshold
    ]

    df = df[
        df['InstitutionId'].isin(institution_departments_df['InstitutionId'].unique())
    ]

    return df


def remap_departments(df):
    # # we're testing
    # if True:
    #     df = df[
    #         df['InstitutionId'] == 8
    #     ]
    #     print('testing')
    #     # df = filter_to_good_institutions_for_testing(df)
    #     # inst_ids = list(df['InstitutionId'])[:5]
    #     # df = df[df['InstitutionId'].isin(inst_ids)]

    department_dfs = []
    rejected_matches_dfs = []
    for institution_id, rows in df.groupby('InstitutionId'):
        print(institution_id)
        department_df, rejected_matches_df = remap_institution_departments(rows)

        department_dfs.append(department_df)
        rejected_matches_dfs.append(rejected_matches_df)


    departments, rejected_matches = assign_department_ids(
        pd.concat(department_dfs),
        pd.concat(rejected_matches_dfs),
    )

    aliases = get_department_aliases(departments)

    departments, rejected_matches = assign_department_names(
        departments,
        rejected_matches,
    )
    
    remapping_is_complete(departments, print_results=True)

    departments = departments[
        [
            'DepartmentId',
            'DepartmentName',
            'InstitutionId',
            'Taxonomy',
            'FakeDepartmentId',
        ]
    ].drop_duplicates()

    departments = clean_taxonomies(departments)

    return departments


def clean_taxonomies(df):
    taxonomies_df = df.copy()[
        [
            'DepartmentId',
            'Taxonomy',
        ]
    ].drop_duplicates()

    df = df.drop(columns=['Taxonomy'])

    df = df.merge(
        taxonomies_df,
        on='DepartmentId',
    )

    return df


def remap_institution_departments(df):
    """
    df contains only departments from a given institution

    1. start by assigning departments with the same name a InterimId
    """
    institution_id = df.iloc[0]['InstitutionId']

    n_department_ids_start = df['FakeDepartmentId'].nunique()

    df = df.copy().drop(columns=[
        'InstitutionId',
    ])

    df = remap_department_ids_by_name_cleaning(df)

    last_n_remaining = None
    n_remaining = df['InterimId'].nunique()

    while last_n_remaining != n_remaining:
        df, rejected_matches = get_remapped_departments(
            df,
            include_same_year_remaps=True,
            include_merges=True,
        )

        last_n_remaining = n_remaining
        n_remaining = remapping_is_complete(df)

    last_n_remaining = None
    while last_n_remaining != n_remaining:
        df, rejected_matches = get_remapped_departments(
            df,
            include_same_year_remaps=True,
            include_merges=True,
            allow_multiple_matches=True,
        )

        last_n_remaining = n_remaining
        n_remaining = remapping_is_complete(df)

    df['InstitutionId'] = institution_id
    rejected_matches['InstitutionId'] = institution_id

    return df, rejected_matches


def assign_department_ids(df, rejected_matches):
    ids_df = df.copy()[
        [
            'InstitutionId',
            'InterimId',
        ]
    ].drop_duplicates()

    ids_df['DepartmentId'] = [i for i in range(len(ids_df))]

    df = df.merge(
        ids_df,
        on=[
            'InstitutionId',
            'InterimId',
        ]
    ).drop(columns=['InterimId'])

    rejected_matches = rejected_matches.merge(
        ids_df.rename(columns={
            'InterimId': 'InterimIdToRemap',
        }),
        on=[
            'InstitutionId',
            'InterimIdToRemap',
        ]
    ).merge(
        ids_df.rename(columns={
            'InterimId': 'MatchCandidateInterimId',
            'DepartmentId': 'MatchCandidateDepartmentId',
        }),
        on=[
            'InstitutionId',
            'MatchCandidateInterimId',
        ]
    ).drop(columns=[
        'MatchCandidateInterimId',
        'InterimIdToRemap',
    ])

    return df, rejected_matches


def get_department_aliases(df):
    aliases_df = df.copy()[
        [
            'DepartmentId',
            'DepartmentName',
        ]
    ].drop_duplicates()

    aliases_df['AliasesCount'] = aliases_df.groupby('DepartmentId')['DepartmentName'].transform('nunique')
    aliases_df = aliases_df[
        aliases_df['AliasesCount'] > 1
    ].drop(columns=['AliasesCount'])

    return aliases_df


def assign_department_names(df, rejected_matches):
    names_df = df.copy()[
        [
            'DepartmentId',
            'DepartmentName',
            'Year',
        ]
    ].drop_duplicates()

    names_df['NameYears'] = names_df.groupby([
        'DepartmentId',
        'DepartmentName'
    ])['Year'].transform('nunique')

    names_df['MaxNameYears'] = names_df.groupby('DepartmentId')['NameYears'].transform('max')

    names_df = names_df[
        names_df['NameYears'] == names_df['MaxNameYears']
    ][
        [
            'DepartmentId',
            'DepartmentName',
        ]
    ].drop_duplicates(subset=['DepartmentId'])

    df = df.drop(columns=['DepartmentName']).merge(
        names_df,
        on='DepartmentId'
    )

    rejected_matches = rejected_matches.merge(
        names_df,
        on='DepartmentId',
    ).merge(
        names_df.rename(columns={
            'DepartmentId': 'MatchCandidateDepartmentId',
            'DepartmentName': 'MatchCandidateName',
        })
    )

    return df, rejected_matches


def remapping_is_complete(df, print_results=False):
    id_column = 'DepartmentId' if 'DepartmentId' in df.columns else 'InterimId'

    n = df[id_column].nunique()

    n_unremappable = df[
        df['Unremappable']
    ][id_column].nunique()

    n_complete = df[
        df['InAllYears']
    ][id_column].nunique()

    n_remaining = n - n_unremappable - n_complete

    p_complete = hnelib.utils.fraction_to_percent(n_complete / n, 1)
    p_remaining = hnelib.utils.fraction_to_percent(n_remaining / n, 1)
    
    if print_results:
        print(f"{n} departments:")
        print(f"\t{n_complete} in all years (complete)")
        print(f"\t{n_unremappable} are unremappable")
        print(f"\t{n_remaining} might be able to be remapped with looser tolerances")

    return n_remaining


def get_remapped_departments(
    df,
    include_same_year_remaps=False,
    include_merges=False,
    allow_multiple_matches=False,
):
    df = annotate_department_events(df)

    remappable_df = df[
        (~df['InAllYears'])
        &
        (~df['Unremappable'])
    ]

    unremappable_departments = set()
    all_candidate_matches = pd.DataFrame()
    matchable_departments = set(df['InterimId'].unique())
    for (interim_id, year), rows in remappable_df.groupby(['InterimId', 'Year']):
        potential_matches_df = get_departments_that_share_taxonomies(
            df,
            matchable_departments,
            interim_id, 
            taxonomies=set(rows['Taxonomy'].unique()),
        )

        if not all_candidate_matches.empty:
            already_checked = set(all_candidate_matches[
                all_candidate_matches['MatchCandidateInterimId'] == interim_id
            ]['InterimIdToRemap'].unique())

            potential_matches_df = potential_matches_df[
                ~potential_matches_df['InterimId'].isin(already_checked)
            ]

        if include_same_year_remaps:
            potential_matches_df = potential_matches_df[
                potential_matches_df['Year'] != year
            ]

        candidate_matches = get_departments_with_overlapping_employment(
            potential_matches_df,
            id_to_match=interim_id, 
            people_to_overlap=set(rows['PersonId'].unique()),
            name=rows.iloc[0]['DepartmentName'],
        )

        if candidate_matches.empty:
            unremappable_departments.add(interim_id)
        else:
            all_candidate_matches = pd.concat([all_candidate_matches, candidate_matches])

    if all_candidate_matches.empty:
        rejected_matches = pd.DataFrame({})
    else:
        accepted_matches, rejected_matches = categorize_matches(
            all_candidate_matches,
            include_merges=include_merges,
            allow_multiple_matches=allow_multiple_matches,
        )
        df = remap_matches(df, accepted_matches)

        unremappable_departments -= set(accepted_matches['InterimIdToRemap'].unique())
        unremappable_departments -= set(accepted_matches['MatchCandidateInterimId'].unique())
        unremappable_departments &= set(df['InterimId'].unique())
        df = annotate_department_events(df, unremappable_departments)

    return df, rejected_matches


def remap_matches(df, matches):
    """
    I'm pretty sure this has to be done sequential, cuz remaps could effect each other
    """
    remaps = matches.copy()[
        [
            'InterimIdToRemap',
            'MatchCandidateInterimId',
        ]
    ].drop_duplicates()

    while not matches.empty:
        remap = matches.iloc[0]
        old_id = matches.iloc[0]['InterimIdToRemap']
        new_id = matches.iloc[0]['MatchCandidateInterimId']
        df, matches = apply_remap(df, matches, old_id, new_id)

    return df


def apply_remap(df, matches, old_interim_id, new_interim_id):
    """
    remap is a single 
    """
    df = df.copy()
    matches = matches.copy()
    remap = {i: i for i in df['InterimId'].unique()}

    remap[old_interim_id] = new_interim_id

    df['InterimId'] = df['InterimId'].apply(remap.get)
    matches['InterimIdToRemap'] = matches['InterimIdToRemap'].apply(remap.get)
    matches['MatchCandidateInterimId'] = matches['MatchCandidateInterimId'].apply(remap.get)

    matches = matches[
        matches['InterimIdToRemap'] != matches['MatchCandidateInterimId']
    ]

    return df, matches


def remap_department_ids_by_name_cleaning(df):
    df['CleanDepartmentName'] = df['DepartmentName'].apply(clean_department_name)

    names_df = df[
        [
            'CleanDepartmentName',
        ]
    ].drop_duplicates()

    names_df['InterimId'] = [i for i in range(len(names_df))]

    df = df.merge(
        names_df,
        on='CleanDepartmentName',
    ).drop(columns=['CleanDepartmentName'])

    return df


def get_departments_that_share_taxonomies(df, matchable_departments, interim_id, taxonomies):
    """
    We're also going to require that potential matches happen in a different
    year. We don't want to merge departments in the same year if we can help
    it.
    """
    return df[
        (df['InterimId'] != interim_id)
        &
        (df['Taxonomy'].isin(taxonomies))
        &
        (df['InterimId'].isin(matchable_departments))
    ].copy()


def categorize_matches(matches, include_merges=False, allow_multiple_matches=False):
    """
    where we're trying to match `InterimIdToRemap` into `MatchCandidateInterimId`.

    Whenever an InterimId has only one possible match, and that meets the
    threshold, we're going to merge them into it.
    
    for now, we're only going to remap when there is a single
    IsOverlap[OtherToMatch] for a given InterimIdToRemap.

    In other words, we're only going to remap small things into bigger ones.

    This is more reasonable than it would have been in the old way of doing
    things, because we allow matches over any year (other than the current year)
    """
    matches = matches.copy()

    matches['LeftMatch'] = matches.groupby(
        'InterimIdToRemap'
    )['MatchCandidateInterimId'].transform('nunique')

    matches['RightMatch'] = matches.groupby(
        'MatchCandidateInterimId'
    )['MatchCandidateInterimId'].transform('nunique')

    matches['Match'] = matches['IsOverlap[ToMatch-Other]']

    if allow_multiple_matches:
        matches = matches[
            matches['IsOverlap[ToMatch-Other]'] == matches.groupby(
                'InterimIdToRemap'
            )['IsOverlap[ToMatch-Other]'].transform('max')
        ]
    else:
        matches['Match'] &= matches['LeftMatch'] == 1

    matches['Accept'] = matches['Match']

    if include_merges:
        matches['MatchMerge'] = matches['IsOverlap[Other-ToMatch]']

        if allow_multiple_matches:
            matches = matches[
                matches['IsOverlap[Other-ToMatch]'] == matches.groupby(
                    'InterimIdToRemap'
                )['IsOverlap[Other-ToMatch]'].transform('max')
            ]
        else:
            matches['MatchMerge'] &= matches['RightMatch'] == 1

        matches['Accept'] |= matches['MatchMerge']

    accepted_matches = matches.copy()[
        matches['Accept']
    ]

    rejected_matches = matches.copy()[
        ~matches['Accept']
    ]

    return accepted_matches, rejected_matches


def get_departments_with_overlapping_employment(df, id_to_match, people_to_overlap, name):
    """
    produces a df with the following:
    - InterimIdToRemap
    - MatchCandidateInterimId
    - Score[ToMatch-Other]
    - Score[Other-ToMatch]
    - IsOverlap[ToMatch-Other]
    - IsOverlap[Other-ToMatch]
    """
    n_people = len(people_to_overlap)

    overlap_rows = []
    for (other_interim_id, year), rows in df.groupby(['InterimId', 'Year']):
        n_other_people = rows.iloc[0]['PeopleInYear']
        year = rows.iloc[0]['Year']

        other_people = set(rows['PersonId'].unique())

        overlap = len(people_to_overlap & other_people)

        to_match_other_overlap = overlap_score(
            numerator=overlap,
            denominator=n_other_people,
            other_denominator=n_people,
        )

        other_to_match_overlap = overlap_score(
            numerator=overlap,
            denominator=n_people,
            other_denominator=n_other_people,
        )

        overlap_rows.append({
            'InterimIdToRemap': id_to_match,
            'NameToMatch': name,
            'MatchCandidateName': rows.iloc[0]['DepartmentName'],
            'MatchCandidateInterimId': other_interim_id,
            'Score[ToMatch-Other]': to_match_other_overlap,
            'Score[Other-ToMatch]': other_to_match_overlap,
            'Year': year,
        })

    df = pd.DataFrame(overlap_rows)

    if df.empty:
        return df

    df['Score[ToMatch-Other]'] = df.groupby([
        'InterimIdToRemap',
        'MatchCandidateInterimId',
    ])['Score[ToMatch-Other]'].transform('max')

    df['Score[Other-ToMatch]'] = df.groupby([
        'InterimIdToRemap',
        'MatchCandidateInterimId',
    ])['Score[Other-ToMatch]'].transform('max')

    df['IsOverlap[ToMatch-Other]'] = df['Score[ToMatch-Other]'] > EMPLOYMENT_OVERLAP_THRESHOLD
    df['IsOverlap[Other-ToMatch]'] = df['Score[Other-ToMatch]'] > EMPLOYMENT_OVERLAP_THRESHOLD

    df = df[
        (df['IsOverlap[ToMatch-Other]'] > EMPLOYMENT_OVERLAP_THRESHOLD)
        |
        (df['IsOverlap[Other-ToMatch]'] > EMPLOYMENT_OVERLAP_THRESHOLD)
    ]

    df = df.drop(columns=['Year']).drop_duplicates()

    df['InterimIdToRemapMatches'] = df.groupby(
        'InterimIdToRemap'
    )['MatchCandidateInterimId'].transform('nunique')

    df['MatchCandidateInterimIdMatches'] = df.groupby(
        'MatchCandidateInterimId'
    )['MatchCandidateInterimId'].transform('nunique')

    return df


def overlap_score(
    numerator,
    denominator,
    other_denominator,
    denominator_threshold=MIN_DEPARTMENT_SIZE_FOR_OVERLAP,
    penalty_score=EMPLOYMENT_OVERLAP_THRESHOLD,
):
    """
    If |people_to_match| < 5 <= |other_people|, then Overlap[ToMatch-Other] will be set to EMPLOYMENT_OVERLAP_THRESHOLD
    If |other_people| < 5 <= |people_to_match|, then Overlap[Other-ToMatch] will be set to EMPLOYMENT_OVERLAP_THRESHOLD

    Also requires the overlap to be > MIN_DEPARTMENT_SIZE_FOR_OVERLAP
    """
    score = numerator / denominator

    if denominator < denominator_threshold <= other_denominator:
        score = min(score, penalty_score)

    return score


def annotate_department_events(df, unremappable_departments=[]):
    events_df = df.copy()[
        [
            'InterimId',
            'Year',
        ]
    ].drop_duplicates()

    years = get_years()

    events_df['Start'] = events_df.groupby('InterimId')['Year'].transform('min')
    events_df['End'] = events_df.groupby('InterimId')['Year'].transform('max')
    events_df['Years'] = events_df.groupby('InterimId')['Year'].transform('nunique')
    events_df['Years'] = events_df.groupby('InterimId')['Year'].transform('nunique')
    events_df['Range'] = 1 + events_df['End'] - events_df['Start']
    events_df['HasHoles'] = events_df['Years'] != events_df['Range']
    events_df['InAllYears'] = events_df['Years'] == len(years)
    events_df['Unremappable'] = events_df['InterimId'].isin(unremappable_departments)

    events_df = events_df[
        [
            'InterimId',
            'InAllYears',
            'HasHoles',
            'Unremappable',
        ]
    ].drop_duplicates()

    df = df[MINIMAL_DEPARTMENT_REMAPPING_COLUMNS].drop_duplicates()
    df = df.merge(
        events_df,
        on='InterimId',
    )

    df['PeopleInYear'] = df.groupby(['InterimId', 'Year'])['PersonId'].transform('nunique')

    return df



if __name__ == '__main__':
    usfhn.utils.print_cleaning_step_start("Department Cleaning")

    df = get_employment()
    validate_department_id_to_institution_id_is_unique(df)

    df = assign_fake_department_ids(df)

    remapped_departments = remap_departments(df)

    raw_employment = get_employment(raw=True)
    raw_employment = assign_fake_department_ids(raw_employment)
    raw_employment = raw_employment.drop(columns=[
        'DepartmentName',
        'Taxonomy',
    ])

    print(raw_employment.nunique())

    employment = raw_employment.merge(
        remapped_departments,
        on=[
            'InstitutionId',
            'FakeDepartmentId',
        ],
    ).drop(columns=['FakeDepartmentId']).drop_duplicates()

    print(employment.nunique())
    print(len(employment))

    employment.to_csv(
        usfhn.constants.AA_2022_DEPARTMENT_CLEANED_EMPLOYMENT_V2_PATH,
        index=False,
    )
