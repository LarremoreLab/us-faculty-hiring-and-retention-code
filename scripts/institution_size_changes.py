import pandas as pd
import usfhn.constants
import usfhn.datasets

if __name__ == '__main__':
    """
    this script is finds anomalies in institution growth over the years
    """
    df = pd.read_csv(usfhn.constants.AA_CLEAN_INSTITUTIONS_EMPLOYMENT_PATH)
    df['InstitutionPeoplePerYear'] = df.groupby(['Year', 'InstitutionId'])['PersonId'].transform('nunique')
    df = df[
        ['InstitutionId', 'InstitutionName', 'Year', 'InstitutionPeoplePerYear']
    ].drop_duplicates()

    percent_threshold = .7
    count_threshold = 100

    stats = []
    for (institution_id), rows in df.groupby('InstitutionId'):
        institution_name = rows.iloc[0]['InstitutionName']
        rows = rows.sort_values(by=['Year'])

        years = sorted(list(rows['Year'].unique()))
        for year_one in years[:-1]:
            year_two = year_one + 1
            if year_two not in years:
                continue

            year_one_count = rows[rows['Year'] == year_one].iloc[0]['InstitutionPeoplePerYear']
            year_two_count = rows[rows['Year'] == year_two].iloc[0]['InstitutionPeoplePerYear']
            min_count = min(year_one_count, year_two_count)
            max_count = max(year_one_count, year_two_count)

            if min_count / max_count < percent_threshold and max_count - min_count > count_threshold:
                stats.append({
                    'InstitutionId': institution_id,
                    'InstitutionName': institution_name,
                    'YearOne': year_one,
                    'YearTwo': year_two,
                    'YearOneCount': year_one_count,
                    'YearTwoCount': year_two_count,
                    'Grew': year_one_count < year_two_count,
                })

    stats = pd.DataFrame(stats)

    # filter to only insts that employ
    df = usfhn.datasets.CURRENT_DATASET.data
    employing_inst_ids = df['InstitutionId'].unique()
    stats = stats[
        stats['InstitutionId'].isin(employing_inst_ids)
    ]

    stats['InstitutionAppearances'] = stats.groupby('InstitutionId')['YearOne'].transform('nunique')
    stats['ChangeTypes'] = stats.groupby('InstitutionId')['Grew'].transform('nunique')
    stats = stats.drop(columns=['Grew'])

    stats.to_csv('institution_changes.csv', index=False)
