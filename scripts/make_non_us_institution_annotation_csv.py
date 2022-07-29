import pandas as pd

import usfhn.constants
import usfhn.institutions


if __name__ == '__main__':
    df = pd.read_csv(usfhn.constants.AA_2022_BASIC_CLEANING_DEGREE_INSTITUTIONS_PATH)
    df = usfhn.institutions.annotate_country(df)
    df = usfhn.institutions.annotate_us(df, drop_country_name=False)

    df = df[
        ~df['US']
    ]

    df = df[
        [
            'CountryName',
            'DegreeInstitutionName',
        ]
    ].drop_duplicates().rename(columns={
        'CountryName': 'Country',
        'DegreeInstitutionName': 'Institution',
    })


    df['Updated Institution'] = ''
    df['Remove'] = ''
    df['Updated Country'] = ''

    df = df.sort_values(by=['Country', 'Institution'])

    df.to_csv(
        usfhn.constants.AA_2022_DEGREE_INSTITUTION_REMAP_TO_ANNOTATE_PATH,
        index=False,
    )
