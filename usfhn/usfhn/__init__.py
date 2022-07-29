# from functools import cached_property
import pandas as pd

import usfhn.constants as constants

class DataFrameAccessor(object):
    def __init__(self):
        self.conditions = {
            'Institution': {
                'PhDGranting': True,
            },
            'Person': {
                'TenureTrack': True,
            },
            'Department': {
                'AllYears': False,
            },
            'DegreeDepartment': {
                'Employing': True
            },
        }

    # @cached_property
    def degrees(self):
        """
        returns a DataFrame with the following columns:
        - PersonId: Int.
        - PersonName: String.
        - Degree: String.
        - DegreeYear: Int. Unknowns are zeros.
        - DegreeInstitutionId: Int. Non-employing institutions may be null here.
        - DegreeInstitutionName: String. Non-employing institutions may be null here.

        Columns do not contain nulls unless stated.

        There is one row per PersonId.
        """
        df = pd.read_csv(constants.AA_CLEAN_PEOPLE_MASTER_DEGREES_PATH).drop_duplicates(
            subset='PersonId'
        )

        df = df.merge(
            self.people,
            on='PersonId'
        )

        if self.conditions['DegreeDepartment']['Employing']:
            df = df[
                df['DegreeInstitutionId'].isin(self.institutions.InstitutionId.unique())
            ]


        return df

    # @cached_property
    def employment(self):
        """
        This is the _cleaned_ but filtered set of employment rows:
        - institutions are US, PhD granting
        - people are tenured or tenure track.

        returns a dataframe with the following columns:
        - PersonId
        - PersonName
        - Year
        - Rank
        - DepartmentId
        - DepartmentName
        - InstitutionId
        - InstitutionName

        All rows here are primary appointments.

        """
        df = pd.read_csv(constants.AA_PRIMARY_APPOINTED_EMPLOYMENT_PATH)

        df = df.merge(
            self.institutions,
            on='InstitutionId'
        )

        df = df.merge(
            self.people,
            on=['PersonId', 'Year', 'Rank']
        )

        df = df.merge(
            self.departments,
            on=['DepartmentId', 'InstitutionId', 'Year']
        )

        return df

    # @cached_property
    def people(self):
        """
        Columns:
        - PersonId
        - PersonName
        - Rank
        - Year
        - TenureTrack
        - Gender (later)
        """
        df = pd.read_csv(constants.AA_PRIMARY_APPOINTED_PEOPLE_PATH)

        if self.conditions['Person']['TenureTrack']:
            df = df[
                df['TenureTrack']
            ]

        return df
        
    @property
    def institutions(self):
        """
        Columns:
        - InstitutionId: Int.
        - InstitutionName: String.
        - PhDGranting: boolean.

        This is generated in `analyses/cleaning/f_employing_institution_annotation.py`
        """
        df = pd.read_csv(constants.AA_EMPLOYING_INSTITUTION_FILTERED_INSTITUTIONS_PATH)

        if self.conditions['Institution']['PhDGranting']:
            df = df[
                df['PhDGranting']
            ]

        return df

    # @cached_property
    def departments(self):
        """
        Columns:
        - DepartmentId: Int.
        - DepartmentName: String.
        - InstitutionId: Int.
        - InstitutionName: String.
        - Subfield
        - Field
        - Area
        - Umbrella

        This is generated in `analyses/cleaning/g_field_annotation.py`
        """
        return pd.read_csv(constants.AA_FIELD_DEFINED_DEPARTMENTS_PATH)


dataframes = DataFrameAccessor()
