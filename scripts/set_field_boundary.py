import argparse
from pathlib import Path
import pandas as pd
from collections import defaultdict

import usfhn
import usfhn.constants as constants


ORDER = [
    "Humanities, General",
    "Performing and Visual Arts, various",
    "Music, General",
    "Music specialties",
    "History",
    "Culture Studies, various",
    "Ancient Studies",
    "American Studies",
    "Asian Studies",
    "European Studies",
    "English Language and Literature",
    "Art History and Criticism",
    "Comparative Literature",
    "Composition, Rhetoric and Writing",
    "Theatre Literature, History and Criticism",
    "Languages, various",
    "Asian Languages",
    "Classics and Classical Languages",
    "French Language and Literature",
    "Germanic Languages and Literatures",
    "Italian Language and Literature",
    "Near and Middle Eastern Languages and Cultures",
    "Slavic Languages and Literatures",
    "Spanish Language and Literature",
    "Linguistics",
    "Philosophy",
    "Religion and Religious Studies",
    "Theology and Theological Studies",
    "Law",
    "Social Sciences, various",
    "Anthropology",
    "Economics, General",
    "Applied Economics",
    "Agricultural Economics",
    "Geography",
    "Gender Studies",
    "Political Science",
    "International Affairs",
    "Psychology, General",
    "Psychology, various",
    "Clinical Psychology",
    "Counseling Psychology",
    "Educational Psychology",
    "School Psychology",
    "Neuroscience",
    "Cognitive Science",
    "Sociology",
    "Criminal Justice and Criminology",
    "Biological Sciences, General",
    "Biological Sciences, various",
    "Biomedical Sciences, General",
    "Biomedical Sciences, various",
    "Anatomy",
    "Biochemistry",
    "Biophysics",
    "Structural Biology",
    "Computational Biology",
    "Biostatistics",
    "Ecology",
    "Evolutionary Biology",
    "Developmental Biology",
    "Entomology",
    "Cell Biology",
    "Microbiology",
    "Molecular Biology",
    "Neuroscience",
    "Cognitive Science",
    "Zoology",
    "Plant Biology",
    "Pathology",
    "Plant Pathology",
    "Chemistry",
    "Chemical Sciences, various",
    "Biochemistry",
    "Biophysics",
    "Structural Biology",
    "Geology",
    "Geophysics",
    "Oceanography",
    "Environmental Sciences",
    "Fisheries Science",
    "Forestry and Forest Resources",
    "Marine Sciences",
    "Natural Resources",
    "Wildlife Science",
    "Physics, General",
    "Geophysics",
    "Applied Physics",
    "Astronomy",
    "Atmospheric Sciences and Meteorology",
    "Computational Sciences, General",
    "Computational Sciences, various",
    "Computer Science",
    "Computer Engineering",
    "Computational Biology",
    "Information Science",
    "Information Technology",
    "Mathematics",
    "Applied Mathematics",
    "Statistics",
    "Biostatistics",
    "Agriculture, various",
    "Agricultural Economics",
    "Agricultural Engineering",
    "Animal Sciences",
    "Agronomy",
    "Horticulture",
    "Plant Sciences",
    "Soil Science",
    "Food Science",
    "Architecture, Design, Planning, various",
    "Architecture",
    "Urban and Regional Planning",
    "Business, various",
    "Accounting",
    "Business Administration",
    "Finance",
    "Management",
    "Management Information Systems",
    "Marketing",
    "Health Professions, various",
    "Consumer Sciences, various",
    "Human Development and Family Sciences, General",
    "Human Development and Family Sciences, various",
    "Communication Disorders and Sciences",
    "Speech and Hearing Sciences",
    "Nursing",
    "Health Promotion, Kinesiology, Exercise Science and Rehab",
    "Health, Physical Education, Recreation",
    "Nutrition Sciences",
    "Public Health",
    "Environmental Health Sciences",
    "Social Work",
    "Medical Sciences, various",
    "Epidemiology",
    "Genetics",
    "Medical Genetics",
    "Molecular Genetics",
    "Immunology",
    "Pharmacology",
    "Pharmacy",
    "Molecular Pharmacology",
    "Oncology",
    "Oral Biology and Craniofacial Science",
    "Pharmaceutical Sciences",
    "Physiology",
    "Toxicology",
    "Veterinary Medical Sciences",
    "Engineering, General",
    "Engineering, various",
    "Engineering Mechanics",
    "Aerospace Engineering",
    "Agricultural Engineering",
    "Biomedical Engineering",
    "Chemical Engineering",
    "Civil Engineering",
    "Electrical Engineering",
    "Computer Engineering",
    "Environmental Engineering",
    "Geological Engineering",
    "Industrial Engineering",
    "Materials Science and Engineering",
    "Mechanical Engineering",
    "Nuclear Engineering",
    "Operations Research",
    "Systems Engineering",
    "Communication",
    "Mass Communications and Media Studies",
    "Public Administration",
    "Public Policy",
    "Education, General",
    "Curriculum and Instruction",
    "Foundations of Education",
    "Education Research",
    "Educational Psychology",
    "Education Administration",
    "Higher Education Administration",
    "Teacher Education Specific Levels",
    "Teacher Education Specific Subject Areas",
    "Counselor Education",
    "Mathematics Education",
    "Science Education",
    "Special Education",
]

if __name__ == '__main__':
    """
    1. What fraction of Univs have a dept with this label?
    2. How often do Univ who have ≥1 such dept have >1 dept with this label?
    3. If we build a FHN at this level, what fraction of hires come from within the label? (closedness)
    4. How big is this field (number of faculty, number of departments)
    """
    parser = argparse.ArgumentParser(description='annotate the taxonomization')
    parser.add_argument("--path", default='annotated_taxonomization.csv', type=str, help="where to put it")
    
    args = parser.parse_args()

    df = usfhn.dataframes.employment.merge(
        usfhn.dataframes.departments.drop(columns=['InstitutionId', 'InstitutionName', 'DepartmentName']),
        on='DepartmentId'        
    )

    df = df[
        ['PersonId', 'DepartmentId', 'InstitutionId', 'Taxonomy']
    ].drop_duplicates()

    df['InstitutionCount'] = df.groupby('Taxonomy')['InstitutionId'].transform('nunique')
    df['DepartmentCount'] = df.groupby('Taxonomy')['DepartmentId'].transform('nunique')
    df['PersonCount'] = df.groupby('Taxonomy')['PersonId'].transform('nunique')

    df['DepartmentTaxonomyCount'] = df.groupby('DepartmentId')['Taxonomy'].transform('nunique')

    df['InstitutionDepartmentCount'] = df.groupby(
        ['Taxonomy', 'InstitutionId']
    )['DepartmentId'].transform('nunique')

    df = df.merge(
        usfhn.dataframes.degrees[['PersonId', 'DegreeInstitutionId']],
        on='PersonId',
        how='left',
    )

    level_to_department_multi_taxonomy_count = defaultdict(int)
    level_to_multi_institution_count = defaultdict(int)
    level_to_closedness = defaultdict(float)
    for taxonomy, rows in df.groupby('Taxonomy'):
        level_to_multi_institution_count[taxonomy] = rows[
            rows['InstitutionDepartmentCount'] > 1
        ]['InstitutionId'].nunique()

        level_to_department_multi_taxonomy_count[taxonomy] = rows[
            rows['DepartmentTaxonomyCount'] > 1
        ]['DepartmentId'].nunique()

        # Closedness
        rows_with_degrees = rows[
            rows['DegreeInstitutionId'].notnull()
        ]

        closedness_denominator = rows_with_degrees['PersonId'].nunique()
        closedness_numerator = rows_with_degrees[
            rows_with_degrees['DegreeInstitutionId'].isin(rows['InstitutionId'].unique())
        ]['PersonId'].nunique()

        if closedness_numerator and closedness_denominator:
            level_to_closedness[taxonomy] = round(closedness_numerator / closedness_denominator, 2)

    df['MultiDepartmentInstitutionCount'] = df['Taxonomy'].apply(level_to_multi_institution_count.get)
    df['MultiTaxonomyDepartmentCount'] = df['Taxonomy'].apply(level_to_department_multi_taxonomy_count.get)
    df['Closedness'] = df['Taxonomy'].apply(level_to_closedness.get)

    df = df.drop(columns=[
        'InstitutionDepartmentCount', 'DepartmentTaxonomyCount',
        'InstitutionId', 'DegreeInstitutionId', 'DepartmentId', 'PersonId'
    ]).drop_duplicates()

    df['SingleTaxonomyDepartmentCount'] = df['DepartmentCount'] - df['MultiTaxonomyDepartmentCount']

    ordering_df = pd.DataFrame(
        [(i + 1, taxonomy) for i, taxonomy in enumerate(ORDER)],
        columns=['Order', 'Taxonomy']
    )

    df = ordering_df.merge(df, on=['Taxonomy']).sort_values(by='Order')

    df.to_csv(Path.cwd().joinpath(args.path), index=False)
