from pathlib import Path

################################################################################
#
#
# Basics
#
#
################################################################################
ROOT = Path(__file__).absolute().parent.parent.parent

DATA_PATH = ROOT.joinpath('data')
OUTPUT_PATH = ROOT.joinpath('output')
FIGURES_PATH = DATA_PATH.joinpath('figures')
PDF_PATH = FIGURES_PATH.joinpath('pdf')

################################################################################
#
#
# Academic Analytics Data
#
#
################################################################################
AA_PATH = DATA_PATH.joinpath('aa')

# raw
AA_RAW_PATH = AA_PATH.joinpath('raw')
FACULTY_LISTS_PATH = AA_RAW_PATH.joinpath('faculty_lists')
DEGREES_PATH = AA_RAW_PATH.joinpath('degree_year_and_place')
META_PATH = AA_PATH.joinpath('meta')

AA_FACULTY_GENDER_2022_DROP_PATH = AA_PATH.joinpath('AAD2020-1969_UniqueFacGender.csv')

AA_MASTER_DEGREES_PATH = AA_RAW_PATH.joinpath('degrees-master.csv')
AA_PRIMARY_APPOINTMENTS_PATH = AA_RAW_PATH.joinpath('primary_appointments.csv')
AA_APPOINTMENT_PROVIDING_INSTITUTIONS_PATH = AA_RAW_PATH.joinpath('primary_appointment_institutions.csv')
SCHOOLS_WITH_PRIMARY_APPOINTMENTS_PATH = AA_PATH.joinpath('schools_with_primary_appointments.csv')

# institution-cleaned
AA_CLEAN_INSTITUTIONS_PATH = AA_PATH.joinpath('clean-institutions')
AA_CLEAN_INSTITUTIONS_EMPLOYMENT_PATH = AA_CLEAN_INSTITUTIONS_PATH.joinpath('employment.gz')
AA_CLEAN_INSTITUTIONS_DEGREES_PATH = AA_CLEAN_INSTITUTIONS_PATH.joinpath('degrees.gz')
AA_CLEAN_INSTITUTIONS_MASTER_DEGREES_PATH = AA_CLEAN_INSTITUTIONS_PATH.joinpath('degrees-master.gz')
AA_CLEAN_INSTITUTIONS_ALIASES_PATH = AA_CLEAN_INSTITUTIONS_PATH.joinpath('aliases.json')
AA_CLEAN_INSTITUTIONS_NAMELESS_DEGREES_PATH = AA_CLEAN_INSTITUTIONS_PATH.joinpath('nameless-degrees.gz')

# department-cleaned
AA_CLEAN_DEPARTMENTS_PATH = AA_PATH.joinpath('clean-departments')
AA_CLEAN_DEPARTMENTS_EMPLOYMENT_PATH = AA_CLEAN_DEPARTMENTS_PATH.joinpath('employment.gz')
AA_CLEAN_DEPARTMENTS_ALIASES_PATH = AA_CLEAN_DEPARTMENTS_PATH.joinpath('aliases.json')

# department events/transitions
DEPARTMENT_EVENTS_PATH = DATA_PATH.joinpath('department_events')
RAW_DEPARTMENT_EVENTS_PATH = DEPARTMENT_EVENTS_PATH.joinpath('raw')
ANNOTATED_DEPARTMENT_EVENTS_PATH = DEPARTMENT_EVENTS_PATH.joinpath('annotated')

# people-cleaned
AA_CLEAN_PEOPLE_PATH = AA_PATH.joinpath('clean-people')
AA_CLEAN_PEOPLE_EMPLOYMENT_PATH = AA_CLEAN_PEOPLE_PATH.joinpath('employment.gz')
AA_CLEAN_PEOPLE_DEGREES_PATH = AA_CLEAN_PEOPLE_PATH.joinpath('degrees.gz')
AA_CLEAN_PEOPLE_MASTER_DEGREES_PATH = AA_CLEAN_PEOPLE_PATH.joinpath('degrees-master.gz')
AA_CLEAN_PEOPLE_STILL_UNCLEAN = AA_CLEAN_PEOPLE_PATH.joinpath('unclean.gz')

# taxonomy-cleaned
AA_TAXONOMY_CLEANED_PATH = AA_PATH.joinpath('clean-taxonomy')
AA_TAXONOMY_CLEANED_DEPARTMENTS_PATH = AA_TAXONOMY_CLEANED_PATH.joinpath('departments.gz')

# primary appointment annotated
AA_PRIMARY_APPOINTED_PATH = AA_PATH.joinpath('primary-appointed')
AA_PRIMARY_APPOINTED_EMPLOYMENT_PATH = AA_PRIMARY_APPOINTED_PATH.joinpath('employment.gz')
AA_PRIMARY_APPOINTED_REMAINDER_PATH = AA_PRIMARY_APPOINTED_PATH.joinpath('remainder.gz')
AA_PRIMARY_APPOINTED_PEOPLE_PATH = AA_PRIMARY_APPOINTED_PATH.joinpath('people.gz')
AA_PRIMARY_APPOINTED_PEOPLE_WHO_WHERE_DROPPED = AA_PRIMARY_APPOINTED_PATH.joinpath('non-primary-people.gz')

# employment-annotated
AA_EMPLOYING_INSTITUTION_FILTERED_PATH = AA_PATH.joinpath('employing-institution-filtered')
AA_EMPLOYING_INSTITUTION_FILTERED_INSTITUTIONS_PATH = AA_EMPLOYING_INSTITUTION_FILTERED_PATH.joinpath('employment.gz')

# taxonomy-cleaned
AA_FIELD_DEFINED_PATH = AA_PATH.joinpath('field-defined')
AA_FIELD_DEFINED_DEPARTMENTS_PATH = AA_FIELD_DEFINED_PATH.joinpath('departments.gz')
AA_FIELD_DEFINED_ALL_DEPARTMENTS_PATH = AA_FIELD_DEFINED_PATH.joinpath('all_departments.gz')
AA_FIELD_DEFINED_TAXONOMIZATION_PATH = AA_FIELD_DEFINED_PATH.joinpath('taxonomization.gz')

# people-gendering
AA_PEOPLE_GENDERED_PATH = AA_PATH.joinpath('people-gendered')
AA_PEOPLE_GENDERED_OLD_EMPLOYMENT_PEOPLE_PATH = AA_PEOPLE_GENDERED_PATH.joinpath('old_people.gz')
AA_PEOPLE_GENDERED_INTERIM_PEOPLE_PATH = AA_PEOPLE_GENDERED_PATH.joinpath('interim.gz')
AA_PEOPLE_GENDERED_PEOPLE_PATH = AA_PEOPLE_GENDERED_PATH.joinpath('people.gz')

AA_ETHNICITIES = AA_PATH.joinpath('people-ethnicities')
AA_RAW_ETHNICITIES = AA_ETHNICITIES.joinpath('20210223-ethnicities.gz')
AA_PEOPLE_ETHNICITIES = AA_ETHNICITIES.joinpath('people.gz')

################################################################################
#
#
# AA 2022 Refresh
#
#
################################################################################
AA_2022_REFRESH_PATH = AA_PATH.joinpath('2022-refresh')
AA_2022_REFRESH_FACULTY_PATH = AA_2022_REFRESH_PATH.joinpath('2022.03.11.Faculty.csv')
AA_2022_REFRESH_CIP_CODES_PATH = AA_2022_REFRESH_PATH.joinpath('2022.03.11.CIP Codes.csv')

AA_2022_BASIC_CLEANING_PATH = AA_2022_REFRESH_PATH.joinpath('basic-cleaning')
AA_2022_BASIC_CLEANING_EMPLOYMENT_PATH = AA_2022_BASIC_CLEANING_PATH.joinpath('employment.gz')
AA_2022_BASIC_CLEANING_DEGREE_INSTITUTIONS_PATH = AA_2022_BASIC_CLEANING_PATH.joinpath('degree-institutions.gz')
AA_2022_BASIC_CLEANING_INSTITUTIONS_PATH = AA_2022_BASIC_CLEANING_PATH.joinpath('institutions.gz')

AA_2022_TENURE_TRACK_FILTERED_PATH = AA_2022_REFRESH_PATH.joinpath('tenure-track-filtered')
AA_2022_TENURE_TRACK_FILTERED_EMPLOYMENT_PATH = AA_2022_TENURE_TRACK_FILTERED_PATH.joinpath('employment.gz')

# degree institution remapping (this will probably go somewhere else)
AA_2022_DEGREE_INSTITUTION_REMAP_PATH = AA_2022_REFRESH_PATH.joinpath('degree-institution-remap')
AA_2022_DEGREE_INSTITUTION_REMAP_TO_ANNOTATE_PATH = AA_2022_DEGREE_INSTITUTION_REMAP_PATH.joinpath('to-annotate.csv')
AA_2022_DEGREE_INSTITUTION_REMAP_ANNOTATED_PATH = AA_2022_DEGREE_INSTITUTION_REMAP_PATH.joinpath('annotated.csv')
AA_2022_DEGREE_INSTITUTION_REMAP_EMPLOYMENT_PATH = AA_2022_DEGREE_INSTITUTION_REMAP_PATH.joinpath('employment.gz')
AA_2022_DEGREE_INSTITUTION_REMAP_INSTITUTION_COUNTRIES_PATH = AA_2022_DEGREE_INSTITUTION_REMAP_PATH.joinpath('institution-countries.csv')
AA_2022_DEGREE_INSTITUTION_REMAP_INSTITUTIONS_PATH = AA_2022_DEGREE_INSTITUTION_REMAP_PATH.joinpath('institutions.csv')
AA_2022_DEGREE_INSTITUTION_REMAP_DEGREE_INSTITUTIONS_PATH = AA_2022_DEGREE_INSTITUTION_REMAP_PATH.joinpath('degree-institutions.csv')
AA_2022_DEGREE_INSTITUTION_REMAP_CANONICAL_INSTITUTION_NAMES = AA_2022_DEGREE_INSTITUTION_REMAP_PATH.joinpath('canonical-institution-names.csv')

# degree filtered
AA_2022_DEGREE_FILTERED_PATH = AA_2022_REFRESH_PATH.joinpath('degree-filtered')
AA_2022_DEGREE_FILTERED_EMPLOYMENT_PATH = AA_2022_DEGREE_FILTERED_PATH.joinpath('employment.gz')
AA_2022_DEGREE_FILTERED_DEGREES_PATH = AA_2022_DEGREE_FILTERED_PATH.joinpath('degrees.gz')
AA_2022_DEGREE_FILTERED_NON_DOCTORAL_DEGREES_PATH = AA_2022_DEGREE_FILTERED_PATH.joinpath('non-doctoral-degrees.gz')
AA_2022_DEGREE_FILTERED_STATS_PATH = AA_2022_DEGREE_FILTERED_PATH.joinpath('stats.json')

AA_2022_DEPARTMENT_CLEANING_PATH = AA_2022_REFRESH_PATH.joinpath('department-cleaning')
AA_2022_DEPARTMENT_EVENTS_TO_ANNOTATE_PATH = AA_2022_DEPARTMENT_CLEANING_PATH.joinpath('events-to-annotate.csv')
AA_2022_ANNOTATED_DEPARTMENT_EVENTS_PATH = AA_2022_DEPARTMENT_CLEANING_PATH.joinpath('annotated-events.csv')
AA_2022_DEPARTMENT_CLEANED_EMPLOYMENT_PATH = AA_2022_DEPARTMENT_CLEANING_PATH.joinpath('employment.gz')
AA_2022_DEPARTMENT_CLEANED_EMPLOYMENT_V2_PATH = AA_2022_DEPARTMENT_CLEANING_PATH.joinpath('employment-v2.gz')

AA_2022_TAXONOMY_CLEANED_PATH = AA_2022_REFRESH_PATH.joinpath('taxonomy-cleaned')
AA_2022_TAXONOMY_CLEANED_DEPARTMENTS_PATH = AA_2022_TAXONOMY_CLEANED_PATH.joinpath('departments.gz')
AA_2022_TAXONOMY_CLEANED_EMPLOYMENT_PATH = AA_2022_TAXONOMY_CLEANED_PATH.joinpath('employment.gz')

AA_2022_MULTI_INSTITUTION_FILTERED_PATH = AA_2022_REFRESH_PATH.joinpath('multi-institution-filtered')
AA_2022_MULTI_INSTITUTION_FILTERED_EMPLOYMENT_PATH = AA_2022_MULTI_INSTITUTION_FILTERED_PATH.joinpath('employment.gz')
AA_2022_MULTI_INSTITUTION_FILTERED_STATS_PATH = AA_2022_MULTI_INSTITUTION_FILTERED_PATH.joinpath('stats.json')

# people-gendering
AA_2022_PEOPLE_GENDERED_PATH = AA_2022_REFRESH_PATH.joinpath('people-gendered')
AA_2022_PEOPLE_GENDERED_OLD_EMPLOYMENT_PEOPLE_PATH = AA_2022_PEOPLE_GENDERED_PATH.joinpath('old_people.gz')
AA_2022_PEOPLE_GENDERED_INTERIM_PEOPLE_PATH = AA_2022_PEOPLE_GENDERED_PATH.joinpath('interim.gz')
AA_2022_PEOPLE_GENDERED_PEOPLE_PATH = AA_2022_PEOPLE_GENDERED_PATH.joinpath('people.gz')

# people-imputed
AA_2022_PEOPLE_IMPUTED_PATH = AA_2022_REFRESH_PATH.joinpath('people-imputed')
AA_2022_PEOPLE_IMPUTED_EMPLOYMENT_PATH = AA_2022_PEOPLE_IMPUTED_PATH.joinpath('employment.gz')
AA_2022_PEOPLE_IMPUTED_STATS_PATH = AA_2022_PEOPLE_IMPUTED_PATH.joinpath('stats.json')

# primary appointment annotated
AA_2022_PRIMARY_APPOINTED_PATH = AA_2022_REFRESH_PATH.joinpath('primary-appointed')
AA_2022_PRIMARY_APPOINTED_EMPLOYMENT_PATH = AA_2022_PRIMARY_APPOINTED_PATH.joinpath('employment.gz')
AA_2022_PRIMARY_APPOINTED_PEOPLE_WHO_WHERE_DROPPED = AA_2022_PRIMARY_APPOINTED_PATH.joinpath('non-primary-people.gz')
AA_2022_PRIMARY_APPOINTED_STATS_PATH = AA_2022_PRIMARY_APPOINTED_PATH.joinpath('stats.json')

# field_defined
AA_2022_FIELD_DEFINED_PATH = AA_2022_REFRESH_PATH.joinpath('field-defined')
AA_2022_FIELD_DEFINED_DEPARTMENTS_PATH = AA_2022_FIELD_DEFINED_PATH.joinpath('departments.gz')
AA_2022_FIELD_DEFINED_TAXONOMIZATION_PATH = AA_2022_FIELD_DEFINED_PATH.joinpath('taxonomization.gz')
AA_2022_FIELD_DEFINED_ALTERNATE_TAXONOMIZATION_PATH = AA_2022_FIELD_DEFINED_PATH.joinpath('field-diagnostics.gz')

# multi-career-move employments
AA_2022_MULTI_CAREER_MOVE_ANNOTATED_PATH = AA_2022_REFRESH_PATH.joinpath('multi-career-move-annotated')
AA_2022_LAST_EMPLOYING_INSTITUTION_ANNOTATIONS_PATH = AA_2022_MULTI_CAREER_MOVE_ANNOTATED_PATH.joinpath('people-institutions.gz')
AA_2022_MULTI_CAREER_MOVE_STATS_PATH = AA_2022_MULTI_CAREER_MOVE_ANNOTATED_PATH.joinpath('stats.json')
################################################################################
#
#
# DATASETS: filtered subsets of the full dataset (or not filtered at all)
#
#
################################################################################
DATASETS_PATH = DATA_PATH.joinpath('datasets')

################################################################################
#
#
# Taxonomization
#
#
################################################################################
TAXONOMIZATION_PATH = DATA_PATH.joinpath('taxonomization')
AA_RAW_TAXONOMIES_PATH = TAXONOMIZATION_PATH.joinpath('aa_raw.csv')
AA_TAXONOMY_RENAMES_PATH = TAXONOMIZATION_PATH.joinpath('aa_renames.csv')
FTAS_TAXONOMY_HIERARCHY_PATH = TAXONOMIZATION_PATH.joinpath('ftas_taxonomy_hierarchy.csv')
CLEAN_FTAS_TAXONOMY_HIERARCHY_PATH = TAXONOMIZATION_PATH.joinpath('clean-ftas-taxonomy-hierarchy.csv')

################################################################################
#
#
# Visualization Data
#
#
################################################################################
VISUALIZATIONS_PATH = DATA_PATH.joinpath('visualization')

VISUALIZATION_TREES_PATH = VISUALIZATIONS_PATH.joinpath('trees')
VISUALIZATION_SANKEYS_PATH = VISUALIZATIONS_PATH.joinpath('sankeys')

################################################################################
#
#
# Field Age
#
#
################################################################################
FIELD_AGE_PATH = DATA_PATH.joinpath('field-age')

FIELD_JOURNAL_AGES = FIELD_AGE_PATH.joinpath('field-journal-ages.csv')

################################################################################
#
#
# WoS PhD departments
#
#
################################################################################
DEPARTMENT_LEVEL_FACULTY_HIRING_PATH = DATA_PATH.joinpath('department_level_faculty_hiring')
PHD_DEPARTMENTS_PATH = DEPARTMENT_LEVEL_FACULTY_HIRING_PATH.joinpath('phd_departments.gz')
PHD_LEVEL_RANKS_PATH = DEPARTMENT_LEVEL_FACULTY_HIRING_PATH.joinpath('ranks.gz')

################################################################################
#
#
# Gender
#
#
################################################################################
GENDER_PATH = DATA_PATH.joinpath('gender')
OFFLINE_GENDER_PATH = GENDER_PATH.joinpath('offline.csv')
GENNI_GENDER_PATH = GENDER_PATH.joinpath('genni.csv')
KNOWN_GENDER_PATH = GENDER_PATH.joinpath('known.csv')
UNKNOWN_GENDER_PATH = GENDER_PATH.joinpath('unknown.csv')

################################################################################
#
#
# Rankings
#
#
################################################################################
RANKING_PATH = DATA_PATH.joinpath('rankings')

################################################################################
#
#
# Stats
#
#
################################################################################
STATS_PATH = DATA_PATH.joinpath('stats')

################################################################################
#
#
# Misc
#
#
################################################################################
# ... aa to wos url remaps
AA_TO_WOS_URL_REMAPPED_DEPARTMENTS_PATH = DATA_PATH.joinpath('aa-to-wos-url-remapped-departments.csv')

FIELD_DEFINITIONS_PATH = DATA_PATH.joinpath('field_definitions.json')

HARDCODED_GENDERS_PATH = DATA_PATH.joinpath('allFaculty_NameGenderDict.json')

SCORECARD_DATA_PATH = DATA_PATH.joinpath("Most-Recent-Cohort-All-Data-Elements.gz")
SCORECARD_CROSSWALK_WITH_DELETIONS_PATH = DATA_PATH.joinpath("AA-scorecard-crosswalk-google.csv")
SCORECARD_CROSSWALK_PATH = DATA_PATH.joinpath("AA-scorecard-crosswalk-final.csv")
SCORECARD_DATA_URL = "https://ed-public-download.app.cloud.gov/" +\
    "downloads/Most-Recent-Cohorts-All-Data-Elements.csv"

################################################################################
#
#
# Variables
#
#
################################################################################
YEAR_UNION = 0
TAXONOMY_UNION = True
TAXONOMY_HIERARCHY_LEVELS = ['Taxonomy', 'Field', 'Area', 'Umbrella', 'Academia']
GENDER_AGNOSTIC = 'All'
GENDER_KNOWN = 'Known'
GENDER_UNKNOWN = 'Unknown'

NULL_MODEL_DRAWS = 1000

# consider attritions for people with careers long than this to be 'retirements';
# younger than this, 'exits'
CAREER_LENGTH_THRESHOLD = 15

FIELD_CRITERIA = {
    'InstitutionThreshold': .25,
    'PersonThreshold': 1000,
    'ClosednessThreshold': .5,
    'ClosedPeopleThreshold': 500,
    'ExcludeVarious': True,
    'TaxonomyExclusions': [
        'Teacher Education Specific Levels',
        'Teacher Education Specific Subjects',
    ],
}

RANKS = ['Assistant Professor', 'Associate Professor', 'Professor']

YEAR_REQUIREMENT = 5

MAX_YEARS_BETWEEN_DEGREE_AND_CAREER_START = 4
MAX_DEGREE_TO_FIRST_JOB_YEAR_GAP = 3

EARLIEST_DEGREE_YEAR = 1980

################################################################################
# AA Constants
################################################################################
RANK_TYPE_ID_TO_RANK = {
    0: "Unknown",
    1: "Professor",
    2: "Associate Professor",
    3: "Assistant Professor",
    4: "Instructor",
    5: "Lecturer",
    9: "Other",
}

ARTS_HUMANITIES_FIELDS = [
    'English Language and Literature',
    'Art History and Criticism',
    'Music',
    'Theatre Literature, History and Criticism',
]

################################################################################
#
#
# Misc
#
#
################################################################################
SCIECO_DATA_PATH = ROOT.parent.joinpath('scieco-data')
SCIECO_DATA_AA_PATH = SCIECO_DATA_PATH.joinpath('data', 'aa', '2022')
SCIECO_DATASET = 'scieco-data'

IPEDS_DATA_PATH = DATA_PATH.joinpath('ipeds')
IPEDS_TO_AA_COMPARISON = IPEDS_DATA_PATH.joinpath('ipeds-aa-differences.csv')

RANK_PREDICTION_PATH = DATA_PATH.joinpath('rank-prediction')
RANK_PREDICTION_EARLY_TO_LATE_PATH = RANK_PREDICTION_PATH.joinpath('early-to-late.gz')

DEPARTMENT_DROPPING_DIAGNOSTIC_CSV_PATH = ROOT.joinpath('department_dropping_diagnostic_csv_path.csv')

DEFAULT_DATASET_PATH = DATA_PATH.joinpath('default-dataset.txt')

################################################################################
# Paper stuff
################################################################################
PAPER_PATH = ROOT.joinpath('paper')
PAPER_FIGURES_PATH = PAPER_PATH.joinpath('figures')
PAPER_TABLES_PATH = PAPER_PATH.joinpath('tables')
PAPER_TEX_PATH = PAPER_PATH.joinpath('main.tex')
PAPER_STATS_PATH = PAPER_PATH.joinpath('stats.tex')

################################################################################
#
#
# Institution location stuff
#
#
################################################################################
INSTITUTION_LOCATION_PATH = DATA_PATH.joinpath('institution-location')

# https://www.ef.com/wwen/epi/
INSTITUTION_LOCATION_COUNTRY_ENGLISH_PROFICIENCIES_PATH = INSTITUTION_LOCATION_PATH.joinpath('english-proficiencies.csv')

INSTITUTION_LOCATION_COUNTRIES_PATH = INSTITUTION_LOCATION_PATH.joinpath('countries.csv')
INSTITUTION_LOCATION_COUNTRY_ALPHA_CODES_PATH = INSTITUTION_LOCATION_PATH.joinpath('country-alpha3-codes.csv')
INSTITUTION_LOCATION_COUNTRY_CONTINENTS_PATH = INSTITUTION_LOCATION_PATH.joinpath('country-continents.csv')

# AA-v1
AA_V1_INSTITUTION_LOCATION_PATH = INSTITUTION_LOCATION_PATH.joinpath('aa-v1')
AA_V1_INSTITUTION_LOCATION_TEST_QUERIES_PATH = AA_V1_INSTITUTION_LOCATION_PATH.joinpath('test_queries.csv')
AA_V1_INSTITUTION_LOCATION_QUERIES_PATH = AA_V1_INSTITUTION_LOCATION_PATH.joinpath('queries.csv')
AA_V1_INSTITUTION_LOCATION_TEST_QUERIES_RESULTS_PATH = AA_V1_INSTITUTION_LOCATION_PATH.joinpath('test_batch_results.csv')
AA_V1_INSTITUTION_LOCATION_OTHER_QUERIES_RESULTS_PATH = AA_V1_INSTITUTION_LOCATION_PATH.joinpath('queries_results.csv')
AA_V1_INSTITUTION_LOCATION_HAND_ANNOTATED_PATH = AA_V1_INSTITUTION_LOCATION_PATH.joinpath('hand_annotated.csv')
AA_V1_INSTITUTION_LOCATION_DIRTY_IN_PROGRESS_PATH = AA_V1_INSTITUTION_LOCATION_PATH.joinpath('dirty_in_progress.csv')
AA_V1_INSTITUTION_LOCATION_RESULTS_PATH = AA_V1_INSTITUTION_LOCATION_PATH.joinpath('results.csv')

# AA-2022 institution location
AA_2022_INSTITUTION_LOCATION_PATH = INSTITUTION_LOCATION_PATH.joinpath('aa-2022')
AA_2022_INSTITUTION_LOCATION_QUERIES_PATH = AA_2022_INSTITUTION_LOCATION_PATH.joinpath('queries.csv')
AA_2022_INSTITUTION_LOCATION_QUERIES_RESULTS_PATH = AA_2022_INSTITUTION_LOCATION_PATH.joinpath('queries_results.csv')
AA_2022_INSTITUTION_LOCATION_HAND_ANNOTATED_PATH = AA_2022_INSTITUTION_LOCATION_PATH.joinpath('hand_annotated.csv')
AA_2022_INSTITUTION_LOCATION_DIRTY_IN_PROGRESS_PATH = AA_2022_INSTITUTION_LOCATION_PATH.joinpath('dirty_in_progress.csv')
AA_2022_INSTITUTION_LOCATION_RESULTS_PATH = AA_2022_INSTITUTION_LOCATION_PATH.joinpath('results.csv')

################################################################################
#
#
# Data Sharing
#
#
################################################################################
DATA_SHARING_ROOT = ROOT.parent.joinpath('us-faculty-hiring-networks', 'data')

################################################################################
#
#
# Nature submission
#
#
################################################################################
NATURE_SUB_3_PATH = Path('/Users/hne/Dropbox/Hunter Faculty Hiring/Submission 4 - Final')
FIGURES_NATURE_SUB_3_PATH = NATURE_SUB_3_PATH.joinpath('figures')
FIGURES_DATA_NATURE_SUB_3_PATH = NATURE_SUB_3_PATH.joinpath('figures-data')

################################################################################
#
#
# visualization data
#
#
################################################################################
VISUALIZATION_DATA_PATH = DATA_PATH.joinpath('visualizations')
WEBS_PATH = VISUALIZATION_DATA_PATH.joinpath('webs')
HOMELAND_COLONY_WEBS = WEBS_PATH.joinpath('homeland-colony-webs')

MISC_PATH = DATA_PATH.joinpath('misc')
CHROMEDRIVER_PATH = MISC_PATH.joinpath('chromedriver')

################################################################################
#
#
# School visualization data path
#
#
################################################################################
SCHOOL_COMPARISON_VISUALIZATION_DATA_PATH = ROOT.parent.joinpath('us-faculty-networks-school-comparison', 'data.json')
