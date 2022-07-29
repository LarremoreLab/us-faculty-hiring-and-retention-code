from pathlib import Path
import importlib
import pandas as pd
import numpy as np
import usfhn.constants

CURRENT_DATASET = None

DATA_VERSION = 2

CRITERIA = {
    'DepartmentsInThresholdYears': True,
    'PrimaryAppointments': True,
    'MinimumTaxonomySize': 3,
    'USPhDGrantingDegreeInstitutions': True,
    'LatestInstitutionEmployment': True,
    'UmbrellaExclusions': [
        'Public Administration and Policy',
        'Journalism, Media, Communication',
    ],
}

DATASETS = {
    'default': CRITERIA,
    'careers': {
        **CRITERIA,
        'LatestInstitutionEmployment': False,
    },
    'dynamics': {
        **CRITERIA,
        'DepartmentsInAllYears': True,
        'DepartmentsInThresholdYears': False,
    },
    'unfiltered-census-pool': {
        **CRITERIA,
        'DepartmentsInThresholdYears': False,
        'UmbrellaExclusions': [],
    },
    'alternate-fields': {
        **CRITERIA,
        'AlternateTaxonomy': True,
    },
    'scieco-data': {},
}


DATASET_OBJECTS = {}


class DataSet(object):
    def __init__(self, name, filters):
        self.name = name
        self.filters = filters

    @property
    def dataset_storage_path(self):
        path = usfhn.constants.DATASETS_PATH.joinpath(self.name)
        if not path.exists():
            path.mkdir()
        return path


    def clean(self):
        paths = [
            self.verbose_data_path,
            self.data_path,
            self.ranks_path,
            self.ranks_by_year_path,
            self.closedness_data_path,
            # self.configuration_models_path,
        ]

        for path in paths:
            if path.exists():
                path.unlink()

    @property
    def verbose_data_path(self):
        return self.dataset_storage_path.joinpath(f"verbose_data.gz")

    @property
    def results_path(self):
        path = self.dataset_storage_path.joinpath(f"results")
        path.mkdir(exist_ok=True, parents=True)
        return path

    @property
    def verbose_data(self):
        if not hasattr(self, '_verbose_data'):
            self._verbose_data = pd.read_csv(self.verbose_data_path, low_memory=False)

        return self._verbose_data.copy()

    @property
    def data_path(self):
        return self.dataset_storage_path.joinpath("data.gz")

    @property
    def table_csvs_path(self):
        path = self.dataset_storage_path.joinpath("table-csvs")
        path.mkdir(exist_ok=True, parents=True)
        return path

    @property
    def gini_subsamples_path(self):
        return self.dataset_storage_path.joinpath(f"gini_subsamples.gz")

    @property
    def gini_subsamples(self):
        if not hasattr(self, '_gini_subsamples'):
            self._gini_subsamples = pd.read_csv(self.gini_subsamples_path)

        return self._gini_subsamples.copy()

    @property
    def placement_predictions_lowess_path(self):
        return self.dataset_storage_path.joinpath('placement_predictions_lowess.gz')

    @property
    def data(self):
        if not hasattr(self, '_data'):
            self._data = pd.read_csv(self.data_path, low_memory=False)

        return self._data.copy()

    @property
    def degrees(self):
        if not hasattr(self, '_degrees'):
            self._degrees = self.data[
                [
                    'PersonId',
                    'DegreeInstitutionId',
                    'DegreeYear',
                ]
            ].drop_duplicates()

        return self._degrees.copy()

    @property
    def closedness_data_path(self):
        """
        there are special criteria for closedness.
        """
        return self.dataset_storage_path.joinpath(f"closedness_data.gz")

    @property
    def closedness_data(self):
        if not hasattr(self, '_closedness_data'):
            self._closedness_data = pd.read_csv(self.closedness_data_path, low_memory=False)

        return self._closedness_data.copy()

    @property
    def figures_path(self):
        path = usfhn.constants.FIGURES_PATH.joinpath(self.name)
        if not path.exists():
            path.mkdir()
        return path

    @property
    def tables_path(self):
        path = self.dataset_storage_path.joinpath('tables')
        if not path.exists():
            path.mkdir()
        return path

    @property
    def stats_path(self):
        path = self.dataset_storage_path.joinpath('stats')
        if not path.exists():
            path.mkdir()
        return path

    @property
    def ranks_path(self):
        return self.dataset_storage_path.joinpath('ranks.gz')

    @property
    def ranks_by_year_path(self):
        return self.dataset_storage_path.joinpath('ranks-by-year.gz')

    @property
    def core_periphery_path(self):
        return self.dataset_storage_path.joinpath('core_periphery.gz')

    @property
    def ranks(self):
        if not hasattr(self, '_ranks'):
            self._ranks = pd.read_csv(self.ranks_path)

        return self._ranks.copy()

    @property
    def ranks_by_year(self):
        if not hasattr(self, '_ranks_by_year'):
            self._ranks_by_year = pd.read_csv(self.ranks_by_year_path)

        return self._ranks_by_year.copy()

    @property
    def configuration_models_path(self):
        path = self.dataset_storage_path.joinpath('configuration_models')
        if not path.exists():
            path.mkdir()
        return path

    @property
    def model_stats_path(self):
        path = self.configuration_models_path.joinpath('model_stats.gz')
        return path

    @property
    def institution_countries(self):
        if not hasattr(self, '_institution_countries'):
            if DATA_VERSION == 1:
                path = usfhn.constants.AA_V1_INSTITUTION_LOCATION_RESULTS_PATH
            elif DATA_VERSION == 2:
                path = usfhn.constants.AA_2022_DEGREE_INSTITUTION_REMAP_INSTITUTION_COUNTRIES_PATH
            
            self._institution_countries = pd.read_csv(path)

        return self._institution_countries.copy()



def set_dataset(dataset_name=None):
    """
    This... deserves comment.

    Here is the problem:

    There are multiple different filters of the data that we want to look at.

    For example, we might be interested in both:
        - an analysis of only those departments that appear all years
        - an analysis of all departments

    This isn't really something that the code that does the analysis needs to be
    aware of, and it shouldn't be aware of it.

    However, that code also uses lru_cache, which means that if we change
    things, we're gonna need to clear the lru_caches.

    This isn't so bad though because _all_ of the functions use a single function:
    `usfhn.dataset.CURRENT_DATASET.data`. This function loads the dataset pointed to by:
    `datasets.CURRENT_DATASET.data_path`

    So this is what we do:
    1. set this file's `CURRENT_DATASET` global variable to the dataset passed.
       Default to the the first item in the `DATASETS` global variable
       (consistent departments)
    2. if this changes the dataset (as opposed to just setting it),
        - for all the python files in the `usfhn` module (except plots):
            - call `cache_clear` on any method that has that attribute
    """
    global CURRENT_DATASET
    global DATASETS

    if not dataset_name:
        dataset_name = get_default_dataset_name()

    assert(dataset_name in DATASETS.keys())

    old_dataset = CURRENT_DATASET
    new_dataset = DataSet(dataset_name, DATASETS[dataset_name])

    module_exclusions = ['plots']

    this_file = Path(__file__).resolve()
    usfhn_modules = [p.stem for p in this_file.parent.glob('*.py') if p != this_file]

    if old_dataset and old_dataset.name != new_dataset.name:
        module_exclusions = ['plots', 'placement_predictor']

        this_file = Path(__file__).resolve()
        usfhn_modules = [p.stem for p in this_file.parent.glob('*.py') if p != this_file]

        exclusions = ['run_mvr']
        usfhn_modules = [module for module in usfhn_modules if module not in exclusions]

        for module in [m for m in usfhn_modules if m not in module_exclusions]:
            try:
                module = importlib.import_module(f"usfhn.{module}")
                for method_name in dir(module):
                    method = getattr(module, method_name)
                    if hasattr(method, 'cache_clear'):
                        method.cache_clear()
            except NameError:
                pass
            except AttributeError:
                pass

    CURRENT_DATASET = DataSet(dataset_name, DATASETS[dataset_name])


def get_dataset_df(df_name, dataset_name=None, by_year=False):
    if not dataset_name:
        if by_year:
            dataset_name = 'dynamics'
        else:
            dataset_name = get_default_dataset_name()

    if dataset_name not in DATASET_OBJECTS:
        DATASET_OBJECTS[dataset_name] = DataSet(dataset_name, DATASETS[dataset_name])

    return getattr(DATASET_OBJECTS[dataset_name], df_name)


def get_dataset_df_via_set_dataset(df_name, dataset_name=None):
    """
    This version uses `set_dataset`, which isn't great, and doesn't really do caching
    """
    if dataset_name:
        set_dataset(dataset_name)

    df = getattr(CURRENT_DATASET, df_name)

    set_dataset()
    return df


def get_default_dataset_name():
    if usfhn.constants.DEFAULT_DATASET_PATH.exists():
        dataset_name = usfhn.constants.DEFAULT_DATASET_PATH.read_text().strip()

        if dataset_name in DATASETS:
            return dataset_name

    return list(DATASETS.keys())[0]


set_dataset()
