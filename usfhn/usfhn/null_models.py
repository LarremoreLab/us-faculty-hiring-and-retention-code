import argparse
import functools
import multiprocessing
import SpringRank
import pandas as pd
import numpy as np
from scipy.sparse import dok_matrix, coo_matrix
from numba import jit
import random
from collections import defaultdict
import time

import usfhn.constants
import usfhn.rank
import usfhn.datasets
import usfhn.plot_utils
import usfhn.fieldwork

################################################################################
# We do some wonky things to save time. It turns out that batching random numbers
# is far more efficient than calling one by one, so what we do is this:
# - make a batch of 500,000,000 random draws.
# - store the last index we've used from it
# - get a random number by doing the following:
#     1. take the first unused random float. Multiply it by a range, convert it to
#        an int
#     2. increment the last used
################################################################################
UNUSED_RANDOM_NUMBER_INDEX = 0
RANDOM_NUMBERS = []
RANDOM_NUMBER_DRAW_SIZE = 10 ** 8


def get_random_index(sequence_length, exclude_value=None):
    """
    The idea behind exclude value is to emulate something like choice without replacement.

    Say you have a sequence of length N.

    You call get_random_index(N, exclude_value=None), you get an index within
    that sequence. Call this index I

    You want to make a second choice, without replacement. (This is my case, but
    the general case of `exclude_values` requires trivial changes.)

    You call get_random_index(N-1, exclude_value=I). If the random index you get
    from this second call is I, then we return N.

    Here are the cases, sequence length 5, possible indexes: [0, 1, 2, 3, 4]:
    - case 1, second draw gets end of sequence:
        - draw 1:
            - args: (5, exclude_value=None)
            - returns: .99 * 5 = 4
        - draw 2:
            - args: (4, exclude_value=4)
            - returns: .99 * 4 = 3
    - case 2, second draw gets excluded value:
        - draw 1:
            - args: (5, exclude_value=None)
            - returns: .99 * 3 = 2
        - draw 2:
            - args: (4, exclude_value=2)
            - returns: .99 * 3 = 2, excluded, so returns 4 (sequence length)

    This ensures that we are sampling randomly.
    """
    global RANDOM_NUMBERS, RANDOM_NUMBER_DRAW_SIZE, UNUSED_RANDOM_NUMBER_INDEX

    if len(RANDOM_NUMBERS) == 0 or UNUSED_RANDOM_NUMBER_INDEX == RANDOM_NUMBER_DRAW_SIZE:
        RANDOM_NUMBERS = draw_random_numbers(RANDOM_NUMBER_DRAW_SIZE)
        UNUSED_RANDOM_NUMBER_INDEX = 0

    index = int(RANDOM_NUMBERS[UNUSED_RANDOM_NUMBER_INDEX] * sequence_length)

    UNUSED_RANDOM_NUMBER_INDEX += 1

    if exclude_value and index == exclude_value:
        index = sequence_length

    return index


@jit(nopython=True)
def draw_random_numbers(size):
    return np.random.random(size=size)



################################################################################
#
#
# Model stuff
#
#
################################################################################
class Model(object):
    def __init__(
        self,
        level,
        level_value,
        draw_count=usfhn.constants.NULL_MODEL_DRAWS,
        year=None,
        exclude_self_loops=True,
    ):
        self.level = level
        self.level_value = level_value
        self.draw_count = draw_count
        self.year = year
        self.exclude_self_loops = exclude_self_loops
        self.set_edges()

    ################################################################################
    # paths shared across the models
    ################################################################################
    @property
    def models_path(self):
        path = usfhn.datasets.CURRENT_DATASET.configuration_models_path
        path.mkdir(exist_ok=True)
        return path

    @property
    def all_edges_path(self):
        path = self.models_path.joinpath('edges')
        path.mkdir(exist_ok=True)
        return path

    @property
    def swap_sequences_path(self):
        path = self.models_path.joinpath('swap_sequences')
        path.mkdir(exist_ok=True)
        return path

    @property
    def edges_at_intervals_path(self):
        path = self.models_path.joinpath('edges_at_intervals')
        path.mkdir(exist_ok=True)
        return path

    @property
    def violations_path(self):
        path = self.models_path.joinpath('violations')
        path.mkdir(exist_ok=True)
        return path

    @property
    def all_gathered_violations_path(self):
        path = self.models_path.joinpath('gathered_violations')
        path.mkdir(exist_ok=True)
        return path

    ################################################################################
    # paths specific to this model
    ################################################################################
    @property
    def model_name(self):
        level_value_string = str(self.level_value).replace(' ', '-')
        level_value_string = level_value_string.replace(',', 'COMMA')

        parts = [
            f'Level-{self.level}',
            f'LevelValue-{level_value_string}'
        ]

        if self.year is not None:
            parts.append(f'Year-{self.year}')

        if not self.exclude_self_loops:
            parts.append(f'Include-self-loops')

        return "_".join(parts)

    def get_qualified_path(self, directory, draw_number=None):
        path = self.model_name

        if draw_number is not None:
            path = f"{path}-{draw_number}"

        path = f"{path}.gz"

        return directory.joinpath(path)

    @property
    def edges_path(self):
        return self.get_qualified_path(self.all_edges_path)

    @property
    def gathered_violations_path(self):
        return self.get_qualified_path(self.all_gathered_violations_path)

    ################################################################################
    # logic functions
    ################################################################################
    def set_edges(self):
        path = self.edges_path

        if not path.exists():
            df = usfhn.datasets.get_dataset_df('data').copy()
            df = df[
                df[self.level] == self.level_value
            ]

            if self.year:
                df = df[
                    df['Year'] == self.year
                ]

            df = df[
                [
                    'DegreeInstitutionId',
                    'InstitutionId',
                    'PersonId',
                ]
            ].drop_duplicates()

            df = usfhn.rank.filter_hiring(df, multigraph=True)

            identifiers = usfhn.utils.identifier_to_integer_id_map(df['InstitutionId'])

            df['i'] = df['DegreeInstitutionId'].apply(identifiers.get)
            df['j'] = df['InstitutionId'].apply(identifiers.get)

            if self.exclude_self_loops:
                df = df[
                    df['i'] != df['j']
                ]

            df.to_csv(path, index=False)

        return pd.read_csv(path)

    @property
    def gathered_violations(self):
        path = self.gathered_violations_path

        if not path.exists():
            dfs = []
            for draw in range(self.draw_count):
                draw_df = pd.read_csv(self.get_qualified_path(self.violations_path, draw))
                draw_df['Draw'] = draw
                dfs.append(draw_df)

            df = pd.concat(dfs)
            df.to_csv(path, index=False)

        return pd.read_csv(path)

    ################################################################################
    # job functions
    ################################################################################
    def get_job_messages(self, message_function):
        return [getattr(self, message_function)(draw) for draw in range(self.draw_count)]

    def model_draw_job_message(self, draw_number):
        return (
            self.level,
            self.level_value,
            draw_number,
            self.edges_path,
            self.get_qualified_path(self.swap_sequences_path, draw_number),
            self.get_qualified_path(self.edges_at_intervals_path, draw_number),
            self.get_qualified_path(self.violations_path, draw_number),
        )

def get_violations(sources, targets):
    """
    i = DegreeInstitutionId
    j = InstitutionId

    so a violation is when Rank[j] > Rank[i]
    """
    matrix = get_matrix_from_sources_and_targets(sources, targets)
    ranks = SpringRank.get_ranks(matrix)
    index_to_rank = {i: r for i, r in enumerate(ranks)}

    df = pd.DataFrame({'DegreeInstitutionId': sources, 'InstitutionId': targets})

    df['DegreeInstitutionRank'] = df['DegreeInstitutionId'].apply(index_to_rank.get)
    df['InstitutionRank'] = df['InstitutionId'].apply(index_to_rank.get)

    df['IsRankViolation'] = df['InstitutionRank'] > df['DegreeInstitutionRank']

    return len(df[df['IsRankViolation'] == True])


def get_matrix_from_sources_and_targets(sources, targets):
    node_count = max(max(sources), max(targets)) + 1
    values = np.ones(len(sources), dtype=np.int64)
    return coo_matrix((values, (sources, targets)), shape=(node_count, node_count))


def draw_configuration_model(
    level,
    level_value,
    draw,
    edges_path,
    swaps_path,
    edges_at_intervals_path,
    violations_path,
):
    if all([p.exists() for p in [edges_path, swaps_path, edges_at_intervals_path, violations_path]]):
        return

    start = time.time()
    edges = pd.read_csv(edges_path)

    step_factor = 1000
    edge_count = len(edges)
    steps = edge_count * step_factor

    edges_at_intervals = np.empty((step_factor, edge_count), dtype=np.int64)
    sources = np.copy(edges['i'].to_numpy())
    targets = np.copy(edges['j'].to_numpy())
    swaps = np.zeros((steps, 2), dtype=np.int64)

    violations = []

    interval = 0
    for step in range(steps):
        double_edge_swap_stub_labelled_multigraph(sources, targets, edge_count, swap=swaps[step])

        if not (step + 1) % edge_count:
            edges_at_intervals[interval] = targets
            violations_count = get_violations(sources, targets)
            violations.append({'Interval': interval, 'NullViolations': violations_count})
            interval += 1

    pd.DataFrame(swaps).to_csv(swaps_path, index=False)
    pd.DataFrame(edges_at_intervals).to_csv(edges_at_intervals_path, index=False)
    pd.DataFrame(violations).to_csv(violations_path, index=False)

    end = time.time()
    print(f"{level}, {level_value}, {draw}: {round(end - start, 5)}")


def double_edge_swap_stub_labelled_multigraph(sources, targets, sequence_length, swap):
    """
    Performs an inplace swap on a stub labelled multigraph.

    Arguments:
        sources: ids of source nodes. length = n
        targets: ids of target nodes. length = n
        sequence_length: number of edges. Doesn't change so don't calculate it over and over.
        reject_self_loops: boolean. Do nothing if the swap will create a self loop
    """
    e1 = get_random_index(sequence_length)
    e2 = get_random_index(sequence_length - 1, exclude_value=e1)
    make_self_loopless_swap(sources, targets, swap, e1, e2)


@jit(nopython=True)
def make_self_loopless_swap(sources, targets, swap, e1, e2):
    s1 = sources[e1]
    s2 = sources[e2]
    t1 = targets[e1]
    t2 = targets[e2]

    if s1 == t2 or s2 == t1:
        e2 = e1
    else:
        targets[e1] = t2
        targets[e2] = t1

    swap[0] = e1
    swap[1] = e2


################################################################################
#
#
#
# Running & etc
#
#
#
################################################################################
def get_models(draw_count=usfhn.constants.NULL_MODEL_DRAWS, year=None, level=None):
    levels = [level] if level else ['Academia', 'Umbrella', 'Area', 'Field']

    df = usfhn.datasets.get_dataset_df('data').copy()
    df = usfhn.fieldwork.linearize_df_taxonomy_hierarchy(df)[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
        ]
    ].drop_duplicates()

    df = df[
        df['TaxonomyLevel'].isin(levels)
    ]

    models = []
    for (level, value), rows in df.groupby(['TaxonomyLevel', 'TaxonomyValue']):
        models.append(Model(level, value, draw_count=draw_count, year=year))

    return models


def get_job_messages(message_function, models, shuffle=True):
    job_messages = []
    for model in models:
        messages = model.get_job_messages(message_function)

        if shuffle:
            random.shuffle(messages)

        job_messages.extend(messages)

    return job_messages


def get_job_from_queue_and_call_function(queue, function):
    function(*queue.get())


def parallelize_jobs(jobs_list, function, cores):
    manager = multiprocessing.Manager()
    queue = manager.Queue()
    pool = multiprocessing.Pool(cores)

    for job in jobs_list:
        queue.put(job)

    readers = []
    for i in range(queue.qsize()):
        readers.append(
            pool.apply_async(get_job_from_queue_and_call_function, (queue, function))
        )

    out = [r.get() for r in readers]

################################################################################
#
# Plot utils/etc
#
################################################################################
@functools.lru_cache()
def get_stats():
    df = pd.read_csv(usfhn.datasets.CURRENT_DATASET.model_stats_path)
    df = df.rename(columns={
        'Level': 'TaxonomyLevel',
        'LevelValue': 'TaxonomyValue',
    })

    df['TaxonomyValue'] = df['TaxonomyValue'].apply(lambda v: 'Academia' if v == 'True' else v)

    # this won't have to be done once we rerun configuration models (20220520)
    df['TaxonomyValue'] = df['TaxonomyValue'].apply(usfhn.utils.rename_general_taxonomy)

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", '-t', default='model_draw', type=str, help="what to do")
    parser.add_argument("--draw_count", '-d', default=usfhn.constants.NULL_MODEL_DRAWS, type=int, help="draws to request")
    parser.add_argument("--year", '-y', default=usfhn.constants.YEAR_UNION, type=int, help="year")
    parser.add_argument("--parallel", '-p', default=True, action='store_false', help="run in parallel")
    parser.add_argument("--cores", '-c', default=multiprocessing.cpu_count(), type=int, help="cores to run on.")
    parser.add_argument("--level", '-l', default=None, type=str, help="levels to run on. If not specified, will run on all levels ('Academia', 'Umbrella', 'Area', 'Field' etc)")
    parser.add_argument("--test", default=False, action='store_true', help="test things; run 1 job")
    args = parser.parse_args()

    if args.test:
        models = [Model('Field', 'Computer Science', draw_count=2, year=0)]
    else:
        models = get_models(args.draw_count, args.year, level=args.level)

    message_function = None

    if args.task == 'model_draw':
        message_function = 'model_draw_job_message'
        job_function = draw_configuration_model
    elif args.task == 'gather_violations':
        for model in models:
            model.gathered_violations

    if message_function:
        jobs = get_job_messages(message_function=message_function, models=models)

        if args.parallel:
            random.shuffle(jobs)
            parallelize_jobs(jobs, job_function, args.cores)
        else:
            for job in jobs:
                job_function(*job)
