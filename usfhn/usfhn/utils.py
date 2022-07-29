import numpy as np
import math
from collections import defaultdict


def rename_general_taxonomy(taxonomy):
    if not isinstance(taxonomy, str):
        return ''
    
    return taxonomy.replace(', General', '')


def print_cleaning_step_start(step):
    step_row_whitespace = 80 - len(step)
    pre_string = " " * (step_row_whitespace // 2)
    print("################################################################################")
    print(f"{pre_string}{step}")
    print("################################################################################")


def clean_institution_name_for_permissive_joining(string):
    """
    lowercase and make institution name strings simpler for more permissive joining
    """

    string = string.lower()

    things_to_remove = [
        'the',
        'of',
        ',',
        "'",

    ]

    for thing in things_to_remove:
        string = string.replace(thing, '')

    # collapse multiple spaces into one
    string = " ".join([w for w in string.split(' ') if len(w)])

    return string


def identifier_to_integer_id_map(identifiers):
    """
    maps each element in a list-like object to a unique 0-indexed integer. 

    in: a list like object
    out: dict. 
        - keys: each unique value in `identifiers`
        - values: a unique 0-indexed integer.  
    """
    identifiers = sorted(list(set(identifiers)))
    return {identifier: i for i, identifier in enumerate(identifiers)}


def convert_hiring_df_to_matrix(
    df,
    row_column='DegreeInstitutionId',
    column_column='InstitutionId',
    value_column='Count',
):
    identifiers = identifier_to_integer_id_map(
        set(df[row_column]) | set(df[column_column])
    )

    df['i'] = df['DegreeInstitutionId'].apply(identifiers.get)
    df['j'] = df['InstitutionId'].apply(identifiers.get)

    matrix = np.zeros((len(identifiers), len(identifiers)), dtype=int)

    for row, column, value in zip(df[row_column], df[column_column], df[value_column]):
        row_index = identifiers[row]
        column_index = identifiers[column]
        matrix[row_index, column_index] = value

    return matrix, identifiers


def get_values_at_index_percentiles(data, percentiles):
    """
    data should be sorted.

    returns list:
    [
        (index, index_value)
    ]
    """
    has_100 = False
    if 100 in percentiles:
        percentiles = [p for p in percentiles if p != 100]
        has_100 = True

    indices = [math.floor(i) for i in np.percentile(range(len(data) + 1), percentiles)]
    values = [data[i] for i in indices]

    if has_100:
        indices.append(len(data) - 1)
        values.append(data[-1])

    return list(zip(indices, values))


class LabelledMatrix(object):
    def __init__(self, edges):
        self.edges = edges
        name_to_index = defaultdict(lambda: len(name_to_index))
        self.name_to_index = name_to_index

        self.matrix = self.make_matrix(self.edges)

        self.index_to_name = {v: k for k, v in self.name_to_index.items()}

    def make_matrix(self, edges):
        nodes = set()
        for i, j, w in edges:
            nodes.update({i, j})

        matrix = np.zeros(shape=(len(nodes), len(nodes)), dtype=float)

        for i, j, w in edges:
            matrix[self.name_to_index[i], self.name_to_index[j]] = w

        return matrix
