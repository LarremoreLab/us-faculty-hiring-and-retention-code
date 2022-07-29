from pathlib import Path
from functools import cached_property
from webweb import Web
import networkx as nx
import pandas as pd

import hnelib.pandas

import usfhn.constants
import usfhn.views
import usfhn.institutions


def get_geodesics_df():
    df = usfhn.stats.runner.get('basics/faculty-hiring-network')[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
        ]
    ].drop_duplicates()

    geodesics_dfs = []
    for (level, value), _ in df.groupby(['TaxonomyLevel', 'TaxonomyValue']):
        print(f"{level}: {value}")
        full_homeland_colony_web = FullHomelandColonyWeb(level=level, value=value)
        geodesics_dfs.append(full_homeland_colony_web.subwebs_df)

    return pd.concat(geodesics_dfs)


class FullHomelandColonyWeb(object):
    def __init__(self, level='Academia', value='Academia'):
        self.level = level
        self.value = value

    @cached_property
    def data(self):
        import usfhn.stats
        df = usfhn.stats.runner.get('basics/faculty-hiring-network')[
            [
                'TaxonomyLevel',
                'TaxonomyValue',
                'InstitutionId',
                'DegreeInstitutionId',
            ]
        ].drop_duplicates()

        ranks = usfhn.stats.runner.get('ranks/df', rank_type='prestige')[
            [
                'TaxonomyLevel',
                'TaxonomyValue',
                'InstitutionId',
                'OrdinalRank',
                'Percentile',
            ]
        ].drop_duplicates().rename(columns={
            'InstitutionId': 'DegreeInstitutionId'
        })

        df = df.merge(
            ranks,
            on=[
                'TaxonomyLevel',
                'TaxonomyValue',
                'DegreeInstitutionId',
            ]
        )


        df = usfhn.institutions.annotate_institution_name(df)
        df = usfhn.institutions.annotate_institution_name(df, 'DegreeInstitutionId', 'DegreeInstitutionName')
        df = df[
            df['DegreeInstitutionId'].isin(df['InstitutionId'].unique())
        ]

        df = usfhn.views.filter_by_taxonomy(df, self.level, self.value)

        return df

    @cached_property
    def subwebs_df(self):
        subweb_dfs = []

        for degree_institution_id in set(self.data['DegreeInstitutionId'].unique()):
            subweb = HomelandColonyWeb(
                self.data,
                self.level,
                self.value,
                degree_institution_id=degree_institution_id,
            )
            subweb_dfs.append(subweb.nodes_df)

        df = pd.concat(subweb_dfs)

        # graph might not be strongly connected, so just use max path length
        df['Diameter'] = max(df['PathLength'])
        df['FractionalPathLength'] = df['PathLength'] / df['Diameter']
        df['MeanFractionalPathLength'] = df.groupby(
            'DegreeInstitutionId'
        )['FractionalPathLength'].transform('mean')

        df = hnelib.pandas.annotate_25th_and_75th_percentiles(
            df,
            'FractionalPathLength',
            ['DegreeInstitutionId'],
        )

        df['OtherCount'] = df.groupby('DegreeInstitutionId')['OtherInstitutionId'].transform('nunique')

        df = df[
            df['OtherCount'] > 5
        ].drop(columns=['OtherCount'])

        df['TaxonomyLevel'] = self.level
        df['TaxonomyValue'] = self.value

        return df


    @cached_property
    def nx_G(self):

        edges = []
        for i, row in self.data.iterrows():
            edges.append([row['DegreeInstitutionId'], row['InstitutionId']])

        G = nx.DiGraph()
        G.add_edges_from(edges)

        return G


class HomelandColonyWeb(object):
    def __init__(self, df, level, value, rank_percentile=None, degree_institution_id=None):
        self.df = df
        self.level = level
        self.value = value

        if degree_institution_id:
            self.degree_institution_id = degree_institution_id
        elif rank_percentile:
            self.degree_institution_id = self.get_degree_institution_id_by_rank_percentile(rank_percentile)

    @staticmethod
    def selenium_driver():
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--window-size=2880x1800")

        driver = webdriver.Chrome(
            chrome_options=chrome_options,
            executable_path=usfhn.constants.CHROMEDRIVER_PATH,
        )

        return driver

    @property
    def png_path(self):
        value_string = value.replace(' ', '-')
        value_string = value.replace(',', '')

        return f"{level}_{value}_{rank_percentile}.png"


    def get_degree_institution_id_by_rank_percentile(self, rank_percentile):
        df = self.df.copy()
        df['DifferenceFromPercentile'] = df['Percentile'] - rank_percentile
        df['DifferenceFromPercentile'] = df['DifferenceFromPercentile'].apply(abs)
        df = df.sort_values(by=['DifferenceFromPercentile'])

        df = df.drop(columns=['DifferenceFromPercentile'])

        return df.iloc[0]['DegreeInstitutionId']

    @cached_property
    def shortest_paths(self):
        edges = []
        for i, row in self.df.iterrows():
            edges.append([row['DegreeInstitutionId'], row['InstitutionId']])

        G = nx.DiGraph()
        G.add_edges_from(edges)

        return nx.single_source_shortest_path(G, self.degree_institution_id)

    @property
    def edges(self):
        sssp_edges = set()
        for path in self.shortest_paths.values():
            for i, j in zip(path[:-1], path[1:]):
                sssp_edges.add((i, j))

        edges = [[int(i), int(j)] for i, j in sssp_edges]
        return edges


    @property
    def nodes_df(self):
        nodes = []
        for node, path in self.shortest_paths.items():
            nodes.append({
                'OtherInstitutionId': node,
                'DegreeInstitutionId': self.degree_institution_id,
                'PathLength': len(path),
            })

        df = self.df.copy()[
            [
                'DegreeInstitutionId',
                'Percentile',
            ]
        ].drop_duplicates()

        df = df.merge(
            pd.DataFrame(nodes),
            on='DegreeInstitutionId',
        )

        return df

    @property
    def nodes(self):
        nodes = {}
        for node, path in self.shortest_paths.items():
            nodes[int(node)] = {
                'pathLength': len(path) - 1,
                'name': usfhn.institutions.get_institution_id_to_name_map()[node],
            }

        return nodes

    def web(self):
        web = Web(adjacency=self.edges, nodes=self.nodes)
        web.display.colorNodesBy = 'PathLength'
        web.show()
