import pandas as pd
from pathlib import Path

from webweb import Web

import hnelib.utils

import usfhn.datasets
import usfhn.measurements
import usfhn.views
import usfhn.institutions
import usfhn.fieldwork


BASE_PATH_TO_SAVE_TO = Path("/Users/hne/Documents/research/LarremoreLab.github.io/us-faculty/")
# BASE_URL = "https://larremorelab.github.io/us-faculty/"
BASE_URL = "./"
# print('debugging')
# BASE_URL = "http://127.0.0.1:4000/us-faculty-networks/"


def get_data():
    df = usfhn.datasets.CURRENT_DATASET.data[
        ['InstitutionId', 'DegreeInstitutionId', 'Taxonomy', 'PersonId']
    ].drop_duplicates()

    df = usfhn.fieldwork.linearize_df_taxonomy_hierarchy(df)
    df = df[
        [
            'PersonId',
            'InstitutionId',
            'DegreeInstitutionId',
            'TaxonomyLevel',
            'TaxonomyValue',
        ]
    ]

    df = df[
        df['TaxonomyLevel'].isin(['Umbrella', 'Field'])
    ]

    df['TaxonomyLevel'] = df['TaxonomyLevel'].apply(lambda l: 'Domain' if l == 'Umbrella' else 'Field')

    ranks = usfhn.datasets.CURRENT_DATASET.ranks
    ranks = usfhn.views.filter_exploded_df(ranks)[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'InstitutionId',
            'OrdinalRank',
        ]
    ].drop_duplicates()

    df = df.merge(
        ranks,
        on=[
            'TaxonomyLevel',
            'TaxonomyValue',
            'InstitutionId',
        ],
    ).merge(
        ranks.rename(columns={
            'InstitutionId': 'DegreeInstitutionId',
            'OrdinalRank': 'DegreeOrdinalRank',
        }),
        on=[
            'TaxonomyLevel',
            'TaxonomyValue',
            'DegreeInstitutionId',
        ],
    )

    # remove self hires for the visualization
    # df = df[
    #     df['InstitutionId'] != df['DegreeInstitutionId']
    # ]

    df = df[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'DegreeInstitutionId',
            'InstitutionId',
            'PersonId',
            'OrdinalRank',
            'DegreeOrdinalRank',
        ]
    ].drop_duplicates()

    df = usfhn.institutions.annotate_institution_name(df)
    df = usfhn.institutions.annotate_institution_name(df, 'DegreeInstitutionId', 'DegreeInstitutionName')

    return df, ranks


def collapse_small_nodes(df, threshold=1):
    """
    threshold is % through the lorenz curve
    """
    df = df.copy()

    institutions = set(df['DegreeInstitutionId'].unique()) | set(df['InstitutionId'].unique())

    df['OrdinalFraction'] = df['OrdinalRank'] / max(df['OrdinalRank'])

    to_collapse = set(df[
        # df['ProductionFractionCumSum'] > threshold
        df['OrdinalFraction'] > threshold
    ]['DegreeInstitutionId'].unique())

    institutions = {i: -1 if i in to_collapse else i for i in institutions}

    df['DegreeInstitutionId'] = df['DegreeInstitutionId'].apply(institutions.get)
    df['InstitutionId'] = df['InstitutionId'].apply(institutions.get)

    ordinal_rank_map = {i: o for i, o in zip(df['DegreeInstitutionId'], df['OrdinalRank'])}
    ordinal_rank_map[-1] = max(df['OrdinalRank']) + 1

    df['OrdinalRank'] = df['DegreeInstitutionId'].apply(ordinal_rank_map.get)

    institution_names = {i: n for i, n in zip(df['InstitutionId'], df['InstitutionName'])}
    institution_names[-1] = f'lower {100 * (1 - threshold)}%'

    df['InstitutionName'] = df['InstitutionId'].apply(institution_names.get)

    degree_institution_names = {i: n for i, n in zip(df['DegreeInstitutionId'], df['DegreeInstitutionName'])}
    degree_institution_names[-1] = f'lower {100 * (1 - threshold)}%'

    df['DegreeInstitutionName'] = df['DegreeInstitutionId'].apply(degree_institution_names.get)

    return df


def make_edges(df):
    df = df.copy()

    df['Placements'] = df.groupby([
        'InstitutionId',
        'DegreeInstitutionId',
    ])['PersonId'].transform('nunique')

    df['OutDegree'] = df.groupby([
        'DegreeInstitutionId',
    ])['PersonId'].transform('nunique')

    df = df[
        [
            'InstitutionId',
            'InstitutionName',
            'DegreeInstitutionId',
            'DegreeInstitutionName',
            'OrdinalRank',
            'DegreeOrdinalRank',
            'Placements',
            'OutDegree',
        ]
    ].drop_duplicates()

    out_df = df[
        [
            'DegreeInstitutionId',
            'DegreeOrdinalRank',
            'OutDegree',
        ]
    ].drop_duplicates()

    out_df = out_df.sort_values(by='DegreeOrdinalRank')

    out_df['CumSum'] = out_df['OutDegree'].cumsum()
    out_df['CumSumFraction'] = out_df['CumSum'] / sum(out_df['OutDegree'])
    out_df['CumSumPercent'] = out_df['CumSumFraction'].apply(lambda f: hnelib.utils.fraction_to_percent(f, 1))

    out_df = out_df[
        [
            'DegreeInstitutionId',
            'CumSumPercent',
        ]
    ].drop_duplicates()

    in_df = df[
        [
            'InstitutionId',
            'OrdinalRank',
        ]
    ].drop_duplicates()

    degree_institution_to_cum_sum_percent = out_df.set_index('DegreeInstitutionId')['CumSumPercent'].to_dict()

    out_institutions = set(out_df['DegreeInstitutionId'])
    institution_to_cum_sum_percent = {}
    for i, row in in_df.sort_values(by=['OrdinalRank']).iterrows():
        inst_id = row['InstitutionId']
        if inst_id in out_institutions:
            cum_sum_percent = degree_institution_to_cum_sum_percent[inst_id]
        else:
            preceding_inst_id = in_df[
                in_df['OrdinalRank'] == row['OrdinalRank'] - 1
            ].iloc[0]['InstitutionId']
            cum_sum_percent = institution_to_cum_sum_percent[preceding_inst_id]

        institution_to_cum_sum_percent[inst_id] = cum_sum_percent

    df = df.sort_values(by='DegreeOrdinalRank', ascending=False).merge(
        out_df.rename(columns={
            'CumSumPercent': 'DegreeCumSumPercent',
        }),
        on='DegreeInstitutionId',
        how='left',
    )

    df['CumSumPercent'] = df['InstitutionId'].apply(institution_to_cum_sum_percent.get)

    # df['CumSumPercent'] = df['CumSumPercent'].fillna(100)
    df['InstitutionId'] = df['InstitutionId'].apply(int)
    df['DegreeInstitutionId'] = df['DegreeInstitutionId'].apply(int)

    return df


def get_nodes(df):
    nodes = {}

    for i, row in df.iterrows():
        degree_name = f"{row['DegreeOrdinalRank'] + 1}. {row['DegreeInstitutionName']}"
        degree_name += f" ({row['DegreeCumSumPercent']}%)"
        nodes[row['DegreeInstitutionId']] = {
            'name': degree_name,
            'OrdinalRank': row['DegreeOrdinalRank'],
            'OutDegree': row['OutDegree'],
        }

        if row['InstitutionId'] not in nodes:
            name = f"{row['OrdinalRank'] + 1}. {row['InstitutionName']}"
            name += f" ({row['CumSumPercent']}%)"
            nodes[row['InstitutionId']] = {
                'name': name,
                'OrdinalRank': row['OrdinalRank'],
                'OutDegree': .01,
            }

    return nodes


def scale_edges(df):
    df = df.copy()[
        [
            'InstitutionId',
            'InstitutionName',
            'DegreeInstitutionId',
            'DegreeInstitutionName',
            'OrdinalRank',
            'DegreeOrdinalRank',
            'Placements',
        ]
    ].drop_duplicates()

    df = df.merge(
        df.copy().rename(columns={
            'InstitutionId': 'DegreeInstitutionId',
            'InstitutionName': 'DegreeInstitutionName',
            'DegreeInstitutionId': 'InstitutionId',
            'DegreeInstitutionName': 'InstitutionName',
            'OrdinalRank': 'DegreeOrdinalRank',
            'DegreeOrdinalRank': 'OrdinalRank',
            'Placements': 'CounterPlacements',
        }), 
        on=[
            'InstitutionId',
            'InstitutionName',
            'DegreeInstitutionId',
            'DegreeInstitutionName',
            'OrdinalRank',
            'DegreeOrdinalRank',
        ],
        how='outer',
    )

    df['Placements'] = df['Placements'].fillna(0)
    df['CounterPlacements'] = df['CounterPlacements'].fillna(0)

    seen_edges = set()

    edges = []
    for i, row in df.sort_values(by='DegreeOrdinalRank').iterrows():
        placements = int(row['Placements'])
        counter_placements = int(row['CounterPlacements'])

        institution_name = row['InstitutionName']
        degree_institution_name = row['DegreeInstitutionName']

        inst_pair_set = (institution_name, degree_institution_name)

        if inst_pair_set in seen_edges:
            continue
        else:
            seen_edges.add(inst_pair_set)

        if institution_name == degree_institution_name:
            total_placements = placements
            value = 0
        else:
            total_placements = placements + counter_placements

            # we might have to reverse this (as in I might have the ordering wrong)
            if placements > counter_placements:
                value = hnelib.utils.fraction_to_percent(placements / total_placements)
            else:
                value = -1 * (hnelib.utils.fraction_to_percent(counter_placements / total_placements))

        string_parts = []

        if placements:
            string_parts.append(f'{degree_institution_name} ➞ {institution_name}: {placements}')

        if counter_placements:
            string_parts.append(f'{institution_name} ➞ {degree_institution_name}: {counter_placements}')

        string = "\n".join(string_parts)

        edges.append((
            row['DegreeInstitutionId'],
            row['InstitutionId'],
            total_placements,
            {
                'direction': value,
                'mouseOverLabel': string
            },
        ))

    return edges


def encode_field_string_for_url(string):
    string = string.lower()
    string = string.replace(',', '')
    string = string.replace(' ', '-')
    return string


def get_url_for_field_string(string):
    string = encode_field_string_for_url(string)
    url = f'{BASE_URL}{string}'
    return url


def get_web_preamble_html(field_names, current_field):
    options = []
    for field_name in field_names:
        option_attributes = [f'href={get_url_for_field_string(field_name)}']

        if field_name == current_field:
            option_attributes.append("Selected=true")

        option_attributes_string = " ".join(option_attributes)

        options.append(f'<option {option_attributes_string}>{field_name}</option>')

    options_string = "\n".join(options)


    html = "<p>don't know what you're looking at? "
    html+= "<a href='https://larremorelab.github.io/faculty/tutorial.html'>open tutorial in new window</a></p>"

    html += """displaying faculty hiring networks from <select onchange="
        function openLink(html) {
            var selectedIndex = html.options.selectedIndex
            var option = html.options[selectedIndex]
            var href = option.attributes.href.nodeValue

            window.location.href = href
        }
        openLink(this)"
    >
    """

    html += options_string
    html += """
    </select>
    """

    return html


def set_web_parameters(web):
    web.display.plotType = 'Chord Diagram'
    web.display.sizeNodesBy = 'OutDegree'
    web.display.sortNodesBy = 'OrdinalRank'
    web.display.edgeColorAttribute = 'direction'
    web.display.edgeColorPalette = 'RdBu'
    web.display.edgeColorFlip = True
    web.display.showNodeNames = True
    web.display.edgeColorRange = [.125, .875]
    # web.display.widgetsToShowByKey = ['networkName', 'preamble']
    web.display.widgetsToShowByKey = ['preamble']

    web.display.chordDiagramEdgeLegend = {
        'location': [-1, 1],
        'colorBox': {
            'width': 25,
            'height': 15,
        },
        'data': [
            {'value': .875, 'text': '100% up the hierarchy'},
            {'value': .6875, 'text': '50% up the hierarchy'},
            {'value': .5, 'text': 'equal exchange'},
            {'value': .3125, 'text': '50% down the hierarchy'},
            {'value': .125, 'text': '100% down the hierarchy'},
        ],
        'pad': {
            'x': 10,
            'y': 10,
        },
    }


if __name__ == '__main__':
    df, ranks = get_data()
    df = usfhn.views.filter_by_taxonomy(df, level='Field')

    # print('testing on cs')
    # df = usfhn.views.filter_by_taxonomy(df, level='Field', value='Computer Science')

    fields = sorted(list(df['TaxonomyValue'].unique()))

    for (level, value), _df in df.groupby(['TaxonomyLevel', 'TaxonomyValue']):
        from_ranks_ordinals = set(usfhn.views.filter_by_taxonomy(ranks, level, value)['OrdinalRank'].unique())
        _df = make_edges(_df)

        nodes = get_nodes(_df)

        _df['FullDegreeInstitutionName'] = _df['DegreeInstitutionId'].apply(lambda i: nodes[i]['name'])
        _df['FullInstitutionName'] = _df['InstitutionId'].apply(lambda i: nodes[i]['name'])

        edges = scale_edges(_df)

        field_name = value.replace(', General', '')

        web = Web(adjacency=edges, nodes=nodes)
        set_web_parameters(web)
        web.display.preamble = get_web_preamble_html(fields, value)

        path = BASE_PATH_TO_SAVE_TO.joinpath(encode_field_string_for_url(value) + ".html")
        web.save(path)

        if value == 'Information Science':
            path = BASE_PATH_TO_SAVE_TO.joinpath("index.html")
            web.save(path)

    # web.save('/Users/hne/Documents/research/LarremoreLab.github.io/us-faculty.html')
    # web.show()
