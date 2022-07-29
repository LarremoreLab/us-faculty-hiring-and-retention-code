import string
import copy
import random
from pathlib import Path
from collections import defaultdict, Counter
from matplotlib.patches import Patch
from matplotlib.colors import to_rgba
import matplotlib.patches as patches
import matplotlib.ticker as mticker
from matplotlib.path import Path as MPath
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.sparse import dok_matrix
from scipy.stats import gaussian_kde, pearsonr
import argparse
import functools
import itertools
import math
import matplotlib.axes
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.proportion import proportion_confint
import seaborn as sns
import palettable
import matplotlib as mpl

import hnelib.plot
import hnelib.pandas
import hnelib.runner

import usfhn
import usfhn.utils
import usfhn.constants as constants
import usfhn.measurements as measurements
import usfhn.views as views
import usfhn.plot_utils as plot_utils
from usfhn.plot_utils import PLOT_VARS
from usfhn.standard_plots import plot_univariate_value_across_taxonomy_levels
import usfhn.datasets
import usfhn.null_models
import usfhn.self_hiring
import usfhn.steeples
import usfhn.closedness
import usfhn.core_periphery
import usfhn.changes
import usfhn.paper_stats
import usfhn.non_us
import usfhn.stats
import usfhn.institutions
import usfhn.self_hiring


def wrap_subplots(function, set_methods):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        fig, axes = function(*args, **kwargs)

        axes_list = axes.reshape(-1) if isinstance(axes, np.ndarray) else [axes]

        for ax in axes_list:
            for set_method in set_methods:
                getattr(ax, set_method['method'])(**set_method['kwargs'])

        return fig, axes

    return wrapper


# Set up matplotlib stylings
@functools.lru_cache(maxsize=1)
def style_matplotlib():
    global PLOT_VARS
    for artist_name, artist_settings in PLOT_VARS['artists'].items():
        for (namespace, method) in artist_settings['functions']:
            partial = functools.partialmethod(
                getattr(namespace, method),
                **artist_settings['kwargs'],
            )

            setattr(namespace, method, partial)

    plt.subplots = wrap_subplots(
        plt.subplots, PLOT_VARS['subplots_set_methods'])

    return


################################################################################
#
#
#
# Production
#
#
#
################################################################################
def plot_degree_breakdown(
    parent_ax=None,
    shrink_ax=True,
    annotation_fontsize=hnelib.plot.FONTSIZES['annotation'],
):
    axis_fontsize = hnelib.plot.FONTSIZES['axis']
    tick_fontsize = hnelib.plot.FONTSIZES['tick']
    field_fontsize = 5
    y_pad = .75
    y_pad_half = .45


    if not parent_ax:
        fig, parent_ax = plt.subplots(figsize=(2.625, 3), tight_layout=True)

    df = usfhn.closedness.get_closednesses()
    df = df[
        df['TaxonomyLevel'].isin(['Academia', 'Umbrella'])
    ][
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'NonUSPhD',
            'NonPhD',
            'FacultyCount'
        ]
    ].drop_duplicates()

    df['NonPhD'] = df['NonPhD'] / df['FacultyCount']
    df['NonPhD'] *= 100
    df['NonUSPhD'] = df['NonUSPhD'] / df['FacultyCount']
    df['NonUSPhD'] *= 100
    df['USPhD'] = 100 - df['NonPhD'] - df['NonUSPhD']

    df = views.annotate_umbrella_color(df, 'Umbrella')

    umbrellas = ['Academia'] + usfhn.views.get_umbrellas()
    umbrellas = list(reversed(umbrellas))

    us_phds = []
    non_us_phds = []
    non_phds = []
    colors = []
    indexes = [i for i in range(len(umbrellas))]

    for umbrella in umbrellas:
        row = df[
            df['TaxonomyValue'] == umbrella
        ].iloc[0]
        us_phds.append(row['USPhD'])
        non_us_phds.append(row['NonUSPhD'])
        non_phds.append(row['NonPhD'])
        colors.append(row['UmbrellaColor'])

    if shrink_ax:
        hnelib.plot.hide_axis(parent_ax)

        ax = parent_ax.inset_axes((0, 0, .9, .9))
    else:
        ax = parent_ax

    ax.set_xlim(0, 100.25)
    ax.set_xticks([0, 20, 40, 60, 80, 100.25])
    ax.set_xticklabels([0, 20, 40, 60, 80, 100])
    ax.set_xlabel('% of faculty')
    ax.set_yticks([])

    left = np.zeros(len(indexes))

    for x in [20, 40, 60, 80]:
        ax.plot(
            [x, x],
            [0, max(indexes)],
            color=PLOT_VARS['colors']['dark_gray'],
            zorder=1,
            alpha=.5,
            lw=.75,
        )

    ax.set_ylim(min(indexes) - .5, max(indexes) + .5)

    ax.barh(
        indexes,
        us_phds,
        color='w',
        edgecolor=colors,
        zorder=2,
    )

    left += np.array(us_phds)

    ax.barh(
        indexes,
        non_us_phds,
        left=left,
        color='w',
        edgecolor=colors,
        zorder=2,
    )

    ax.barh(
        indexes,
        non_us_phds,
        left=left,
        color=hnelib.plot.set_alpha_on_colors(colors, .35),
        edgecolor=colors,
        zorder=2,
    )

    left += np.array(non_us_phds)

    ax.barh(
        indexes,
        non_phds,
        color='w',
        edgecolor=colors,
        hatch='xx',
        left=left,
        zorder=2,
    )

    for index, umbrella, color in zip(indexes, umbrellas, colors):
        ax.annotate(
            plot_utils.clean_taxonomy_string(umbrella),
            (5, index), 
            ha='left',
            va='center',
            color=color,
            fontsize=field_fontsize,
        )

    y = max(indexes) + .4

    padded_y = y + y_pad
    half_padded_y = y + y_pad_half

    academia_row = df[df['TaxonomyValue'] == 'Academia'].iloc[0]
    us_phd =  academia_row['USPhD']
    non_us_phd =  academia_row['NonUSPhD']
    no_phd =  academia_row['NonPhD']

    no_doctorate_x = us_phd + non_us_phd + (no_phd / 2)

    us_doctorate_x = 35

    non_us_doctorate_x = us_phd + (non_us_phd  / 2)
    non_us_doctorate_text_x = us_phd
    non_us_doctorate_text_x = np.mean([no_doctorate_x, us_doctorate_x])

    annotations = [
        {
            'text': 'no\ndoctorate',
            'xy_start': (no_doctorate_x, y),
            'xy_end': (no_doctorate_x, padded_y),
        },
        {
            'text': 'non-U.S.\ndoctorate',
            'xy_start': (non_us_doctorate_x, y),
            'xy_end': (non_us_doctorate_x, half_padded_y),
            'xy_text': (non_us_doctorate_text_x, padded_y),
            'arrowprops': {
                **hnelib.plot.BASIC_ARROW_PROPS,
                'shrinkA': 0,
            },
        },
        {
            'text': 'U.S.\ndoctorate',
            'xy_start': (us_doctorate_x, y),
            'xy_end': (us_doctorate_x, padded_y),
        },
    ]

    for annotation in annotations:
        arrowprops = annotation.get('arrowprops', hnelib.plot.BASIC_ARROW_PROPS)
        ax.annotate(
            '',
            xy=annotation['xy_start'],
            xytext=annotation['xy_end'],
            arrowprops=arrowprops,
            annotation_clip=False,
        )

        ax.annotate(
            annotation['text'],
            xy=annotation.get('xy_text', annotation['xy_end']),
            va=annotation.get('va', 'bottom'),
            ha=annotation.get('ha', 'center'),
            annotation_clip=False,
            fontsize=annotation_fontsize,
        )

    ax.annotate(
        "",
        xy=(non_us_doctorate_x, half_padded_y),
        xytext=(non_us_doctorate_text_x, half_padded_y),
        arrowprops=hnelib.plot.HEADLESS_ARROW_PROPS,
        annotation_clip=False,
    )

    ax.annotate(
        "",
        xy=(non_us_doctorate_text_x, half_padded_y),
        xytext=(non_us_doctorate_text_x, padded_y),
        arrowprops={
            **hnelib.plot.HEADLESS_ARROW_PROPS,
            'shrinkA': .2,
        },
        annotation_clip=False,
    )

    hnelib.plot.finalize(
        ax, 
        axis_fontsize=axis_fontsize,
        tick_fontsize=tick_fontsize,
    )

    return ax


def plot_degree_breakdown_no_deg_us_deg_non_us_deg(parent_ax=None):
    y_pad = .65

    if not parent_ax:
        fig, parent_ax = plt.subplots(figsize=(2.625, 3), tight_layout=True)
        y_pad = 1.2

    df = usfhn.closedness.get_closednesses()
    df = df[
        df['TaxonomyLevel'].isin(['Academia', 'Umbrella'])
    ][
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'NonUSPhD',
            'NonPhD',
            'FacultyCount'
        ]
    ].drop_duplicates()

    df['NonPhD'] = df['NonPhD'] / df['FacultyCount']
    df['NonPhD'] *= 100
    df['NonUSPhD'] = df['NonUSPhD'] / df['FacultyCount']
    df['NonUSPhD'] *= 100
    df['USPhD'] = 100 - df['NonPhD'] - df['NonUSPhD']

    df = views.annotate_umbrella_color(df, 'Umbrella')

    umbrellas = ['Academia'] + usfhn.views.get_umbrellas()
    umbrellas = list(reversed(umbrellas))

    us_phds = []
    non_us_phds = []
    non_phds = []
    colors = []
    indexes = [i for i in range(len(umbrellas))]

    for umbrella in umbrellas:
        row = df[
            df['TaxonomyValue'] == umbrella
        ].iloc[0]
        us_phds.append(row['USPhD'])
        non_us_phds.append(row['NonUSPhD'])
        non_phds.append(row['NonPhD'])
        colors.append(row['UmbrellaColor'])

    data_df = usfhn.views.filter_to_academia_and_domain(df)
    data_df = usfhn.plot_utils.annotate_color(data_df)
    data_df = data_df[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'NonUSPhD',
            'USPhD',
            'NonPhD',
            'Color',
        ]
    ].drop_duplicates()

    plot_data = []
    element_id = 0
    for i, row in data_df.iterrows():
        for col in data_df.columns:
            plot_data.append({
                'Element': element_id,
                'Attribute': col,
                'Value': row[col],
            })

        element_id += 1


    hnelib.plot.hide_axis(parent_ax)

    ax = parent_ax.inset_axes((0, 0, .9, .9))

    ax.set_xlim(0, 100.25)
    ax.set_xticks([0, 20, 40, 60, 80, 100.25])
    ax.set_xticklabels([0, 20, 40, 60, 80, 100])
    ax.set_xlabel('% of faculty')
    ax.set_yticks([])

    left = np.zeros(len(indexes))

    for x in [20, 40, 60, 80]:
        ax.plot(
            [x, x],
            [0, max(indexes)],
            color=PLOT_VARS['colors']['dark_gray'],
            zorder=1,
            alpha=.5,
            lw=.5,
        )

    ax.set_ylim(min(indexes) - .5, max(indexes) + .5)

    ax.barh(
        indexes,
        non_phds,
        color=hnelib.plot.set_alpha_on_colors(colors, .35),
        edgecolor=colors,
        left=left,
        zorder=2,
        lw=.5,
    )

    left += np.array(non_phds)

    ax.barh(
        indexes,
        us_phds,
        color='w',
        edgecolor=colors,
        zorder=2,
        left=left,
        lw=.5,
    )

    left += np.array(us_phds)

    ax.barh(
        indexes,
        non_us_phds,
        left=left,
        color='w',
        hatch='xxx',
        edgecolor=colors,
        zorder=2,
        lw=.5,
    )

    for index, umbrella, color in zip(indexes, umbrellas, colors):
        ax.annotate(
            plot_utils.clean_taxonomy_string(umbrella),
            (50, index), 
            ha='center',
            va='center',
            color=color,
            fontsize=hnelib.plot.FONTSIZES['medium'],
        )

    y = max(indexes) + .4

    padded_y = y + y_pad

    academia_row = df[df['TaxonomyValue'] == 'Academia'].iloc[0]
    us_phd =  academia_row['USPhD']
    non_us_phd =  academia_row['NonUSPhD']
    no_phd =  academia_row['NonPhD']

    no_doctorate_x = no_phd / 2

    us_doctorate_x = 50

    non_us_doctorate_x = no_phd + us_phd + (non_us_phd  / 2)

    annotations = [
        {
            'text': 'no\ndoctorate',
            'xy_start': (no_doctorate_x, y),
            'xy_end': (no_doctorate_x, padded_y),
        },
        {
            'text': 'non-U.S.\ndoctorate',
            'xy_start': (non_us_doctorate_x, y),
            'xy_end': (non_us_doctorate_x, padded_y),
        },
        {
            'text': 'U.S.\ndoctorate',
            'xy_start': (us_doctorate_x, y),
            'xy_end': (us_doctorate_x, padded_y),
        },
    ]

    for annotation in annotations:
        arrowprops = annotation.get('arrowprops', hnelib.plot.BASIC_ARROW_PROPS)
        ax.annotate(
            '',
            xy=annotation['xy_start'],
            xytext=annotation['xy_end'],
            arrowprops=arrowprops,
            annotation_clip=False,
        )

        ax.annotate(
            annotation['text'],
            xy=annotation.get('xy_text', annotation['xy_end']),
            va=annotation.get('va', 'bottom'),
            ha=annotation.get('ha', 'center'),
            annotation_clip=False,
            fontsize=hnelib.plot.FONTSIZES['annotation'],
        )

    hnelib.plot.finalize(ax)

    return pd.DataFrame(plot_data)


def draw_umbrella_insets_on_production_plot(ax):
    """
    instead of:
    App Sci | Ed  | Eng     | Hum
    Math    | Med | Nat Sci | Soc Sci

    we're going to set things up so we get:

    App Sci | Eng  | Math | Nat Sci
    Ed      | Hum  | Med  | Soc Sci

    because then we can have `Mathematics & Computing` (on two lines) instead of `Math & Computing`
    """
    # alphabetical
    # row_one = ['Applied Sciences', 'Education', 'Engineering', 'Humanities']
    # row_two = ['Mathematics and Computing', 'Medicine and Health', 'Natural Sciences', 'Social Sciences']

    row_one = ['Applied Sciences', 'Natural Sciences', 'Mathematics and Computing', 'Medicine and Health']
    row_two = ['Education', 'Engineering', 'Humanities', 'Social Sciences']

    umbrellas = row_one + row_two

    left_umbrellas = [row_one[0], row_two[0]]
    bottom_umbrellas = row_two

    ticks = [0, .2, .4, .6, .8, 1]
    ticklabels = [0, 20, 40, 60, 80, 100]

    df = usfhn.stats.runner.get('basics/lorenz')

    umbrella_df = usfhn.views.filter_by_taxonomy(df, 'Umbrella')

    field_df = usfhn.views.filter_by_taxonomy(df, 'Field')

    field_df = views.annotate_umbrella_color(field_df, taxonomization_level='Field')

    element_id = 0
    plot_data = []

    umbrella_to_inset = add_umbrella_inset_axs(ax, umbrellas)
    for umbrella, inset_ax in umbrella_to_inset.items():
        _umbrella_df = usfhn.views.filter_by_taxonomy(umbrella_df, value=umbrella)

        inset_ax.plot(
            _umbrella_df['X'], _umbrella_df['Y'],
            color=PLOT_VARS['colors']['dark_gray'],
            zorder=2,
            alpha=.5,
            lw=2,
        )

        plot_data_umbrella_df = _umbrella_df.copy()
        plot_data_umbrella_df['Subplot'] = umbrella
        for i, row in plot_data_umbrella_df.iterrows():
            for col in ['TaxonomyValue', 'TaxonomyLevel', 'X', 'Y', 'Subplot']:
                plot_data.append({
                    'Element': element_id,
                    'Attribute': col,
                    'Value': row[col],
                })

            element_id += 1

        annotation_point = _umbrella_df[
            _umbrella_df['Y'] <= .8
        ].sort_values(by='X', ascending=False).iloc[0]

        inset_ax.scatter(
            [annotation_point['X']],
            [annotation_point['Y']],
            zorder=5,
            color=PLOT_VARS['colors']['dark_gray'],
        )

        _field_df = field_df[
            field_df['Umbrella'] == umbrella
        ]

        for _, rows in _field_df.groupby('TaxonomyValue'):
            rows = rows.sort_values(by='X')
            inset_ax.plot(
                rows['X'], rows['Y'],
                color=rows.iloc[0]['UmbrellaColor'],
                zorder=3,
                lw=.5,
            )

            plot_data_field_df = rows.copy()
            plot_data_field_df['Line'] = umbrella
            for i, row in plot_data_field_df.iterrows():
                for col in ['TaxonomyValue', 'TaxonomyLevel', 'X', 'Y', 'Line']:
                    plot_data.append({
                        'Element': element_id,
                        'Attribute': col,
                        'Value': row[col],
                    })

                element_id += 1

        yticks = []
        yticklabels = []
        xticks = []
        xticklabels = []

        if umbrella in left_umbrellas:
            inset_ax.set_yticks(ticks)
            inset_ax.set_yticklabels(ticklabels)
        else:
            inset_ax.set_yticks([])

        if umbrella in bottom_umbrellas:
            inset_ax.set_xticks(ticks)
            inset_ax.set_xticklabels(ticklabels)
        else:
            inset_ax.set_xticks([])

        inset_ax.tick_params('both', labelsize=hnelib.plot.FONTSIZES['tick'])

        inset_ax.plot(
            [0, 1], [0, 1],
            color='black',
            lw=.5,
            zorder=1,
        )

        umbrella_title = umbrella
        inset_ax.set_xlim(0, 1.005)
        inset_ax.set_ylim(0, 1.005)
        inset_ax.set_title(
            plot_utils.clean_taxonomy_string(umbrella_title),
            size=hnelib.plot.FONTSIZES['axis'],
            color=PLOT_VARS['colors']['umbrellas'][umbrella],
            pad=4,
        )

        hnelib.plot.add_gridlines(
            inset_ax,
            xs=ticks,
            ys=ticks,
            lw=.5,
            alpha=.2,
        )

        inset_ax.set_aspect('equal', adjustable='box')

    halfway_down_lorenz_span = (.75 / 2) + .25
    halfway_across_lorenz_span = (.75 / 2) + .25

    ax.annotate(
        '% of U.S.-trained faculty',
        (.21, halfway_down_lorenz_span),
        size=hnelib.plot.FONTSIZES['axis'],
        xycoords='axes fraction',
        ha='left',
        va='center',
        rotation=90,
    )

    ax.annotate(
        '% of U.S. universities',
        (halfway_across_lorenz_span, .195),
        size=hnelib.plot.FONTSIZES['axis'],
        xycoords='axes fraction',
        ha='center',
        va='top',
    )

    return pd.DataFrame(plot_data)


def plot_academia_production_with_quartiles(ax):
    df = usfhn.stats.runner.get('basics/production')

    df = df[
        (df['TaxonomyLevel'] == 'Academia')
        &
        (df['ProductionPercent'].notnull())
    ][
        [
            'DegreeInstitutionId',
            'TaxonomyLevel',
            'TaxonomyValue',
            'TotalProduction',
            'ProductionPercent',
        ]
    ].drop_duplicates()

    df = usfhn.institutions.annotate_institution_name(df, id_col='DegreeInstitutionId')

    df = df.sort_values(by=['ProductionPercent'], ascending=False)
    production_percents = df['ProductionPercent']
    current_cum_sum = 0
    colors = []
    color_1 = PLOT_VARS['colors']['color_1']
    color_2 = PLOT_VARS['colors']['dark_gray']
    current_color = color_1
    bracket_indexes = [[0]]
    color_changes = [20, 40, 60, 80]
    production_percents = list(df['ProductionPercent'])

    for i, percent in enumerate(production_percents):
        current_cum_sum += percent
        colors.append(current_color)

        if color_changes and current_cum_sum > color_changes[0]:
            current_color = color_1 if current_color == color_2 else color_2
            color_changes.pop(0)

            bracket_indexes[-1].append(i)
            bracket_indexes.append([i + 1])

    bracket_indexes[-1].append(len(production_percents) - 1)

    element_id = 0
    plot_data = []
    for i, row in df.iterrows():
        plot_data.append({
            'Element': element_id,
            'Attribute': 'X',
            'Value': 100 * element_id / len(df),
        })

        plot_data.append({
            'Element': element_id,
            'Attribute': 'Y',
            'Value': row['ProductionPercent'],
        })

        element_id += 1

    bracket_pad = .025
    for i, (start, end) in enumerate(bracket_indexes):
        y1_start = production_percents[start] + bracket_pad
        y2_start = production_percents[end] + bracket_pad
        y_end = production_percents[start] + (bracket_pad * 2)

        if i == len(bracket_indexes) - 1:
            previous_start, previous_end = bracket_indexes[i - 1]
            text_x = start + ((previous_end - previous_start) / 2)
            end += 1
        else:
            text_x = start + ((end - start) / 2)

        ax.plot(
            [start, start],
            [y1_start, y_end],
            color=PLOT_VARS['colors']['dark_gray'],
            lw=.5,
        )

        ax.plot(
            [end, end],
            [y2_start, y_end],
            color=PLOT_VARS['colors']['dark_gray'],
            lw=.5,
        )
        ax.plot(
            [start, end],
            [y_end, y_end],
            color=PLOT_VARS['colors']['dark_gray'],
            lw=.5,
        )

        if end == bracket_indexes[-1][1] + 1:
            text = f"{end - start}"
        else:
            text = f"{1 + end - start}"

        ax.annotate(
            text,
            (text_x, y_end + bracket_pad),
            ha='center',
            va='bottom',
            fontsize=hnelib.plot.FONTSIZES['medium'],
        )

        if not i:
            ax.annotate(
                f"universities",
                (end, y_end + bracket_pad),
                ha='left',
                va='bottom',
                fontsize=hnelib.plot.FONTSIZES['medium'],
            )

    df['Color'] = colors

    total_institutions = usfhn.datasets.CURRENT_DATASET.data['InstitutionId'].nunique()
    producing_institutions = df['DegreeInstitutionId'].nunique()
    non_producing_institutions = total_institutions - producing_institutions

    tick_values = []
    tick_annotations = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for tick in tick_annotations:
        integer_tick = int(producing_institutions * tick / 100)
        tick_values.append(integer_tick)

    ax.set_xticks(tick_values)
    ax.set_xticklabels(tick_annotations)

    n_institutions = len(df)
    ax.bar(list(range(n_institutions)), df['ProductionPercent'], color=df['Color'])

    ax.set_ylim(0, 4)
    ax.set_yticks([0, 1, 2, 3, 4])

    ax.set_ylabel(f"% of U.S.-trained faculty produced")
    ax.set_xlabel("% of U.S. universities")
    ax.use_sticky_edges = False
    ax.margins(x=0.01, y=0)

    return pd.DataFrame(plot_data)


def plot_academia_production():
    fig, ax = plt.subplots(figsize=(hnelib.plot.WIDTHS['2-col'], 3.986), tight_layout=True)

    panel_a_df = plot_academia_production_with_quartiles(ax)
    panel_a_df['Subfigure'] = 'A'

    panel_b_df = draw_umbrella_insets_on_production_plot(ax)
    panel_b_df['Subfigure'] = 'B'

    hnelib.plot.finalize(
        [ax, ax],
        [-.03, .21],
        plot_label_y_pad=1.08,
    )

    return pd.concat([panel_a_df, panel_b_df])


def add_umbrella_inset_axs(parent_ax, umbrellas):
    ncols = 4
    nrows = math.ceil(len(umbrellas) / ncols)

    # we want to go:
    # x x x x
    # x x x x
    # _ _ x x
    # `last_row_start_col` tells us how many to skip
    last_row_start_col = nrows * ncols - len(umbrellas)

    # x_start = .275
    # x_stop = .975
    # x_pad_fraction = .15

    # y_start = .3
    # y_stop = 1
    # y_pad_fraction = .15

    x_start = .25
    x_stop = 1
    x_pad_fraction = .10

    y_start = .25
    y_stop = 1
    y_pad_fraction = .12

    x_span = x_stop - x_start
    y_span = y_stop - y_start

    x_pad_total = x_pad_fraction * x_span
    y_pad_total = y_pad_fraction * y_span

    inset_width_total = x_span - x_pad_total
    inset_height_total = y_span - y_pad_total

    inset_width = inset_width_total / ncols
    inset_height = inset_height_total / nrows

    x_inset_pad = x_pad_total / (ncols - 1)
    y_inset_pad = y_pad_total / (nrows - 1)

    insets = []
    for row in reversed(range(nrows)):
        y = y_start + row * (inset_height + y_inset_pad)

        col_start = 0 if row else last_row_start_col

        for col in range(col_start, ncols):
            x = x_start + col * (inset_width + x_inset_pad)
            ax_inset = parent_ax.inset_axes(
                (
                    x,
                    y,
                    inset_width,
                    inset_height,
                )
            )
            insets.append(ax_inset)

    umbrella_to_inset = {}
    for ax, umbrella in zip(insets, umbrellas):
        umbrella_to_inset[umbrella] = ax

    return umbrella_to_inset



def plot_gini_subsampling():
    ranks = usfhn.datasets.CURRENT_DATASET.data
    ranks = ranks[
        [
            'PersonId',
            'Rank',
        ]
    ].drop_duplicates()

    ranks['PeopleOfRank'] = ranks.groupby('Rank')['PersonId'].transform('nunique')
    ranks['FractionOfPeopleOfRank'] = ranks['PeopleOfRank'] / len(ranks)

    ranks = ranks[
        [
            'Rank',
            'FractionOfPeopleOfRank'
        ]
    ].drop_duplicates()

    assistant_professor_fraction = ranks[
        ranks['Rank'] == 'Assistant Professor'
    ].iloc[0]['FractionOfPeopleOfRank']

    df = usfhn.datasets.CURRENT_DATASET.gini_subsamples

    df['Mean'] = df.groupby('SampleFraction')['GiniCoefficient'].transform('mean')
    df = hnelib.pandas.add_error(df, groupby_col='SampleFraction', value_col='GiniCoefficient')

    df = df[
        [
            'SampleFraction',
            'Mean',
            'Error',
        ]
    ].drop_duplicates()

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot(
        list(df['SampleFraction']),
        list(df['Mean']),
    )
    ax.errorbar(
        list(df['SampleFraction']),
        list(df['Mean']),
        fmt='none',
        yerr=list(df['Error']),
        zorder=3,
    )

    print(assistant_professor_fraction)

    ax.axvline(
        assistant_professor_fraction,
        lw=1,
        ls='--',
        label='fraction of assistant professors',
    )
    ax.legend()

    ax.set_ylabel('Gini coefficient')
    ax.set_xlabel('fraction sampled')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    hnelib.plot.add_gridlines_on_ticks(ax)


def plot_lorenz_comparison(level='Academia', value='Academia'):
    df = usfhn.stats.runner.get('non-new-hire/lorenz')

    new_hires_df = usfhn.stats.runner.get('new-hire/lorenz')

    umbrellas = usfhn.views.get_umbrellas()
    levels = ['Academia'] + ['Umbrella' for i in range(len(umbrellas))]
    umbrellas = ['Academia'] + umbrellas
    
    fig, ax = plt.subplots(figsize=(6, 6), tight_layout=True)

    for level, value in zip(levels, umbrellas):
        _df = usfhn.views.filter_by_taxonomy(df, level=level, value=value)
        _df = usfhn.views.annotate_umbrella_color(_df, 'Umbrella')
        _new_hires_df = usfhn.views.filter_by_taxonomy(new_hires_df, level=level, value=value)
        _new_hires_df = usfhn.views.annotate_umbrella_color(_new_hires_df, 'Umbrella')

        ax.plot(
            _df['X'], _df['Y'],
            color=_df.iloc[0]['UmbrellaColor'],
            zorder=2,
            lw=1,
            label='everyone',
        )

        ax.plot(
            _new_hires_df['X'], _new_hires_df['Y'],
            color=_df.iloc[0]['UmbrellaColor'],
            zorder=2,
            lw=1,
            ls='--',
            alpha=.5,
            label='new hires',
        )

    ax.set_xlabel('fraction of institutions')
    ax.set_ylabel('fraction of faculty')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ticks = [0, .2, .4, .6, .8, 1]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(hnelib.plot.stringify_numbers_without_ugly_zeros(ticks))
    ax.set_yticklabels(hnelib.plot.stringify_numbers_without_ugly_zeros(ticks))
    hnelib.plot.add_gridlines_on_ticks(ax)


def plot_gini_multi_plot():
    fig, axes = plt.subplots(
        1, 3,
        figsize=(hnelib.plot.WIDTHS['2-col'], 1.772),
        gridspec_kw={
            'width_ratios': [2, .75, .9],
            'wspace': .25,
        }
    )

    tadpole_ax = axes[0]
    time_ax = axes[1]
    slope_ax = axes[2]

    legend_fontsize = 7

    ################################################################################
    # ginis
    ################################################################################

    df = usfhn.stats.runner.get('ginis/by-new-hire/df')[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'GiniCoefficient',
            'NewHire',
        ]
    ].drop_duplicates()

    df = hnelib.pandas.aggregate_df_over_column(
        df,
        agg_col='NewHire',
        join_cols=[
            'TaxonomyLevel',
            'TaxonomyValue',
        ],
        value_cols=['GiniCoefficient'],
        agg_value_to_label={
            True: 'NewHire',
            False: 'OldHire',
        }
    )

    plot_data = []
    element_id = 0
    for i, row in df.iterrows():
        for col in df.columns:
            plot_data.append({
                'Subfigure': 'A',
                'Element': element_id,
                'Attribute': col,
                'Value': row[col],
            })

        element_id += 1

    start_is, end_is, lines, ulabels = usfhn.standard_plots.plot_connected_two_values_by_taxonomy_level(
        df,
        'NewHireGiniCoefficient',
        'OldHireGiniCoefficient',
        open_label='existing faculty',
        closed_label='new faculty',
        y_label='Gini coefficient',
        y_min=0,
        y_max=1,
        y_step=.25,
        ax=tadpole_ax,
        add_gridlines=False,
        add_taxonomy_lines=False,
    )

    tadpole_ax.set_ylim(0, .8)
    yticks = [0, .2, .4, .6, .8]
    tadpole_ax.set_yticks(yticks)
    tadpole_ax.set_yticklabels(hnelib.plot.stringify_numbers_without_ugly_zeros(yticks))

    annotation_offset = .08
    arrow_pad = .01
    start = start_is['Academia']
    end = end_is['Academia']

    annotation_row = df[
        df['TaxonomyLevel'] == 'Academia'
    ].iloc[0]

    labels = ['new faculty', 'existing faculty']
    xs = [start, end]
    y_starts = [
        annotation_row['NewHireGiniCoefficient'] - arrow_pad,
        annotation_row['OldHireGiniCoefficient'] - arrow_pad,
    ]
    y_ends = [.62, .62]

    annotation_kwargs = {
        'ha': 'center',
        'va': 'top',
        'color': PLOT_VARS['colors']['dark_gray'],
        'rotation': 90,
        'annotation_clip': False,
        'fontsize': hnelib.plot.FONTSIZES['medium'],
    }

    for label, x, y_start, y_end in zip(labels, xs, y_starts, y_ends):
        tadpole_ax.annotate(
            "",
            (x, y_start),
            xytext=(x, y_end),
            arrowprops=hnelib.plot.BASIC_ARROW_PROPS,
        )

        tadpole_ax.annotate(
            label,
            (x, y_end),
            **annotation_kwargs,
        )

    hnelib.plot.add_gridlines(tadpole_ax, ys=[.2, .8])

    for y in [.4, .6]:
        tadpole_ax.plot(
            [lines[1], tadpole_ax.get_xlim()[1]],
            [y, y],
            lw=.5,
            alpha=.5,
            zorder=1,
            color=PLOT_VARS['colors']['dark_gray'],
        )

    df = usfhn.stats.runner.get('ginis/by-year/df')

    element_id = 0
    df = usfhn.views.filter_to_academia_and_domain(df)
    for i, row in df.sort_values(by=['TaxonomyLevel', 'TaxonomyValue', 'Year']).iterrows():
        for col in df.columns:
            plot_data.append({
                'Subfigure': 'B',
                'Element': element_id,
                'Attribute': col,
                'Value': row[col],
            })
        element_id += 1

    usfhn.standard_plots.plot_domain_level_values_over_time(
        df=df,
        column='GiniCoefficient',
        ylabel='Gini coefficient',
        ylim=[0, .8],
        yticks=[0, .2, .4, .6, .8],
        ax=time_ax,
        gridlines=False,
    )

    time_ax.set_xticks([2012, 2016, 2020])

    hnelib.plot.add_gridlines_on_ticks(time_ax, x=False)

    umbrellas = usfhn.views.get_umbrellas()
    levels = ['Academia'] + ['Umbrella' for i in range(len(umbrellas))]
    values = ['Academia'] + umbrellas

    element_id = 0
    for level, value in zip(levels, values):
        Xs, Ys = plot_ranked_attrition(slope_ax, level=level, value=value, rank_type='production')

        for x, y in zip(Xs, Ys):
            plot_data.append({
                'Subfigure': 'C',
                'Element': element_id,
                'Attribute': 'X',
                'Value': x,
            })

            plot_data.append({
                'Subfigure': 'C',
                'Element': element_id,
                'Attribute': 'Y',
                'Value': y,
            })

            plot_data.append({
                'Subfigure': 'C',
                'Element': element_id,
                'Attribute': 'TaxonomyLevel',
                'Value': level,
            })

            plot_data.append({
                'Subfigure': 'C',
                'Element': element_id,
                'Attribute': 'TaxonomyValue',
                'Value': value,
            })

            element_id += 1

    slope_ax.set_xlim(0, 1.005)
    xticks = [0, .2, .4, .6, .8, 1]
    xticklabels = [0, 20, 40, 60, 80, 100]
    slope_ax.set_xticks(xticks)
    slope_ax.set_xticklabels(xticklabels)

    slope_ax.set_ylim(0, .12)
    yticks = [0, .03, .06, .09, .12]
    slope_ax.set_yticks(yticks)
    slope_ax.set_yticklabels(hnelib.plot.stringify_numbers_without_ugly_zeros(yticks))

    slope_ax.set_xlabel('production rank')
    slope_ax.set_ylabel('average annual attrition risk')

    usfhn.plot_utils.add_gridlines_and_annotate_rank_direction(
        slope_ax,
        rank_type='production',
        fontsize=hnelib.plot.FONTSIZES['annotation'],
        break_height=.01,
        x_gridlines_to_break=[.2, .4, .6],
    )

    legend = plot_utils.add_umbrella_legend(
        slope_ax,
        get_umbrella_legend_handles_kwargs={
            'style': 'none',
            'include_academia': True,
        },
        legend_kwargs={
            'fontsize': hnelib.plot.FONTSIZES['legend'],
            'loc': 'center left',
            'bbox_to_anchor': (.85, .5),
            'bbox_transform': slope_ax.transAxes,
        },
    )

    if time_ax:
        axes = [tadpole_ax, time_ax, slope_ax]
        x_pads = [-.085, -.2325, -.23]
    else:
        axes = [tadpole_ax, slope_ax]
        x_pads = [-.08, -.18]

    hnelib.plot.finalize(axes, x_pads)

    return pd.DataFrame(plot_data)


def plot_gender_multi_plot():
    fig, axes = plt.subplots(
        1, 3,
        figsize=(hnelib.plot.WIDTHS['2-col'], 1.772),
        gridspec_kw={
            'width_ratios': [.75, 2, .9],
            'wspace': .25,
        }
    )

    time_ax = axes[0]
    tadpole_ax = axes[1]
    aging_ax = axes[2]

    legend_fontsize = 7

    ################################################################################
    # 
    ################################################################################
    df = usfhn.stats.runner.get('gender/by-year/df')[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'Year',
            'FractionFemale'
        ]
    ].drop_duplicates()

    df['PercentFemale'] = 100 * df['FractionFemale']
    df = df.drop(columns=['FractionFemale'])

    element_id = 0
    plot_data = []
    for i, row in df.iterrows():
        for col in df.columns:
            plot_data.append({
                'Subfigure': 'A',
                'Element': element_id,
                'Attribute': col,
                'Value': row[col],
            })

        element_id += 1

    usfhn.standard_plots.plot_domain_level_values_over_time(
        df,
        'PercentFemale',
        ylabel='% women',
        ylim=[0, 100],
        yticks=[0, 25, 50, 75, 100],
        ax=time_ax,
    )

    time_ax.set_xticks([2012, 2016, 2020])

    ################################################################################
    # 
    ################################################################################
    df = usfhn.stats.runner.get('attrition/risk/gender')[
        [
            'Gender',
            'TaxonomyLevel',
            'TaxonomyValue',
            'Events',
            'AttritionEvents',
        ]
    ].drop_duplicates()

    df = hnelib.pandas.aggregate_df_over_column(
        df,
        agg_col='Gender',
        join_cols=[
            'TaxonomyLevel',
            'TaxonomyValue',
        ],
        value_cols=[
            'Events',
            'AttritionEvents',
        ],
    )

    df['TotalEvents'] = df['MaleEvents'] + df['FemaleEvents']
    df['FemaleFractionOfAttritions'] = df['FemaleAttritionEvents'] / df['TotalEvents']
    df['FemalePercentOfAttritions'] = 100 * df['FemaleFractionOfAttritions']

    df = df[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'FemalePercentOfAttritions',
        ]
    ].drop_duplicates()

    new_hires_df = usfhn.stats.runner.get('gender/by-new-hire/df')
    new_hires_df = new_hires_df[
        new_hires_df['NewHire']
    ][
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'FractionFemale',
        ]
    ]

    new_hires_df['NewHiresPercentFemale'] = 100 * new_hires_df['FractionFemale']
    new_hires_df = new_hires_df.drop(columns=['FractionFemale'])

    df = df.merge(
        new_hires_df,
        on=[
            'TaxonomyLevel',
            'TaxonomyValue',
        ]
    )

    element_id = 0
    for i, row in df.iterrows():
        for col in df.columns:
            plot_data.append({
                'Subfigure': 'B',
                'Element': element_id,
                'Attribute': col,
                'Value': row[col],
            })

        element_id += 1

    start_is, end_is, lines, ulabels = usfhn.standard_plots.plot_connected_two_values_by_taxonomy_level(
        df,
        'NewHiresPercentFemale',
        'FemalePercentOfAttritions',
        open_label='% of new hires',
        closed_label='% of attritions',
        y_label='% women',
        y_min=0,
        y_max=100,
        y_step=25,
        ax=tadpole_ax,
        add_gridlines=False,
        add_taxonomy_lines=False,
    )

    annotation_offset = 8
    arrow_pad = 1
    start = start_is['Academia']
    end = end_is['Academia']

    annotation_row = df[
        df['TaxonomyLevel'] == 'Academia'
    ].iloc[0]

    annotation_kwargs = {
        'ha': 'center',
        'va': 'bottom',
        'color': PLOT_VARS['colors']['dark_gray'],
        'rotation': 90,
        'annotation_clip': False,
        'fontsize': hnelib.plot.FONTSIZES['annotation'],
    }

    labels = ['% of new hires', '% of attritions']
    xs = [start, end]
    y_starts = [
        annotation_row['NewHiresPercentFemale'] + arrow_pad,
        annotation_row['FemalePercentOfAttritions'] + arrow_pad * 5,
    ]
    y_ends = [55, 55]

    for label, x, y_start, y_end in zip(labels, xs, y_starts, y_ends):
        tadpole_ax.annotate(
            "",
            (x, y_start),
            xytext=(x, y_end),
            arrowprops=hnelib.plot.BASIC_ARROW_PROPS,
        )

        tadpole_ax.annotate(
            label,
            (x, y_end),
            **annotation_kwargs,
        )

    for y in [25, 50, 75, 100]:
        x = lines[1]

        tadpole_ax.plot(
            [x, tadpole_ax.get_xlim()[1]],
            [y, y],
            lw=.5,
            alpha=.5,
            zorder=1,
            color=PLOT_VARS['colors']['dark_gray'],
        )

    df = usfhn.stats.runner.get('gender/by-career-age/df')

    df = df[
        df['TaxonomyLevel'].isin(['Academia', 'Umbrella'])
    ]

    df = usfhn.views.annotate_umbrella_color(df, 'Umbrella')
    df['PercentFemale'] = 100 * df['FractionFemale']

    df = hnelib.pandas.add_proportions_confidence_interval(
        df,
        count_col='FemaleFaculty',
        observations_col='Faculty',
        groupby_cols=['CareerAge', 'TaxonomyValue'],
        as_percent=True,
    )

    umbrellas = usfhn.views.get_umbrellas() + ['Academia']
    df['Index'] = df['TaxonomyValue'].apply(umbrellas.index)

    for _, rows in df.groupby(['Index', 'TaxonomyValue']):
        rows = rows.copy()
        rows = rows.sort_values(by='CareerAge')


        aging_ax.plot(
            rows['CareerAge'],
            rows['PercentFemale'],
            color=rows.iloc[0]['UmbrellaColor'],
            lw=.75,
            zorder=2,
        )

        aging_ax.fill_between(
            rows['CareerAge'],
            y1=rows['LowerConf'],
            y2=rows['UpperConf'],
            color=rows.iloc[0]['FadedUmbrellaColor'],
            zorder=1,
            alpha=.25,
        )

    element_id = 0
    for i, row in df.sort_values(by=['TaxonomyLevel', 'TaxonomyValue', 'CareerAge']).iterrows():
        for col in df.columns:
            plot_data.append({
                'Subfigure': 'C',
                'Element': element_id,
                'Attribute': col,
                'Value': row[col],
            })

        element_id += 1

    aging_ax.set_ylim(0, 100)
    aging_ax.set_yticks([0, 25, 50, 75, 100])
    aging_ax.set_ylabel('% women')

    aging_ax.set_xlim(1, 40)
    aging_ax.set_xticks([1, 10, 20, 30, 40])
    aging_ax.set_xlabel('career age (years since doctorate)')

    hnelib.plot.add_gridlines_on_ticks(aging_ax)

    legend = plot_utils.add_umbrella_legend(
        axes[-1],
        get_umbrella_legend_handles_kwargs={
            'style': 'none',
            'include_academia': True,
        },
        legend_kwargs={
            'fontsize': hnelib.plot.FONTSIZES['legend'],
            'loc': 'center left',
            'bbox_to_anchor': (.85, .5),
            'bbox_transform': axes[-1].transAxes,
        },
    )

    hnelib.plot.finalize(axes, [-.30, -.11, -.245])

    return pd.DataFrame(plot_data)


def plot_production_versus_employment(style='percents'):
    employment = usfhn.stats.runner.get('taxonomy/institutions').rename(columns={
        'Count': 'FacultyCount',
        'Fraction': 'FacultyFraction',
    })

    production = usfhn.stats.runner.get('basics/production')[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'DegreeInstitutionId', 
            'ProductionFraction',
            'ProductionCount',
        ]
    ].drop_duplicates().rename(columns={
        'DegreeInstitutionId': 'InstitutionId',
    }).drop_duplicates()

    df = employment.merge(
        production,
        on=['InstitutionId', 'TaxonomyLevel', 'TaxonomyValue'],
    ).drop_duplicates()

    df['FacultyCount'] = df['FacultyCount'].fillna(0)
    df['FacultyFraction'] = df['FacultyFraction'].fillna(0)

    df['ProductionCount'] = df['ProductionCount'].fillna(0)
    df['ProductionFraction'] = df['ProductionFraction'].fillna(0)

    if style == 'percents':
        df['X'] = df['FacultyFraction'] * 100
        df['Y'] = df['ProductionFraction'] * 100
    elif style == 'integers':
        df['X'] = df['FacultyCount']
        df['Y'] = df['ProductionCount']

    fig, ax = plt.subplots(1, figsize=(9, 9))

    df = df[
        df['TaxonomyLevel'] == 'Academia'
    ]

    ax.scatter(df['X'], df['Y'], alpha=.5, s=5, zorder=2)

    max_max = max(ax.get_ylim()[1], ax.get_xlim()[1])

    ax.set_xlim(0, max_max)
    ax.set_ylim(0, max_max)
    ax.plot(
        [0, max_max],
        [0, max_max],
        color=PLOT_VARS['colors']['dark_gray'],
        alpha=.5,
        zorder=1,
        lw=1,
    )

    max_employed = usfhn.institutions.annotate_institution_name(
        df[df['X'] == max(df['X'])]
    ).iloc[0]

    production_line_y_start = .05 if style == 'percents' else 100
    production_line_y_offset = .7 if style == 'percents' else 1250

    ax.annotate(
        max_employed['InstitutionName'],
        xy=(
            max_employed['X'],
            max_employed['Y'] + production_line_y_offset,
        ),
        xytext=(0, 5),
        textcoords='offset points',
        ha='center',
        va='bottom',
    )

    ax.plot(
        [max_employed['X'], max_employed['X']],
        [max_employed['Y'] + production_line_y_start, max_employed['Y'] + production_line_y_offset],
        color=PLOT_VARS['colors']['dark_gray'], alpha=.5, zorder=1, lw=1
    )

    max_produced = usfhn.institutions.annotate_institution_name(
        df[df['Y'] == max(df['Y'])]
    ).iloc[0]
    ax.annotate(
        max_produced['InstitutionName'],
        xy=(
            max_produced['X'],
            max_produced['Y'],
        ),
        xytext=(0, 5),
        textcoords='offset points',
        ha='center',
        va='bottom',
    )

    label = '%' if style == 'percents' else 'count'

    ax.set_xlabel(f'{label} of faculty employed')
    ax.set_ylabel(f'{label} of faculty produced')
    ax.set_title(f'{label} of faculty produced vs employed\nby institution')

    ax.set_aspect('equal', adjustable='box')


def histogram_production_per_employment():
    production = usfhn.stats.runner.get('basics/production')[
        [
            'DegreeInstitutionId',
            'TaxonomyLevel',
            'TaxonomyValue',
            'ProductionCount',
        ]
    ].drop_duplicates().rename(columns={
        'DegreeInstitutionId': 'InstitutionId',
    }).drop_duplicates()

    df = usfhn.stats.runner.get('taxonomy/institutions').rename(columns={
        'Count': 'FacultyCount',
    }).merge(
        production,
        on=['InstitutionId', 'TaxonomyLevel', 'TaxonomyValue'],
    ).drop_duplicates()

    df['FacultyCount'] = df['FacultyCount'].fillna(0)
    df['ProductionCount'] = df['ProductionCount'].fillna(0)
    df['ProducedOverEmployed'] = df['ProductionCount'] / df['FacultyCount']

    df = df[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'InstitutionId',
            'ProducedOverEmployed'
        ]
    ].drop_duplicates()

    df = df[
        df['TaxonomyLevel'] == 'Academia'
    ]

    df = usfhn.institutions.annotate_institution_name(df)

    fig, ax = plt.subplots(1, figsize=(9, 9))

    min_fraction = min(df['ProducedOverEmployed'])
    max_fraction = max(df['ProducedOverEmployed'])

    bins = [1/32, 1/24, 1/16, 1/12, 1/8, 1/6, 1/4, 1/3, 1/2, 3/4, 1, 1.5, 2, 3, 4, 6, 8]

    mean = df['ProducedOverEmployed'].mean()
    ax.axvline(x=mean, label=f'mean = {round(mean, 2)}')

    ax.hist(
        df['ProducedOverEmployed'],
        bins=bins,
        alpha=.75,
    )

    ax.set_xlabel('# produced / # employed')
    ax.set_ylabel('# of institutions')
    ax.set_title('faculty produced per faculty member employed\nby institution')
    ax.legend()

    max_rate = usfhn.institutions.annotate_institution_name(
        df[df['ProducedOverEmployed'] == max_fraction]
    ).iloc[0]
    ax.annotate(
        max_rate['InstitutionName'],
        xy=(
            max_rate['ProducedOverEmployed'], 
            1,
        ),
        xytext=(0, 50),
        textcoords='offset points',
        ha='center',
        va='bottom',
        arrowprops=dict(color=PLOT_VARS['colors']['dark_gray'], width=1, headwidth=6, shrink=0.05),
    )

    ax.set_xscale('log', base=2)
    x_min, x_max = ax.get_xlim()
    ax.set_xticks([1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8])
    ax.set_xticklabels(['1:32', '1:16', "1:8", "1:4", "1:2", '1:1', "2:1", "4:1", '8:1'])


def plot_institutions_in_top_x(taxonomization_level):
    fig, ax = plt.subplots(1, figsize=(8, 8))
    df = usfhn.datasets.get_dataset_df('ranks')
    df = views.filter_generalized_df(df, taxonomy_level=taxonomization_level)

    df = df[df['Year'] == constants.YEAR_UNION]

    df['InstitutionsInTopX'] = df.groupby('OrdinalRank')['InstitutionId'].transform('nunique')
    ordinal_rank_to_institutions_at_or_below = []
    for ordinal_rank, rows in df.groupby('OrdinalRank'):
        institutions_at_or_below = set(rows['InstitutionId'].unique())

        if len(ordinal_rank_to_institutions_at_or_below):
            institutions_at_or_below |= ordinal_rank_to_institutions_at_or_below[ordinal_rank - 1]

        ordinal_rank_to_institutions_at_or_below.append(institutions_at_or_below)

    ax.plot(
        [i for i in range(len(ordinal_rank_to_institutions_at_or_below))],
        [len(insts) for insts in ordinal_rank_to_institutions_at_or_below],
    )

    ax.set_xlabel('department rank (ordinal)')
    ax.set_ylabel('institutions')
    ax.set_title(f'Institutions with a department of rank x or higher\n({taxonomization_level})')


def one_plot_rank_change(taxonomization_level):
    df = usfhn.stats.runner.get('ranks/gender/placements')

    df = df[
        df['NormalizedRankDifference'] != 0
    ]

    fig, ax = plt.subplots(figsize=(5, 8))

    step = .01

    median_texts = []
    gender_to_mean = {}
    max_half_density = 0
    for gender, rows in df.groupby('Gender'):
        min_change = min(rows['NormalizedRankDifference'])
        max_change = max(rows['NormalizedRankDifference'])
        bins = np.arange(min_change, max_change + step, step)
        kernel = gaussian_kde(rows['NormalizedRankDifference'])
        densities = [kernel(x) for x in bins]
        half_density = (max(densities)[0] - min(densities)[0]) / 2
        max_half_density = max(half_density, max_half_density)

        color = PLOT_VARS['colors']['gender'][gender]

        gender_to_mean[gender] = rows['NormalizedRankDifference'].mean()
        if gender == 'All':
            ax.plot(densities, bins, lw=2, color=color, zorder=2)
        else:
            ax.plot(densities, bins, lw=2, color=color, zorder=3, alpha=.5)

    for gender, mean in gender_to_mean.items():
        ax.plot(
            [0, max_half_density],
            [mean, mean],
            lw=2,
            color=PLOT_VARS['colors']['gender'][gender],
            zorder=1,
            linestyle='--',
            label=f'{gender} ({round(mean, 2)})',
        )


    ax.axhline(0, lw=1, color=PLOT_VARS['colors']['dark_gray'], zorder=1)

    ax.set_ylabel('normalized rank difference')
    ax.set_xlabel('density')

    ax.set_xticks([])
    ax.set_title(f"prestige change from PhD\nto faculty job ({taxonomization_level})", size=36)

    ax.legend()
    plt.tight_layout()


def plot_rank_change_taxonomy_level(taxonomization_level):
    df = usfhn.stats.runner.get('ranks/placements')
    df = df[
        df['NormalizedRankDifference'] != 0
    ]

    df = views.annotate_umbrella_color(df, taxonomization_level=taxonomization_level)

    step = .01
    fig, ax = plt.subplots(figsize=(5, 8))
    medians = []
    colors = []
    for value, rows in df.groupby('TaxonomyValue'):
        min_change = min(rows['NormalizedRankDifference'])
        max_change = max(rows['NormalizedRankDifference'])
        bins = np.arange(min_change, max_change + step, step)
        kernel = gaussian_kde(rows['NormalizedRankDifference'])
        densities = [kernel(x) for x in bins]

        color = rows.iloc[0]['UmbrellaColor']

        medians.append(np.median(rows['NormalizedRankDifference']))
        colors.append(color)
        ax.plot(densities, bins, lw=2, color=color, zorder=3, alpha=.5)

        ax.axhline(0, lw=1, color=PLOT_VARS['colors']['dark_gray'], zorder=1)

    x_min, x_max = ax.get_xlim()
    x_middle = np.mean([x_min, x_max])
    for median, color in zip(medians, colors):
        ax.plot([x_min, x_middle], [median, median], lw=2, color=color, zorder=2, linestyle='--')

    legend_elements = [
        Line2D(
            [0, 1], [0, 1],
            label='mean',
            linestyle='--',
            color=PLOT_VARS['colors']['dark_gray'],
        ),
    ]
    if taxonomization_level != 'Academia':
        legend_elements += plot_utils.get_umbrella_legend_handles()

    ax.legend(handles=legend_elements, handletextpad=.1, prop={'size': 10}, frameon=True)

    ax.set_ylabel('normalized rank difference')
    ax.set_xlabel('density')

    ax.set_xticks([])
    ax.set_title(f"prestige change from\nPhD to faculty job", size=36)

    for y in np.arange(-.75, 1, step=.25):
        ax.axhline(y, lw=1, color=PLOT_VARS['colors']['dark_gray'], alpha=.25)

    plt.tight_layout()


def plot_rank_change_umbrella_group(taxonomization_level='Area'):
    umbrellas = sorted(views.get_umbrellas())

    n_umbrellas = len(umbrellas)
    rows = 2
    columns = 4
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    axes = axes.flatten()

    df = usfhn.stats.runner.get('ranks/placements')
    df = df[
        df['NormalizedRankDifference'] != 0
    ]

    df = views.annotate_umbrella_color(df, taxonomization_level=taxonomization_level)
    df = df.sort_values(by=['Umbrella', 'TaxonomyValue'])

    # social sciences has 7
    linestyles = [
        '-',                        # solid
        'dotted',                   # dotted
        '--',                       # dashed
        '-.',                       # dashdot
        (0, (1, 10)),               # loosely dotted
        (0, (5, 10)),               # loosely dashed
        (0, (3, 10, 1, 10)),        # loosely dashdotted
        (0, (3, 10, 1, 10, 1, 10)), # loosely dashdotdotted
    ]

    umbrella_to_taxonomy = defaultdict(list)
    umbrella_to_taxonomy_mean = defaultdict(dict)
    umbrella_to_taxonomy_linestyle = defaultdict(dict)
    umbrella_to_color = {}
    step = .01
    for (umbrella, taxonomy), rows in df.groupby(['Umbrella', 'TaxonomyValue']):
        umbrella_to_color[umbrella] = rows.iloc[0]['UmbrellaColor']
        umbrella_to_taxonomy[umbrella].append(taxonomy)

        taxonomy_index = umbrella_to_taxonomy[umbrella].index(taxonomy)

        umbrella_to_taxonomy_mean[umbrella][taxonomy] = np.mean(rows['NormalizedRankDifference'])
        umbrella_to_taxonomy_linestyle[umbrella][taxonomy] = linestyles[taxonomy_index]

        ax = axes[umbrellas.index(umbrella)]

        bins = np.arange(
            min(rows['NormalizedRankDifference']),
            max(rows['NormalizedRankDifference']) + step,
            step,
        )

        kernel = gaussian_kde(rows['NormalizedRankDifference'])
        densities = [kernel(x) for x in bins]

        ax.plot(
            densities, bins,
            lw=2,
            linestyle=umbrella_to_taxonomy_linestyle[umbrella][taxonomy],
            color=umbrella_to_color[umbrella],
            zorder=3,
            alpha=.5,
            label=taxonomy,
        )

    for ax, umbrella in zip(axes, umbrellas):
        ax.axhline(0, lw=1, color=PLOT_VARS['colors']['dark_gray'], zorder=1)
        ax.set_title(umbrella)

        x_min, x_max = ax.get_xlim()
        x_middle = np.mean([x_min, x_max])

        for taxonomy in umbrella_to_taxonomy[umbrella]:
            mean = umbrella_to_taxonomy_mean[umbrella][taxonomy]
            ax.plot(
                [x_min, x_middle],
                [mean, mean],
                lw=2,
                color=umbrella_to_color[umbrella],
                zorder=2,
                linestyle=umbrella_to_taxonomy_linestyle[umbrella][taxonomy],
            )

        ax.axhline(0, lw=1, color=PLOT_VARS['colors']['dark_gray'])
        for y in np.arange(-.75, 1.25, step=.25):
            ax.axhline(y, lw=1, color=PLOT_VARS['colors']['dark_gray'], alpha=.25)

        ax.legend(prop={'size': 12}, handlelength=3)
        ax.set_ylim(-1, 1)
        ax.set_xticks([])

    axes[0].set_ylabel('normalized rank difference')
    axes[4].set_ylabel('normalized rank difference')
    for ax in axes[4:]:
        ax.set_xlabel('density')

    plt.suptitle(f"prestige change from PhD to faculty job", size=36)

    plt.tight_layout()


def plot_steepness_difference(taxonomization_level='Area'):
    print('this wont work yet, need to move to new stats')
    # df = usfhn.null_models.get_stats_at_taxonomization_level_for_plotting(taxonomization_level)
    steepness = measurements.get_steepness_by_taxonomy()

    steepness = views.filter_exploded_df(steepness)
    df = df.merge(
        steepness,
        on=['TaxonomyLevel', 'TaxonomyValue']
    )

    fig, ax = plt.subplots(1, figsize=(8, 8))

    together = list(df['NullSteepnessMean']) + list(df['Steepness'])
    max_max = max(together)
    min_min = min(together)

    df['DifferenceFromNull'] = df['Violations'] - df['NullViolationsMean']
    df['DifferenceFromNullPercent'] = df['DifferenceFromNull'] / df['NullViolationsMean']
    df['DifferenceFromNullPercent'] *= 100

    ax.axhline(0, zorder=1, lw=1)
    
    plot_utils.scatter_with_fill_by_null_hierarchy_threshold(
        ax, df, 'Violations', 'DifferenceFromNullPercent'
    )

    scatter_fill_legend_handles = plot_utils.get_legend_handles_for_scatter_filled_by_threshold()
    umbrella_legend_handles = plot_utils.get_umbrella_legend_handles()

    ax.legend(
        loc='lower right',
        handles=scatter_fill_legend_handles + umbrella_legend_handles,
        prop={'size': 15}
    )

    hnelib.plot.annotate_pearson(ax, df['DifferenceFromNullPercent'], df['Violations'])

    ax.set_xlabel('empirical fraction of upward hires')
    ax.set_ylabel('% difference from null model')
    ax.set_title("\n".join([
        f'empirical fraction of upward hires',
        f'vs difference from null model',
        f'({taxonomization_level})'
    ]))


def plot_steepness_vs_gini(taxonomization_level='Area'):
    fig, ax = plt.subplots(1, figsize=(8, 8))

    df = measurements.get_steepness_by_taxonomy()
    df = views.filter_exploded_df(df)
    df = df[
        df['TaxonomyLevel'] == taxonomization_level
    ]

    ginis = usfhn.stats.runner.get('ginis/df')

    df = df.merge(ginis, on=['TaxonomyLevel', 'TaxonomyValue'])

    print('this wont work yet, need to move to new stats')
    # null_models = usfhn.null_models.get_stats_at_taxonomization_level_for_plotting(taxonomization_level)[
    #     ['TaxonomyLevel', 'TaxonomyValue', 'MoreHierarchicalCount']
    # ].drop_duplicates()

    df = df.merge(null_models, on=['TaxonomyLevel', 'TaxonomyValue'])

    df = views.annotate_umbrella_color(df, taxonomization_level)

    plot_utils.scatter_with_fill_by_null_hierarchy_threshold(
        ax, df, 'Steepness', 'GiniCoefficient'
    )

    scatter_fill_legend_handles = plot_utils.get_legend_handles_for_scatter_filled_by_threshold()
    umbrella_legend_handles = plot_utils.get_umbrella_legend_handles()

    ax.legend(
        loc='lower right',
        handles=scatter_fill_legend_handles + umbrella_legend_handles,
        prop={'size': 15}
    )

    y_min, y_max = ax.get_ylim()
    ax.set_ylim(.25, .8)
    ax.set_xlim(.65, .95)

    hnelib.plot.annotate_pearson(ax, df['Steepness'], df['GiniCoefficient'])

    ax.set_xlabel('Fraction of downard hires')
    ax.set_title('fraction of  violations\nvs\nGini coefficient')

    ax.set_xlabel('empirical fraction of downward hires')
    ax.set_ylabel('Gini coefficient')
    ax.set_title("\n".join([
        f'empirical fraction of downward hires',
        f'vs Gini coefficient',
        f'({taxonomization_level})'
    ]))


def plot_steepness_vs_gender_ratio():
    df = views.filter_exploded_df(measurements.get_taxonomy_gender_ratios())
    df = df.merge(
        views.filter_exploded_df(measurements.get_steepness_by_taxonomy()),
        on=['TaxonomyLevel', 'TaxonomyValue']
    )

    plot_relationship_across_taxonomy_levels(
        df,
        x_column='FractionFemale',
        y_column='Steepness',
        title=f'fraction of downward edges vs fraction female',
        x_label='fraction female',
        y_label='fraction of downward edges',
        x_lim=[0, 1],
    )


def plot_rank_change_gender_difference_vs_fraction_female(direction='Down'):
    df = usfhn.stats.runner.get('ranks/gender/placements')
    df = df[
        df['NormalizedRankDifference'] != 0
    ].merge(
        usfhn.stats.runner.get('gender/taxonomy'),
        on=['TaxonomyLevel', 'TaxonomyValue']
    )

    df = df[
        df['Year'] == constants.YEAR_UNION
    ].drop(columns=['Year'])

    df['Direction'] = df['NormalizedRankDifference'].apply(lambda d: 'Up' if d > 0 else 'Down')

    df = df[
        df['Direction'] == direction
    ].drop(columns=['Direction'])

    df['MeanRankChange'] = df.groupby(
        ['TaxonomyLevel', 'TaxonomyValue', 'Gender']
    )['NormalizedRankDifference'].transform('mean')

    df['MeanRankChange'] = df['MeanRankChange'].apply(abs)

    df = df[
        ['TaxonomyLevel', 'TaxonomyValue', 'Gender', 'MeanRankChange', 'FractionFemale']
    ].drop_duplicates()

    male_df = df[
        df['Gender'] == 'Male'
    ].copy().drop(columns=['Gender', 'FractionFemale']).rename(columns={
        'MeanRankChange': 'MaleMeanRankChange'
    })
    female_df = df[
        df['Gender'] == 'Female'
    ].copy().drop(columns=['Gender']).rename(columns={
        'MeanRankChange': 'FemaleMeanRankChange'
    })

    df = male_df.merge(
        female_df,
        on=['TaxonomyLevel', 'TaxonomyValue']
    )

    df['MeanRankChangeDifference'] = df['MaleMeanRankChange'] - df['FemaleMeanRankChange']

    df = df[
        ['TaxonomyLevel', 'TaxonomyValue', 'MeanRankChangeDifference', 'FractionFemale']
    ].drop_duplicates()

    plot_relationship_across_taxonomy_levels(
        df,
        x_column='FractionFemale',
        y_column='MeanRankChangeDifference',
        title=f'gender difference in mean rank change {direction} vs fraction female',
        x_label='fraction female',
        y_label=f'male - female\nmean rank change{direction}',
        x_lim=[0, 1],
        y_keep_zero_line=True,
    )

def plot_rank_change_taxonomy(taxonomy_level='Academia', taxonomy_value='Academia'):
    df = usfhn.stats.runner.get('rank/placements')
    df = usfhn.views.filter_by_taxonomy(df, level=taxonomy_level, value=taxonomy_value)

    # df = df.sample(frac=.05)
    df = views.annotate_umbrella_color(df, taxonomy_level)

    fig, ax = plt.subplots(figsize=(6, 6))
    step = .01
    color = df.iloc[0]['UmbrellaColor']

    min_change = min(df['NormalizedRankDifference'])
    max_change = max(df['NormalizedRankDifference'])

    bins = np.arange(min_change, max_change + step, step)
    kernel = gaussian_kde(df['NormalizedRankDifference'])
    densities = [kernel(x) for x in bins]

    half_density = (max(densities)[0] - min(densities)[0]) / 2

    mean = df['NormalizedRankDifference'].mean()
    ax.plot(densities, bins, lw=2, color=color, zorder=2)

    ax.plot(
        [0, half_density],
        [mean, mean],
        lw=2,
        color=color,
        zorder=1,
        linestyle='--',
        label=f'mean = {round(mean, 2)}',
    )


def plot_rank_change_beeswarm(
    taxonomy_level='Academia',
    taxonomy_value='Academia',
    ax=None,
    limit=2500,
    draw_beeswarm=True,
):
    df = usfhn.stats.runner.get('ranks/placements')
    df = usfhn.views.filter_by_taxonomy(df, taxonomy_level, taxonomy_value)

    percent_self_hires = round(100 * len(df[df['NormalizedRankDifference'] == 0]) / len(df))
    percent_up = round(100 * len(df[df['NormalizedRankDifference'] > 0]) / len(df))
    percent_down = round(100 * len(df[df['NormalizedRankDifference'] < 0]) / len(df))

    mean = -1 * usfhn.paper_stats.mean_movement_on_hierarchy() / 100

    df = df[
        df['NormalizedRankDifference'] != 0
    ]

    if len(df) > limit:
        df = df.sample(n=limit)

    df = views.annotate_umbrella_color(df, taxonomy_level)

    if not ax:
        fig, ax = plt.subplots(figsize=(3, 6))

    width = 4/5
    sub_ax = ax.inset_axes((0, 0, width, 1))

    annotations_ax = ax.inset_axes((width, 0, 1, 1))
    hnelib.plot.hide_axis(annotations_ax)
    hnelib.plot.hide_axis(ax)

    ax = sub_ax

    plot_data = []
    element_id = 0
    for i, row in df.iterrows():
        plot_data.append({
            'Element': element_id,
            'Attribute': 'NormalizedRankDifference',
            'Value': row['NormalizedRankDifference'],
        })
        element_id += 1

    if draw_beeswarm:
        sns.swarmplot(
            x=[0 for i in range(len(df))],
            y=df['NormalizedRankDifference'],
            color=df.iloc[0]['UmbrellaColor'],
            orient='v',
            size=.5,
            ax=ax,
            zorder=2,
        )

    percent_up = usfhn.paper_stats.p_movement_at_level_and_value(direction='up')
    percent_self_hires = usfhn.paper_stats.p_movement_at_level_and_value(direction='self-hires')
    percent_down = usfhn.paper_stats.p_movement_at_level_and_value(direction='down')

    # brackets
    x_min, x_max = ax.get_xlim()
    x_range = x_max - x_min
    bracket_x = x_max / 2
    bracket_side_length = x_range * .05
    bracket_pad = .02
    brackets = [
        {
            'y_min': bracket_pad,
            'y_max': max(df['NormalizedRankDifference']) + bracket_pad,
            'label': f'{percent_up}%\nmove\nup',
            'hatch': 'XXXXX',
            'facecolor': 'none',
        },
        {
            'y_min': 0,
            'y_max': 0,
            'label': f'{percent_self_hires}%\nare\nself-hires',
            'hatch': None,
            'facecolor': hnelib.plot.set_alpha_on_colors(PLOT_VARS['colors']['color_1']),
        },
        {
            'y_min': -1 + bracket_pad,
            'y_max': -1 * bracket_pad,
            'label': f'{percent_down}%\nmove\ndown',
            'hatch': None,
            'facecolor': 'none',
        },
    ]

    xlim = ax.get_xlim()
    ax.plot(
        [xlim[0], bracket_x],
        [0, 0],
        lw=.5,
        color=PLOT_VARS['colors']['dark_gray'],
    )
    ax.set_xlim(-bracket_x - .01, bracket_x + .01)
    
    ax.set_ylim(-1, .75)
    yticks = [-1, -.75, -.5, -.25, 0, .25, .5, .75]
    ax.set_yticks(yticks)
    ax.set_yticklabels(hnelib.plot.stringify_numbers_without_ugly_zeros(yticks))
    ax.set_ylabel("prestige change from\nU.S. doctorate to faculty job")

    annotations_ax.set_xlim(0, 1)
    annotations_ax.set_ylim(ax.get_ylim())

    rectangle_width = .1
    rectangle_height = .055
    for bracket in brackets:
        bracket_side_x = bracket_x - bracket_side_length
        y_min = bracket['y_min']
        y_max = bracket['y_max']
        y_mid = ((y_max - y_min) / 2) + y_min

        y_min = max(.99 * ax.get_ylim()[0], y_min)
        y_max = min(.99 * ax.get_ylim()[1], y_max)

        ax.plot(
            [bracket_x, bracket_side_x],
            [y_min, y_min],
            color=PLOT_VARS['colors']['dark_gray'],
            lw=.5,
        )
        ax.plot(
            [bracket_x, bracket_side_x],
            [y_max, y_max],
            color=PLOT_VARS['colors']['dark_gray'],
            lw=.5,
        )
        ax.plot(
            [bracket_x, bracket_x],
            [y_min, y_max],
            color=PLOT_VARS['colors']['dark_gray'],
            lw=.5,
        )


        annotations_ax.annotate(
            bracket['label'],
            xy=(.025, y_mid),
            ha='left',
            va='center',
            fontsize=hnelib.plot.FONTSIZES['annotation'],
        )

        annotations_ax.add_patch(
            mpatches.Rectangle(
                xy=(.05 + bracket_side_length * 2.25, y_mid + .07),
                width=rectangle_width,
                height=rectangle_height,
                lw=.5,
                facecolor=bracket['facecolor'],
                edgecolor=PLOT_VARS['colors']['color_1'],
                hatch=bracket['hatch'],
            )
        )

    x_min, x_max = ax.get_xlim()
    y_min, y_may = ax.get_ylim()

    mean_x_fraction = .55
    mean_x_min = mean_x_fraction * x_min
    mean_x_max = mean_x_fraction * x_max
    y_line = .6 * y_min

    ax.plot(
        [mean_x_min, mean_x_max],
        [mean, mean],
        color=PLOT_VARS['colors']['dark_gray'],
        ls='--',
        lw=.5,
    )

    # downward line
    ax.plot(
        [mean_x_min, mean_x_min],
        [mean - .01, y_line],
        color=PLOT_VARS['colors']['dark_gray'],
        alpha=.5,
        lw=.5,
    )

    ax.annotate(
        f"mean\nchange is\n-.{abs(int(round(100 * mean, 2)))}",
        (mean_x_min, y_line),
        ha='center',
        va='top',
        fontsize=hnelib.plot.FONTSIZES['annotation'],
    )

    ax.set_xticks([])

    return ax, pd.DataFrame(plot_data)

def plot_rank_change_multi_plot():
    fig, axes = plt.subplots(
        1, 4, figsize=(hnelib.plot.WIDTHS['2-col'], 1.772),
        gridspec_kw={'wspace': .35},
    )

    axes[0], panel_a_df = plot_rank_change_beeswarm(ax=axes[0])
    panel_a_df['Subfigure'] = 'A'

    panel_b_df = plot_umbrella_rank_breakdown(axes[1])
    panel_b_df['Subfigure'] = 'B'

    panel_c_df = plot_field_level_null_models(ax=axes[2])
    panel_c_df['Subfigure'] = 'C'

    panel_d_df = plot_systematic_prestige(parent_ax=axes[3])
    panel_d_df['Subfigure'] = 'D'

    hnelib.plot.finalize(
        [axes[0], axes[1], axes[1], axes[3]],
        plot_label_x_pads=[-.39, -.17, 1.08, -.22],
    )

    return pd.concat([
        panel_a_df,
        panel_b_df,
        panel_c_df,
        panel_d_df,
    ])


def plot_umbrella_rank_breakdown(ax, label_yaxis=False):
    df = usfhn.stats.runner.get('ranks/hierarchy-stats', rank_type='prestige')
    df['MovementFraction'] *= 100

    up_df = df.copy()[
        df['MovementType'] == 'Upward'
    ][
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'MovementFraction',
        ]
    ].rename(columns={
        'MovementFraction': 'Violations'
    })

    down_df = df.copy()[
        df['MovementType'] == 'Downward'
    ][
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'MovementFraction',
        ]
    ].rename(columns={
        'MovementFraction': 'Steepness'
    })

    self_hires_df = df.copy()[
        df['MovementType'] == 'Self-Hire'
    ][
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'MovementFraction',
        ]
    ].rename(columns={
        'MovementFraction': 'SelfHiresFraction'
    })

    df = up_df.merge(
        down_df,
        on=['TaxonomyLevel', 'TaxonomyValue']
    ).merge(
        self_hires_df,
        on=['TaxonomyLevel', 'TaxonomyValue']
    )

    df = df[
        df['TaxonomyLevel'] == 'Umbrella'
    ]

    df = views.annotate_umbrella_color(df, 'Umbrella')

    bottoms = []
    downwards = []
    self_hires = []
    upwards = []
    colors = []
    umbrellas = []
    indexes = [i for i in range(df['Umbrella'].nunique())]

    plot_data = []
    element_id = 0

    for umbrella, rows in df.groupby('Umbrella'):
        row = rows.iloc[0]
        downwards.append(row['Steepness'])
        self_hires.append(row['SelfHiresFraction'])
        upwards.append(row['Violations'])
        colors.append(row['UmbrellaColor'])
        umbrellas.append(umbrella)

        plot_data.append({
            'Element': element_id,
            'Attribute': 'UpwardHires',
            'Value': row['Violations'],
        })

        plot_data.append({
            'Element': element_id,
            'Attribute': 'DownwardHires',
            'Value': row['Steepness'],
        })

        plot_data.append({
            'Element': element_id,
            'Attribute': 'SelfHires',
            'Value': row['SelfHiresFraction'],
        })

        plot_data.append({
            'Element': element_id,
            'Attribute': 'Color',
            'Value': row['UmbrellaColor'],
        })

        plot_data.append({
            'Element': element_id,
            'Attribute': 'TaxonomyLevel',
            'Value': 'Domain',
        })

        plot_data.append({
            'Element': element_id,
            'Attribute': 'TaxonomyValue',
            'Value': umbrella,
        })

    bottom = np.zeros(len(indexes))

    ax.set_ylim(0, 100.25)

    yticks = [0, 25, 50, 75, 100]
    ax.set_yticks(yticks)

    if label_yaxis:
        ax.set_ylabel('% of U.S.-trained faculty')
    else:
        ax.set_yticklabels([f'{t}%' for t in yticks])
        ax.set_xlabel('hierarchy transitions')

    ax.set_xticks([])

    ax.bar(
        indexes,
        downwards,
        color='none',
        edgecolor=colors,
        lw=.5,
    )

    bottom += np.array(downwards)

    ax.bar(
        indexes,
        self_hires,
        bottom=bottom,
        color=hnelib.plot.set_alpha_on_colors(colors),
        edgecolor=colors,
        lw=.5,
    )

    bottom += np.array(self_hires)

    ax.bar(
        indexes,
        upwards,
        color='none',
        edgecolor=colors,
        hatch='xx',
        bottom=bottom,
        lw=.5,
    )

    for index, umbrella, color in zip(indexes, umbrellas, colors):
        ax.annotate(
            plot_utils.clean_taxonomy_string(umbrella),
            (index, 5), 
            ha='center',
            va='bottom',
            color=color,
            rotation=90,
            fontsize=hnelib.plot.FONTSIZES['annotation'],
        )

    return pd.DataFrame(plot_data)

def plot_field_level_null_models(ax=None, alpha=.05):
    df = usfhn.null_models.get_stats()[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'NullSteepnessMean',
            'MoreHierarchicalCount',
        ]
    ].drop_duplicates()

    df = usfhn.views.filter_by_taxonomy(df, level='Field')
    df['NullSteepnessMean'] *= 100

    df['P'] = df['MoreHierarchicalCount'] / usfhn.constants.NULL_MODEL_DRAWS
    df = usfhn.stats_utils.correct_multiple_hypotheses(df, alpha=alpha)
    df['Significant'] = df['PCorrected'] < alpha

    steepness_df = usfhn.stats.runner.get('ranks/hierarchy-stats', rank_type='prestige')

    steepness_df['MovementFraction'] *= 100
    steepness_df = hnelib.pandas.aggregate_df_over_column(
        steepness_df,
        agg_col='MovementType',
        join_cols=[
            'TaxonomyLevel',
            'TaxonomyValue',
        ],
        value_cols=['MovementFraction'],
        agg_value_to_label={
            'Self-Hire': 'SelfHire',
            'Upward': 'Violations',
        },
    ).rename(columns={
        'SelfHireMovementFraction': 'SelfHiresFraction',
        'ViolationsMovementFraction': 'Violations',
    })

    df = df.merge(
        steepness_df,
        on=['TaxonomyLevel', 'TaxonomyValue']
    )

    df['NullViolations'] = 100 - df['NullSteepnessMean'] - df['SelfHiresFraction']

    data_df = df.copy()[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'NullViolations',
            'Violations',
            'PCorrected',
        ]
    ].drop_duplicates()
    data_df = usfhn.plot_utils.annotate_color(data_df)

    plot_data = []
    element_id = 0
    for i, row in data_df.iterrows():
        for col in data_df.columns:
            plot_data.append({
                'Element': element_id,
                'Attribute': col,
                'Value': row[col],
            })

        element_id += 0

    df = df[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'Violations',
            'NullViolations',
            'Significant',
        ]
    ].drop_duplicates()

    df = views.annotate_umbrella_color(df, 'Field')

    if not ax:
        fig, ax = plt.subplots(figsize=(4, 4), tight_layout=True)

    label = "% up-hierarchy hires"

    lim = [0, 30.25]
    ticks = [0, 10, 20, 30]

    insig_df = df[
        ~df['Significant']
    ]

    ax.scatter(
        insig_df['Violations'],
        insig_df['NullViolations'],
        marker='x',
        color=hnelib.plot.set_alpha_on_colors(insig_df['FadedUmbrellaColor'], .5),
        zorder=2,
        s=8,
        lw=.65,
    )

    sig_df = df[
        df['Significant']
    ]

    ax.scatter(
        sig_df['Violations'],
        sig_df['NullViolations'],
        facecolor=sig_df['FadedUmbrellaColor'],
        edgecolor=sig_df['UmbrellaColor'],
        s=8,
        lw=.65,
        zorder=2,
    )

    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(hnelib.plot.stringify_numbers_without_ugly_zeros(ticks))
    ax.set_yticklabels(hnelib.plot.stringify_numbers_without_ugly_zeros(ticks))
    ax.set_xlabel(f"{label}\n(empirical)")
    ax.set_ylabel(f"{label}\n(null model)")

    ################################################################################
    # start testing annotations
    ################################################################################

    annotations = []

    # max violations: Animal Sciences
    max_violations_row = df[
        df['NullViolations'] == max(df['NullViolations'])
    ].iloc[0]

    annotations.append({
        'row': max_violations_row,
        'x-multiplier': 0,
        'y-multiplier': 1,
    })

    # min violations annotation: Religious Studies
    min_violations_row = df[
        df['NullViolations'] == min(df['NullViolations'])
    ].iloc[0]

    annotations.append({
        'row': min_violations_row,
        'x-multiplier': 0,
        'y-multiplier': -1,
    })

    # medicine and health: pharmacy
    not_significant_annotation = df[
        (df['Umbrella'] == 'Medicine and Health')
        &
        (~df['Significant'])
        &
        (df['Violations'] > 12)
    ].sort_values(by=['Violations']).iloc[0]

    annotations.append({
        'row': not_significant_annotation,
        'x-multiplier': 1,
        'y-multiplier': 0,
        'point-pad': 0,
    })

    # mathematics and computing: computer science
    other_row = df[
        (df['Umbrella'] == 'Mathematics and Computing')
    ].sort_values(by=['NullViolations']).iloc[-1]

    annotations.append({
        'row': other_row,
        'x-multiplier': 0,
        'y-multiplier': 1,
    })

    # annotations
    annotation_pad = .5
    # diagonal_annotation_pad = .35
    # diagonal_arrow_pad = 1.65
    straight_pad = 2.25

    point_pad = .2
    arrow_length = 3.5
    text_negative_pad = -.5

    for annotation in annotations:
        row = annotation['row']
        x = row['Violations']
        y = row['NullViolations']

        point_pad = annotation.get('point-pad', point_pad)

        x_multiplier = annotation.get('x-multiplier', 0)
        y_multiplier = annotation.get('y-multiplier', 0)

        arrow_x_start = x + (x_multiplier * point_pad)
        arrow_y_start = y + (y_multiplier * point_pad)

        arrow_x_end = arrow_x_start + (x_multiplier * arrow_length)
        arrow_y_end = arrow_y_start + (y_multiplier * arrow_length)

        text_x = arrow_x_end + (x_multiplier * text_negative_pad)
        text_y = arrow_y_end + (y_multiplier * text_negative_pad)

        if x_multiplier == 1:
            ha = 'left'
        elif x_multiplier == -1:
            ha = 'right'
        else:
            ha = 'center'

        if y_multiplier == 1:
            va = 'bottom'
        elif y_multiplier == -1:
            va = 'top'
        else:
            va = 'center'

        text = str(row['TaxonomyValue']).replace(' ', '\n')

        ax.annotate(
            '',
            (arrow_x_start, arrow_y_start),
            xytext=(arrow_x_end, arrow_y_end),
            arrowprops=PLOT_VARS['arrowprops'],
            fontsize=hnelib.plot.FONTSIZES['annotation'],
        )

        ax.annotate(
            text,
            (text_x, text_y),
            ha=ha,
            va=va,
            fontsize=hnelib.plot.FONTSIZES['annotation'],
        )

    ax.set_aspect('equal', adjustable='box')

    line_kwargs = {
        'color': PLOT_VARS['colors']['dark_gray'],
        'alpha': .5,
        'zorder': 1,
    }

    lines = [
        # horizontal
        [[0, 30], [10, 10]],
        [[0, 30], [20, 20]],
        [[0, 19], [30, 30]],
        [[26, 30], [30, 30]],
        # vertical
        [[10, 10], [0, 23]],
        [[10, 10], [27.5, 30]],
        [[20, 20], [0, 11.5]],
        [[20, 20], [14, 27]],
        [[30, 30], [0, 30]],
        # diagonal
        [[0, 2], [0, 2]],
        [[6.5, 30], [6.5, 30]],

    ]

    for p1, p2 in lines:
        ax.plot(
            p1,
            p2,
            **line_kwargs
        )

    hnelib.plot.finalize(ax)

    return pd.DataFrame(plot_data)


def plot_systematic_prestige(parent_ax=None, label_umbrellas=False):
    if parent_ax:
        hnelib.plot.hide_axis(parent_ax)
        new_parent_ax = parent_ax.inset_axes((-.075, 0, 1.075, 1))
        parent_ax = new_parent_ax
        hnelib.plot.hide_axis(parent_ax)
        ax = parent_ax.inset_axes((.15, 0, .85, 1))
        colorbar_ax = parent_ax.inset_axes((.075, .15, .075, .75))
        parent_ax.set_ylabel("Pearson correlation\nbetween ranks")
    else:
        fig, axes = plt.subplots(
            1, 2,
            figsize=(5.75, 5),
            gridspec_kw={'width_ratios': [5, .25]},
            tight_layout=True,
        )

        ax = axes[0]
        colorbar_ax = axes[1]

    df = usfhn.stats.runner.get('ranks/institution-rank-correlations', rank_type='prestige')
    df = views.annotate_umbrella(df, 'Field', 'TaxonomyValueOne').rename(columns={
        'Umbrella': 'UmbrellaOne'
    })

    df = df[
        df['UmbrellaOne'].notnull()
    ]

    field_to_umbrella = df[
        ['TaxonomyValueOne', 'UmbrellaOne']
    ].drop_duplicates()

    field_to_umbrella = field_to_umbrella.sort_values(by=['UmbrellaOne'])
    field_to_umbrella['IndexOne'] = [i for i in range(len(field_to_umbrella))]

    df = df.merge(field_to_umbrella, on=['TaxonomyValueOne', 'UmbrellaOne'])

    field_to_umbrella = field_to_umbrella.rename(columns={
        'UmbrellaOne': 'UmbrellaTwo',
        'TaxonomyValueOne': 'TaxonomyValueTwo',
        'IndexOne': 'IndexTwo',
    }).sort_values(by=['IndexTwo'])

    df = df.merge(field_to_umbrella, on='TaxonomyValueTwo')

    n_fields = df['TaxonomyValueOne'].nunique()

    df['IndexOne'] = n_fields - df['IndexOne'] - 1

    mean = df['Pearson'].mean()

    plot_data = []
    element_id = 0
    data_df = df.copy()[
        [
            'TaxonomyValueOne',
            'TaxonomyValueTwo',
            'Pearson',
            'P',
            'IndexOne',
            'IndexTwo',
        ]
    ].drop_duplicates()

    for i, row in data_df.iterrows():
        for col in data_df.columns:
            plot_data.append({
                'Element': element_id,
                'Attribute': col,
                'Value': row[col],
            })

        element_id += 1

    matrix = np.zeros((n_fields, n_fields))
    for i, row in df.iterrows():
        i = row['IndexOne']
        j = row['IndexTwo']

        # value = row['Pearson'] if row['Pearson'] >= -.20 else -.20
        value = row['Pearson']
        matrix[i][j] = value

    cmap_name = 'viridis'
    cmap = plt.get_cmap(cmap_name)
    line_color = 'black'

    plot = ax.pcolor(matrix, cmap=cmap)
    kwargs = {}

    if colorbar_ax:
        hnelib.plot.hide_axis(colorbar_ax)
        kwargs['cax'] = colorbar_ax.inset_axes((.15, .15, .85, .7))
    else:
        kwargs['location'] = 'left'

    cbar = plt.colorbar(
        plot,
        ax=[ax],
        **kwargs,
    )

    if not parent_ax:
        cbar.set_label(
            'Pearson correlation between rankings',
            size=hnelib.plot.FONTSIZES['axis'],
        )

    cbar.ax.tick_params(labelsize=hnelib.plot.FONTSIZES['annotation']) 

    if parent_ax:
        kwargs['cax'].yaxis.set_ticks_position('left') 

    new_ticks = []
    for tick in cbar.get_ticks():
        new_ticks.append(round(tick, 3))

    new_ticklabels = [-.6, -.4, -.20, 0, .20, .4, .6, .8, 1]
    new_ticks = [t for t in new_ticklabels]

    cbar.set_ticks(new_ticks)
    cbar.set_ticklabels(hnelib.plot.stringify_numbers_without_ugly_zeros(new_ticklabels))

    duplicated_umbrellas = list(field_to_umbrella['UmbrellaTwo'])

    umbrella_to_abbreviation = {
        'Applied Sciences': 'Applied Sci.',
        'Education': 'Education',
        'Engineering': 'Engineering',
        'Mathematics and Computing': 'Math. + Comp.',
        'Humanities': 'Humanities',
        'Medicine and Health': 'Med. + Health',
        'Natural Sciences': 'Natural Sci.',
        'Social Sciences': 'Social Sci.',
    }

    annotations = []
    annotation = {}
    for i, umbrella in enumerate(duplicated_umbrellas):
        if not annotation:
            annotation = {
                'text': umbrella_to_abbreviation[umbrella],
                'start': i,
                'color': PLOT_VARS['colors']['umbrellas'][umbrella],
            }

        if i == len(duplicated_umbrellas) - 1 or duplicated_umbrellas[i + 1] != umbrella:
            annotation['end'] = i
            annotations.append(annotation)
            annotation = {}

    line_kwargs = {
        'lw': 2.75,
    }

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    edge_kwargs = {
        'color': line_color,
        'lw': .25,
    }

    # border lines
    ax.plot(
        [xmin, xmax],
        [ymin, ymin],
        **edge_kwargs,
        clip_on=False,
    )

    ax.plot(
        [xmin, xmax],
        [ymax, ymax],
        **edge_kwargs,
        clip_on=False,
    )

    ax.plot(
        [xmin, xmin],
        [ymin, ymax],
        **edge_kwargs,
        clip_on=False,
    )

    ax.plot(
        [xmax, xmax],
        [ymin, ymax],
        **edge_kwargs,
        clip_on=False,
    )

    swatch_pad_amount = 3 if parent_ax else 1.5

    ax.set_frame_on(False)
    fontsize = 7
    for annotation in annotations:
        start_i = annotation['start']
        end_i = annotation['end']

        if start_i not in [0, n_fields - 1]:
            ax.plot(
                [start_i, start_i],
                [0, n_fields],
                **edge_kwargs,
            )
            ax.plot(
                [0, n_fields],
                [n_fields - start_i, n_fields - start_i],
                **edge_kwargs,
            )

        start = start_i + .5
        end = end_i + .5

        # side
        x = -1 * swatch_pad_amount
        y_text = np.mean([start, end])

        y_start = n_fields - start + 1 - .5
        y_end = n_fields - end - 1 + .5

        ax.plot(
            [x, x],
            [y_start, y_end],
            **line_kwargs,
            color=annotation['color'],
        )

        # bottom
        y = n_fields + swatch_pad_amount
        x_text = np.mean([start, end])

        ax.plot(
            [start - .5, end + .5],
            [y, y],
            **line_kwargs,
            color=annotation['color'],
        )

        # if label_umbrellas:
        ax.annotate(
            annotation['text'],
            (x_text, y + swatch_pad_amount),
            color=annotation['color'],
            fontsize=hnelib.plot.FONTSIZES['annotation'],
            ha='center',
            va='top',
            annotation_clip=False,
            rotation=90,
        )

    ax.set_ylim(reversed(ax.get_ylim()))
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_aspect('equal', adjustable='box')

    if parent_ax:
        hnelib.plot.finalize(parent_ax)

    return pd.DataFrame(plot_data)


def plot_top_ten_by_field_lorenz(ax=None):
    if not ax:
        fig, ax = plt.subplots(figsize=(5, 5))

    truncate = .1

    for umbrella in views.get_umbrellas():
        plot_top_ten_umbrella_lorenz(ax, umbrella, truncate=truncate)

    plot_top_ten_umbrella_lorenz(ax, truncate=truncate)

    ax.set_xlabel('fraction of institutions')
    ax.set_ylabel('fraction of top 10 departments')
    ax.set_xlim([0, truncate * 1.01])
    ax.set_ylim([0, 1.01])

    ax.set_xticks([0, .025, .05, .075, .1])
    ax.set_xticklabels(["0", ".025", ".05", ".075", ".1"])
    ax.set_yticks([0, .2, .4, .6, .8, 1])

    hnelib.plot.add_gridlines(ax, ys=[.6, .8, 1])

    ys = [.6, 1]
    for x in [.025, .05, .075, .1]:
        ax.plot(
            [x, x,],
            ys,
            color=PLOT_VARS['colors']['dark_gray'],
            zorder=1,
            alpha=.5,
            lw=1,
        )


    handles = plot_utils.get_umbrella_legend_handles(
        style='line',
        include_academia=True,
        extra_kwargs={'alpha': 1},
    )

    legend = ax.legend(
        handles=handles,
        loc='lower center',
    )

    for text in legend.get_texts():
        text.set_color(PLOT_VARS['colors']['umbrellas'][text.get_text()])

    return ax


def plot_top_ten_umbrella_lorenz(ax, umbrella=None, truncate=1):
    inst_df = usfhn.datasets.CURRENT_DATASET.ranks

    if umbrella:
        inst_df = views.filter_by_taxonomy(inst_df, level='Umbrella', value=umbrella)

    institutions = set(inst_df['InstitutionId'].unique())
    n_institutions = len(institutions)

    df = usfhn.steeples.get_absolute_steeples()
    df = views.filter_exploded_df(df)
    df = views.filter_by_taxonomy(df, level='Field')
    df = views.annotate_umbrella(df, 'Field')
    
    if umbrella:
        df = df[
            df['Umbrella'] == umbrella
        ]

    df = df[
        df['InTop10'] == True
    ]

    df['InstitutionCount'] = df.groupby('InstitutionId')['InstitutionId'].transform('count')
    df = df[
        [
            'InstitutionId',
            'InstitutionCount',
        ]
    ].drop_duplicates()

    institutions_without_steeples = institutions - set(df['InstitutionId'].unique())

    extra_rows = []
    for institution in institutions_without_steeples:
        extra_rows.append({
            'InstitutionId': institution,
            'InstitutionCount': 0,
        })

    df = pd.concat([df, pd.DataFrame(extra_rows)])

    df = df.sort_values(by='InstitutionCount', ascending=False)
    df['CumSumInstitutionCount'] = df['InstitutionCount'].cumsum()
    df['CumSumInstitutionCountFraction'] = df['CumSumInstitutionCount'] / df['InstitutionCount'].sum()

    df['InstitutionIndex'] = [i for i in range(len(df))]
    df['CumSumInstitution'] = df['InstitutionIndex'].cumsum()
    df['CumSumInstitutionFraction'] = df['CumSumInstitution'] / df['InstitutionIndex'].sum()

    df = df[
        df['CumSumInstitutionFraction'] <= truncate * 1.01
    ]

    if umbrella:
        color = PLOT_VARS['colors']['umbrellas'][umbrella]
    else:
        color = PLOT_VARS['colors']['umbrellas']['Academia']

    ax.plot(
        df['CumSumInstitutionFraction'],
        df['CumSumInstitutionCountFraction'],
        color=color,
        lw=2,
    )


def plot_top_ten_by_field(ax=None):
    if not ax:
        fig, ax = plt.subplots(figsize=(5, 5))

    df = usfhn.steeples.get_absolute_steeples()
    df = df[
        df['InTop10'] == True
    ].drop(columns=['InTop10'])

    df = views.annotate_umbrella_color(df, 'Field')

    umbrella_df = df[
        [
            'TaxonomyValue',
            'Umbrella'
        ]
    ].drop_duplicates().sort_values(by=[
        'Umbrella',
        'TaxonomyValue',
    ])

    umbrella_df['Index'] = [i for i in range(len(umbrella_df))]

    df = df.merge(
        umbrella_df,
        on=['Umbrella', 'TaxonomyValue']
    )

    df['TotalInField'] = df.groupby('TaxonomyValue')['InstitutionId'].transform('nunique')
    steeple_df = df[
        df['Steeple'] == True
    ][
        [
            'TaxonomyValue',
            'InstitutionId',
        ]
    ]

    steeple_df['SteepleCount'] = steeple_df.groupby('TaxonomyValue')['InstitutionId'].transform('nunique')

    steeple_df = steeple_df.drop(columns=['InstitutionId']).drop_duplicates()
    df = df.merge(
        steeple_df,
        on='TaxonomyValue',
        how='left',
    )[
        [
            'Umbrella',
            'UmbrellaColor',
            'Index',
            'TaxonomyValue',
            'TotalInField',
            'SteepleCount',
        ]
    ].drop_duplicates()

    df['SteepleCount'] = df['SteepleCount'].fillna(0)
    df['NonSteepleCount'] = df['TotalInField'] - df['SteepleCount']
    df['NonSteepleCount'] = df['NonSteepleCount'].fillna(0)

    df['SteepleFraction'] = df['SteepleCount'] / df['TotalInField']
    df['SteepleFraction'] = df['SteepleFraction'].fillna(0)
    df['NonSteepleFraction'] = df['NonSteepleCount'] / df['TotalInField']
    df['NonSteepleFraction'] = df['NonSteepleFraction'].fillna(0)

    df['FaceColor'] = df['UmbrellaColor'].apply(lambda c: to_rgba(c, .5))

    df = df.sort_values(by='Index')

    ax.barh(
        df['Index'],
        df['NonSteepleFraction'],
        color=df['FaceColor'],
        lw=.5,
        edgecolor=df['UmbrellaColor'],
    )

    ax.barh(
        df['Index'],
        df['SteepleFraction'],
        color='w',
        lw=.5,
        edgecolor=df['UmbrellaColor'],
        left=df['NonSteepleFraction'],
        # hatch='xxxxx',
    )

    ax.set_ylim(max(df['Index']) + .5, -.5)
    ax.set_yticks([])
    ax.set_xticks([])
    return ax


def plot_self_hiring_by_gender(ax=None):
    if not ax:
        fig, ax = plt.subplots(1, figsize=(7, 4), tight_layout=True)

    tick_fontsize = 10
    axis_label_fontsize = 13
    annotation_fontsize = 9

    spacer_amount = 5

    df = usfhn.stats.runner.get('self-hire/by-gender/df').rename(columns={
        'SelfHiresFraction': 'SelfHireRate',
    })[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'SelfHireRate',
            'Gender',
        ]
    ].drop_duplicates()

    df = df[
        df['TaxonomyLevel'].isin(['Academia', 'Umbrella'])
    ]

    df['Color'] = df['TaxonomyValue'].apply(PLOT_VARS['colors']['umbrellas'].get)

    indexes = {u: spacer_amount * (i + 1) for i, u in enumerate(views.get_umbrellas())}
    indexes['Academia'] = 0

    df['Index'] = df['TaxonomyValue'].apply(indexes.get)

    ungendered_df = views.filter_exploded_df(df)

    female_df = df[df['Gender'] == 'Female'].copy()
    female_df['Index'] += 1
    male_df = df[df['Gender'] == 'Male'].copy()
    male_df['Index'] += 2

    ax.bar(
        ungendered_df['Index'],
        ungendered_df['SelfHireRate'],
        facecolor='none',
        edgecolor=ungendered_df['Color'],
        zorder=2,
    )

    ax.bar(
        female_df['Index'],
        female_df['SelfHireRate'],
        color=female_df['Color'],
        zorder=2,
    )

    ax.bar(
        male_df['Index'],
        male_df['SelfHireRate'],
        facecolor='none',
        edgecolor=male_df['Color'],
        hatch='xxx',
        zorder=2,
    )

    yticks = [0, .05, .1, .15, .2]
    ax.set_yticks(yticks)
    ax.set_yticklabels(hnelib.plot.stringify_numbers_without_ugly_zeros(yticks))
    ax.set_ylabel('self-hire rate', size=axis_label_fontsize)

    ax.set_ylim(0, .22)

    male_to_annotate = male_df[
        male_df['TaxonomyValue'] == 'Medicine and Health'
    ].iloc[0]

    female_to_annotate = female_df[
        female_df['TaxonomyValue'] == 'Medicine and Health'
    ].iloc[0]

    overall_to_annotate = ungendered_df[
        ungendered_df['TaxonomyValue'] == 'Medicine and Health'
    ].iloc[0]

    side_pad = .5

    overall_y = overall_to_annotate['SelfHireRate'] - .005

    x_pad = 3

    labels = ['men', 'women', 'overall']
    label_xs = [male_to_annotate['Index'], female_to_annotate['Index'], overall_to_annotate['Index']]
    ys = [.125, .175, overall_y]
    sides = ['right', 'right', 'left']

    for label, label_x, y, side in zip(labels, label_xs, ys, sides):
        x = label_x - side_pad if side == 'left' else label_x + side_pad
        x_text = label_x - x_pad if side == 'left' else label_x + x_pad

        ax.annotate(
            label,
            (x, y), 
            xytext=(x_text, y),
            arrowprops=hnelib.plot.BASIC_ARROW_PROPS,
            ha='left' if side == 'right' else 'right',
            va='center',
            color=PLOT_VARS['colors']['dark_gray'],
            fontsize=axis_label_fontsize - 5,
        )

    hnelib.plot.add_gridlines(
        ax,
        ys=[tick for tick in ax.get_yticks() if tick not in [.05, .1]],
    )

    ax.set_xticks([])

    for i, row in female_df.iterrows():
        ax.annotate(
            plot_utils.clean_taxonomy_string(row['TaxonomyValue']),
            (row['Index'] - 2.05, .0075), 
            ha='center',
            va='bottom',
            color=row['Color'],
            rotation=90,
            fontsize=annotation_fontsize,
        )

    math_and_comp_x1 = indexes['Mathematics and Computing'] - 2.05
    math_and_comp_x2 = indexes['Mathematics and Computing'] - .05

    xmin, xmax = ax.get_xlim()
    for x1, x2 in [(xmin, math_and_comp_x1 - .5), (math_and_comp_x2 + .5, xmax)]:
        ax.plot(
            [x1, x2],
            [.1, .1],
            alpha=.5,
            color=PLOT_VARS['colors']['dark_gray'],
            zorder=1,
            lw=1,
        )

    ax.set_xlim(xmin, xmax)

def plot_self_hiring_by_gender_horizontal(ax=None):
    if not ax:
        fig, ax = plt.subplots(1, figsize=(7, 4), tight_layout=True)

    taxonomy_annotatio_fontsize = 9
    annotation_fontsize = 8

    spacer_amount = 5

    plot_data = []
    element_id = 0

    df = usfhn.stats.runner.get('self-hires/by-gender/df').rename(columns={
        'SelfHiresFraction': 'SelfHireRate',
    })[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'SelfHireRate',
            'Gender',
        ]
    ].drop_duplicates()

    df = df[
        df['TaxonomyLevel'].isin(['Academia', 'Umbrella'])
    ]

    df['Color'] = df['TaxonomyValue'].apply(PLOT_VARS['colors']['umbrellas'].get)

    df['PercentSelfHires'] = 100 * df['SelfHireRate']

    # adding in self-hires bar
    expectations = usfhn.self_hiring.get_expected_self_hiring_rates_and_compare_to_actual()[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'E[SelfHires]',
        ]
    ].drop_duplicates()

    expectations['E[SelfHires]'] *= 100

    df = df.merge(
        expectations,
        on=[
            'TaxonomyLevel',
            'TaxonomyValue',
        ],
    )

    plot_df = df.copy()[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'SelfHireRate',
            'Gender',
            'E[SelfHires]',
        ]
    ].rename(columns={
        'E[SelfHires]': 'ExpectedSelfHireRate',
    }).drop_duplicates()
    for i, row in plot_df.iterrows():
        for col in plot_df.columns:
            plot_data.append({
                'Element': element_id,
                'Attribute': col,
                'Value': row[col],
            })

        element_id += 1

    umbrellas = ['Academia'] + usfhn.views.get_umbrellas()
    umbrellas = reversed(umbrellas)

    indexes = {u: spacer_amount * (i + 1) for i, u in enumerate(umbrellas)}

    df['Index'] = df['TaxonomyValue'].apply(indexes.get)

    ungendered_df = views.filter_exploded_df(df)

    female_df = df[df['Gender'] == 'Female'].copy()
    female_df['Index'] += 1

    male_df = df[df['Gender'] == 'Male'].copy()
    male_df['Index'] += 2

    ax.barh(
        ungendered_df['Index'],
        ungendered_df['PercentSelfHires'],
        lw=.5,
        facecolor='w',
        edgecolor=ungendered_df['Color'],
        zorder=2,
    )

    ax.scatter(
        ungendered_df['E[SelfHires]'],
        ungendered_df['Index'],
        facecolor='w',
        edgecolor=PLOT_VARS['colors']['dark_gray'],
        zorder=3,
        s=12,
        lw=.65,
    )

    ax.barh(
        female_df['Index'],
        female_df['PercentSelfHires'],
        color=female_df['Color'],
        zorder=2,
        lw=.5,
    )

    ax.barh(
        male_df['Index'],
        male_df['PercentSelfHires'],
        facecolor='w',
        edgecolor=male_df['Color'],
        hatch='xxxxx',
        zorder=2,
        lw=.5,
    )

    xticks = [0, 5, 10, 15, 20]
    ax.set_xticks(xticks)
    ax.set_xticklabels(hnelib.plot.stringify_numbers_without_ugly_zeros(xticks))
    ax.set_xlabel('% self-hires')

    ax.set_xlim(0, 22)

    for i, row in female_df.iterrows():
        ax.annotate(
            plot_utils.clean_taxonomy_string(row['TaxonomyValue']),
            (.2, row['Index'] + 2.2), 
            ha='left',
            va='center',
            color=row['Color'],
            fontsize=hnelib.plot.FONTSIZES['annotation'],
        )

    male_to_annotate = male_df[
        male_df['TaxonomyValue'] == 'Medicine and Health'
    ].iloc[0]

    female_to_annotate = female_df[
        female_df['TaxonomyValue'] == 'Medicine and Health'
    ].iloc[0]

    overall_to_annotate = ungendered_df[
        ungendered_df['TaxonomyValue'] == 'Medicine and Health'
    ].iloc[0]

    vert_pad = .25
    vert_pad_text = .2

    y_pad = 2.5

    labels = ['men', 'women', 'overall']
    label_ys = [male_to_annotate['Index'], female_to_annotate['Index'], overall_to_annotate['Index']]
    xs = [12.5, 17.5, 12.5]
    vas = ['bottom', 'bottom', 'top']

    for label, label_y, x, va in zip(labels, label_ys, xs, vas):
        multiplier = -1 if va == 'top' else 1

        y_start = label_y + (multiplier * vert_pad)
        y_end = label_y + (multiplier * y_pad)
        y_text = y_end + (multiplier * vert_pad_text)

        ax.annotate(
            label,
            (x, y_text), 
            ha='center',
            va=va,
            color=PLOT_VARS['colors']['dark_gray'],
            fontsize=hnelib.plot.FONTSIZES['annotation'],
        )

        ax.annotate(
            "",
            (x, y_start), 
            xytext=(x, y_end),
            arrowprops=hnelib.plot.ZERO_SHRINK_A_ARROW_PROPS,
        )

    expected_annotation = ungendered_df[
        ungendered_df['TaxonomyValue'] == 'Social Sciences'
    ].iloc[0]

    x_pad = .8
    x_pad_text = .4

    y = expected_annotation['Index']
    y_start = y - vert_pad
    y_end = y - y_pad
    x_start = expected_annotation['E[SelfHires]']
    x_end = expected_annotation['E[SelfHires]'] + x_pad
    x_text = x_end + x_pad_text

    ax.annotate(
        "",
        (x_start, y_start), 
        xytext=(x_start, y_end),
        arrowprops=hnelib.plot.ZERO_SHRINK_A_ARROW_PROPS,
    )

    ax.annotate(
        "",
        (x_start, y_end), 
        xytext=(x_end, y_end),
        arrowprops=hnelib.plot.HEADLESS_ARROW_PROPS,
    )

    ax.annotate(
        "network null model",
        (x_text, y_end), 
        ha='left',
        va='center',
        color=PLOT_VARS['colors']['dark_gray'],
        fontsize=hnelib.plot.FONTSIZES['annotation'],
    )

    ax.set_ylim(min(male_df['Index']) - 1, ax.get_ylim()[1])
    ax.set_ylim(min(ungendered_df['Index']) - 4, ax.get_ylim()[1])

    hnelib.plot.add_gridlines(
        ax,
        xs=[tick for tick in ax.get_xticks() if tick not in [5, 10]],
    )

    ymin, ymax = ax.get_ylim()

    ys = [
        # [y_start - .5, indexes['Medicine and Health'] + 2.45],
        [y_start - .5, indexes['Mathematics and Computing'] - .5],
        [indexes['Mathematics and Computing'] - .5, indexes['Mathematics and Computing'] + 2],
        [indexes['Humanities'] - .5, ymax],
    ]

    for y1, y2 in ys:
        ax.plot(
            [10, 10],
            [y1, y2],
            alpha=.5,
            color=PLOT_VARS['colors']['dark_gray'],
            zorder=1,
            lw=.5,
        )

    ax.set_ylim(ymin, ymax)

    ax.set_yticks([])

    return pd.DataFrame(plot_data)


def plot_self_hiring_by_prestige(ax=None):
    if not ax:
        fig, ax = plt.subplots(1, figsize=(3.5, 4), tight_layout=True)

    tick_fontsize = 10
    axis_label_fontsize = 13
    annotation_fontsize = 9

    df = usfhn.self_hiring.compare_self_hire_rate_of_top_institutions_vs_rest()

    field_df = usfhn.views.filter_by_taxonomy(df, 'Field')

    field_df = views.annotate_umbrella_color(field_df, 'Field')

    umbrellas = views.get_umbrellas()
    for (umbrella), rows in field_df.groupby('Umbrella'):
        umbrella_index = umbrellas.index(umbrella)
        ax.scatter(
            [umbrella_index for i in range(len(rows))],
            rows['Ratio'],
            facecolor='none',
            edgecolor=rows['UmbrellaColor'],
            zorder=2,
        )

    ax.set_xticks([])
    ax.set_ylim(-.01, 3.5)
    ax.set_yticks([0, .5, 1, 1.5, 2, 2.5, 3, 3.5])
    ax.set_yticklabels(["0", ".5", "1", 1.5, '2', 2.5, '3', 3.5])

    hnelib.plot.add_gridlines(
        ax,
        xs=[i + .5 for i in range(len(umbrellas))],
        lw=.5,
    )

    xlim = [-.5, len(umbrellas) - .5]
    annotation_pad = 7
    annotation_xlim = [xlim[0], xlim[1] + annotation_pad]
    ax.set_xlim(annotation_xlim)

    ax.spines['bottom'].set_visible(False)

    for y in ax.get_yticks():
        if y == 1:
            lw = .75
            xs = annotation_xlim
            color = 'black'
            alpha = 1
        else:
            lw = .5
            xs = xlim
            color = PLOT_VARS['colors']['dark_gray']
            alpha = .5

        ax.plot(
            xs,
            [y, y],
            color=color,
            alpha=alpha,
            lw=lw,
            zorder=1,
        )

    y_offset = .05
    y_text_offset = .3
    annotation_x = xlim[1] + (annotation_pad / 2)


    ax.annotate(
        "prestigious universities\nself-hire more than\nother universities",
        (annotation_x, 1 + y_offset + y_text_offset),
        ha='center',
        va='bottom',
        color = PLOT_VARS['colors']['dark_gray'],
        fontsize=axis_label_fontsize - 5,
    )

    ax.annotate(
        '',
        (annotation_x, 1 + y_offset + y_text_offset),
        xytext=(annotation_x, 1 + y_offset),
        arrowprops=hnelib.plot.BASIC_ARROW_PROPS,
        color=PLOT_VARS['colors']['dark_gray'],
    )


    ax.annotate(
        "prestigious universities\nself-hire less than\nother universities",
        (annotation_x, 1 - y_offset - y_text_offset),
        ha='center',
        va='top',
        color = PLOT_VARS['colors']['dark_gray'],
        fontsize=axis_label_fontsize - 5,
    )

    ax.annotate(
        '',
        (annotation_x, 1 - y_offset - y_text_offset),
        xytext=(annotation_x, 1 - y_offset),
        arrowprops=hnelib.plot.BASIC_ARROW_PROPS,
        color=PLOT_VARS['colors']['dark_gray'],
    )

    ax.set_ylabel(
        r"$\frac{\mathrm{average\ self-hire\ rate\ at\ top\ 5\ universities}}{\mathrm{average\ self-hire\ rate\ at\ other\ universities}}$".replace("-", u"\u2010"),
        # r"$\frac{\mathrm{mean\ self-hire\ rate\ at\ top\ 5\ universities\ by\ prestige}}{\mathrm{mean\ self-hire\ rate\ at\ all\ other\ universities}}$".replace("-", u"\u2010"),
        size=axis_label_fontsize,
    )


def plot_self_hiring_multi_plot():
    """
    1. self hiring rates by gender at umbrella and academia (bar)
    2. top 5 vs rest self hire rate
    """
    fig, axes = plt.subplots(
        1, 2,
        figsize=(10.5, 4),
        gridspec_kw={'width_ratios': [1.5, 1]},
        tight_layout=True,
    )

    tick_fontsize = 10
    axis_label_fontsize = 13
    annotation_fontsize = 9

    ################################################################################
    # self hire rate by gender
    ################################################################################
    ax = axes[0]

    plot_self_hiring_by_gender(axes[0])

    ################################################################################
    # top 5 vs rest plot
    ################################################################################
    ax = axes[1]
    plot_self_hiring_by_prestige(axes[1])

    hnelib.plot.annotate_plot_letters(axes, [-.1075, -.18])

    hnelib.plot.set_label_fontsize(axes, axis_label_fontsize)
    hnelib.plot.set_ticklabel_fontsize(axes, tick_fontsize)


def plot_gender():
    fig, ax = plt.subplots(figsize=(4, 2.5))

    tick_fontsize = 9
    axis_label_fontsize = 12

    df = usfhn.stats.runner.get('gender/df')

    df['PercentFemale'] = df['FractionFemale'] * 100

    field_df = views.filter_by_taxonomy(df, 'Field')
    # print(field_df.head())
    # import sys; sys.exit()
    field_df = views.annotate_umbrella_color(field_df, 'Field')

    umbrellas = ['Academia'] + views.get_umbrellas()
    umbrellas = list(reversed(umbrellas))

    for umbrella in umbrellas:
        umbrella_index = umbrellas.index(umbrella)

        if umbrella == 'Academia':
            umbrella_row = views.filter_by_taxonomy(df, 'Academia', 'Academia')
            color = PLOT_VARS['colors']['umbrellas']['Academia']
        else:
            rows = field_df[
                field_df['Umbrella'] == umbrella
            ]
            ax.scatter(
                rows['PercentFemale'],
                [umbrella_index for i in range(len(rows))],
                facecolor='none',
                edgecolor=rows['UmbrellaColor'],
                zorder=2,
            )

            umbrella_row = views.filter_by_taxonomy(df, 'Umbrella', umbrella)
            color = PLOT_VARS['colors']['dark_gray']

        ax.scatter(
            umbrella_row['PercentFemale'],
            [umbrella_index],
            color=color,
            zorder=3,
        )

    x_pad = 1.5
    ax.set_ylim(-.5, len(umbrellas) - .5, )
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlim(0, 100)
    for i, umbrella in enumerate(umbrellas):
        ax.set_yticks([])
        ax.annotate(
            plot_utils.clean_taxonomy_string(umbrella),
            (-2.5, i),
            ha='right',
            va='center',
            color=PLOT_VARS['colors']['umbrellas'][umbrella],
            size=tick_fontsize,
            annotation_clip=False,
            
        )

    ax.axvline(
        50,
        color=PLOT_VARS['colors']['dark_gray'],
        lw=1,
        zorder=1,
    )

    hnelib.plot.add_gridlines(
        ax,
        xs=[25, 50, 75, 100],
        ys=[i + .5 for i in range(len(umbrellas))],
        lw=.5,
    )

    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax.set_xlabel('% women', size=axis_label_fontsize)

def plot_field_level_exclusions_by_domain():
    fig, ax = plt.subplots(1, figsize=(8.5, 4), tight_layout=True)

    tick_fontsize = 10
    axis_label_fontsize = 13
    annotation_fontsize = 9

    df = usfhn.stats.runner.get('pool-reduction/umbrella-stats')
    taxonomy_df = usfhn.stats.runner.get('pool-reduction/excluded-taxonomy-sizes').rename(columns={
        'Faculty': 'TaxonomyFaculty',
    })

    df = df.merge(
        taxonomy_df,
        on='Umbrella',
    )

    df = df[
        df['Umbrella'].notna()
    ]

    df['UmbrellaColor'] = df['Umbrella'].apply(PLOT_VARS['colors']['umbrellas'].get)
    df['FadedColor'] = df['UmbrellaColor'].apply(hnelib.plot.set_alpha_on_colors)

    umbrellas = sorted(list(df['Umbrella'].unique()))

    df['UmbrellaIndex'] = df['Umbrella'].apply(umbrellas.index)

    ax.scatter(
        df['ExcludedFaculty'],
        df['UmbrellaIndex'],
        facecolor=PLOT_VARS['colors']['dark_gray'],
        edgecolor=PLOT_VARS['colors']['dark_gray'],
        zorder=3,
    )

    ax.scatter(
        df['TaxonomyFaculty'],
        df['UmbrellaIndex'],
        facecolors='w',
        edgecolors=df['UmbrellaColor'],
    )

    y_pad = .5
    ax.set_ylim(-y_pad, len(umbrellas) - y_pad)
    ax.set_xlim(0, 17500)

    ax.set_yticks([i for i in range(len(umbrellas))])
    ax.set_yticklabels([u for u in umbrellas], va='center')
    for tick, u in zip(ax.get_yticklabels(), umbrellas):
        tick.set_color(PLOT_VARS['colors']['umbrellas'][u])

    ax.tick_params(axis='y', which='both', length=0)


def plot_faculty_ranks_over_time(plot_counts=True, use_raw_data=False):
    if use_raw_data:
        df = pd.read_csv(usfhn.constants.AA_2022_DEGREE_FILTERED_EMPLOYMENT_PATH)
    else:
        df = usfhn.datasets.get_dataset_df('closedness_data', by_year=True)

    df = df[
        [
            'PersonId',
            'Rank',
            'Year',
        ]
    ].drop_duplicates()

    df['Faculty'] = df.groupby(['Year', 'Rank'])['PersonId'].transform('nunique')
    df = df.drop(columns=['PersonId']).drop_duplicates()

    ranks = ['Assistant Professor', 'Associate Professor', 'Professor']

    rank_to_abbreviation = {
        'Assistant Professor': 'Assistant',
        'Associate Professor': 'Associate',
        'Professor': 'Professor',
    }
    rank_abbreviations = list(rank_to_abbreviation.values())
    abbreviation_to_rank = {a: r for r, a in rank_to_abbreviation.items()}

    years = list(sorted(df['Year'].unique()))

    df = hnelib.pandas.aggregate_df_over_column(
        df,
        agg_col='Rank',
        join_cols=[
            'Year',
        ],
        value_cols=['Faculty'],
        agg_value_to_label=rank_to_abbreviation,
    )

    df['TotalFaculty'] = df['AssistantFaculty'] + df['AssociateFaculty'] + df['ProfessorFaculty']
    df['BottomFaculty-high'] = 0
    df['BottomPercent-high'] = 0

    for i, rank in enumerate(rank_abbreviations):
        df[f'{rank}Percent'] = df[f"{rank}Faculty"] / df['TotalFaculty']
        df[f'{rank}Percent'] *= 100

        lower_rank = rank_abbreviations[i - 1] if i else 'Bottom'

        for val_type in ['Faculty', 'Percent']:
            df[f'{rank}{val_type}-low'] = df[f'{lower_rank}{val_type}-high']
            df[f'{rank}{val_type}-high'] = df[f'{rank}{val_type}-low'] + df[f"{rank}{val_type}"]

    if plot_counts:
        fig, axes = plt.subplots(1, 2, figsize=(8, 4), tight_layout=True)
        count_ax = axes[0]
        percent_ax = axes[1]
        val_types = ['Faculty', 'Percent']
    else:
        fig, percent_ax = plt.subplots(figsize=(6, 4), tight_layout=True)
        axes = [percent_ax]
        val_types = ['Percent']


    xticks = list(range(2012, 2021, 2))

    if use_raw_data:
        ytick_max = 220000
    else:
        ytick_max = 181000

    ytick_step = 30000
    yticks = list(range(0, ytick_max + 1, ytick_step))
    yticklabels = [f"{v/1000}K" if v else v for v in yticks]

    df = df.sort_values(by='Year')

    for rank_abbreviation in reversed(rank_abbreviations):
        rank = abbreviation_to_rank[rank_abbreviation]

        color = PLOT_VARS['colors']['ranks'][rank]

        for ax, val_type in zip(axes, val_types):
            ax.plot(
                years,
                df[f'{rank_abbreviation}{val_type}-high'],
                color=color,
                lw=1.5,
                zorder=2,
                label=rank,
            )

            ax.fill_between(
                years,
                df[f'{rank_abbreviation}{val_type}-low'],
                df[f'{rank_abbreviation}{val_type}-high'],
                color=color,
                alpha=.3,
                zorder=1,
            )

    if plot_counts:
        count_ax.set_ylim(yticks[0], ytick_max)
        count_ax.set_yticks(yticks)
        count_ax.set_yticklabels(yticklabels)
        count_ax.set_ylabel('# of faculty')

    percent_ax.set_ylim(0, 100.25)
    percent_ax.set_yticks([0, 25, 50, 75, 100])
    percent_ax.set_ylabel('% of faculty')

    for ax in axes:
        ax.set_xlim(min(years), max(years))
        ax.set_xticks(xticks)
        ax.set_xlabel('year')

        hnelib.plot.add_gridlines_on_ticks(ax, x=False)

    percent_ax.legend(
        bbox_to_anchor=(1, .5),
        bbox_transform=percent_ax.transAxes,
        loc='center left',
    )

    if plot_counts:
        axis_label_x_pads = [-.275, -.185]
    else:
        axis_label_x_pads = []

    hnelib.plot.finalize(axes, axis_label_x_pads)


def plot_change_in_things():
    tick_fontsize = 10
    axis_label_fontsize = 15

    fig, axes = plt.subplots(1, 4, figsize=(14, 6), tight_layout=True)

    umbrellas = ['Academia'] + views.get_umbrellas()

    # how do we space things?
    # 0 1 2
    # |s e|
    #   label
    space = 2
    lines = [i * space for i in range(len(umbrellas) + 1)]
    start_is = {u: umbrellas.index(u) * space + .5 for u in umbrellas}
    end_is = {u: umbrellas.index(u) * space + 1.5 for u in umbrellas}
    umbrella_to_label_position = {u: np.mean([start_is[u], end_is[u]]) for u in umbrellas}

    ################################################################################
    # ginis
    ################################################################################
    df = usfhn.changes.get_gini_coefficients_for_rank_subset(
        ranks=('Associate Professor', 'Professor'),
        outcolumn='AllGiniCoefficient',
    ).merge(
        usfhn.changes.get_gini_coefficients_for_rank_subset(
            ranks=('Assistant Professor',),
            outcolumn='AsstGiniCoefficient',
        ),
        on=['TaxonomyLevel', 'TaxonomyValue'],
    )

    draw_lines_for_change_plot(
        axes[0],
        df,
        umbrellas,
        start_is,
        end_is,
        'AllGiniCoefficient',
        'AsstGiniCoefficient',
    )

    for umbrella in umbrellas:
        axes[0].annotate(
            plot_utils.clean_taxonomy_string(umbrella),
            (umbrella_to_label_position[umbrella], .39), 
            ha='center',
            va='top',
            color=PLOT_VARS['colors']['umbrellas'][umbrella],
            rotation=90,
            fontsize=axis_label_fontsize,
        )

    axes[0].set_yticks([0, .2, .4, .6, .8])
    axes[0].set_yticklabels(["0", ".20", ".40", ".60", ".80"])
    axes[0].set_ylim(0, .8)
    hnelib.plot.add_gridlines(axes[0], ys=[.4, .6, .8], lw=.5)

    axes[0].set_ylabel("Gini coefficient", size=axis_label_fontsize)

    ################################################################################
    # gender
    ################################################################################
    df = usfhn.changes.get_gender_percents_for_rank_subset(
        ranks=('Associate Professor', 'Professor'),
        outcolumn='AllPercentFemale',
    ).merge(
        usfhn.changes.get_gender_percents_for_rank_subset(
            ranks=('Assistant Professor',),
            outcolumn='AsstPercentFemale',
        ),
        on=['TaxonomyLevel', 'TaxonomyValue'],
    )

    draw_lines_for_change_plot(
        axes[1],
        df,
        umbrellas,
        start_is,
        end_is,
        'AllPercentFemale',
        'AsstPercentFemale',
    )

    axes[1].set_yticks([0, 25, 50, 75, 100])
    axes[1].set_yticklabels(["0", "25", "50", "75", "100"])
    axes[1].set_ylim(0, 100)

    hnelib.plot.add_gridlines(axes[1], ys=[25, 50, 100], lw=.5)
    axes[1].set_ylabel("% women", size=axis_label_fontsize)

    field_df = df[
        df['TaxonomyLevel'] == 'Field'
    ]
    field_df = views.annotate_umbrella(field_df, taxonomization_level='Field')

    annotation_df = field_df[
        field_df['Umbrella'] == 'Education'
    ]

    annotation_offset = 5
    arrow_pad = 1
    start = start_is['Education']
    end = end_is['Education']

    y_val = max(annotation_df['AllPercentFemale'])
    axes[1].annotate(
        'full + associate professors',
        (start, y_val + arrow_pad), 
        xytext=(start, y_val + arrow_pad + annotation_offset),
        ha='center',
        va='bottom',
        color=PLOT_VARS['colors']['dark_gray'],
        rotation=90,
        arrowprops=dict(arrowstyle='->', color=PLOT_VARS['colors']['dark_gray']),
    )

    y_val = max(annotation_df['AsstPercentFemale'])
    axes[1].annotate(
        'assistant professors',
        (end, y_val + arrow_pad), 
        xytext=(end, y_val + arrow_pad + annotation_offset),
        ha='center',
        va='bottom',
        color=PLOT_VARS['colors']['dark_gray'],
        rotation=90,
        arrowprops=dict(arrowstyle='->', color=PLOT_VARS['colors']['dark_gray']),
    )

    ################################################################################
    # steepness
    ################################################################################
    df = usfhn.changes.get_steepness_for_rank_subset(
        ranks=('Associate Professor', 'Professor'),
        outcolumn='AllViolations',
    ).merge(
        usfhn.changes.get_steepness_for_rank_subset(
            ranks=('Assistant Professor',),
            outcolumn='AsstViolations',
        ),
        on=['TaxonomyLevel', 'TaxonomyValue'],
    )

    draw_lines_for_change_plot(
        axes[2],
        df,
        umbrellas,
        start_is,
        end_is,
        'AllViolations',
        'AsstViolations',
    )

    axes[2].set_ylim(0, .25)
    axes[2].set_yticks([0, .05, .1, .15, .20, .25])
    axes[2].set_yticklabels(["0", ".05", ".10", ".15", ".20", ".25"])

    hnelib.plot.add_gridlines(axes[2], ys=axes[2].get_yticks(), lw=.5)
    axes[2].set_ylabel("fraction of up-the-hierarchy hires", size=axis_label_fontsize)

    ################################################################################
    # self hire rates
    ################################################################################
    df = usfhn.changes.get_self_hire_rate_for_rank_subset(
        ranks=('Associate Professor', 'Professor'),
        outcolumn='AllSelfHireRate',
    ).merge(
        usfhn.changes.get_self_hire_rate_for_rank_subset(
            ranks=('Assistant Professor',),
            outcolumn='AsstSelfHireRate',
        ),
        on=['TaxonomyLevel', 'TaxonomyValue'],
    )

    draw_lines_for_change_plot(
        axes[3],
        df,
        umbrellas,
        start_is,
        end_is,
        'AllSelfHireRate',
        'AsstSelfHireRate',
    )

    axes[3].set_yticks([0, .1, .2, .3, .4])
    axes[3].set_yticklabels(["0", ".1", ".2", ".3", ".4"])
    axes[3].set_ylim(0, .4)

    hnelib.plot.add_gridlines(axes[3], ys=axes[3].get_yticks(), lw=.5)
    axes[3].set_ylabel("self-hire rate", size=axis_label_fontsize)

    for ax in axes:
        ax.set_xlim(lines[0], lines[-1] * 1.001)
        ax.set_xticks([])
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        hnelib.plot.add_gridlines(ax, xs=lines, lw=.5)

    for _ax, label, x_pad in zip(axes, ['A', 'B', 'C', 'D'], [-.16, -.16, -.16, -.16]):
        _ax.text(
            x_pad,
            1.1,
            label,
            transform=_ax.transAxes,
            fontname='Arial',
            fontsize=18,
            fontweight='bold',
            va='top',
            ha='left',
        )


def draw_lines_for_change_plot(ax, df, umbrellas, start_is, end_is, start_col, end_col):
    umbrella_df = df[
        df['TaxonomyLevel'] == 'Umbrella'
    ]

    academia_df = df[
        df['TaxonomyLevel'] == 'Academia'
    ]

    umbrella_df = pd.concat([academia_df, umbrella_df])

    field_df = df[
        df['TaxonomyLevel'] == 'Field'
    ]
    field_df = views.annotate_umbrella_color(field_df, taxonomization_level='Field')

    for umbrella in umbrellas:
        start = start_is[umbrella]
        end = end_is[umbrella]

        field_rows = field_df[
            field_df['Umbrella'] == umbrella
        ]

        for _, row in field_rows.iterrows():
            ax.plot(
                [start, end],
                [row[start_col], row[end_col]],
                color=row['UmbrellaColor'],
                lw=1.5,
                alpha=.75,
                zorder=1,
            )

            ax.scatter(
                [start],
                [row[start_col]],
                color=row['UmbrellaColor'],
                zorder=2,
                s=17,
            )

            ax.scatter(
                [end],
                [row[end_col]],
                facecolor='white',
                edgecolor=row['UmbrellaColor'],
                zorder=2,
                s=17,
            )

        umbrella_row = umbrella_df[
            (umbrella_df['TaxonomyValue'] == umbrella)
        ].iloc[0]

        if umbrella == 'Academia':
            color = PLOT_VARS['colors']['umbrellas']['Academia']
        else:
            color = PLOT_VARS['colors']['dark_gray']

        ax.plot(
            [start, end],
            [umbrella_row[start_col], umbrella_row[end_col]],
            color=color,
            lw=2.5,
            zorder=3,
        )

        ax.scatter(
            [start],
            [umbrella_row[start_col]],
            color=color,
            zorder=4,
        )

        ax.scatter(
            [end],
            [umbrella_row[end_col]],
            facecolor='white',
            edgecolor=color,
            zorder=4,
        )

def plot_permeability_multi_plot():
    fig, axes = plt.subplots(
        1, 2,
        figsize=(5, 3.5),
        tight_layout=True,
    )

    tick_fontsize = 10
    axis_label_fontsize = 12

    ################################################################################
    # out of U.S.
    ################################################################################
    df = views.filter_exploded_df(usfhn.closedness.get_closednesses())
    df['NonUSPhDPercent'] = df['NonUSPhD'] / df['FacultyCount']
    df['NonUSPhDPercent'] *= 100

    field_df = df[
        df['TaxonomyLevel'] == 'Field'
    ]

    field_df = views.annotate_umbrella_color(field_df, 'Field')

    ax = axes[0]

    umbrellas = views.get_umbrellas()
    for (umbrella), rows in field_df.groupby('Umbrella'):
        umbrella_index = umbrellas.index(umbrella)
        ax.scatter(
            [umbrella_index for i in range(len(rows))],
            rows['NonUSPhDPercent'],
            facecolor='none',
            edgecolor=rows['UmbrellaColor'],
            zorder=2,
        )

        umbrella_row = df[
            (df['TaxonomyLevel'] == 'Umbrella')
            &
            (df['TaxonomyValue'] == umbrella)
        ].iloc[0]

        ax.scatter(
            [umbrella_index],
            umbrella_row['NonUSPhDPercent'],
            color=PLOT_VARS['colors']['dark_gray'],
            zorder=3,
        )

    x_pad = .5
    ax.set_xlim(-x_pad, len(umbrellas) - x_pad)
    ax.set_yticks([0, 10, 20, 30])
    ax.set_ylim(0, 51)
    # ax.set_yticks([0, 10, 20, 30, 40])
    # ax.set_ylim(0, 40)
    for i, umbrella in enumerate(umbrellas):
        ax.set_xticks([])
        ax.annotate(
            plot_utils.clean_taxonomy_string(umbrella),
            (i, 32.7), 
            ha='center',
            va='bottom',
            color=PLOT_VARS['colors']['umbrellas'][umbrella],
            rotation=90,
        )


    hnelib.plot.add_gridlines(
        ax,
        ys=ax.get_yticks(),
        xs=[i + .5 for i in range(len(umbrellas) - 1)],
        lw=.5,
    )

    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax.set_ylabel('% of faculty trained outside the U.S.', size=axis_label_fontsize)

    ################################################################################
    # out of field
    ################################################################################
    df = views.filter_exploded_df(usfhn.closedness.get_closednesses())
    df['USPhDOutOfFieldPercent'] = df['USPhDOutOfField'] / df['FacultyCount']
    df['USPhDOutOfFieldPercent'] *= 100
    df['USPhDInFieldPercent'] = df['USPhDInField'] / df['FacultyCount']
    df['USPhDInFieldPercent'] *= 100

    field_df = df[
        df['TaxonomyLevel'] == 'Field'
    ]

    field_df = views.annotate_umbrella_color(field_df, 'Field')

    ax = axes[1]

    umbrellas = views.get_umbrellas()
    for (umbrella), rows in field_df.groupby('Umbrella'):
        umbrella_index = umbrellas.index(umbrella)
        ax.scatter(
            [umbrella_index for i in range(len(rows))],
            rows['USPhDOutOfFieldPercent'],
            facecolor='none',
            edgecolor=rows['UmbrellaColor'],
            zorder=2,
        )

        umbrella_row = df[
            (df['TaxonomyLevel'] == 'Umbrella')
            &
            (df['TaxonomyValue'] == umbrella)
        ].iloc[0]

        ax.scatter(
            [umbrella_index],
            umbrella_row['USPhDOutOfFieldPercent'],
            color=PLOT_VARS['colors']['dark_gray'],
            zorder=3,
        )

    x_pad = .5
    ax.set_xlim(-x_pad, len(umbrellas) - x_pad)
    ax.set_yticks([0, 25, 50])
    ax.set_ylim(0, 85)
    for i, umbrella in enumerate(umbrellas):
        ax.set_xticks([])
        ax.annotate(
            plot_utils.clean_taxonomy_string(umbrella),
            (i, 55), 
            ha='center',
            va='bottom',
            color=PLOT_VARS['colors']['umbrellas'][umbrella],
            rotation=90,
        )


    hnelib.plot.add_gridlines(
        ax,
        ys=ax.get_yticks(),
        xs=[i + .5 for i in range(len(umbrellas) - 1)],
        lw=.5,
    )

    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax.set_ylabel('% of faculty trained out of field\n(lower bound)', size=axis_label_fontsize)

    for _ax, label, x_pad in zip(axes, ['A', 'B'], [-.33, -.29]):
        _ax.text(
            x_pad,
            1.2,
            label,
            transform=_ax.transAxes,
            fontname='Arial',
            fontsize=18,
            fontweight='bold',
            va='top',
            ha='left',
        )

def plot_permeability_multi_plot_horizontal():
    # DOUBLE WIDE
    # fig, axes = plt.subplots(
    #     1, 2,
    #     figsize=(10, 2.5),
    #     tight_layout=True,
    # )

    fig, axes = plt.subplots(
        2, 1,
        figsize=(4.5, 5),
        tight_layout=True,
    )

    tick_fontsize = 10
    axis_label_fontsize = 12

    ################################################################################
    # out of U.S.
    ################################################################################
    df = views.filter_exploded_df(usfhn.closedness.get_closednesses())
    df['NonUSPhDPercent'] = df['NonUSPhD'] / df['FacultyCount']
    df['NonUSPhDPercent'] *= 100

    field_df = df[
        df['TaxonomyLevel'] == 'Field'
    ]

    field_df = views.annotate_umbrella_color(field_df, 'Field')

    ax = axes[0]

    umbrellas = list(reversed(views.get_umbrellas()))
    for (umbrella), rows in field_df.groupby('Umbrella'):
        umbrella_index = umbrellas.index(umbrella)
        ax.scatter(
            rows['NonUSPhDPercent'],
            [umbrella_index for i in range(len(rows))],
            facecolor='none',
            edgecolor=rows['UmbrellaColor'],
            zorder=2,
        )

        umbrella_row = df[
            (df['TaxonomyLevel'] == 'Umbrella')
            &
            (df['TaxonomyValue'] == umbrella)
        ].iloc[0]

        ax.scatter(
            umbrella_row['NonUSPhDPercent'],
            [umbrella_index],
            color=PLOT_VARS['colors']['dark_gray'],
            zorder=3,
        )

    y_pad = .5
    ax.set_ylim(-y_pad, len(umbrellas) - y_pad)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_xlim(0, 35)

    ax.set_yticks([i for i in range(len(umbrellas))])
    ax.set_yticklabels([u for u in umbrellas], va='center')
    for tick, u in zip(ax.get_yticklabels(), umbrellas):
        tick.set_color(PLOT_VARS['colors']['umbrellas'][u])

    ax.tick_params(axis='y', which='both', length=0)

    hnelib.plot.add_gridlines(
        ax,
        ys=[i + .5 for i in range(len(umbrellas))],
        xs=ax.get_xticks(),
        lw=.5,
    )

    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax.set_xlabel(
        '% of faculty trained outside the U.S.',
        size=axis_label_fontsize
    )

    ################################################################################
    # out of field
    ################################################################################
    df = views.filter_exploded_df(usfhn.closedness.get_closednesses())
    df['USPhDOutOfFieldPercent'] = df['USPhDOutOfField'] / df['FacultyCount']
    df['USPhDOutOfFieldPercent'] *= 100
    df['USPhDInFieldPercent'] = df['USPhDInField'] / df['FacultyCount']
    df['USPhDInFieldPercent'] *= 100

    field_df = df[
        df['TaxonomyLevel'] == 'Field'
    ]

    field_df = views.annotate_umbrella_color(field_df, 'Field')

    ax = axes[1]

    umbrellas = list(reversed(views.get_umbrellas()))
    for (umbrella), rows in field_df.groupby('Umbrella'):
        umbrella_index = umbrellas.index(umbrella)
        ax.scatter(
            rows['USPhDOutOfFieldPercent'],
            [umbrella_index for i in range(len(rows))],
            facecolor='none',
            edgecolor=rows['UmbrellaColor'],
            zorder=2,
        )

        umbrella_row = df[
            (df['TaxonomyLevel'] == 'Umbrella')
            &
            (df['TaxonomyValue'] == umbrella)
        ].iloc[0]

        ax.scatter(
            umbrella_row['USPhDOutOfFieldPercent'],
            [umbrella_index],
            color=PLOT_VARS['colors']['dark_gray'],
            zorder=3,
        )

    y_pad = .5
    ax.set_ylim(-y_pad, len(umbrellas) - y_pad)
    ax.set_xticks([0, 15, 30, 45, 60])
    ax.set_xlim(0, 60)

    ax.set_yticks([i for i in range(len(umbrellas))])
    ax.set_yticklabels([u for u in umbrellas], va='center')
    for tick, u in zip(ax.get_yticklabels(), umbrellas):
        tick.set_color(PLOT_VARS['colors']['umbrellas'][u])

    ax.tick_params(axis='y', which='both', length=0)

    hnelib.plot.add_gridlines(
        ax,
        ys=[i + .5 for i in range(len(umbrellas))],
        xs=ax.get_xticks(),
        lw=.5,
    )

    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax.set_xlabel('% of faculty trained out of field (lower bound)', size=axis_label_fontsize)

    for _ax, label, x_pad in zip(axes, ['A', 'B'], [-.4, -.4]):
        _ax.text(
            x_pad,
            1.23,
            label,
            transform=_ax.transAxes,
            fontname='Arial',
            fontsize=18,
            fontweight='bold',
            va='top',
            ha='left',
        )


def plot_umbrella_out_of_field_to_non_us(ax=None, axis_label_fontsize=12, tick_fontsize=12):
    df = usfhn.closedness.get_closednesses()
    df = views.filter_exploded_df(df)

    df = df[
        df['TaxonomyLevel'] == 'Umbrella'
    ].drop_duplicates()

    df = views.annotate_umbrella_color(df, taxonomization_level='Umbrella')

    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4), tight_layout=True)

    df['OutOfField/NonUS'] = df['USPhDOutOfField'] / df['NonUSPhD']

    df = df.sort_values(by='Umbrella')
    umbrellas = views.get_umbrellas()

    pad_before_umbrella = 1.5
    pad_after_umbrella = 1.5
    pad_between_stats = 1
    x = 0

    width = .75
    text_offset = width / 2

    colors = df['UmbrellaColor']

    out_of_field_xs = []
    out_of_field_ys = []

    non_us_xs = []
    non_us_ys = []

    tick_positions = []
    annotation_positions = []
    for umbrella in umbrellas:
        x += pad_before_umbrella
        annotation_positions.append(x - (.1 * pad_after_umbrella))
        u_row = df[df['TaxonomyValue'] == umbrella].iloc[0]
        out_of_field_xs.append(x)
        out_of_field_ys.append(100 * u_row['USPhDOutOfField'] / u_row['FacultyCount'])

        tick_positions.append(x + (pad_between_stats / 2))

        x += pad_between_stats

        non_us_xs.append(x)
        non_us_ys.append(100 * u_row['NonUSPhD'] / u_row['FacultyCount'])
        x += pad_after_umbrella

    ax.bar(
        out_of_field_xs,
        out_of_field_ys,
        width=width,
        color=colors,
        zorder=2,
    )

    ax.bar(
        non_us_xs,
        non_us_ys,
        width=width,
        edgecolor=colors,
        facecolor='none',
        hatch='xx',
        zorder=2,
    )

    ax.legend(
        handles=[
            mpatches.Patch(
                edgecolor=PLOT_VARS['colors']['dark_gray'],
                facecolor='none',
                hatch='xx',
                label='% of non-U.S. PhDs',
            ),
            mpatches.Patch(
                color=PLOT_VARS['colors']['dark_gray'],
                label='% of inter-field PhDs',
            ),
        ],
        loc='upper left',
        prop={'size': 9}
    )

    for umbrella, x_pos, color in zip(df['TaxonomyValue'], annotation_positions, colors):
        ax.annotate(
            plot_utils.clean_taxonomy_string(umbrella),
            (x_pos - text_offset, .15 * pad_before_umbrella), 
            ha='right',
            va='bottom',
            color=color,
            rotation=90,
        )

    ax.set_xticks([])
    ax.set_yticks([0, 5, 10, 15, 20])
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax.set_ylabel('% of faculty', size=axis_label_fontsize)
    hnelib.plot.add_gridlines(ax, ys=[5, 10, 15, 20], lw=.5, alpha=.2)

    return ax


def plot_self_hiring_null_model(taxonomy_level, taxonomy_value=None):
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    for i, null_function in enumerate(['uniform', 'max_other', 'prestige_neighbor', 'chung_lu']):
        df = usfhn.self_hiring.null_model(null_function)
        df = df[
            (df['Year'] == constants.YEAR_UNION)
            &
            (df['Gender'] == constants.GENDER_AGNOSTIC)
        ].drop(columns=[
            'Year', 'Gender'
        ])

        df = df[
            df['TaxonomyLevel'] == taxonomy_level
        ]

        if not taxonomy_value:
            taxonomy_value = sorted(list(df['TaxonomyValue'].unique()))[0]
            
        df = df[
            df['TaxonomyValue'] == taxonomy_value
        ]

        df = usfhn.institutions.annotate_institution_name(df)
        df = views.annotate_umbrella_color(df, taxonomy_level)

        df = df[
            ['OrdinalRank', 'ActualOverExpected', 'Hires', 'SelfHiresDelta', 'InstitutionName', 'UmbrellaColor']
        ].drop_duplicates()

        ratio_ax = axes[0, i]
        percent_excess_ax = axes[1, i]

        ratio_ax.scatter(
            df['OrdinalRank'],
            df['ActualOverExpected'],
            alpha=.5,
            color=df['UmbrellaColor'],
        )
        ratio_ax.set_yscale('log')
        ratio_ax.set_ylim(10**-1, ratio_ax.get_ylim()[1])

        df['PercentExcess'] = 100 * (df['SelfHiresDelta'] / df['Hires'])

        percent_excess_ax.scatter(
            df['OrdinalRank'],
            df['PercentExcess'],
            alpha=.5,
            color=df['UmbrellaColor'],
        )

        y_min, y_max = percent_excess_ax.get_ylim()
        percent_excess_ax.set_yticks(np.arange(0, y_max, 10))

        # excess annotations
        percent_excess_annotation = df[df['OrdinalRank'] < 100]
        if not percent_excess_annotation.empty:
            percent_excess_annotation = percent_excess_annotation[
                percent_excess_annotation['PercentExcess'] == max(percent_excess_annotation['PercentExcess'])
            ].iloc[0]
            percent_excess_ax.annotate(
                text=percent_excess_annotation['InstitutionName'],
                xy=(
                    percent_excess_annotation['OrdinalRank'] + 5,
                    percent_excess_annotation['PercentExcess'],
                ),
                ha='left',
                va='center',
            )

        if not i:
            ratio_ax.set_ylabel('% change in self hire rate\nfrom null')
            percent_excess_ax.set_ylabel('excess self hires\nas % of total faculty')

        percent_excess_ax.set_xlabel('ordinal rank\nhigh rank ' + r'$\rightarrow$' + ' low')

    axes[0, 0].set_title('E[SH] =\n1/#institutions')
    axes[0, 1].set_title('E[SH] =\nmax(hired from other)')
    axes[0, 2].set_title('E[SH] =\navg(neighbors)')
    axes[0, 3].set_title('E[SH] =\nChung-Lu')

    if taxonomy_level == 'Academia':
        title = f"{taxonomy_level} self hiring vs null"
    else:
        title = f"{taxonomy_value} self hiring vs null ({taxonomy_level} level)"

    plt.suptitle(title, size=36)
    plt.tight_layout()


def plot_self_hiring_null_model_umbrella():
    umbrellas = sorted(list(views.get_taxonomization()['Umbrella'].unique()))
    n_umbrellas = len(umbrellas)
    per_umbrella_height = 5
    fig, axes = plt.subplots(n_umbrellas, 4, figsize=(20, 1 + (per_umbrella_height * n_umbrellas)))

    for i, umbrella in enumerate(umbrellas):
        for j, null_function in enumerate(['uniform', 'max_other', 'prestige_neighbor', 'chung_lu']):
            ax = axes[i, j]

            df = usfhn.self_hiring.null_model(null_function)
            df = df[
                (df['Year'] == constants.YEAR_UNION)
                &
                (df['Gender'] == constants.GENDER_AGNOSTIC)
            ].drop(columns=[
                'Year', 'Gender'
            ])

            df = df[
                (df['TaxonomyLevel'] == 'Umbrella')
                &
                (df['TaxonomyValue'] == umbrella)
            ]

            df = views.annotate_umbrella_color(df, 'Umbrella')

            df = df[
                ['OrdinalRank', 'ActualOverExpected', 'UmbrellaColor']
            ].drop_duplicates()

            ax.scatter(
                df['OrdinalRank'],
                df['ActualOverExpected'],
                alpha=.5,
                color=df['UmbrellaColor'],
            )
            ax.set_yscale('log')

            if not j:
                ax.set_ylabel(f'% change in self hire rate\nfrom null ({umbrella})')

    axes[0, 0].set_title('E[SH] =\n1/#institutions')
    axes[0, 1].set_title('E[SH] =\nmax(hired from other)')
    axes[0, 2].set_title('E[SH] =\navg(neighbors)')
    axes[0, 3].set_title('E[SH] =\nChung-Lu')

    for null_type_axes in axes.T:
        y_mins, y_maxs = zip(*[ax.get_ylim() for ax in null_type_axes])
        y_min = min(y_mins)
        y_max = max(y_maxs)

        for ax in null_type_axes:
            ax.set_ylim(y_min, y_max)


    # plt.suptitle(f"self hiring vs null (umbrella level)", size=36)
    plt.tight_layout()

def plot_institutional_self_hiring_versus_prestige():
    df = usfhn.stats.runner.get('self-hire/by-institution').rename(columns={
        'SelfHireFraction': 'SelfHireRate',
    })
    df = df.merge(
        usfhn.stats.runner.get('ranks/df', rank_type='prestige'),
        on=['InstitutionId', 'TaxonomyLevel', 'TaxonomyValue']
    )

    df = df[
        (df['Year'] == constants.YEAR_UNION)
        &
        (df['Gender'] == constants.GENDER_AGNOSTIC)
    ].drop(columns=[
        'Year', 'Gender'
    ])

    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes[0, 0].set_ylabel('self-hire rate')
    axes[1, 0].set_ylabel('self-hire rate')
    axes[1, 1].set_xlabel('prestige\n' + r'low $\rightarrow$ high')
    axes[1, 0].set_xlabel('prestige\n' + r'low $\rightarrow$ high')
    axes = axes.flatten()

    for ax, level in zip(axes, ['Academia', 'Umbrella', 'Area', 'Field']):
        level_df = df[df['TaxonomyLevel'] == level]
        level_df = level_df[
            ['InstitutionId', 'SelfHireRate', 'TaxonomyValue', 'NormalizedRank']
        ].drop_duplicates()

        level_df = views.annotate_umbrella_color(level_df, level)

        ax.set_title(f'{level.lower()}')
        ax.scatter(
            level_df['NormalizedRank'], level_df['SelfHireRate'],
            alpha=.2,
            color=level_df['UmbrellaColor'],
            s=8,
        )
        ax.set_ylim(0, .8)

    plt.suptitle('prestige vs self-hire rate, by institution', size=36)


def plot_institutional_self_hiring_versus_size():
    df = usfhn.stats.runner.get('self-hire/by-institution').rename(columns={
        'SelfHireFraction': 'SelfHireRate',
    })
    hires = usfhn.stats.runner.get('basics/faculty-hiring-network')[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'InstitutionId',
            'InDegree',
        ]
    ].rename(columns={
        'InDegree': 'FacultyCount'
    })
    
    df = df.merge(
        hires,
        on=[
            'TaxonomyLevel',
            'TaxonomyValue',
            'InstitutionId',
        ]
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes[0, 0].set_ylabel('self-hire rate')
    axes[1, 0].set_ylabel('self-hire rate')
    axes[1, 1].set_xlabel('number of faculty')
    axes[1, 0].set_xlabel('number of faculty')
    axes = axes.flatten()

    for ax, level in zip(axes, ['Academia', 'Umbrella', 'Area', 'Field']):
        level_df = df[df['TaxonomyLevel'] == level]
        level_df = level_df[
            ['InstitutionId', 'SelfHireRate', 'TaxonomyValue', 'FacultyCount']
        ].drop_duplicates()

        level_df = views.annotate_umbrella_color(level_df, level)

        ax.set_title(f'{level.lower()}')
        ax.scatter(
            level_df['FacultyCount'], level_df['SelfHireRate'],
            alpha=.2,
            color=level_df['UmbrellaColor'],
            s=8,
        )
        ax.set_ylim(0, .8)

    plt.suptitle(f'number of faculty vs self-hire rate, by institution', size=36)


def plot_gender_fractions():
    fig, axes, legend_ax = plot_univariate_value_across_taxonomy_levels(
        measurements.get_taxonomy_gender_ratios(),
        'FractionFemale',
        'fraction female',
        y_min=0,
        y_max=1,
        y_step=.1,
    )

def plot_taxonomy_field_sizes():
    df = usfhn.datasets.CURRENT_DATASET.ranks.copy()[
        ['TaxonomyLevel', 'TaxonomyValue', 'InstitutionId']
    ].drop_duplicates()

    df = df[
        df['TaxonomyLevel'] != 'Taxonomy'
    ]

    df['InstitutionsInField'] = df.groupby(
        ['TaxonomyValue', 'TaxonomyLevel']
    )['InstitutionId'].transform('nunique')
    df = df.drop(columns=['InstitutionId']).drop_duplicates()
    df = df.sort_values(by=['InstitutionsInField'])

    fig, axes, legend_ax = plot_univariate_value_across_taxonomy_levels(
        df,
        'InstitutionsInField',
        'institutional participation',
        y_min=0,
        y_max=400,
        y_step=50,
    )


def plot_taxonomy_gender_ratios():
    fig, axes, legend_ax = plot_univariate_value_across_taxonomy_levels(
        measurements.get_taxonomy_gender_ratios(),
        'FractionFemale',
        'fraction female',
        y_min=0,
        y_max=1,
        y_step=.1,
    )

def plot_self_hiring_by_taxonomy():
    plot_univariate_value_across_taxonomy_levels(
        usfhn.stats.runner.get('self-hire/df'),
        'SelfHireFraction',
        'self-hire rate',
        y_min=0,
        y_max=.4,
        y_step=.05,
    )

def plot_self_hiring_at_top_vs_bottom_by_taxonomy():
    plot_univariate_value_across_taxonomy_levels(
        usfhn.self_hiring.compare_self_hire_rate_of_top_institutions_vs_rest(),
        'Ratio',
        'self hiring rate of top 5/rest',
        y_min=0,
        y_max=5,
        y_step=1,
    )

def plot_self_hiring_at_bottom_vs_rest_by_taxonomy():
    plot_univariate_value_across_taxonomy_levels(
        usfhn.self_hiring.compare_self_hire_rate_of_bottom_institutions_vs_rest(),
        'Ratio',
        'self hiring rate of top(n)/bottom 50',
        y_min=0,
        y_max=5,
        y_step=1,
    )


def plot_steepness_by_taxonomy():
    plot_univariate_value_across_taxonomy_levels(
        views.filter_exploded_df(measurements.get_steepness_by_taxonomy()),
        'Steepness',
        'fraction of downward edges',
        y_min=.4,
        y_max=1,
        y_step=.1,
    )

def plot_up_down_ratio_by_taxonomy():
    df = measurements.get_steepness_by_taxonomy()
    df['RatioOfDownToUp'] = df['DownwardEdges'] / df['UpwardEdges']

    plot_univariate_value_across_taxonomy_levels(
        measurements.get_steepness_by_taxonomy(),
        'RatioOfDownToUp',
        "ratio of down-hierarchy to up-hierarchy edges",
        
        y_label='downward:upward ratio',
        y_min=0,
        y_max=20,
        y_step=5,
    )

def plot_rank_change_by_taxonomy():
    df = usfhn.stats.runner.get('ranks/placements')

    df['MeanRankChange'] = df.groupby(
        ['TaxonomyLevel', 'TaxonomyValue']
    )['NormalizedRankDifference'].transform('mean')

    df = df[
        ['TaxonomyLevel', 'TaxonomyValue', 'MeanRankChange']
    ].drop_duplicates()

    plot_univariate_value_across_taxonomy_levels(
        df,
        'MeanRankChange',
        'PhD - employment prestige (mean)',
        y_min=0,
        y_max=-.5,
        y_step=.1,
    )

def plot_closedness_by_taxonomy():
    plot_univariate_value_across_taxonomy_levels(
        usfhn.closedness.get_closednesses(),
        'Closedness',
        'closedness',
        y_min=.35,
        y_max=.95,
        y_step=.1,
    )

def plot_non_us_closedness_by_taxonomy():
    df = usfhn.closedness.get_closednesses().copy()
    df['NonUSPhD'] = df['NonUSPhD'] / df['FacultyCount']

    plot_univariate_value_across_taxonomy_levels(
        df,
        'NonUSPhD',
        'Non-US',
        y_min=0,
        y_max=.3,
        y_step=.05,
    )

def plot_us_phd_in_field_closedness_by_taxonomy():
    df = usfhn.closedness.get_closednesses().copy()
    df['USPhDInField'] = df['USPhDInField'] / df['FacultyCount']

    plot_univariate_value_across_taxonomy_levels(
        df,
        'USPhDInField',
        'US PhD trained in-field',
        y_min=.4,
        y_max=1,
        y_step=.1,
    )

def plot_us_phd_out_of_field_closedness_by_taxonomy():
    df = usfhn.closedness.get_closednesses().copy()
    df['USPhDOutOfField'] = df['USPhDOutOfField'] / df['FacultyCount']
    plot_univariate_value_across_taxonomy_levels(
        df,
        'USPhDOutOfField',
        'US PhD trained out-of-field',
        y_min=0,
        y_max=.6,
        y_step=.05,
    )

def plot_us_phd_in_field_vs_non_us():
    df = usfhn.closedness.get_closednesses()
    df['USPhDInField'] = df['USPhDInField'] / df['FacultyCount']
    df['NonUSPhD'] = df['NonUSPhD'] / df['FacultyCount']

    plot_relationship_across_taxonomy_levels(
        usfhn.closedness.get_closednesses(),
        'USPhDInField',
        'NonUSPhD',
        title=f'in-field US v.s. Non-US PhD',
        x_label='in-field US',
        y_label='Non-US',
        x_lim=[.4, 1],
        y_lim=[0, .3],
    )

def plot_closedness_to_field_size():
    df = usfhn.closedness.get_closednesses().merge(
        usfhn.stats.runner.get('taxonomy/institutions'),
        on=['TaxonomyLevel', 'TaxonomyValue']
    )

    df = views.filter_exploded_df(df)

    df = df[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'Count',
            'Closedness',
        ]
    ].drop_duplicates()

    plot_relationship_across_taxonomy_levels(
        df,
        'Count',
        'Closedness',
        title=f'closedness v.s. taxonomy size',
        x_label='taxonomy size',
        y_label='closedness',
        x_lim=[0, 400],
        y_lim=[.35, .95],
    )




def plot_relationship_across_taxonomy_levels(
    df,
    x_column,
    y_column,
    title,
    x_lim=None,
    y_lim=None,
    xticks=None,
    yticks=None,
    xticklabels=None,
    yticklabels=None,
    x_label=None,
    y_label=None,
    s_column=None,
    color_column='UmbrellaColor',
    facecolor_column='none',
    edgecolor_column='none',
    scatter_kwargs={},
    figsize=(12, 7),
    set_equal_aspect=False,
    x_keep_zero_line=False,
    y_keep_zero_line=False,
    filter_kwargs={},
    fill_academia=False,
    extra_legend_handles=[],
):
    df = views.filter_exploded_df(df, **filter_kwargs)

    fig, axes = plt.subplots(1, 4, figsize=figsize, gridspec_kw={'width_ratios': [1, 1, 1, .25]})
    axes = axes.flatten()
    legend_ax = axes[-1]
    axes = axes[:-1]

    legend_handles = plot_utils.get_umbrella_legend_handles() + [
        Line2D(
            [], [],
            color='none',
            marker='o',
            markerfacecolor=PLOT_VARS['colors']['Academia'] if fill_academia else 'none',
            markeredgecolor=PLOT_VARS['colors']['Academia'],
            label='academia',
        )
    ]

    if extra_legend_handles:
        legend_handles.extend(extra_legend_handles)

    legend_ax.legend(handles=legend_handles, loc='center', prop={'size': 12})
    legend_ax.axis('off')

    for ax, level in zip(axes, ['Field', 'Area', 'Umbrella']):
        level_df = df[
            df['TaxonomyLevel'] == level
        ].copy().drop_duplicates()

        if color_column == 'UmbrellaColor' and 'UmbrellaColor' not in level_df.columns:
            level_df = views.annotate_umbrella_color(level_df, level)

        ax.set_title(level.lower())
        for umbrella, rows in level_df.groupby('Umbrella'):
            rows = rows.drop_duplicates(subset=[x_column, y_column])

            _scatter_kwargs = {**scatter_kwargs}
            if s_column:
                _scatter_kwargs['s'] = rows[s_column]

            if color_column:
                _scatter_kwargs['color'] = rows[color_column]
            elif facecolor_column and edgecolor_column:
                _scatter_kwargs['facecolor'] = rows[facecolor_column]
                _scatter_kwargs['edgecolor'] = rows[edgecolor_column]

            ax.scatter(
                rows[x_column],
                rows[y_column],
                alpha=.5,
                zorder=2,
                **_scatter_kwargs,
            )


    if len(df[df['TaxonomyLevel'] == 'Academia']):
        academia_row = df[
            df['TaxonomyLevel'] == 'Academia'
        ].iloc[0]

        _scatter_kwargs = {**scatter_kwargs}

        if s_column:
            _scatter_kwargs['s'] = academia_row[s_column]

        axes[-1].scatter(
            [academia_row[x_column]],
            [academia_row[y_column]],
            alpha=1,
            zorder=2,
            color=PLOT_VARS['colors']['academia'],
            facecolor=PLOT_VARS['colors']['academia'] if fill_academia else 'w',
            **_scatter_kwargs,
        )

    if not x_lim:
        min_x = min([ax.get_xlim()[0] for ax in axes])
        max_x = max([ax.get_xlim()[1] for ax in axes])
        x_lim = [min_x, max_x]

    if not y_lim:
        min_y = min([ax.get_ylim()[0] for ax in axes])
        max_y = max([ax.get_ylim()[1] for ax in axes])
        y_lim = [min_y, max_y]

    x_label = x_label if x_label else x_column
    y_label = y_label if y_label else y_column

    for i, ax in enumerate(axes):
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        if set_equal_aspect:
            ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel(x_label)

        if xticks:
            ax.set_xticks(xticks)

            if xticklabels:
                ax.set_xticklabels(xticklabels)

        if yticks:
            ax.set_yticks(yticks)

            if yticklabels:
                ax.set_yticklabels(yticklabels)

        for y_tick in ax.get_yticks():
            if not y_keep_zero_line:
                if round(y_tick, 4) == 0:
                    continue

            ax.plot(
                ax.get_xlim(),
                [y_tick, y_tick],
                color=PLOT_VARS['colors']['dark_gray'],
                alpha=.5,
                zorder=1,
                lw=1,
            )

        for x_tick in ax.get_xticks():
            if not x_keep_zero_line:
                if round(x_tick, 4) == 0:
                    continue

            # ax.axvline(
            ax.plot(
                [x_tick, x_tick],
                ax.get_ylim(),
                color=PLOT_VARS['colors']['dark_gray'],
                alpha=.5,
                zorder=1,
                lw=1,
            )

    axes[0].set_ylabel(y_label)

    plt.suptitle(f'{title}\nby taxonomic level', size=36)
    plt.tight_layout()
    return fig, axes, legend_ax


def plot_hierarchy_steepness_vs_mean_rank_change(): 
    df = usfhn.stats.runner.get('ranks/placements')
    df = df[
        (df['NormalizedRankDifference'] != 0)
    ][
        ['TaxonomyLevel', 'TaxonomyValue', 'PersonId', 'NormalizedRankDifference']
    ].drop_duplicates()

    df['MeanRankDifference'] = df.groupby(
        ['TaxonomyLevel', 'TaxonomyValue']
    )['NormalizedRankDifference'].transform('mean')

    df = df.merge(
        views.filter_exploded_df(measurements.get_steepness_by_taxonomy()),
        on=['TaxonomyLevel', 'TaxonomyValue'],
    )

    plot_relationship_across_taxonomy_levels(
        df,
        'MeanRankDifference',
        'Steepness',
        title=f'fraction of downward edges vs\nprestige change from PhD to faculty job',
        x_label='prestige change\nfrom\nPhD to faculty job',
        y_label='fraction of downward edges',
        x_lim=[-.45, -.05],
        xticks=[-.4, -.3, -.2, -.1],
    )


def plot_self_hire_multiplot():
    fig, axes = plt.subplots(
        1, 2,
        figsize=(hnelib.plot.WIDTHS['1-col'], 2.953),
        gridspec_kw={'width_ratios': [3, 3]},
        tight_layout=True,
    )

    panel_a_df = plot_self_hiring_by_gender_horizontal(axes[0])
    panel_a_df['Subfigure'] = 'A'

    panel_b_df = plot_self_hire_risk_ratios(axes[1])
    panel_b_df['Subfigure'] = 'B'

    hnelib.plot.finalize(
        axes,
        [-.02, -.195],
        plot_label_y_pad=1.1
    )

    return pd.concat([panel_a_df, panel_b_df])


def plot_attritions_multiplot():
    fig, axes = plt.subplots(3, 1, figsize=(4, 9), tight_layout=True)

    sub_axes, _ = plot_english_vs_non_english_non_us_attrition_risk_ratio(axes=[axes[0], axes[1]])
    plot_self_hire_risk_ratios(axes[2], annotate_significance=False)


def plot_self_hire_risk_ratios(ax=None, annotate_significance=True):
    df = usfhn.self_hiring.get_self_hire_non_self_hire_risk_ratios()

    df['Ratio'] = df['Ratio'].apply(np.log2)

    title = r"$\frac{\mathrm{self-hires}}{\mathrm{non-self-hires}}$".replace("-", u"\u2010")

    data_df = df.copy()[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'Ratio',
            'PCorrected',
        ]
    ].rename(columns={
        'Ratio': 'RiskRatio',
        'PCorrected': 'P',
    })
    data_df = usfhn.plot_utils.annotate_color(data_df)

    plot_data = []
    element_id = 0
    for i, row in data_df.iterrows():
        for col in data_df.columns:
            plot_data.append({
                'Element': element_id,
                'Attribute': col,
                'Value': row[col],
            })

        element_id += 1

    fig, ax, legend_ax = usfhn.standard_plots.plot_univariate_value_across_taxonomy_levels_single_plot(
        df,
        value_column='Ratio',
        y_label='relative annual attrition',
        x_for_non_significant=True,
        ax=ax,
    )

    ax.set_title(title, fontsize=hnelib.plot.FONTSIZES['title'])

    ytick_pad_top = .05

    yticks = [np.log2(x) for x in [3/4, 1, 5/4, 6/4, 7/4, 2]]
    yticklabels = ['3:4', '1:1', '5:4', '6:4', '7:4', '2:1']

    y_min = -.5
    y_max = yticks[-1] + ytick_pad_top

    umbrellas = ['Academia'] + usfhn.views.get_umbrellas()

    ax.set_ylim(y_min, y_max)

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)

    ax.spines['bottom'].set_visible(False)

    hnelib.plot.add_gridlines_on_ticks(ax, x=False)

    ax.axhline(
        np.log2(1),
        color='black',
        lw=1,
        zorder=1,
    )

    hnelib.plot.finalize([ax])

    if not annotate_significance:
        return

    annotations = []
    fields_df = usfhn.views.filter_by_taxonomy(df, 'Field')
    fields_df = usfhn.views.annotate_umbrella(fields_df, 'Field')

    sig_df = fields_df[
        fields_df['Umbrella'] == 'Engineering'
    ]

    sig_row = sig_df[
        sig_df['Ratio'] == max(sig_df['Ratio'])
    ].iloc[0]

    umbrellas = ['Academia'] + usfhn.views.get_umbrellas()

    annotations.append({
        'label': 'significant',
        'y': sig_row['Ratio'],
        'x': umbrellas.index(sig_row['Umbrella']),
        'y-pad-text': .005,
    })

    insig_df = fields_df[
        fields_df['Umbrella'] == 'Applied Sciences'
    ].sort_values(by=['Ratio'])

    insig_row = insig_df.iloc[0]

    annotations.append({
        'label': 'not significant',
        'y': insig_row['Ratio'],
        'x': umbrellas.index(insig_row['Umbrella']),
        'x-text': umbrellas.index(insig_row['Umbrella']) + .1,
        'direction': 'down',
        'y-pad': .01,
    })

    y_pad = .015
    y_len = .065
    y_pad_text = .005

    x_pad = .05
    x_pad_text = .075

    for annotation in annotations:
        if annotation.get('direction', 'up') == 'up':
            y_multiplier = 1
            va = 'bottom'
        else:
            y_multiplier = -1
            va = 'top'

        y_start = annotation['y'] + (annotation.get('y-pad', y_pad) * y_multiplier)
        y_end = y_start + (annotation.get('y-len', y_len) * y_multiplier)

        x = annotation['x']
        x_text = annotation.get('x-text', x)
        y_text = y_end + (y_multiplier * annotation.get('y-pad-text', y_pad_text))

        ax.annotate(
            '',
            xy=(x, y_start),
            xytext=(x, y_end),
            arrowprops=hnelib.plot.ZERO_SHRINK_A_ARROW_PROPS,
            annotation_clip=False,
        )

        ax.annotate(
            annotation['label'],
            xy=(x_text, y_text),
            va=va,
            ha='center',
            annotation_clip=False,
            color=PLOT_VARS['colors']['dark_gray'],
            fontsize=hnelib.plot.FONTSIZES['annotation'],
        )

    return pd.DataFrame(plot_data)


def plot_hierarchy_steepness_vs_self_hiring_rate():

    df = usfhn.stats.runner.get('self-hire/df')[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'SelfHireFraction',
        ]
    ].drop_duplicates()

    df = df.merge(
        views.filter_exploded_df(measurements.get_steepness_by_taxonomy()),
        on=['TaxonomyLevel', 'TaxonomyValue'],
    )

    plot_relationship_across_taxonomy_levels(
        df,
        'SelfHireFraction',
        'Steepness',
        title=f'self-hire rate vs fraction of downward edges',
        x_label='self-hire rate',
        y_label='fraction of downward edges',
        x_lim=[0, .35],
        y_lim=[.4, 1],
    )

def plot_level_to_level_prestige_correlations(taxonomization_level='Field'):
    df = usfhn.stats.runner.get('ranks/institution-rank-correlations', rank_type='prestige')
    df = views.annotate_umbrella(
        df,
        taxonomization_level=taxonomization_level,
        taxonomization_column='TaxonomyValueOne'
    )

    df = df[
        df['TaxonomyLevel'] == taxonomization_level
    ]

    df['ValuesPerUmbrella'] = df.groupby('Umbrella')['TaxonomyValueOne'].transform('nunique')

    umbrellas_to_values = {}
    for umbrella, rows in df.groupby('Umbrella'):
        umbrellas_to_values[umbrella] = sorted(list(rows['TaxonomyValueOne'].unique()))

    line_annotations = []
    values = []
    for umbrella in sorted(list(df['Umbrella'].unique())):
        umbrella_values = umbrellas_to_values[umbrella]
        values += umbrella_values
        line_annotations.append({
            'annotation': umbrella,
            'start': umbrella_values[0],
            'end': umbrella_values[-1],
        })

    n_values = len(values)
    matrix = np.zeros((n_values, n_values))
    flat_correlations = []
    max_correlation = -1
    min_correlation = 1
    for value_one, value_two in itertools.product(values, values):
        value_to_value_df = df[
            (df['TaxonomyValueOne'] == value_one)
            &
            (df['TaxonomyValueTwo'] == value_two)
        ]

        if value_to_value_df.empty:
            continue

        i = values.index(value_one)
        j = values.index(value_two)
        correlation = value_to_value_df.iloc[0]['Pearson']
        matrix[i, j] = correlation

        if i != j:
            flat_correlations.append(correlation)

        max_correlation = max(correlation, max_correlation)
        min_correlation = min(correlation, min_correlation)

    mean_correlation = np.mean(flat_correlations)
    for i, j in itertools.product(range(len(values)), range(len(values))):
        matrix[i, j] -= mean_correlation

    fig, ax = plt.subplots(figsize=(16, 11))

    ax.set_title(
        f"{taxonomization_level}-to-{taxonomization_level} prestige correlations " + r"(Pearson's $\rho$)"
    )
    cmap_name = 'RdBu'
    label = r'$\rho$'

    cmap = plt.get_cmap(cmap_name)

    plot = ax.pcolor(
        matrix,
        cmap=cmap,
        vmin=min_correlation - mean_correlation,
        vmax=max_correlation - mean_correlation,
    )

    colorbar = fig.colorbar(plot, label=label)

    ticks = colorbar.ax.get_yticks().tolist()
    colorbar.ax.yaxis.set_major_locator(mticker.FixedLocator(ticks))
    label_format = '{:,.0f}'
    colorbar.ax.set_yticklabels([round(tick + mean_correlation, 1) for tick in ticks])

    line_kwargs = {
        'lw': 1.0,
        'color': PLOT_VARS['colors']['dark_gray'],
    }

    ax.set_frame_on(False)

    taxonomization_level_to_pad = {
        'Umbrella': {
            'p0': -1/4,
            'p1': -1/8,
            'annotation': -1/8,
            # 'annotation': -1.5/4,
        },
        'Area': {
            'p0': -1/2,
            'p1': -1/4,
            'annotation': -1/4,
        },
        'Field': {
            'p0': -1,
            'p1': -1/2,
            'annotation': -1/2,
        },
    }

    pads = taxonomization_level_to_pad[taxonomization_level]

    for annotation in line_annotations:
        annotation['annotation'] = plot_utils.clean_taxonomy_string(
            annotation['annotation']
        ).strip().replace(' ', '\n')

        start_i = values.index(annotation['start'])
        end_i = values.index(annotation['end'])

        if start_i not in [0, n_values - 1]:
            ax.plot(
                [start_i, start_i],
                [0, n_values],
                **line_kwargs,
            )
            ax.plot(
                [0, n_values],
                [start_i, start_i],
                **line_kwargs,
            )

        start = start_i + .5
        end = end_i + .5

        y = np.mean([start, end])

        # side
        p0 = pads['p0']
        p1 = pads['p1']
        annotation_pad = pads['annotation']
        ax.plot([p0, p0], [start, end], **line_kwargs)
        ax.plot([p0, p1], [start, start], **line_kwargs)
        ax.plot([p0, p1], [end, end], **line_kwargs)

        ax.annotate(
            annotation['annotation'],
            xy=(0, y),
            xytext=(p0 + annotation_pad, y),
            fontsize=PLOT_VARS['text']['annotations']['labelsize'],
            ha='right',
            va='center',
        )

        x = np.mean([start, end])

        if taxonomization_level == 'Area' and annotation['annotation'] == 'Engineering':
            p0 -= 1.1

        ax.plot([start, end], [p0, p0], **line_kwargs)
        ax.plot([start, start], [p0, p1], **line_kwargs)
        ax.plot([end, end], [p0, p1], **line_kwargs)

        ax.annotate(
            annotation['annotation'],
            xy=(x, 0),
            xytext=(x, p0 + annotation_pad),
            fontsize=PLOT_VARS['text']['annotations']['labelsize'],
            va='top',
            ha='center',
            annotation_clip=False,
        )

    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()


def plot_level_to_level_prestige_rank_changes():
    lower_levels = ['Field', 'Area', 'Umbrella']
    upper_levels = ['Area', 'Umbrella', 'Academia']

    df = usfhn.datasets.CURRENT_DATASET.ranks
    df = df[
        (df['Year'] == constants.YEAR_UNION)
        &
        (df['Gender'] == constants.GENDER_AGNOSTIC)
    ][
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'InstitutionId',
            'NormalizedRank',
        ]
    ].drop_duplicates()

    taxonomization = views.get_taxonomization()

    new_df = []
    for lower_level, upper_level in zip(lower_levels, upper_levels):
        level_to_level_taxonomy = taxonomization[
            [lower_level, upper_level]
        ].copy().drop_duplicates().rename(columns={
            upper_level: 'UpperLevelTaxonomyValue',
            lower_level: 'LowerLevelTaxonomyValue',
        })

        lower_level_df = df[
            df['TaxonomyLevel'] == lower_level
        ].copy().rename(columns={
            'TaxonomyLevel': 'LowerLevelTaxonomyLevel',
            'TaxonomyValue': 'LowerLevelTaxonomyValue',
            'NormalizedRank': 'LowerLevelNormalizedRank'
        }).merge(
            level_to_level_taxonomy,
            on='LowerLevelTaxonomyValue'
        )

        upper_level_df = df[
            df['TaxonomyLevel'] == upper_level
        ].copy().rename(columns={
            'TaxonomyLevel': 'UpperLevelTaxonomyLevel',
            'TaxonomyValue': 'UpperLevelTaxonomyValue',
            'NormalizedRank': 'UpperLevelNormalizedRank',
        })

        level_to_level_df = lower_level_df.merge(
            upper_level_df,
            on=['InstitutionId', 'UpperLevelTaxonomyValue']
        )

        new_df.append(level_to_level_df)

    df = pd.concat(new_df)

    df['LowerToUpperDifference'] = df['UpperLevelNormalizedRank'] - df['LowerLevelNormalizedRank']
    df['MeanInstitutionDifference'] = df.groupby([
        'UpperLevelTaxonomyLevel',
        'UpperLevelTaxonomyValue',
        'InstitutionId'
    ])['LowerToUpperDifference'].transform('mean')

    df['LowerLevelCount'] = df.groupby([
        'UpperLevelTaxonomyLevel',
        'UpperLevelTaxonomyValue',
    ])['LowerLevelTaxonomyValue'].transform('nunique')

    df = df[
        df['LowerLevelCount'] > 1
    ]

    df = df[
        [
            'UpperLevelTaxonomyLevel',
            'UpperLevelTaxonomyValue',
            'InstitutionId',
            'MeanInstitutionDifference',
        ]
    ].drop_duplicates()

    # 17 areas
    # 7 umbrellas
    fig, axes = plt.subplots(1, 4, figsize=(13, 8), gridspec_kw={'width_ratios': [17, 7, 1, 3]})
    axes = axes.flatten()
    legend_ax = axes[-1]
    axes = axes[:-1]

    legend_handles = plot_utils.get_umbrella_legend_handles() + [
        Patch(
            facecolor='none',
            edgecolor='black',
            label='academia',
        ),
    ]

    legend_ax.legend(handles=legend_handles, loc='center', prop={'size': 12})
    legend_ax.axis('off')

    umbrellas = sorted(list(df[df['UpperLevelTaxonomyLevel'] == 'Umbrella']['UpperLevelTaxonomyLevel'].unique()))
    umbrella_colors = [PLOT_VARS['colors']['umbrellas'].get(u) for u in umbrellas]
    
    level_patches = []
    level_colors = []

    for ax, level in zip(axes, upper_levels):
        if level == 'Academia':
            continue

        level_df = df[
            df['UpperLevelTaxonomyLevel'] == level
        ].copy()

        level_df = views.annotate_umbrella_color(
            level_df,
            level,
            taxonomization_column='UpperLevelTaxonomyValue',
        )
        ax.set_title(level.lower())
        boxes = []
        colors = []
        labels = []
        level_df = level_df.sort_values(by='Umbrella')
        for (umbrella, upper_level_value), rows in level_df.groupby(['Umbrella', 'UpperLevelTaxonomyValue']):
            boxes.append(rows['MeanInstitutionDifference'])
            colors.append(rows['UmbrellaColor'].iloc[0])
            labels.append(rows['UpperLevelTaxonomyValue'].iloc[0])

        patches = ax.boxplot(
            boxes,
            patch_artist=True,
            showfliers=False,
            labels=labels,
        )

        level_colors.append(colors)
        level_patches.append(patches)


    academia_ax = axes[-1]
    patches = ax.boxplot(
        df[df['UpperLevelTaxonomyLevel'] == 'Academia']['MeanInstitutionDifference'],
        patch_artist=True,
        showfliers=False,
        widths=.8,
        labels=['Academia'],
    )
    academia_ax.set_xlim(.25, 1.75)

    level_patches.append(patches)
    level_colors.append([PLOT_VARS['colors']['color_1']])

    x_min = min([ax.get_xlim()[0] for ax in axes])
    x_max = max([ax.get_xlim()[1] for ax in axes])
    y_min = min([ax.get_ylim()[0] for ax in axes])
    y_max = max([ax.get_ylim()[1] for ax in axes])

    y_min, y_max = -.35, .35
    
    y_ticks = [-.3, -.2, -.1, 0, .1, .2, .3]

    for ax, patches, colors in zip(axes, level_patches, level_colors):
        ax.set_ylim(y_min, y_max)

        for y_tick in y_ticks:
            ax.plot(
                ax.get_xlim(),
                [y_tick, y_tick],
                color=PLOT_VARS['colors']['dark_gray'],
                alpha=.5,
                zorder=1,
                lw=1,
            )

        ax.set_yticks([])
        for patch, color in zip(patches['boxes'], colors):
            patch.set(facecolor=color)

        for tick, color in zip(ax.get_xticklabels(), colors):
            tick.set_color(color)
            tick.set_rotation(90)


    axes[0].set_title('area - field', size=20)
    axes[0].set_ylabel("area rank\n- field rank (mean)")
    axes[0].set_yticks(y_ticks)

    axes[1].set_title('umbrella - area', size=20)
    axes[1].set_ylabel("umbrella rank\n- area rank (mean)")

    axes[2].set_title('academia - umbrella', size=20)
    axes[2].set_ylabel("academia rank\n- umbrella rank (mean)")

    plt.suptitle(f'level-to-level change in institution rank', size=36)
    plt.tight_layout()
    # return fig, axes, legend_ax

def plot_field_r2s(r2s):
    fig, ax = plt.subplots(figsize=(16, 11))

    fields = sorted(list(r2s.FieldOne.unique()))
    values = []

    matrix = np.zeros((len(fields), len(fields)))
    r2s = r2s.sort_values(by=['FieldOne', 'FieldTwo'])
    for pair in r2s.itertuples():
        i = FIELD_ORDERING.index(pair.FieldOne)
        j = FIELD_ORDERING.index(pair.FieldTwo)

        matrix[i][j] = pair.R2

    plot_field_pair_pcolor(fig, ax, matrix, 'RdBu', r'$R^2$')
    ax.set_xlabel('Predictor field')
    ax.set_ylabel(f'Predicted field')
    ax.set_title(r'Field-to-field prestige fits (OLS $R^2$)')


def plot_field_pair_sizes(ranks):
    fig, ax = plt.subplots(figsize=(16, 11))

    values = []

    field_count = ranks.FieldOne.nunique()
    matrix = np.zeros((field_count, field_count))
    ranks = ranks.copy()
    ranks['Size'] = ranks.groupby(['FieldOne', 'FieldTwo'])['FieldOneRank'].transform('count')
    ranks = ranks[
        ['FieldOne', 'FieldTwo', 'Size']
    ].drop_duplicates().sort_values(
        by=['FieldOne', 'FieldTwo']
    )

    for pair in ranks.itertuples():
        i = FIELD_ORDERING.index(pair.FieldOne)
        j = FIELD_ORDERING.index(pair.FieldTwo)
        
        matrix[i][j] = pair.Size

    plot_field_pair_pcolor(fig, ax, matrix, 'Blues', 'size')
    ax.set_title('Field pair Institution counts')

def plot_field_pair_pcolor(fig, ax, matrix, cmap_name, label, line_annotations=None):
    cmap = plt.get_cmap(cmap_name)

    field_count = len(FIELD_ORDERING)

    plot = ax.pcolor(matrix, cmap=cmap)
    fig.colorbar(plot, label=label)

    line_annotations = [
        {
            'annotation': 'engineering',
            'start': 'Aerospace Engineering',
            'end': 'Mechanical Engineering',
        },
        {
            'annotation': 'stm',
            'start': 'Astronomy',
            'end': 'Statistics',
        },
        {
            'annotation': 'lit & culture',
            'start': 'Asian Languages and Cultures',
            'end': 'Specific Languages and Cultures',
        },
        {
            'annotation': 'other',
            'start': 'Anthropology',
            'end': 'Visual Arts',
        },
    ]

    line_kwargs = {
        'lw': 1.0,
        'color': PLOT_VARS['colors']['dark_gray'],
    }

    ax.set_frame_on(False)

    for annotation in line_annotations:
        start_i = FIELD_ORDERING.index(annotation['start'])
        end_i = FIELD_ORDERING.index(annotation['end'])

        if start_i not in [0, field_count - 1]:
            ax.plot(
                [start_i, start_i],
                [0, field_count],
                **line_kwargs,
            )
            ax.plot(
                [0, field_count],
                [start_i, start_i],
                **line_kwargs,
            )

        start = start_i + .5
        end = end_i + .5

        x = -1
        y = np.mean([start, end])

        # side
        ax.plot(
            [x, x],
            [start, end],
            **line_kwargs
        )

        ax.plot(
            [x, x / 2],
            [start, start],
            **line_kwargs
        )

        ax.plot(
            [x, x / 2],
            [end, end],
            **line_kwargs
        )

        ax.annotate(
            annotation['annotation'],
            xy=(0, y),
            xytext=(x * 1.5, y),
            fontsize=PLOT_VARS['text']['annotations']['labelsize'],
            ha='right',
            va='center',
        )

        y = field_count + 1
        x = np.mean([start, end])

        # top
        ax.plot(
            [start, end],
            [y, y],
            **line_kwargs
        )

        ax.plot(
            [start, start],
            [y, y - .5],
            **line_kwargs
        )

        ax.plot(
            [end, end],
            [y, y - .5],
            **line_kwargs
        )

        ax.annotate(
            annotation['annotation'],
            xy=(x, y + 1.5),
            xytext=(x, y + 1.5),
            fontsize=PLOT_VARS['text']['annotations']['labelsize'],
            va='center',
            ha='center',
        )

    ax.set_xticks([])
    ax.set_yticks([])


def plot_institution_prestige_to_field_count(ranks, rank_column='PercentageScalarSpringRank'):
    fig, ax = plt.subplots(figsize=(16, 11))

    # institution
    # field
    # prestige
    ranks['FieldCount'] = ranks.groupby(['InstitutionId', 'Year'])['Field'].transform('nunique')
    average_prestige = ranks.groupby(['InstitutionId', 'Year'])[rank_column].mean()
    average_prestige = average_prestige.rename("AverageRank")
    ranks = ranks.merge(average_prestige, on=['InstitutionId', 'Year'])

    ranks = ranks[
        ['InstitutionId', 'AverageRank', 'FieldCount']
    ].drop_duplicates()

    ax.set_title("Fields participating in vs average institution prestige")
    ax.set_ylabel('# of fields')
    ax.set_xlabel(f'Averaged Institution Prestige (0 to 1)')

    ax.scatter(ranks['AverageRank'], ranks['FieldCount'])
    ax.set_xticks([0, .25, .5, .75, 1])
    yticklabels = ax.get_yticklabels()

    max_field_count = max(ranks['FieldCount'])
    if max_field_count not in yticklabels:
        yticklabels.append(max_field_count)
        ax.set_yticklabels(yticklabels)


def plot_institution_prestige_distributions(ranks, rank_column='PercentageScalarSpringRank'):
    fig, ax = plt.subplots(figsize=(16, 11))

    xs = np.linspace(0, 1, num=100)

    ranks = ranks[
        ['InstitutionId', rank_column]
    ].drop_duplicates()

    ranks = ranks[
        ~pd.isnull(ranks[rank_column])
    ]

    for institution_id, rows in ranks.groupby('InstitutionId'):
        if len(rows) > 1:
            kernel = gaussian_kde(rows[rank_column])
            institution_min_x = min(rows[rank_column])
            institution_max_x = max(rows[rank_column])

            institution_xs = [x for x in xs if institution_min_x <= x <= institution_max_x]

            ys = [kernel(x) for x in institution_xs]
            ax.plot(institution_xs, ys, color=PLOT_VARS['colors']['dark_gray'], alpha=.3)

    ax.set_title("department prestige distribution by institution")
    ax.set_ylabel('density (kde)')
    ax.set_xlabel(f'Averaged Institution Prestige (0 to 1)')


def plot_institution_prestige_curves(ranks, rank_column='PercentageScalarSpringRank'):
    fig, ax = plt.subplots(figsize=(16, 11))

    xs = np.linspace(0, 1, num=100)

    ranks = ranks[
        ['InstitutionId', 'Field', rank_column]
    ].drop_duplicates(subset=['InstitutionId', 'Field'])
    ranks = ranks[
        ~pd.isnull(ranks[rank_column])
    ]

    for institution_id, rows in ranks.groupby('InstitutionId'):
        prestiges = sorted(list(rows[rank_column]))
        xs = [i for i in range(len(prestiges))]
        ax.plot(xs, prestiges, color=PLOT_VARS['colors']['dark_gray'], alpha=.3)

    ax.set_title("Institution field prestige curves")
    ax.set_ylabel('Field Prestige (0 to 1)')
    ax.set_xlabel(f'Within-institution ordinal prestige rank')


def plot_self_hiring_contours(self_hires, by='Field', x_column='Closedness', size=None):
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), sharey=True)

    valid_plots = {
        'Field': ['Closedness', 'Faculty', 'Prestige'],
        'InstitutionId': ['Faculty', 'Prestige'],
    }

    if x_column not in valid_plots[by]:
        print(f"can't plot {x_column} against {by}")
        return

    i = 0
    max_maxx = 0
    max_maxy = 0
    for category, rows in self_hires.groupby('Category'):
        rows = rows.copy()
        if by == 'Field':
            rows['Denominator'] = rows.groupby([x_column, by])['Faculty'].transform('sum')
            rows['Numerator'] = rows.groupby([x_column, by])['SelfHires'].transform('sum')
        elif by == 'InstitutionId':
            rows['Denominator'] = rows.groupby(by)['Faculty'].transform('sum')
            rows['Numerator'] = rows.groupby(by)['SelfHires'].transform('sum')

            if x_column == 'Faculty':
                rows[x_column] = rows['Denominator']
            elif x_column == 'Prestige':
                rows[x_column] = rows.groupby(by)['Prestige'].transform(np.mean)

            rows['Category'] = None

        rows['Fraction'] = rows['Numerator'] / rows['Denominator']

        rows = rows.drop_duplicates(subset=[x_column, 'Fraction'])
        x = list(rows[x_column])
        y = list(rows['Fraction'])

        max_maxx = max(max_maxx, max(x))
        max_maxy = max(max_maxy, max(y))

        contour_plot_helper(axes[i], list(rows[x_column]), list(rows['Fraction']))
        axes[i].set_title(category)
        axes[i].set_xlabel(f'{x_column.lower()}')
        i += 1

    for i, ax in enumerate(axes):
        ax.set_xlim([0, max_maxx])
        ax.set_ylim([0, max_maxy])

        if not i:
            axes[i].set_ylabel('self-hire rate')


    plt.suptitle(f'Self-hire rate versus {x_column.lower()}\nby {by.lower()}')


def self_hire_aggregator(rows, by, x_column):
    rows = rows.copy()

    if by == 'Field':
        rows['Denominator'] = rows.groupby([x_column, by])['Faculty'].transform('sum')
        rows['Numerator'] = rows.groupby([x_column, by])['SelfHires'].transform('sum')
    elif by == 'InstitutionId':
        rows['Denominator'] = rows.groupby(by)['Faculty'].transform('sum')
        rows['Numerator'] = rows.groupby(by)['SelfHires'].transform('sum')

        if x_column == 'Faculty':
            rows[x_column] = rows['Denominator']
        elif x_column == 'Prestige':
            rows[x_column] = rows.groupby(by)['Prestige'].transform(np.mean)

        rows['Category'] = None

    rows['Fraction'] = rows['Numerator'] / rows['Denominator']

    rows = rows.drop_duplicates(subset=[x_column, 'Fraction'])
    return rows


def contour_plot_helper(ax, x, y):
    ngridx = 100
    ngridy = 100

    from scipy.interpolate import griddata
    z_func = gaussian_kde([x, y])
    z = [z_func((x, y))[0] for x, y in zip(x, y)]

    xi = np.linspace(min(x), max(x), ngridx)
    yi = np.linspace(min(y), max(y), ngridy)
    # zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='linear')
    triang = tri.Triangulation(x, y)
    interpolator = tri.LinearTriInterpolator(triang, z)
    Xi, Yi = np.meshgrid(xi, yi)
    zi = interpolator(Xi, Yi)

    ax.contour(xi, yi, zi, levels=10, linewidths=0.5, colors='k', alpha=.5)
    cntr1 = ax.contourf(xi, yi, zi, levels=10, cmap="RdBu_r", alpha=.5)

################################################################################
#
#
#
# Department Level
#
#
#
################################################################################
def department_sample_summary_by_degree():
    """
    We want to see how many people we have departments for by degree year
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    degrees = views.get_degrees().sort_values(by='DegreeYear')
    phd_departments = views.get_degree_departments()
    people_with_phd_departments = set(phd_departments.PersonId)

    years = []
    year_degree_counts = []
    year_department_counts = []
    year_degree_department_percentages = []
    for year, rows in degrees.groupby('DegreeYear'):
        years.append(year)

        year_degree_people = set(rows['PersonId'])
        year_department_people = year_degree_people & people_with_phd_departments
        year_degree_department_percentage = round(
            100 * len(year_department_people) / len(year_degree_people),
            2,
        )

        year_degree_counts.append(len(year_degree_people))
        year_department_counts.append(len(year_department_people))
        year_degree_department_percentages.append(year_degree_department_percentage)

    denominator = ax.plot(years, year_degree_counts, c='#4c4c4c', label='total', alpha=.5)
    numerator = ax.plot(years, year_department_counts, c='#a30005', label='with phd department', alpha=.5)

    ax_fraction = ax.twinx()
    fraction = ax_fraction.plot(
        years, year_degree_department_percentages,
        linestyle='--',
        c='#0e6101',
        alpha=.5,
        label="% with phd department"
    )

    scatters = denominator + numerator + fraction
    labels = [x.get_label() for x in scatters]

    ax_fraction.set_ylabel("percentage")
    ax_fraction.spines['right'].set_visible(True)

    ax.set_ylabel("# of degrees")
    ax.set_xlabel("year")
    ax.legend(scatters, labels)


def department_sample_summary_new_hires_in_year(degrees, phd_departments, employment):
    fig, ax = plt.subplots(figsize=(16, 8))
    degrees = degrees.sort_values(by='DegreeYear')
    employment = employment[employment.PersonId.isin(degrees.PersonId.unique())]
    people_with_phd_departments = set(phd_departments.PersonId)

    employed_people_by_year = dict()
    for year, rows in employment.sort_values(by='Year').groupby('Year'):
        employed_people_by_year[year] = set(rows.PersonId.unique())

    first_year = min(employed_people_by_year.keys())
    already_employed_people = employed_people_by_year.pop(first_year)

    years = []
    hires = []
    with_departments = []
    with_departments_percentages = []
    for year, people_employed_this_year in employed_people_by_year.items():
        hires_this_year = people_employed_this_year - already_employed_people
        with_departments_this_year = people_with_phd_departments & hires_this_year

        years.append(year)
        hires.append(len(hires_this_year))
        with_departments.append(len(with_departments_this_year))

        with_departments_percentages.append(
            round(100 * with_departments[-1] / hires[-1], 2)
        )

        already_employed_people |= hires_this_year

    denominator = ax.plot(
        years, hires,
        label='total', 
        c=PLOT_VARS['colors']['department-level']['black'],
        alpha=.5
    )

    numerator = ax.plot(
        years, with_departments,
        label='with phd department', 
        c=PLOT_VARS['colors']['department-level']['red'],
        alpha=.5
    )

    ax_fraction = ax.twinx()
    fraction = ax_fraction.plot(
        years, with_departments_percentages,
        linestyle='--',
        c=PLOT_VARS['colors']['department-level']['green'],
        alpha=.5,
        label="% with phd department"
    )

    scatters = denominator + numerator + fraction
    labels = [x.get_label() for x in scatters]

    ax_fraction.set_ylabel("percentage")
    ax_fraction.spines['right'].set_visible(True)
    ax_fraction.set_ylim([0, 50])
    ax_fraction.set_yticks([0, 10, 20, 30, 40, 50])

    ax.set_ylabel("# of hires")
    ax.set_xlabel("year")
    ax.legend(scatters, labels)


def lineplot_hierarchy_arcs():
    groupby_columns = ['TaxonomyLevel', 'TaxonomyValue']

    ranks = usfhn.stats.runner.get('ranks/df', rank_type='prestige')

    df = usfhn.stats.runner.get('basics/faculty-hiring-network')[
        groupby_columns + ['InstitutionId', 'DegreeInstitutionId', 'Count']
    ].drop_duplicates().copy().merge(
        ranks,
        on=groupby_columns + ['InstitutionId'],
    ).merge(
        ranks.rename(columns={
            'InstitutionId': 'DegreeInstitutionId',
            'NormalizedRank': 'DegreeNormalizedRank',
        }),
        on=groupby_columns + ['DegreeInstitutionId'],
    )

    df = df[
        (df['TaxonomyLevel'] == 'Academia')
    ]

    other_side_df = df.copy()[
        ['InstitutionId', 'DegreeInstitutionId', 'Count']
    ].rename(columns={
        'InstitutionId': 'InstitutionId2',
        'DegreeInstitutionId': 'InstitutionId1',
        'Count': 'CountFrom1To2',
    })

    df = df.rename(columns={
        'InstitutionId': 'InstitutionId1',
        'DegreeInstitutionId': 'InstitutionId2',
        'NormalizedRank': 'NormalizedRank1',
        'DegreeNormalizedRank': 'NormalizedRank2',
        'Count': 'CountFrom2To1',
    }).merge(
        other_side_df,
        on=['InstitutionId1', 'InstitutionId2'],
        how='outer'
    ).drop_duplicates()

    df['CountFrom1To2'] = df['CountFrom1To2'].fillna(0)
    df['CountFrom2To1'] = df['CountFrom2To1'].fillna(0)
    
    df = df[
        (df['InstitutionId1'] != df['InstitutionId2'])
        &
        (df['NormalizedRank1'] < df['NormalizedRank2'])
        &
        (
            (df['CountFrom1To2'] > 0)
            |
            (df['CountFrom2To1'] > 0)
        )
    ]

    # 1 is the lower prestige institution
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.axis('off')
    
    df['TotalCount'] = df['CountFrom1To2'] + df['CountFrom2To1']
    df['FractionUp'] = df['CountFrom1To2'] / df['TotalCount']
    df['FractionDown'] = df['CountFrom2To1'] / df['TotalCount']

    df['MoreUp'] = df['FractionUp'] > df['FractionDown']
    bins = [0, .2, .4, .6, .8, 1]
    labels = bins[1:]
    df['FractionUpBin'] = pd.cut(df['FractionUp'], bins, labels=labels)
    df['FractionDownBin'] = pd.cut(df['FractionDown'], bins, labels=labels)
    df['Direction'] = df['MoreUp'].apply({True: 'Up', False:'Down'}.get)
    df['BinColumn'] = df['MoreUp'].apply({True: 'FractionUpBin', False: 'FractionDownBin'}.get)
    scale_value_for_color = lambda v: (v/2) + .5

    # cmap=plt.get_cmap(name))
    cmap = matplotlib.cm.get_cmap('seismic')

    path_bins = {
        # 'Up': {b: {'coords': [], 'codes': []} for b in bins},
        # 'Down': {b: {'coords': [], 'codes': []} for b in bins},
        'Up': defaultdict(defaultdict(list).copy),
        'Down': defaultdict(defaultdict(list).copy),
    }
    

    # df = df.head(5000)
    coords = []
    codes = []
    for i, row in df.iterrows():
        y_start = row['NormalizedRank1']
        y_end = row['NormalizedRank2']
        half_distance = (y_end - y_start) / 2

        path_bin = path_bins[row['Direction']][row[row['BinColumn']]]
        path_bin['coords'] += [(0, y_start), (half_distance, y_start), (0, y_end)]
        path_bin['codes'] += [MPath.MOVETO, MPath.CURVE3, MPath.CURVE3]

    for direction, direction_bins in path_bins.items():
        for color_fraction in direction_bins:
            coords = direction_bins[color_fraction]['coords']
            codes = direction_bins[color_fraction]['codes']

            if not len(coords) or not len(codes):
                continue

            if direction == 'Up':
                color_fraction *= -1
            scaled_value = scale_value_for_color(color_fraction)
            color = cmap(scaled_value)

            patch = patches.PathPatch(MPath(coords, codes), facecolor='none', lw=.2, edgecolor=color, alpha=.5)
            ax.add_patch(patch)


    ax.plot([0, 0], [0, 1])
    # y_start = 0
    # y_end = 1

    # first_arc = .25
    # second_arc = y_end - first_arc

    # path = MPath(
    #     [(0, 0), (.5, 0), (0, 1)],
    #     [MPath.MOVETO, MPath.CURVE3, MPath.CURVE3]
    # )
    # arrowstyle = f"fancy,head_length={4},head_width={4},tail_width={1}"

    # patch = FancyArrowPatch(path=path)

    # ax.add_patch(patch)
    # ax.plot([0.3], [.3], "ro")

    # second_arc = .7

    # ax.add_patch(Arc((0, first_arc), .3, 2 * first_arc, angle=0, theta1=180, theta2=270))
    # ax.add_patch(Arc((0, first_arc), .3, 2 * second_arc, angle=0, theta1=90, theta2=180))
    # path = patches.PathPatch(
    #     MPath(
    #         [(0, 0), (.5, 0), (0, 1)],
    #         [MPath.MOVETO, MPath.CURVE3, MPath.CURVE3]
    #     ),
    #     fc="none",
    #     transform=ax.transData
    # )

    ax.set_aspect('equal', adjustable='box')



def annotate_fields_matrix(ax, line_annotations, item_count):
    """
    line_annotations = [
        {
            'start_i': integer,
            'end_i': integer,
            'annotation': str,
        }
    ]
    """
    line_kwargs = {
        'lw': 1.0,
        'color': '#4B917D',
    }

    margin = 50

    for annotation in line_annotations:
        start_i = annotation['start_i']
        end_i = annotation['end_i']

        if start_i not in [0, item_count - 1]:
            ax.plot(
                [start_i, start_i],
                [0, item_count],
                **line_kwargs,
            )
            ax.plot(
                [0, item_count],
                [start_i, start_i],
                **line_kwargs,
            )

        start = start_i
        end = end_i

        x = item_count
        y = np.mean([start, end])

        ax.annotate(
            annotation['annotation'],
            xy=(x, y),
            xytext=(x + margin, y),
            fontsize=PLOT_VARS['text']['annotations']['labelsize'],
            ha='left',
            va='center',
        )

        y = item_count
        x = np.mean([start, end])

        ax.annotate(
            annotation['annotation'],
            xy=(x, y),
            xytext=(x, y + margin),
            fontsize=PLOT_VARS['text']['annotations']['labelsize'],
            va='top',
            ha='center',
        )



def plot_absolute_steeples_of_excellence_single_institution(institution_name='University of Colorado Boulder'):
    df = usfhn.steeples.get_steeples()
    df = usfhn.institutions.annotate_institution_name(df)
    df = views.annotate_umbrella_color(df, taxonomization_level='Field')
    df = df[
        df['InstitutionName'] == institution_name
    ]

    mean_rank = df['NormalizedRank'].mean()

    fields = sorted(list(df['TaxonomyValue'].unique()))
    n_fields = len(fields)

    if n_fields == 0:
        return

    field_width = .2
    total_fields_width = field_width * n_fields
    legend_width = 1
    fig, (ax, legend_ax) = plt.subplots(
        1, 2,
        figsize=(total_fields_width + legend_width, 6),
        gridspec_kw={'width_ratios': [total_fields_width, 1/total_fields_width]},
    )

    umbrella_handles = plot_utils.get_umbrella_legend_handles(style='scatter')

    steeple_marker = '+'

    legend_handles = umbrella_handles + [
        mlines.Line2D(
            [], [],
            color=PLOT_VARS['colors']['dark_gray'],
            linestyle='--',
            label=f'mean ({round(mean_rank, 2)})',
        ),
        mlines.Line2D(
            [], [],
            color='w',
            marker=steeple_marker,
            markerfacecolor=PLOT_VARS['colors']['dark_gray'],
            markeredgecolor=PLOT_VARS['colors']['dark_gray'],
            label='steeple',
        ),
    ]

    legend_ax.legend(handles=legend_handles, loc='center', prop={'size': 8}, labelcolor='markerfacecolor')
    legend_ax.axis('off')

    df = df.sort_values(by=['Umbrella', 'TaxonomyValue']).reset_index()

    x_ticks = list(range(n_fields))
    x_ticklabels = df['TaxonomyValue']
    x_ticklabel_colors = list(df['UmbrellaColor'])

    umbrellas = list(df['Umbrella'])
    umbrella_set = sorted(list(set(umbrellas)))
    x_start = 0
    x_pad = .5

    scatter_points = defaultdict(defaultdict(list).copy)
    for i, row in df.iterrows():
        if row['Steeple']:
            key = 'steeples'
        elif row['Basement']:
            key = 'basements'
        else:
            key = 'others'

        scatter_points[key]['ranks'].append(row['NormalizedRank'])
        scatter_points[key]['xs'].append(i)
        scatter_points[key]['colors'].append(row['UmbrellaColor'])

    ax.scatter(
        scatter_points['others']['xs'],
        scatter_points['others']['ranks'],
        color=scatter_points['others']['colors'],
        alpha=.5,
        zorder=2,
    )
    ax.scatter(
        scatter_points['steeples']['xs'],
        scatter_points['steeples']['ranks'],
        color=scatter_points['steeples']['colors'],
        marker=steeple_marker,
        alpha=.5,
        zorder=2,
    )
    # ax.scatter(
    #     scatter_points['basements']['xs'],
    #     scatter_points['basements']['ranks'],
    #     color=scatter_points['basements']['colors'],
    #     marker=basement_marker,
    #     alpha=.5,
    #     zorder=2,
    # )

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticklabels, size=8)
    ax.set_xlim(-1 * x_pad, len(fields) - 1 + x_pad)

    ax.plot(
        ax.get_xlim(),
        [mean_rank, mean_rank],
        linestyle='--',
        color=PLOT_VARS['colors']['dark_gray'],
        zorder=1,
    )

    hello = {
        1: True,
        2: False
    }

    for tick, color in zip(ax.get_xticklabels(), x_ticklabel_colors):
        tick.set_color(color)
        tick.set_rotation(90)

    
    # ax.set_ylim(min(ax.get_ylim()), min(1.05, max(ax.get_ylim())))
    ax.set_ylim(-.05, 1.05)
    y_lines = [round(t, 1) for t in ax.get_yticks() if round(t, 1) == round(t, 2)]
    hnelib.plot.add_gridlines(ax, ys=[round(t, 1) for t in ax.get_yticks() if round(t, 1) == round(t, 2)])
    y_min, y_max = ax.get_ylim()

    for i, umbrella in enumerate(umbrella_set):
        n_in_umbrella = umbrellas.count(umbrella)
        ax.add_patch(
            mpatches.Rectangle(
                xy=(x_start - x_pad, y_min),
                width=n_in_umbrella - 1 + 2 * x_pad,
                height=y_max - y_min,
                color=x_ticklabel_colors[x_start],
                alpha=.1,
                zorder=0,
                edgecolor=None,
            )
        )

        x_start += n_in_umbrella

    ax.set_title(f'{institution_name} field prestige ranks')
    ax.set_ylabel('prestige rank')
    plt.tight_layout()


def plot_linear_placement_predictions(taxonomy_level='Academia', taxonomy_value='Academia'):
    df, lowess_df  = usfhn.placement_predictor.get_placement_predictions()

    df = df[
        (df['TaxonomyLevel'] == taxonomy_level)
        &
        (df['TaxonomyValue'] == taxonomy_value)
    ]

    df = views.annotate_umbrella_color(df, taxonomy_level)

    slope = df['Slope'].iloc[0]
    intercept = df['Intercept'].iloc[0]
    color = df['UmbrellaColor'].iloc[0]

    fig, ax = plt.subplots(1, figsize=(6, 6))

    ticks = [0, .25, .5, .75, 1]
    tick_labels = ["0", ".25", ".5", ".75", "1"]

    step = .01
    xs = np.arange(0, 1 + step, step=step)

    ax.plot(
        xs,
        [slope * x + intercept for x in xs],
        color=PLOT_VARS['colors']['dark_gray'],
    )

    ax.scatter(
        df['NormalizedDegreeInstitutionRank'],
        df['NormalizedInstitutionRank'],
        s=3,
        alpha=.3,
        color=color,
    )

    ax.set_xlim([1, 0])
    ax.set_ylim([1, 0])

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)

    ax.set_ylabel("employing institution prestige\n" + r'high $\rightarrow$ low')
    ax.set_xlabel("PhD institution prestige\n" + r'high $\rightarrow$ low')
    ax.set_title(f"PhD to employing institution prestige\nin {taxonomy_value} ({taxonomy_level})")

    hnelib.plot.annotate(ax, text=r"$R^2 = $" + f"{round(df['R^2'].iloc[0], 2)}", xy_loc=(.1, .9))

def plot_lowess_placement_predictions(taxonomy_level='Academia', taxonomy_value='Academia'):
    df, lowess_df = usfhn.placement_predictor.get_placement_predictions()

    df = df[
        (df['TaxonomyLevel'] == taxonomy_level)
        &
        (df['TaxonomyValue'] == taxonomy_value)
    ]

    df = views.annotate_umbrella_color(df, taxonomy_level)

    fig, ax = plt.subplots(1, figsize=(6, 6))

    ticks = [0, .25, .5, .75, 1]
    tick_labels = ["0", ".25", ".5", ".75", "1"]

    step = .01
    xs = np.arange(0, 1 + step, step=step)
    color = df['UmbrellaColor'].iloc[0]

    ax.scatter(
        df['DegreeInstitutionPercentile'],
        df['RankPredicted'],
        s=3,
        alpha=.3,
        facecolor='none',
        edgecolor=PLOT_VARS['colors']['dark_gray'],
    )

    ax.scatter(
        df['DegreeInstitutionPercentile'],
        df['InstitutionPercentile'],
        s=3,
        alpha=.3,
        color=color,
        # label='predicted',
    )

    legend_handles = [
        mlines.Line2D(
            [], [],
            color='none',
            label='actual',
            marker='o',
            markerfacecolor='none',
            markeredgecolor=PLOT_VARS['colors']['dark_gray'],
        ),
        mlines.Line2D(
            [], [],
            color='none',
            label='predicted',
            marker='o',
            markerfacecolor=color,
            markeredgecolor=color,
        ),
    ]

    ax.legend(handles=legend_handles)
    # ax.set_xlim([1, 0])
    # ax.set_ylim([1, 0])

    # ax.set_xticks(ticks)
    # ax.set_yticks(ticks)

    # ax.set_xticklabels(tick_labels)
    # ax.set_yticklabels(tick_labels)

    ax.set_ylabel("employing institution prestige\n" + r'high $\rightarrow$ low')
    ax.set_xlabel("PhD institution prestige\n" + r'high $\rightarrow$ low')
    ax.set_title(f"PhD to employing institution prestige\nin {taxonomy_value} ({taxonomy_level})")

    # hnelib.plot.annotate(ax, text=r"$R^2 = $" + f"{round(df['R^2'].iloc[0], 2)}", xy_loc=(.1, .9))


def compare_junior_senior_changes_over_time(measurement='ginis'):
    fig, axes = plt.subplots(
        1, 3,
        figsize=(10, 4),
        tight_layout=True,
        gridspec_kw={'width_ratios': [4, 4, 2]},
    )

    ranks = {
        'assistant': ('Assistant Professor',),
        'senior faculty': ('Associate Professor', 'Professor'),
    }

    # plot_gender_over_time and plot_self_hires_over_time don't work yet
    # model them after plot_ginis_over_time

    measurement_to_plot_function = {
        'ginis': plot_gini_over_time,
        # 'gender': plot_gender_over_time,
        # 'self-hires': plot_self_hires_over_time,
    }

    for ax, faculty_set in zip(axes, ranks.keys()):
        measurement_to_plot_function[measurement](
            ax=ax,
            ranks=ranks[faculty_set],
            gridlines=False,
        )

        ax.set_title(faculty_set)

    hnelib.plot.set_lims_to_max(axes[:2], x=False)
    for ax in axes[:2]:
        hnelib.plot.add_gridlines_on_ticks(ax, x=False)

    hnelib.plot.hide_axis(axes[2])
    axes[2].legend(
        handles=plot_utils.get_umbrella_legend_handles(
            style='line',
            include_academia=True,
            extra_kwargs={
                'alpha': 1,
            }
        ),
        loc='center'
    )


def slopes_over_time(measurement='ginis', faculty_set='assistant'):
    import usfhn.changes
    df = usfhn.stats.runner.get(f'{measurement}/by-time/by-seniority/slopes')

    seniority = False if faculty_set == 'assistant' else True

    df = df[
        df['Senior'] == seniority
    ]

    df = df[
        ['TaxonomyLevel', 'TaxonomyValue', 'Slope']
    ].drop_duplicates()

    if measurement == 'ginis':
        if faculty_set == 'assistant':
            kwargs = {
                'y_min': -.0125,
                'y_max': .0125,
                'y_step': .00625
            }
        elif faculty_set == 'senior-faculty':
            kwargs = {
                'y_min': -.00625,
                'y_max': .00625,
                'y_step': .003125,
            }
    elif measurement == 'gender':
        if faculty_set == 'assistant':
            kwargs = {
                'y_min': -.025,
                'y_max': .025,
                'y_step': .0125,
            }
        elif faculty_set == 'senior-faculty':
            kwargs = {
                'y_min': -.015,
                'y_max': .015,
                'y_step': .005,
            }
    elif measurement == 'self-hires':
        kwargs = {
            'y_min': -.0125,
            'y_max': .0125,
            'y_step': .00625
        }

    fig, axes, legend_ax = plot_univariate_value_across_taxonomy_levels(
        df,
        f'Slope',
        f'slope of {measurement} change ({faculty_set})',
        **kwargs,
    )

    for ax in axes:
        ax.axhline(
            0,
            lw=1,
            color='black',
        )


def plot_gini_over_time(ax=None, gridlines=True, ranks=()):
    df = usfhn.changes.get_gini_coefficients_for_rank_subset(ranks, drop_other_columns=False, by_year=True)

    df = df[
        df['TaxonomyLevel'].isin(['Academia', 'Umbrella'])
    ]

    df = usfhn.views.annotate_umbrella_color(df, taxonomization_level='Umbrella')

    if not ax:
        fig, ax = plt.subplots(1, figsize=(4, 6))

    column = 'GiniCoefficient'
    for value, rows in df.groupby('TaxonomyValue'):
        rows = rows.sort_values(by='Year')
        ax.plot(
            rows['Year'],
            rows[column],
            color=rows.iloc[0]['UmbrellaColor'],
            zorder=1,
        )

        fill_color = rows.iloc[0]['UmbrellaColor']

        if value != 'Academia':
            fill_color = hnelib.plot.set_alpha_on_colors(fill_color)

        ax.scatter(
            rows['Year'],
            rows[column],
            color='w',
            zorder=2,
        )

        ax.scatter(
            rows['Year'],
            rows[column],
            facecolor=fill_color,
            edgecolor=rows.iloc[0]['UmbrellaColor'],
            zorder=3,
        )

    ax.set_xlabel('year')
    ax.set_ylabel('Gini coefficient')

    # ax.set_ylim(.65, .8)
    # ax.set_yticks([.65, .7, .75, .8])
    # ax.set_yticklabels([".65", ".7", ".75", ".8"])

    if gridlines:
        hnelib.plot.add_gridlines_on_ticks(ax, x=False)


def plot_gini_over_time_alone():
    df = usfhn.stats.runner.get('ginis/by-year/df')

    usfhn.standard_plots.plot_domain_level_values_over_time(
        df,
        'GiniCoefficient',
        'Gini coefficient',
    )

def plot_gender_over_time_alone():
    df = usfhn.stats.runner.get('gender/by-year/df')

    usfhn.standard_plots.plot_domain_level_values_over_time(
        df,
        'FractionFemale',
        'fraction female',
    )

def plot_self_hires_over_time_alone():
    df = usfhn.stats.runner.get('self-hire/by-year/df')

    usfhn.standard_plots.plot_domain_level_values_over_time(
        df,
        'SelfHiresFraction',
        'fraction self-hires',
    )


def plot_non_us_institution_makeup():
    df = usfhn.stats.runner.get('non-us/by-institution/df')

    df = usfhn.views.filter_by_taxonomy(df, value='Academia')

    df = usfhn.institutions.annotate_institution_name(df)

    bins = np.arange(0, .6, .01)

    fig, ax = plt.subplots(figsize=(6, 6), tight_layout=True)

    ax.hist(
        df['NonUSFraction'],
        bins=bins,
        alpha=.75,
    )

    ax.set_xlim(0, .6)

    xticks = [0, .1, .2, .3, .4, .5, .6]
    ax.set_xticks(xticks)
    ax.set_xticklabels(hnelib.plot.stringify_numbers_without_ugly_zeros(xticks))

    ax.set_xlabel('% non-U.S.')
    ax.set_ylabel('# of institutions')

    max_val = max(df['NonUSFraction'])
    max_inst_row = df[
        df['NonUSFraction'] == max_val
    ].iloc[0]

    ax.annotate(
        max_inst_row['InstitutionName'],
        xy=(max_val - .005, 2),
        xytext=(max_val - .005, 7),
        arrowprops=hnelib.plot.BASIC_ARROW_PROPS,
        ha='center',
        va='bottom',
    )


def plot_non_us_taxonomy_by_time():
    df = usfhn.stats.runner.get('non-us/by-time/df')

    usfhn.standard_plots.plot_domain_level_values_over_time(
        df,
        'NonUSFraction',
        'fraction of non-U.S. PhDs',
        ylim=[0, .2],
        yticks=[0, .05, .1, .15, .2],
    )


def plot_non_us_institution_makeup_domain_level():
    df = usfhn.stats.runner.get('non-us/by-institution/df')

    df = df[
        df['TaxonomyLevel'].isin(['Umbrella', 'Academia'])
    ]

    umbrellas = usfhn.views.get_umbrellas()

    fig, axes = plt.subplots(2, 5, figsize=(14, 6), tight_layout=True)

    layout = [
        ['Academia'] + umbrellas[:4],
        [None] + umbrellas[4:],
    ]

    bins = np.arange(0, .6, .01)

    for i, layout_row in enumerate(layout):
        for j, taxonomy_value in enumerate(layout_row):
            ax = axes[i, j]

            if not taxonomy_value:
                hnelib.plot.hide_axis(ax)
                continue

            _df = df[
                df['TaxonomyValue'] == taxonomy_value
            ]

            color = PLOT_VARS['colors']['umbrellas'][taxonomy_value]

            ax.hist(
                _df['NonUSFraction'],
                bins=bins,
                facecolor=hnelib.plot.set_alpha_on_colors(color),
                edgecolor=color,
                zorder=2,
            )

            mean = _df['NonUSFraction'].mean()

            ax.axvline(
                mean,
                lw=.5,
                color=color,
                zorder=1,
            )

            if taxonomy_value == 'Academia':
                ax.annotate(
                    "mean",
                    xy=(mean, 38),
                    xytext=(mean + .1, 38),
                    ha='left',
                    va='center',
                    arrowprops=PLOT_VARS['arrowprops'],
                )

            ax.set_title(
                usfhn.plot_utils.clean_taxonomy_string(taxonomy_value),
                color=color,
            )

            ax.set_xlabel('% non-U.S.')
            ax.set_ylabel('# of institutions')

            ax.set_xlim(0, .6)

            xticks = [0, .1, .2, .3, .4, .5, .6]
            ax.set_xticks(xticks)
            ax.set_xticklabels(hnelib.plot.stringify_numbers_without_ugly_zeros(xticks))

            ax.set_ylim(0, 40)
            ax.set_yticks([0, 10, 20, 30, 40])


def plot_non_us_vs_prestige(scatter=True):
    df = usfhn.stats.runner.get('non-us/by-institution/df')

    df = df.merge(
        usfhn.stats.runner.get('ranks/df', rank_type='prestige'),
        on=[
            'InstitutionId',
            'TaxonomyLevel',
            'TaxonomyValue',
        ]
    )[
        [
            'NonUSFraction',
            'Percentile',
            'InstitutionId',
            'TaxonomyLevel',
            'TaxonomyValue',
        ]
    ].drop_duplicates()

    df = df[
        df['TaxonomyLevel'].isin(['Academia', 'Umbrella'])
    ]

    df['Percentile'] *= 100
    df['NonUSPercent'] = df['NonUSFraction'] * 100

    df = usfhn.views.annotate_umbrella_color(df, taxonomization_level='Umbrella')
    df = usfhn.institutions.annotate_institution_name(df)

    fig, axes = plt.subplots(1, 2, figsize=(6, 4), gridspec_kw={'width_ratios': [2/3, 1/3]})

    ax = axes[0]
    for value, rows in df.groupby('TaxonomyValue'):
        rows = rows.sort_values(by=['Percentile'])

        lowess_xs, lowess_ys = zip(*sm.nonparametric.lowess(
            rows['NonUSPercent'],
            rows['Percentile'],
            frac=.4,
        ))

        color = rows.iloc[0]['UmbrellaColor']
        ax.plot(
            lowess_xs,
            lowess_ys,
            color=color,
        )

        if scatter:
            ax.scatter(
                rows['Percentile'],
                rows['NonUSPercent'],
                color=color,
                # facecolor=hnelib.plot.set_alpha_on_colors(color),
                # edgecolor=color,
                s=.5,
            )

    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlim(0, 100)

    if scatter:
        ax.set_yticks([0, 15, 30, 45, 60])
        ax.set_ylim(0, 60)
    else:
        ax.set_yticks([0, 5, 10, 15, 20, 25])
        ax.set_ylim(0, 25)

    hnelib.plot.add_gridlines_on_ticks(ax)

    ax.set_xlabel(r'prestige (high $\rightarrow$ low)')
    ax.set_ylabel('% non-U.S.')

    hnelib.plot.hide_axis(axes[1])
    legend = axes[1].legend(
        handles=usfhn.plot_utils.get_umbrella_legend_handles(
            style='line',
            include_academia=True,
            extra_kwargs={'alpha': 1},
        ),
        loc='center',
    )

    usfhn.plot_utils.set_umbrella_legend_text_colors(legend)


def plot_non_us_vs_prestige_correlation_across_levels():
    df = usfhn.stats.runner.get('non-us/by-institution/df')

    df = df.merge(
        usfhn.stats.runner.get('ranks/df', rank_type='prestige'),
        on=[
            'InstitutionId',
            'TaxonomyLevel',
            'TaxonomyValue',
        ]
    )[
        [
            'NonUSFraction',
            'Percentile',
            'InstitutionId',
            'TaxonomyLevel',
            'TaxonomyValue',
        ]
    ].drop_duplicates()

    df['Percentile'] *= 100

    correlations = []
    for (level, value), rows in df.groupby(['TaxonomyLevel', 'TaxonomyValue']):
        correlations.append({
            'TaxonomyLevel': level,
            'TaxonomyValue': value,
            'Correlation': pearsonr(rows['Percentile'], rows['NonUSFraction'])[0],
        })

    df = pd.DataFrame(correlations)

    fig, axes, legend_ax = plot_univariate_value_across_taxonomy_levels(
        df,
        'Correlation',
        'correlation between prestige and % non-U.S.',
        y_min=-.75,
        y_max=.5,
        y_step=.25,
    )

    for ax in axes:
        ax.axhline(
            0,
            color='black',
            lw=1,
            zorder=1,
        )

def plot_non_us_by_country(n=15, academia_only=False):
    df = usfhn.stats.runner.get('non-us/by-country/overall')

    academia_df = usfhn.views.filter_by_taxonomy(df, level='Academia', value='Academia')

    ordering_rows = []
    for (level, value), rows in academia_df.groupby(['TaxonomyLevel', 'TaxonomyValue']):
        rows = rows.sort_values(by='CountryFraction', ascending=False)

        for i, country_name in enumerate(list(rows['CountryName'])):
            ordering_rows.append({
                'CountryName': country_name,
                'Ordering': i,
            })

    ordering_df = pd.DataFrame(ordering_rows)
    df = df.merge(
        ordering_df,
        on=[
            'CountryName',
        ]
    )

    if academia_only:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 8), gridspec_kw={'width_ratios': [8, 2]})
        ax = axes[0]

    df = df[
        df['TaxonomyLevel'].isin(['Academia', 'Umbrella'])
    ]

    df = usfhn.views.annotate_umbrella_color(df, taxonomization_level='Umbrella')
    df['FadedColor'] = df['UmbrellaColor'].apply(hnelib.plot.set_alpha_on_colors)

    df = df[
        df['Ordering'] < n
    ]

    if academia_only:
        df = df[
            df['TaxonomyValue'] == 'Academia'
        ]

    for umbrella, rows in df.groupby('TaxonomyValue'):
        rows = rows.sort_values(by='Ordering')
        ax.scatter(
            rows['Ordering'],
            rows['CountryFraction'],
            facecolors=rows['FadedColor'],
            edgecolors=rows['UmbrellaColor'],
        )

    df = df.sort_values(by='Ordering')
    ax.set_xticks(df['Ordering'])
    ax.set_xticklabels(df['CountryName'], rotation=90)

    ax.set_ylabel('fraction of faculty from country')

    if not academia_only:
        hnelib.plot.hide_axis(axes[1])
        axes[1].legend(
            handles=plot_utils.get_umbrella_legend_handles(
                style='scatter',
                include_academia=True,
            ),
            loc='center',
        )

def plot_non_us_by_continent():
    df = usfhn.stats.runner.get('non-us/by-country/by-continent')

    continents = reversed(sorted(list(df['Continent'].unique())))
    continent_ordering = {c: i for i, c in enumerate(continents)}

    df['Ordering'] = df['Continent'].apply(continent_ordering.get)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={'width_ratios': [8, 2]})
    ax = axes[0]

    df = df[
        df['TaxonomyLevel'].isin(['Academia', 'Umbrella'])
    ]

    df = usfhn.views.annotate_umbrella_color(df, taxonomization_level='Umbrella')
    df['FadedColor'] = df['UmbrellaColor'].apply(hnelib.plot.set_alpha_on_colors)

    for umbrella, rows in df.groupby('TaxonomyValue'):
        rows = rows.sort_values(by='Ordering')
        ax.scatter(
            rows['ContinentFraction'],
            rows['Ordering'],
            facecolors=rows['FadedColor'],
            edgecolors=rows['UmbrellaColor'],
        )

    df = df.sort_values(by='Ordering')
    ax.set_yticks(df['Ordering'])
    ax.set_yticklabels(df['Continent'])

    ax.set_xlabel('fraction of non-U.S. faculty from continent')

    ax.set_xlim(0, .75)
    ticks = [0, .25, .5, .75]
    ax.set_xticks(ticks)
    ax.set_xticklabels(hnelib.plot.stringify_numbers_without_ugly_zeros(ticks))
    hnelib.plot.add_gridlines_on_ticks(ax, y=False)

    hnelib.plot.hide_axis(axes[1])
    axes[1].legend(
        handles=plot_utils.get_umbrella_legend_handles(
            style='scatter',
            include_academia=True,
        ),
        loc='center',
    )


def plot_non_us_by_continent_separate_domains():
    df = usfhn.stats.runner.get('non-us/by-country/by-continent')

    continents = reversed(sorted(list(df['Continent'].unique())))
    continent_ordering = {c: i for i, c in enumerate(continents)}

    df['Ordering'] = df['Continent'].apply(continent_ordering.get)

    fig, axes = plt.subplots(2, 5, figsize=(14, 6), tight_layout=True)

    layout = plot_utils.get_academia_umbrella_grid()

    df = df[
        df['TaxonomyLevel'].isin(['Academia', 'Umbrella'])
    ]

    visible_axes = []

    for i, layout_row in enumerate(layout):
        for j, taxonomy_value in enumerate(layout_row):
            ax = axes[i, j]

            if not taxonomy_value:
                hnelib.plot.hide_axis(ax)
                continue

            visible_axes.append(ax)

            _df = df[
                df['TaxonomyValue'] == taxonomy_value
            ]

            color = PLOT_VARS['colors']['umbrellas'][taxonomy_value]
            faded_color = hnelib.plot.set_alpha_on_colors(color)

            print(_df.head())

            ax.barh(
                _df['Ordering'],
                _df['ContinentFraction'],
                facecolor=color,
                edgecolor=faded_color,
            )

            if taxonomy_value in ['Academia', 'Humanities']:
                ax.set_yticks(_df['Ordering'])
                ax.set_yticklabels(_df['Continent'])
            else:
                ax.set_yticks([])
            
            ax.set_xlim(0, .75)
            ticks = [0, .25, .5, .75]
            ax.set_xticks(ticks)
            ax.set_xticklabels(hnelib.plot.stringify_numbers_without_ugly_zeros(ticks))

            hnelib.plot.add_gridlines_on_ticks(ax, y=False)
    # ax.set_xlabel('fraction of non-U.S. faculty from continent')


    # axes[1].legend(
    #     handles=plot_utils.get_umbrella_legend_handles(
    #         style='scatter',
    #         include_academia=True,
    #     ),
    #     loc='center',
    # )


def plot_non_us_by_country_compare_to_academia_fraction(n=15):
    df = usfhn.stats.runner.get('non-us/by-country/overall')

    academia_df = usfhn.views.filter_by_taxonomy(df, level='Academia', value='Academia')

    ordering_rows = []
    for (level, value), rows in academia_df.groupby(['TaxonomyLevel', 'TaxonomyValue']):
        rows = rows.sort_values(by='CountryNonUSFraction', ascending=False)

        for i, (country, fraction) in enumerate(zip(rows['CountryName'], rows['CountryNonUSFraction'])):
            ordering_rows.append({
                'CountryName': country,
                'AcademiaCountryNonUSFraction': fraction,
                'Ordering': i,
            })

    ordering_df = pd.DataFrame(ordering_rows)
    df = df.merge(
        ordering_df,
        on='CountryName',
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 8), gridspec_kw={'width_ratios': [8, 2]})
    ax = axes[0]

    df = df[
        df['TaxonomyLevel'].isin(['Umbrella'])
    ]

    df['Value'] = df['CountryNonUSFraction'] / df['AcademiaCountryNonUSFraction']

    df = usfhn.views.annotate_umbrella_color(df, taxonomization_level='Umbrella')
    df['FadedColor'] = df['UmbrellaColor'].apply(hnelib.plot.set_alpha_on_colors)

    df = df[
        df['Ordering'] < n
    ]

    for umbrella, rows in df.groupby('TaxonomyValue'):
        rows = rows.sort_values(by='Ordering')
        ax.scatter(
            rows['Ordering'],
            rows['Value'],
            facecolors=rows['FadedColor'],
            edgecolors=rows['UmbrellaColor'],
            zorder=2,
        )

    ax.axhline(
        1,
        color='black',
        zorder=1,
        lw=1,
    )

    df = df.sort_values(by='Ordering')
    ax.set_ylim([0, 2.5])
    ax.set_xticks(df['Ordering'])
    ax.set_xticklabels(df['CountryName'], rotation=90)

    ax.set_ylabel('enrichment/depletion of faculty from country\ncompared to academia')

    hnelib.plot.hide_axis(axes[1])
    axes[1].legend(
        handles=plot_utils.get_umbrella_legend_handles(style='scatter'),
        loc='center',
    )

def plot_non_us_by_gender_across_levels():
    df = usfhn.stats.runner.get('non-us/gender')[
        [
            'NonUSFraction',
            'TaxonomyLevel',
            'TaxonomyValue',
            'Gender',
        ]
    ].drop_duplicates()

    male_df = df[
        df['Gender'] == 'Male'
    ].rename(columns={
        'NonUSFraction': 'NonUSFraction-Male',
    }).drop(columns=['Gender'])

    female_df = df[
        df['Gender'] == 'Female'
    ].rename(columns={
        'NonUSFraction': 'NonUSFraction-Female',
    }).drop(columns=['Gender'])

    df = male_df.merge(
        female_df,
        on=[
            'TaxonomyLevel',
            'TaxonomyValue',
        ]
    )

    usfhn.standard_plots.plot_connected_two_values_by_taxonomy_level(
        df,
        open_column='NonUSFraction-Male',
        closed_column='NonUSFraction-Female',
        open_label='men',
        closed_label='women',
        y_label='fraction Non U.S. PhDs',
        y_min=0,
        y_max=.3,
        y_step=.075,
    )

def plot_self_hires_career_age():
    fig, ax = plt.subplots(figsize=(6, 6), tight_layout=True)
    df = usfhn.stats.runner.get('self-hires/by-career-age/df')

    df = df[
        df['TaxonomyLevel'].isin(['Academia', 'Umbrella'])
    ]

    df = usfhn.views.annotate_umbrella_color(df, 'Umbrella')
    df['PercentSelfHires'] = 100 * df['SelfHiresFraction']

    df = hnelib.pandas.add_proportions_confidence_interval(
        df,
        count_col='SelfHires',
        observations_col='Faculty',
        groupby_cols=['CareerAge', 'TaxonomyValue'],
        as_percent=True,
    )

    umbrellas = usfhn.views.get_umbrellas() + ['Academia']
    df['Index'] = df['TaxonomyValue'].apply(umbrellas.index)

    for _, rows in df.groupby(['Index', 'TaxonomyValue']):
        rows = rows.copy()
        rows = rows.sort_values(by='CareerAge')


        ax.plot(
            rows['CareerAge'],
            rows['PercentSelfHires'],
            color=rows.iloc[0]['UmbrellaColor'],
            lw=1,
            zorder=2,
        )

        ax.fill_between(
            rows['CareerAge'],
            y1=rows['LowerConf'],
            y2=rows['UpperConf'],
            color=rows.iloc[0]['FadedUmbrellaColor'],
            zorder=1,
            alpha=.25,
        )

    ax.set_ylim(0, 30)
    ax.set_yticks([0, 5, 10, 15, 20, 25, 30])
    ax.set_ylabel('% self-hires')

    ax.set_xlim(1, 40)
    ax.set_xticks([1, 10, 20, 30, 40])
    ax.set_xlabel('career age (years since doctorate)')

    hnelib.plot.add_gridlines_on_ticks(ax)

def plot_career_length_prestige_self_hires_logits(rank_type='employing-institution'):
    df = usfhn.stats.runner.get('self-hires/by-rank/binned-logits', rank_type=rank_type)

    df = usfhn.views.filter_to_academia_and_domain(df)

    df = usfhn.views.annotate_umbrella_color(df, 'Umbrella')

    df['Significant'] = df['Percentile-P'] < .05

    df['Percentile'] /= -10

    data_df = df.copy()[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'Percentile',
            'Percentile-P',
            'CareerAgeBin',
            'UmbrellaColor',
        ]
    ].drop_duplicates().rename(columns={
        'Percentile': 'LogOdds',
        'Percentile-P': 'P',
        'UmbrellaColor': 'Color',
    })

    element_id = 0
    plot_data = []
    for i, row in data_df.iterrows():
        for col in data_df.columns:
            plot_data.append({
                'Element': element_id,
                'Attribute': col,
                'Value': row[col],
            })
        element_id += 1

    fig, ax = plt.subplots(figsize=(hnelib.plot.WIDTHS['1-col'], 2.36), tight_layout=True)

    career_age_bins = sorted(list(df['CareerAgeBin'].unique()))

    df['BinIndex'] = df['CareerAgeBin'].apply(career_age_bins.index)

    xticks = []
    xticklabels = []

    for taxonomy_value, rows in df.groupby('TaxonomyValue'):
        rows = rows.sort_values(by='BinIndex')

        ax.plot(
            rows['BinIndex'],
            rows['Percentile'],
            color=rows.iloc[0]['UmbrellaColor'],
            lw=.5,
            zorder=1,
        )

    for career_age_bin, rows in df.groupby('CareerAgeBin'):
        index = career_age_bins.index(career_age_bin)
        xticks.append(index)
        xticklabels.append(career_age_bin)

        ax.scatter(
            rows['BinIndex'],
            rows['Percentile'],
            color='w',
            zorder=2,
            s=8,
            lw=.65,
        )


        for i, row in rows.iterrows():
            if row['Significant']:
                ax.scatter(
                    [row['BinIndex']],
                    [row['Percentile']],
                    edgecolor=[row['UmbrellaColor']],
                    facecolor=[row['FadedUmbrellaColor']],
                    s=8,
                    lw=.5,
                    zorder=3,
                )
            else:
                ax.scatter(
                    [row['BinIndex']],
                    [row['Percentile']],
                    color=[row['UmbrellaColor']],
                    marker='x',
                    s=8,
                    lw=.5,
                    zorder=3,
                )

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel('career age\n(years since doctorate)')
    ax.set_ylabel('change in log odds of being a self-hire\nfor a one-decile increase in prestige')

    legend = plot_utils.add_umbrella_legend(
        ax,
        get_umbrella_legend_handles_kwargs={
            'include_academia': True,
        },
        legend_kwargs={
            # 'fontsize': legend_fontsize,
            'loc': 'center left',
            'bbox_to_anchor': (1, .5),
            'bbox_transform': ax.transAxes,
            'fontsize': hnelib.plot.FONTSIZES['annotation'],
        },
        extra_legend_handles=[
            Line2D(
                [0], [0],
                color='none',
                marker='o',
                markerfacecolor=hnelib.plot.set_alpha_on_colors('black'),
                markeredgecolor='black',
                markersize=4,
                markeredgewidth=.65,
                label='P < .05',
            ),
            Line2D(
                [0], [0],
                color='none',
                marker='x',
                markerfacecolor='w',
                markeredgecolor='black',
                markersize=4,
                markeredgewidth=.65,
                label='P >= .05',
            ),
        ]
    )

    ax.set_yticks(ax.get_yticks())
    
    ax.set_yticks([-0.05, 0, .05, .1, .15, .2, .25])
    ax.set_yticklabels(["-.05", "0", ".05", ".10", ".15", ".20", ".25"])
    ax.set_xlim(-.5, 3.5)
    hnelib.plot.add_gridlines_on_ticks(ax, x=False)
    ax.set_ylim(-.052, .27)

    ax.axhline(
        0,
        color=hnelib.plot.COLORS['dark_gray'],
        lw=1,
        zorder=2,
    )

    hnelib.plot.finalize(ax)

    return pd.DataFrame(plot_data)


def plot_non_us_career_age():
    fig, ax = plt.subplots(figsize=(6, 6), tight_layout=True)
    df = usfhn.stats.runner.get('non-us/by-career-age/df')

    df = df[
        df['TaxonomyLevel'].isin(['Academia', 'Umbrella'])
    ]

    df = usfhn.views.annotate_umbrella_color(df, 'Umbrella')
    df['PercentNonUS'] = 100 * df['NonUSFraction']

    df = hnelib.pandas.add_proportions_confidence_interval(
        df,
        count_col='NonUSFacultyCount',
        observations_col='FacultyCount',
        groupby_cols=['CareerAge', 'TaxonomyValue'],
        as_percent=True,
    )

    umbrellas = usfhn.views.get_umbrellas() + ['Academia']
    df['Index'] = df['TaxonomyValue'].apply(umbrellas.index)

    for _, rows in df.groupby(['Index', 'TaxonomyValue']):
        rows = rows.copy()
        rows = rows.sort_values(by='CareerAge')


        ax.plot(
            rows['CareerAge'],
            rows['PercentNonUS'],
            color=rows.iloc[0]['UmbrellaColor'],
            lw=1,
            zorder=2,
        )

        ax.fill_between(
            rows['CareerAge'],
            y1=rows['LowerConf'],
            y2=rows['UpperConf'],
            color=rows.iloc[0]['FadedUmbrellaColor'],
            zorder=1,
            alpha=.25,
        )

    ax.set_ylim(0, 30)
    ax.set_yticks([0, 5, 10, 15, 20, 25, 30])
    ax.set_ylabel('% non-U.S.')

    ax.set_xlim(1, 40)
    ax.set_xticks([1, 10, 20, 30, 40])
    ax.set_xlabel('career age (years since doctorate)')

    hnelib.plot.add_gridlines_on_ticks(ax)

def plot_non_us_career_age_by_continent():
    fig, axes = plt.subplots(1, 6, figsize=(16, 4), tight_layout=True)

    df = usfhn.stats.runner.get('non-us/by-career-age/by-continent/df')

    df = df[
        df['TaxonomyLevel'].isin(['Academia', 'Umbrella'])
    ]

    df = usfhn.views.annotate_umbrella_color(df, 'Umbrella')
    df['PercentNonUS'] = 100 * df['NonUSFraction']

    continents = sorted(df[df['Continent'].notna()]['Continent'].unique())

    for continent, rows in df.groupby('Continent'):
        ax = axes[continents.index(continent)]

        rows = hnelib.pandas.add_proportions_confidence_interval(
            rows,
            count_col='NonUSFacultyCount',
            observations_col='FacultyCount',
            groupby_cols=['CareerAge', 'TaxonomyValue'],
            as_percent=True,
        )

        umbrellas = usfhn.views.get_umbrellas() + ['Academia']
        rows['Index'] = rows['TaxonomyValue'].apply(umbrellas.index)

        for _, _rows in rows.groupby(['Index', 'TaxonomyValue']):
            _rows = _rows.copy()
            _rows = _rows.sort_values(by='CareerAge')

            ax.plot(
                _rows['CareerAge'],
                _rows['PercentNonUS'],
                color=_rows.iloc[0]['UmbrellaColor'],
                lw=1,
                zorder=2,
            )

            ax.fill_between(
                _rows['CareerAge'],
                y1=_rows['LowerConf'],
                y2=_rows['UpperConf'],
                color=_rows.iloc[0]['FadedUmbrellaColor'],
                zorder=1,
                alpha=.25,
            )

        ax.set_ylim(0, 25)
        ax.set_yticks([0, 5, 10, 15, 20, 25])
        ax.set_ylabel('% non-U.S.')

        ax.set_xlim(1, 40)
        ax.set_xticks([1, 10, 20, 30, 40])
        ax.set_xlabel('career age (years since doctorate)')
        ax.set_title(continent)

        hnelib.plot.add_gridlines_on_ticks(ax)

def plot_career_trajectory():
    ranks = usfhn.stats.runner.get('ranks/df', rank_type='prestige')
    ranks = usfhn.views.filter_by_taxonomy(ranks, level='Academia')[
        [
            'InstitutionId',
            'Percentile'
        ]
    ].drop_duplicates()

    df = usfhn.stats.runner.get('careers/df')

    df = df.merge(
        ranks,
        on='InstitutionId',
    )

    df['JobCount'] = df.groupby('PersonId')['InstitutionId'].transform('nunique')
    df = df[
        df['JobCount'] > 2
    ]

    phds = df[
        df['CareerStep'] == 0
    ].copy().rename(columns={
        'Percentile': 'PhDRank'
    })[
        [
            'PersonId',
            'PhDRank',
        ]
    ].drop_duplicates()

    first_jobs = df[
        df['CareerStep'] == 1
    ].copy().rename(columns={
        'Percentile': 'FirstJobRank'
    })[
        [
            'PersonId',
            'FirstJobRank',
        ]
    ].drop_duplicates()


    df['LastJob'] = df.groupby('PersonId')['CareerStep'].transform('max')

    last_jobs = df[
        df['CareerStep'] == df['LastJob']
    ].copy().rename(columns={
        'Percentile': 'LastJobRank'
    })[
        [
            'PersonId',
            'LastJobRank',
        ]
    ].drop_duplicates()

    df = phds.merge(
        first_jobs,
        on='PersonId',
    ).merge(
        last_jobs,
        on='PersonId',
    )

    df['FirstJobRank'] = 100 - df['FirstJobRank']
    df['LastJobRank'] = 100 - df['LastJobRank']
    df['PhDRank'] = 100 - df['PhDRank']

    # df = df.head()
    df['Delta1'] = df['FirstJobRank'] - df['PhDRank']
    df['Delta2'] = df['LastJobRank'] - df['FirstJobRank']

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.scatter(
        df['Delta1'],
        df['Delta2'],
        s=1,
        facecolor='w',
        edgecolor=hnelib.plot.set_alpha_on_colors(plot_utils.PLOT_VARS['colors']['color_1'], .25),
    )

    min_lim = min([ax.get_xlim()[0], ax.get_ylim()[0]])
    max_lim = min([ax.get_xlim()[1], ax.get_ylim()[1]])
    max_lim = max([min_lim, max_lim])

    ax.set_xlim(-max_lim, max_lim)
    ax.set_ylim(-max_lim, max_lim)

    ax.plot(
        [ax.get_xlim()[0], ax.get_ylim()[1]],
        [ax.get_xlim()[1], ax.get_ylim()[0]],
        color='red',
        lw=1,
    )

    ax.plot(
        [0, 0],
        ax.get_ylim(),
        color='black',
        lw=1,
    )

    ax.plot(
        ax.get_xlim(),
        [0, 0],
        color='black',
        lw=1,
    )

    ax.set_xlabel("prestige (first job - phd)")
    ax.set_ylabel("prestige (last job - first job)")

    # ax.scatter(
    #     [1 for i in range(len(df))],
    #     df['PhDRank'],
    # )

    # ax.scatter(
    #     [2 for i in range(len(df))],
    #     df['FirstJobRank'],
    # )

    # ax.scatter(
    #     [3 for i in range(len(df))],
    #     df['LastJobRank'],
    # )

    # for i, row in df.iterrows():

def plot_attrition_academia_and_domain(df, column, ylabel, ylim, yticks):
    df = df[
        df['TaxonomyLevel'].isin(['Academia', 'Umbrella'])
    ]

    df = df[
        [
            'Year',
            column,
            'TaxonomyLevel',
            'TaxonomyValue',
        ]
    ].drop_duplicates()

    df = usfhn.views.annotate_umbrella_color(df, taxonomization_level='Umbrella')
    df['FadedColor'] = df['UmbrellaColor'].apply(hnelib.plot.set_alpha_on_colors)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), gridspec_kw={'width_ratios': [5, 3]})

    ax = axes[0]
    umbrellas = usfhn.views.get_umbrellas() + ['Academia']
    for umbrella in umbrellas:
        rows = df[
            df['TaxonomyValue'] == umbrella
        ]
        rows = rows.sort_values(by='Year')
        color = rows.iloc[0]['UmbrellaColor']
        faded_color = rows.iloc[0]['FadedColor']

        hnelib.plot.plot_connected_scatter(
            ax,
            rows,
            'Year',
            column,
            color,
        )

    ax.set_xlabel('year')
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim)

    ax.set_yticks(yticks)
    ax.set_yticklabels(hnelib.plot.stringify_numbers_without_ugly_zeros(yticks))

    hnelib.plot.add_gridlines_on_ticks(ax, x=False)

    hnelib.plot.hide_axis(axes[1])
    axes[1].legend(
        handles=plot_utils.get_umbrella_legend_handles(
            style='line',
            include_academia=True,
            extra_kwargs={
                'alpha': 1,
            }
        ),
        loc='center'
    )

    return axes

def plot_attrition_risk_by_taxonomy():
    df = usfhn.stats.runner.get('attrition/risk/taxonomy')

    df = df[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'AttritionRisk',
        ]
    ].drop_duplicates()

    fig, axes, legend_ax = plot_univariate_value_across_taxonomy_levels(
        df,
        'AttritionRisk',
        f'attrition risk',
        y_min=0,
        y_max=.6,
        y_step=.1,
    )


def plot_attrition_risk_over_time():
    df = usfhn.stats.runner.get('attrition/risk/by-time/taxonomy')

    df = df[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'Year',
            'AttritionRisk',
        ]
    ].drop_duplicates()


    plot_attrition_academia_and_domain(
        df,
        column='AttritionRisk',
        ylabel='attrition risk',
        ylim=[0, .12],
        yticks=[0, .03, .06, .09, .12],
    )


def plot_exit_vs_retirement_attrition_rates():
    df = usfhn.stats.runner.get('attrition/risk/by-time/by-career-stage/taxonomy')

    df = df[
        [
            'AttritionRisk',
            'TaxonomyLevel',
            'TaxonomyValue',
            'CareerStage',
            'Year'
        ]
    ].drop_duplicates()

    early_df = df[df['CareerStage'] == 'Early']
    late_df = df[df['CareerStage'] == 'Late']

    usfhn.standard_plots.compare_two_values_at_the_domain_level_over_time(
        early_df, 
        late_df,
        column='AttritionRisk',
        ylabel='attrition risk',
        title1='early exit',
        title2='retirement',
        ylim=[0, .20],
        yticks=[0, .05, .1, .15, .2],
    )


def plot_exit_vs_retirement_attrition_rates_risk_ratio():
    df = usfhn.stats.runner.get('attrition/risk/by-time/by-career-stage/taxonomy')

    df = df[
        [
            'AttritionRisk',
            'TaxonomyLevel',
            'TaxonomyValue',
            'CareerStage',
            'Year'
        ]
    ].drop_duplicates()

    early_df = df[
        df['CareerStage'] == 'Early'
    ].rename(columns={
        'AttritionRisk': 'EarlyAttritionRisk'
    }).drop(columns=['CareerStage'])

    late_df = df[
        df['CareerStage'] == 'Late'
    ].rename(columns={
        'AttritionRisk': 'LateAttritionRisk'
    }).drop(columns=['CareerStage'])

    df = late_df.merge(
        early_df,
        on=[
            'TaxonomyValue',
            'TaxonomyLevel',
            'Year',
        ]
    )

    df['RiskRatio'] = df['EarlyAttritionRisk'] / df['LateAttritionRisk']

    usfhn.standard_plots.plot_domain_level_values_over_time(
        df,
        column='RiskRatio',
        ylabel='attrition risk (early exit) / attrition risk (retirement)',
        ylim=[0, .9],
        yticks=[0, .15, .30, .45, .6, .75, .9],
    )


def plot_self_hire_attrition_by_time():
    df = usfhn.stats.runner.get('attrition/risk/by-time/self-hires')

    df = df[
        df['SelfHire']
    ]

    usfhn.standard_plots.plot_domain_level_values_over_time(
        df,
        column='AttritionRisk',
        ylabel='fraction of attritions that were self-hires',
        ylim=[0, .25],
        yticks=[0, .05, .10, .15, .2, .25],
    )


def moves_over_time_by_gender(gender='Female'):
    df = usfhn.stats.runner.get('careers/by-year/gender')

    df = df[
        df['Gender'] == gender
    ]

    fig, ax = plt.subplots(figsize=(6, 6), tight_layout=True)
    hnelib.plot.plot_connected_scatter(
        ax,
        df,
        'Year',
        'MoveRisk',
        color=PLOT_VARS['colors']['dark_gray'],
    )

    ax.set_xlabel('year')
    ax.set_ylabel('career move risk')

def plot_institution_mcm_risk():
    df = usfhn.stats.runner.get('careers/institution-risk')
    print(df.head())

    fig, ax = plt.subplots(figsize=(6, 6), tight_layout=True)

    ax.scatter(
        df['LeavingRisk'],
        df['JoiningRisk'],
        facecolor=hnelib.plot.set_alpha_on_colors(PLOT_VARS['colors']['color_1']),
        color=PLOT_VARS['colors']['color_1'],
        s=5,
        zorder=2,
    )

    # ax.set_xlim(0, ax.get_xlim()[1])
    # ax.set_ylim(0, ax.get_ylim()[1])
    ax.set_xlabel('leave risk')
    ax.set_ylabel('join risk')
    hnelib.plot.square_axis(ax)

    ax.plot(
        ax.get_xlim(),
        ax.get_xlim(),
        color=PLOT_VARS['colors']['dark_gray'],
        lw=1,
        zorder=1,
    )

def plot_hierarchy_changes_from_mcms(movement_type='Self-Hire'):
    df = usfhn.stats.runner.get('careers/hierarchy-changes-from-mcms')
    df = df[
        df['MovementType'] == movement_type
    ]

    if movement_type == 'Self-Hire':
        y_min = -.25
        y_max = .25
        y_step = .125

    y_min = -.25
    y_max = .25
    y_step = .125

    fig, axes, legend_ax = plot_univariate_value_across_taxonomy_levels(
        df,
        'MovementFraction-Difference',
        f'post mcm - pre mcm {usfhn.plot_utils.MOVEMENT_TO_STRING[movement_type]} rate',
        y_min=y_min,
        y_max=y_max,
        y_step=y_step,
    )


def plot_non_doctorates_by_taxonomy():
    df = usfhn.stats.runner.get('demographics/non-doctorates-by-taxonomy')

    df['Percent'] = 100 * df['Fraction']

    umbrellas = ['Academia'] + usfhn.views.get_umbrellas()

    data_df = df.copy()
    data_df = usfhn.plot_utils.annotate_color(data_df, add_umbrella=True)
    data_df['Index'] = data_df['Umbrella'].apply(umbrellas.index)

    data_df = data_df[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'Index',
            'Umbrella',
            'Percent',
        ]
    ].drop_duplicates().rename(columns={
        'Index': 'X',
        'Percent': 'Y',
    })

    plot_data = []
    element_id = 0
    for i, row in data_df.iterrows():
        for col in data_df.columns:
            plot_data.append({
                'Element': element_id,
                'Attribute': col,
                'Value': row[col],
            })

        element_id += 1

    fig, ax, legend_ax = usfhn.standard_plots.plot_univariate_value_across_taxonomy_levels_single_plot(
        df,
        'Percent',
        y_label=f'% of faculty without a doctorate',
        y_min=0,
        y_max=70,
        y_step=10,
        figsize=(hnelib.plot.WIDTHS['1-col'], 2.025),
    )

    df = usfhn.views.filter_by_taxonomy(df, 'Field')
    df = df.sort_values(by='Percent', ascending=False)

    df = usfhn.views.annotate_umbrella(df, taxonomization_level='Field')

    df['TaxonomyValue'] = df['TaxonomyValue'].apply(usfhn.plot_utils.main_text_taxonomy_string)

    y_pad = 1.6
    x_pad = .2

    n_annotations = 6

    for i in range(n_annotations):
        row = df.iloc[i]

        ax.annotate(
            plot_utils.clean_taxonomy_string(row['TaxonomyValue']),
            xy=(umbrellas.index(row['Umbrella']), row['Percent'] + y_pad),
            ha='center',
            va='bottom',
            fontsize=hnelib.plot.FONTSIZES['annotation'],
        )

    ax.set_ylim(0, 76)
    ax.set_yticks([0, 25, 50, 75])
    hnelib.plot.add_gridlines_on_ticks(ax, x=False)

    hnelib.plot.finalize(ax)

    return pd.DataFrame(plot_data)


def plot_career_length_by_taxonomy():
    df = usfhn.stats.runner.get('careers/length/taxonomy')

    fig, axes, legend_ax = plot_univariate_value_across_taxonomy_levels(
        df,
        'MeanCareerLength',
        f'mean career age',
        y_min=15,
        y_max=30,
        y_step=5,
    )

def plot_career_length_by_taxonomy_and_time(measure='mean'):
    df = usfhn.stats.runner.get('careers/length/by-year/taxonomy')

    if measure == 'mean':
        ylabel = 'mean career age'
        col = 'MeanCareerLength'
        ylim = [17, 25]
    elif measure == 'median':
        ylabel = 'median career length'
        col = 'MedianCareerLength'
        ylim = [14, 23]


    usfhn.standard_plots.plot_domain_level_values_over_time(
        df,
        col,
        ylabel,
        ylim=ylim,
        s=15,
    )


def plot_demographic_change_multiplot():
    fig, axes = plt.subplots(
        1, 2,
        figsize=(6, 5),
        gridspec_kw={'width_ratios': [3, 3]},
        tight_layout=True,
    )

    plot_taxonomy_size_percent_change(ax=axes[0], draw_legend=False)
    plot_career_length_change(ax=axes[1])

    hnelib.plot.finalize(axes, [-.26, -.045], plot_label_y_pad=1.1)


def plot_taxonomy_size_percent_change(ax=None, draw_legend=True):
    df = usfhn.stats.runner.get('taxonomy/by-year/df')[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'Year',
            'Fraction',
            'Count',
        ]
    ].drop_duplicates()

    min_year_df = df[
        df['Year'] == min(df['Year'])
    ].copy().rename(columns={
        'Fraction': 'StartingFraction',
        'Count': 'StartingCount',
    }).drop(columns=['Year'])

    df = df.merge(
        min_year_df,
        on=[
            'TaxonomyLevel',
            'TaxonomyValue',
        ]
    )

    # df['Growth'] = df['Count'] / df['StartingCount']
    df['Growth'] = df['Fraction'] / df['StartingFraction']
    df['Growth'] *= 100

    df = df[
        ~df['TaxonomyLevel'].isin(['Academia'])
    ]

    df = df[
        df['TaxonomyLevel'].isin(['Umbrella', 'Academia'])
    ]

    if not ax:
        fig, ax = plt.subplots(figsize=(6, 3), tight_layout=True)

    ax.axhline(
        1,
        lw=1,
        color='black',
        zorder=1,
    )

    for level, _df in df.groupby('TaxonomyLevel'):
        _df = usfhn.views.annotate_umbrella_color(_df, level)

        for umbrella, rows in _df.groupby('TaxonomyValue'):
            rows = rows.sort_values(by='Year')
            hnelib.plot.plot_disconnected_scatter(
                ax,
                rows,
                'Year',
                'Growth',
                color=rows.iloc[0]['UmbrellaColor'],
                lw=1.5,

            )

    if draw_legend:
        print('hey')
        legend = plot_utils.add_umbrella_legend(
            ax,
            get_umbrella_legend_handles_kwargs={
                'style': 'none',
                'include_academia': True,
            },
            legend_kwargs={
                'loc': 'center left',
                'bbox_to_anchor': (.9, .5),
                'bbox_transform': ax.transAxes,
            },
        )

    ax.set_ylim(90, 110)
    yticks = [90, 95, 100, 105, 110]
    ax.set_yticks(yticks)
    ax.set_yticklabels(hnelib.plot.stringify_numbers_without_ugly_zeros(yticks))

    ax.set_ylabel("% of market share relative to 2011")

    ax.set_xlim(2011, 2020.1)
    xticks = [2012, 2014, 2016, 2018, 2020]
    ax.set_xticks(xticks)

    ax.set_xlabel('year')

    hnelib.plot.add_gridlines_on_ticks(ax, x=False)


def plot_domain_level_career_length_changes():
    df = usfhn.stats.runner.get('careers/length/by-year/df')[
        [
            'PersonId',
            'TaxonomyLevel',
            'TaxonomyValue',
            'CareerYear',
            'Year',
        ]
    ].drop_duplicates()

    df = df[
        df['TaxonomyLevel'].isin(['Umbrella', 'Academia'])
    ]

    df = df[
        (df['Year'] == min(df['Year']))
        |
        (df['Year'] == max(df['Year']))
    ]

    df['PeoplePerCareerYear'] = df.groupby([
        'TaxonomyLevel',
        'TaxonomyValue',
        'Year',
        'CareerYear',
    ])['PersonId'].transform('nunique')

    df['PeopleInYear'] = df.groupby([
        'TaxonomyLevel',
        'TaxonomyValue',
        'Year',
    ])['PersonId'].transform('nunique')

    df['PeoplePerCareerYearFraction'] = df['PeoplePerCareerYear'] / df['PeopleInYear']

    df = df[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'Year',
            'PeoplePerCareerYearFraction',
            'CareerYear',
        ]
    ].drop_duplicates()

    df = df[
        df['CareerYear'] <= 50
    ]

    umbrellas = usfhn.views.get_umbrellas()

    fig, axes = plt.subplots(2, 5, figsize=(14, 6), tight_layout=True)

    layout = [
        ['Academia'] + umbrellas[:4],
        [None] + umbrellas[4:],
    ]

    bins = np.arange(1, max(df['CareerYear']) + 1, 1)

    for i, layout_row in enumerate(layout):
        for j, taxonomy_value in enumerate(layout_row):
            ax = axes[i, j]

            if not taxonomy_value:
                hnelib.plot.hide_axis(ax)
                continue

            start_df = df[
                (df['Year'] == min(df['Year']))
                &
                (df['TaxonomyValue'] == taxonomy_value)
            ]

            end_df = df[
                (df['Year'] == max(df['Year']))
                &
                (df['TaxonomyValue'] == taxonomy_value)
            ]

            color = PLOT_VARS['colors']['umbrellas'][taxonomy_value]

            ax.bar(
                start_df['CareerYear'],
                start_df['PeoplePerCareerYearFraction'],
                facecolor=hnelib.plot.set_alpha_on_colors(color),
                edgecolor=color,
                zorder=2,
            )

            ax.bar(
                end_df['CareerYear'],
                end_df['PeoplePerCareerYearFraction'],
                facecolor='w',
                edgecolor=PLOT_VARS['colors']['dark_gray'],
                zorder=2,
            )

            # ax.hist(
            #     start_df['CareerYear'],
            #     bins=bins,
            #     facecolor=hnelib.plot.set_alpha_on_colors(color),
            #     edgecolor=color,
            #     zorder=2,
            # )

            # ax.hist(
            #     start_df['CareerYear'],
            #     bins=bins,
            #     facecolor='w',
            #     edgecolor=color,
            #     zorder=2,
            # )

            # mean = _df['NonUSFraction'].mean()

            # ax.axvline(
            #     mean,
            #     lw=.5,
            #     color=color,
            #     zorder=1,
            # )

            # if taxonomy_value == 'Academia':
            #     ax.annotate(
            #         "mean",
            #         xy=(mean, 38),
            #         xytext=(mean + .1, 38),
            #         ha='left',
            #         va='center',
            #         arrowprops=PLOT_VARS['arrowprops'],
            #     )

            ax.set_title(
                usfhn.plot_utils.clean_taxonomy_string(taxonomy_value),
                color=color,
            )

            ax.set_xlabel('career age (years since doctorate)')
            ax.set_ylabel('density')

            # ax.set_xlim(0, .6)

            # xticks = [0, .1, .2, .3, .4, .5, .6]
            # ax.set_xticks(xticks)
            # ax.set_xticklabels(hnelib.plot.stringify_numbers_without_ugly_zeros(xticks))

            # ax.set_ylim(0, 40)
            # ax.set_yticks([0, 10, 20, 30, 40])


def plot_career_length_change(ax=None):
    if not ax:
        fig, ax = plt.subplots(figsize=(3, 5))

    df = usfhn.stats.runner.get('careers/length/by-year/taxonomy')[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'Year',
            'MeanCareerLength',
        ]
    ].drop_duplicates()

    min_year = min(df['Year'])
    max_year = max(df['Year'])

    df = df[
        df['Year'].isin([min_year, max_year])
    ]

    df = hnelib.pandas.aggregate_df_over_column(
        df,
        agg_col='Year',
        join_cols=[
            'TaxonomyLevel',
            'TaxonomyValue',
        ],
        value_cols=['MeanCareerLength'],
        agg_value_to_label={
            min_year: 'Start',
            max_year: 'End',
        }
    )

    spacer_amount = 4
    annotation_fontsize = 9
    axis_label_fontsize = 13

    df = df[
        df['TaxonomyLevel'].isin(['Academia', 'Umbrella'])
    ]

    df['Color'] = df['TaxonomyValue'].apply(PLOT_VARS['colors']['umbrellas'].get)

    umbrellas = ['Academia'] + usfhn.views.get_umbrellas()
    umbrellas = reversed(umbrellas)

    indexes = {u: spacer_amount * (i + 1) for i, u in enumerate(umbrellas)}

    df['StartIndex'] = df['TaxonomyValue'].apply(indexes.get)
    df['EndIndex'] = df['StartIndex'] + 1
    df['LabelIndex'] = df['EndIndex'] + 1.2

    ax.barh(
        df['StartIndex'],
        df['StartMeanCareerLength'],
        facecolor='w',
        edgecolor=df['Color'],
        lw=.75,
        zorder=2,
    )

    ax.barh(
        df['EndIndex'],
        df['EndMeanCareerLength'],
        color=df['Color'],
        edgecolor=df['Color'],
        lw=.75,
        zorder=2,
    )

    ax.set_yticks([])
    for i, row in df.iterrows():
        ax.annotate(
            plot_utils.clean_taxonomy_string(row['TaxonomyValue']),
            (.5, row['LabelIndex']), 
            ha='left',
            va='center',
            color=row['Color'],
            fontsize=annotation_fontsize,
        )

    ax.set_xlabel('mean career age\n(years since doctorate)')

    ax.set_ylim(min(df['StartIndex']) - .75, ax.get_ylim()[1])

    row = df[
        df['TaxonomyValue'] == 'Academia'
    ].iloc[0]

    labels = [f'in {min_year}', f'in {max_year}']
    label_ys = [row['StartIndex'], row['EndIndex']]
    ys = [row['StartIndex'], row['EndIndex']]
    xs = [row['StartMeanCareerLength'], row['EndMeanCareerLength']]
    directions = ['down', 'up']
    # xs = [25, 25]

    arrow_x_end = 24.5
    arrow_pad = 1
    annotation_pad = .3
    up_annotation_pad = annotation_pad - .05
    down_annotation_pad = annotation_pad

    for x, y, direction, label in zip(xs, ys, directions, labels):
        ax.annotate(
            '',
            (x, y), 
            xytext=(arrow_x_end, y),
            arrowprops={
                **hnelib.plot.BASIC_ARROW_PROPS,
                'shrinkA': 0,
            },
            fontsize=axis_label_fontsize - 5,
        )

        y_end = y + arrow_pad if direction == 'up' else y - arrow_pad

        ax.plot(
            [arrow_x_end, arrow_x_end],
            [y, y_end],
            lw=.5,
            color=PLOT_VARS['colors']['dark_gray'],
        )

        y_annotation = y_end + up_annotation_pad if direction == 'up' else y_end - down_annotation_pad

        ax.annotate(
            label,
            (arrow_x_end, y_annotation),
            ha='center',
            va='bottom' if direction == 'up' else 'top',
            color=PLOT_VARS['colors']['dark_gray'],
            fontsize=axis_label_fontsize - 5,
        )

    hnelib.plot.add_gridlines(
        ax,
        xs=[20],
        lw=1,
        alpha=.5,
        zorder=1,
    )


def plot_taxonomy_size_by_year():
    df = usfhn.stats.runner.get('taxonomy/by-year/df')
    usfhn.standard_plots.plot_domain_level_values_over_time(
        df,
        'Fraction',
        'fraction of faculty',
        include_academia=False,
        ylim=[0, .3],
    )

def plot_taxonomy_size_by_year_field_level():
    df = usfhn.stats.runner.get('taxonomy/by-year/df')
    start_year = df['Year'].min()
    end_year = df['Year'].max()

    start_df = df.copy()[
        df['Year'] == start_year
    ][
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'Year',
            'Fraction',
        ]
    ].drop(columns=['Year']).rename(columns={
        'Fraction': 'StartFraction',
    })

    end_df = df.copy()[
        df['Year'] == end_year
    ][
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'Year',
            'Fraction',
        ]
    ].drop(columns=['Year']).rename(columns={
        'Fraction': 'EndFraction',
    })

    df = start_df.merge(
        end_df,
        on=[
            'TaxonomyLevel',
            'TaxonomyValue',
        ]
    )

    usfhn.standard_plots.plot_connected_two_values_by_taxonomy_level(
        df,
        open_column='StartFraction',
        closed_column='EndFraction',
        open_label=f'in {start_year}',
        closed_label=f'in {end_year}',
        y_label='fraction of faculty',
        y_min=0,
        y_max=.3,
        y_step=.075,
        include_academia=False,
    )

def plot_gender_attrition_risk_over_time(gender):
    df = usfhn.stats.runner.get('attrition/risk/by-time/gender')

    df = df[
        df['Gender'] == gender
    ]

    plot_attrition_academia_and_domain(
        df,
        column='AttritionRisk',
        ylabel=f'attrition risk ({usfhn.plot_utils.GENDER_TO_STRING[gender]})',
        ylim=[0, .15],
        yticks=[0, .03, .06, .09, .12, .15]
    )

def plot_gender_attrition_risk_ratio_over_time():
    df = usfhn.stats.runner.get('attrition/risk/by-time/gender')

    male_df = df[
        df['Gender'] == 'Male'
    ].copy().rename(columns={
        'AttritionRisk': 'MaleAttritionRisk'
    }).drop(columns=[
        'Gender'
    ]).drop_duplicates()

    female_df = df[
        df['Gender'] == 'Female'
    ].copy().rename(columns={
        'AttritionRisk': 'FemaleAttritionRisk'
    }).drop(columns=[
        'Gender'
    ]).drop_duplicates()

    df = male_df.merge(
        female_df,
        on=[
            'Year',
            'TaxonomyLevel',
            'TaxonomyValue',
        ],
    ).drop_duplicates()

    df['RiskRatio'] = df['FemaleAttritionRisk'] / df['MaleAttritionRisk']

    plot_attrition_academia_and_domain(
        df,
        column='RiskRatio',
        ylabel=f'female risk of attrition / male risk of attrition',
        ylim=[.5, 1.5],
        yticks=[.5, .75, 1, 1.25, 1.5],
    )


def plot_gender_exit_vs_retirement_attrition_rates():
    df = usfhn.stats.runner.get('attrition/risk/by-time/by-career-stage/gender')

    df = df[
        [
            'Gender',
            'AttritionRisk',
            'TaxonomyLevel',
            'TaxonomyValue',
            'CareerStage',
            'Year'
        ]
    ].drop_duplicates()

    male_df = df[
        df['Gender'] == 'Male'
    ].copy().rename(columns={
        'AttritionRisk': 'MaleAttritionRisk'
    }).drop(columns=[
        'Gender'
    ]).drop_duplicates()

    female_df = df[
        df['Gender'] == 'Female'
    ].copy().rename(columns={
        'AttritionRisk': 'FemaleAttritionRisk'
    }).drop(columns=[
        'Gender'
    ]).drop_duplicates()

    df = male_df.merge(
        female_df,
        on=[
            'Year',
            'TaxonomyLevel',
            'TaxonomyValue',
            'CareerStage',
        ],
    ).drop_duplicates()

    df['RiskRatio'] = df['FemaleAttritionRisk'] / df['MaleAttritionRisk']

    early_df = df[df['CareerStage'] == 'Early']
    late_df = df[df['CareerStage'] == 'Late']

    usfhn.standard_plots.compare_two_values_at_the_domain_level_over_time(
        early_df, 
        late_df,
        column='RiskRatio',
        ylabel='attrition risk (women) / attrition risk (men) ratio',
        title1='early exit',
        title2='retirement',
        ylim=[.5, 1.75],
        yticks=[.5, .75, 1, 1.25, 1.5, 1.75],
    )


def plot_gender_exit_vs_retirement_attrition_rates_other_risk_ratio():
    df = usfhn.stats.runner.get('attrition/risk/by-time/by-career-stage/gender')

    early_df = df[
        df['CareerStage'] == 'Early'
    ].rename(columns={
        'AttritionRisk': 'EarlyAttritionRisk'
    }).drop(columns=['CareerStage'])

    late_df = df[
        df['CareerStage'] == 'Late'
    ].rename(columns={
        'AttritionRisk': 'LateAttritionRisk'
    }).drop(columns=['CareerStage'])

    df = late_df.merge(
        early_df,
        on=[
            'TaxonomyValue',
            'TaxonomyLevel',
            'Year',
            'Gender',
        ]
    )

    df['RiskRatio'] = df['EarlyAttritionRisk'] / df['LateAttritionRisk']

    usfhn.standard_plots.compare_two_values_at_the_domain_level_over_time(
        df[df['Gender'] == 'Female'], 
        df[df['Gender'] == 'Male'], 
        column='RiskRatio',
        ylabel='attrition risk (early exit) / attrition risk (retirement)',
        title1='women',
        title2='men',
    )

def plot_gender_exit_vs_retirement_attrition_rates_quad():
    df = usfhn.stats.runner.get('attrition/risk/by-time/by-career-stage/gender')

    df = df[
        [
            'Gender',
            'AttritionRisk',
            'TaxonomyLevel',
            'TaxonomyValue',
            'CareerStage',
            'Year'
        ]
    ].drop_duplicates()

    men = df['Gender'] == 'Male'
    women = df['Gender'] == 'Female'
    early = df['CareerStage'] == 'Early'
    late = df['CareerStage'] == 'Late'

    dfs = [
        df[women & early],
        df[men & early],
        df[women & late],
        df[men & late],
    ]

    titles = [
        'early exit risk (women)',
        'early exit risk (men)',
        'retirement risk (women)',
        'retirement risk (men)',
    ]

    usfhn.standard_plots.compare_four_values_at_the_domain_level_over_time(
        dfs=dfs,
        column='AttritionRisk',
        ylabel='attrition risk',
        titles=titles,
    )


def plot_self_hire_non_self_hire_attrition_risk_ratio_by_time():
    df = usfhn.stats.runner.get('attrition/risk/by-time/self-hires')

    self_hire_df = df[
        df['SelfHire']
    ].copy().rename(columns={
        'AttritionRisk': 'SelfHireAttritionRisk'
    }).drop(columns=[
        'SelfHire'
    ]).drop_duplicates()

    non_self_hire_df = df[
        ~df['SelfHire']
    ].copy().rename(columns={
        'AttritionRisk': 'NonSelfHireAttritionRisk'
    }).drop(columns=[
        'SelfHire'
    ]).drop_duplicates()

    df = self_hire_df.merge(
        non_self_hire_df,
        on=[
            'Year',
            'TaxonomyLevel',
            'TaxonomyValue',
        ],
    ).drop_duplicates()

    df['RiskRatio'] = df['SelfHireAttritionRisk'] / df['NonSelfHireAttritionRisk']

    axes = usfhn.standard_plots.plot_domain_level_values_over_time(
        df,
        column='RiskRatio',
        ylabel=f'self-hire risk of attrition / non-self-hire risk of attrition',
        ylim=[.5, 2.5],
        yticks=[.5, 1, 1.5, 2, 2.5],
    )

    axes[0].axhline(
        1,
        color='black',
        zorder=2,
        lw=1,
    )

def plot_self_hire_exit_vs_retirement_attrition_rates():
    df = usfhn.stats.runner.get('attrition/risk/by-time/by-career-stage/self-hires')

    df = df[
        [
            'SelfHire',
            'AttritionRisk',
            'TaxonomyLevel',
            'TaxonomyValue',
            'CareerStage',
            'Year'
        ]
    ].drop_duplicates()

    self_hire_df = df[
        df['SelfHire']
    ].copy().rename(columns={
        'AttritionRisk': 'SelfHireAttritionRisk'
    }).drop(columns=[
        'SelfHire'
    ]).drop_duplicates()

    non_self_hire_df = df[
        ~df['SelfHire']
    ].copy().rename(columns={
        'AttritionRisk': 'NonSelfHireAttritionRisk'
    }).drop(columns=[
        'SelfHire'
    ]).drop_duplicates()

    df = self_hire_df.merge(
        non_self_hire_df,
        on=[
            'Year',
            'TaxonomyLevel',
            'TaxonomyValue',
            'CareerStage',
        ],
    ).drop_duplicates()

    df['RiskRatio'] = df['SelfHireAttritionRisk'] / df['NonSelfHireAttritionRisk']

    early_df = df[df['CareerStage'] == 'Early']
    late_df = df[df['CareerStage'] == 'Late']

    usfhn.standard_plots.compare_two_values_at_the_domain_level_over_time(
        early_df, 
        late_df,
        column='RiskRatio',
        ylabel='self-hire risk / non-self-hire risk ratio',
        title1='early exit',
        title2='retirement',
    )


def plot_self_hire_exit_vs_retirement_attrition_rates_other_risk_ratio():
    df = usfhn.stats.runner.get('attrition/risk/by-time/by-career-stage/self-hires')

    df = df[
        [
            'SelfHire',
            'AttritionRisk',
            'TaxonomyLevel',
            'TaxonomyValue',
            'CareerStage',
            'Year'
        ]
    ].drop_duplicates()

    early_df = df[
        df['CareerStage'] == 'Early'
    ].rename(columns={
        'AttritionRisk': 'EarlyAttritionRisk'
    }).drop(columns=['CareerStage'])

    late_df = df[
        df['CareerStage'] == 'Late'
    ].rename(columns={
        'AttritionRisk': 'LateAttritionRisk'
    }).drop(columns=['CareerStage'])

    df = late_df.merge(
        early_df,
        on=[
            'TaxonomyValue',
            'TaxonomyLevel',
            'Year',
            'SelfHire',
        ]
    )

    df['RiskRatio'] = df['EarlyAttritionRisk'] / df['LateAttritionRisk']

    usfhn.standard_plots.compare_two_values_at_the_domain_level_over_time(
        df[df['SelfHire']], 
        df[~df['SelfHire']], 
        column='RiskRatio',
        ylabel='attrition risk (early exit) / attrition risk (retirement)',
        title1='self-hires',
        title2='non-self-hires',
    )


def plot_self_hire_exit_vs_retirement_attrition_rates_quad():
    df = usfhn.stats.runner.get('attrition/risk/by-time/by-career-stage/self-hires')

    df = df[
        [
            'SelfHire',
            'AttritionRisk',
            'TaxonomyLevel',
            'TaxonomyValue',
            'CareerStage',
            'Year'
        ]
    ].drop_duplicates()

    self_hire = df['SelfHire']
    non_self_hire = ~df['SelfHire']
    early = df['CareerStage'] == 'Early'
    late = df['CareerStage'] == 'Late'

    dfs = [
        df[non_self_hire & early],
        df[self_hire & early],
        df[non_self_hire & late],
        df[self_hire & late],
    ]

    titles = [
        'early exit risk (non-self-hires)',
        'early exit risk (self-hires)',
        'retirement risk (non-self-hires)',
        'retirement risk (self-hires)',
    ]

    usfhn.standard_plots.compare_four_values_at_the_domain_level_over_time(
        dfs=dfs,
        column='AttritionRisk',
        ylabel='attrition risk',
        titles=titles,
    )

def plot_field_level_by_gender_attrition_risk():
    df = usfhn.stats.runner.get('attrition/risk/gender')

    df = df[
        [
            'Gender',
            'AttritionRisk',
            'TaxonomyLevel',
            'TaxonomyValue',
        ]
    ].drop_duplicates()

    male_df = df[
        df['Gender'] == 'Male'
    ].copy().rename(columns={
        'AttritionRisk': 'MaleAttritionRisk'
    }).drop(columns=[
        'Gender'
    ]).drop_duplicates()

    female_df = df[
        df['Gender'] == 'Female'
    ].copy().rename(columns={
        'AttritionRisk': 'FemaleAttritionRisk'
    }).drop(columns=[
        'Gender'
    ]).drop_duplicates()

    df = male_df.merge(
        female_df,
        on=[
            'TaxonomyLevel',
            'TaxonomyValue',
        ],
    ).drop_duplicates()

    usfhn.standard_plots.plot_connected_two_values_by_taxonomy_level(
        df,
        open_column='MaleAttritionRisk',
        closed_column='FemaleAttritionRisk',
        open_label='men',
        closed_label='women',
        y_label='attrition risk',
        y_min=0,
        y_max=.6,
        y_step=.1,
    )


def plot_attrition_by_rank_academia_and_domain(rank_type='production'):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), gridspec_kw={'width_ratios': [5, 3]})

    for umbrella in usfhn.views.get_umbrellas():
        plot_ranked_attrition(axes[0], level='Umbrella', value=umbrella, rank_type=rank_type)

    plot_ranked_attrition(axes[0], level='Academia', value='Academia', rank_type=rank_type)

    hnelib.plot.hide_axis(axes[1])
    axes[1].legend(
        handles=plot_utils.get_umbrella_legend_handles(
            style='line',
            include_academia=True,
            extra_kwargs={
                'alpha': 1,
            }
        ),
        loc='center'
    )

    ax = axes[0]
    ax.set_xlim(0, 1.005)
    # ax.set_ylim(0, 1.005)
    ax.set_ylim(0, .15)
    print('hey')
    ticks = [0, .2, .4, .6, .8, 1]
    ax.set_xticks(ticks)
    # ax.set_yticks(ticks)
    ax.set_xticklabels(hnelib.plot.stringify_numbers_without_ugly_zeros(ticks))
    # ax.set_yticklabels(hnelib.plot.stringify_numbers_without_ugly_zeros(ticks))
    hnelib.plot.add_gridlines_on_ticks(ax)

    ax.set_ylabel('attrition risk')

    if rank_type == 'production':
        xlabel = 'production rank'
    elif rank_type == 'doctoral-institution':
        xlabel = 'prestige rank (doctoral institution)'
    elif rank_type == 'employing-institution':
        xlabel = 'prestige rank (employing institution)'

    ax.set_xlabel(xlabel)


def plot_attrition_by_rank_academia_and_domain_by_gender(rank_type='production'):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), gridspec_kw={'width_ratios': [5, 3]})

    umbrellas = usfhn.views.get_umbrellas()
    levels = ['Academia'] + ['Umbrella' for i in range(len(umbrellas))]
    values = ['Academia'] + umbrellas

    for level, value in zip(levels, values):
        plot_ranked_attrition(axes[0], level=level, value=value, rank_type=rank_type, gender='Male', ls='-')
        plot_ranked_attrition(axes[0], level=level, value=value, rank_type=rank_type, gender='Female', ls='--')

    hnelib.plot.hide_axis(axes[1])
    axes[1].legend(
        handles=plot_utils.get_umbrella_legend_handles(
            style='line',
            include_academia=True,
            extra_kwargs={
                'alpha': 1,
            }
        ),
        loc='center'
    )

    ax = axes[0]
    ax.set_xlim(0, 1.005)
    # ax.set_ylim(0, 1.005)
    ax.set_ylim(0, .15)
    ticks = [0, .2, .4, .6, .8, 1]
    ax.set_xticks(ticks)
    # ax.set_yticks(ticks)
    ax.set_xticklabels(hnelib.plot.stringify_numbers_without_ugly_zeros(ticks))
    # ax.set_yticklabels(hnelib.plot.stringify_numbers_without_ugly_zeros(ticks))
    hnelib.plot.add_gridlines_on_ticks(ax)

    ax.set_ylabel('attrition risk')

    if rank_type == 'production':
        xlabel = 'production rank'
    elif rank_type == 'doctoral-institution':
        xlabel = 'prestige rank (doctoral institution)'
    elif rank_type == 'employing-institution':
        xlabel = 'prestige rank (employing institution)'

    ax.set_xlabel(xlabel)

def plot_ranked_attrition(ax, level, value, rank_type='production', gender=None, by_self_hire=False, ls='-'):
    df = usfhn.stats.runner.get('attrition/by-rank/institution/risk', rank_type=rank_type)

    df = usfhn.views.filter_by_taxonomy(df, level=level, value=value)
    df = usfhn.views.annotate_umbrella_color(df, taxonomization_level=level)
    df['FadedColor'] = df['UmbrellaColor'].apply(hnelib.plot.set_alpha_on_colors)
    df = df.sort_values(by='Percentile')

    color = df.iloc[0]['UmbrellaColor']
    faded_color = df.iloc[0]['FadedColor']

    if gender:
        logit_df = usfhn.stats.runner.get('attrition/by-rank/institution/gendered-logits', rank_type=rank_type)
        logit_df = logit_df[
            logit_df['Gender'] == gender
        ]
    elif by_self_hire:
        logit_df = usfhn.stats.runner.get(
            'attrition/by-rank/institution/by-self-hire/logits',
            rank_type=rank_type,
        )
        logit_df = logit_df[
            logit_df['Gender'] == gender
        ]
    else:
        logit_df = usfhn.stats.runner.get('attrition/by-rank/institution/logits', rank_type=rank_type)

    logit_row = usfhn.views.filter_by_taxonomy(logit_df, level=level, value=value).iloc[0]

    Xs = list(np.arange(0, 1.01, .01))
    constant_Xs = sm.add_constant(Xs)
    model = sm.Logit([1 for i in range(len(Xs))], Xs)
    Ys = model.predict([logit_row['const'], logit_row['Percentile']], constant_Xs)

    ax.plot(
        Xs,
        Ys,
        color=color,
        ls=ls,
        lw=1.5,
        zorder=3,
    )

    return Xs, Ys


def plot_in_or_out_of_field_vs_ranks_logit(ax, level, value, rank_type):
    df = usfhn.stats.runner.get('interdisciplinarity/institution-with-ranks', rank_type=rank_type)

    df = usfhn.views.filter_by_taxonomy(df, level, value)

    if df.empty:
        return

    df = usfhn.views.annotate_umbrella_color(df, taxonomization_level=level)
    df['FadedColor'] = df['UmbrellaColor'].apply(hnelib.plot.set_alpha_on_colors)
    df = df.sort_values(by='Percentile')

    color = df.iloc[0]['UmbrellaColor']
    faded_color = df.iloc[0]['FadedColor']

    # ax.scatter(
    #     df['Percentile'],
    #     df['NonUSFraction'],
    #     color='white',
    #     zorder=2,
    #     linewidths=.5,
    #     s=2,
    # )

    # ax.scatter(
    #     df['Percentile'],
    #     df['NonUSFraction'],
    #     facecolor=faded_color,
    #     edgecolor=color,
    #     linewidths=.5,
    #     zorder=2,
    #     s=2,
    # )

    logit_df = usfhn.stats.runner.get('interdisciplinarity/logits', rank_type=rank_type)

    logit_row = usfhn.views.filter_by_taxonomy(logit_df, level=level, value=value).iloc[0]

    Xs = df['Percentile']
    Xs = sm.add_constant(Xs)
    model = sm.Logit([1 for i in range(len(Xs))], Xs)
    Ys = model.predict([logit_row['B0'], logit_row['B1']], Xs)
    Ys = [1 - y for y in Ys]

    ls = '--' if logit_row['B1-P'] > .05 else '-'

    ax.plot(
        df['Percentile'],
        Ys,
        color=color,
        lw=2,
        ls=ls,
        zorder=3,
    )

def plot_in_or_out_of_field_vs_ranks_logit_slopes_with_p_values(rank_type='production'):
    fig, axes = plt.subplots(1, 2, figsize=(6, 4), gridspec_kw={'width_ratios': [4, 2]})

    for umbrella in usfhn.views.get_umbrellas():
        plot_in_or_out_of_field_vs_ranks_logit(axes[0], level='Umbrella', value=umbrella, rank_type=rank_type)

    hnelib.plot.hide_axis(axes[1])
    axes[1].legend(
        handles=plot_utils.get_umbrella_legend_handles(
            style='line',
            # include_academia=True,
            extra_kwargs={
                'alpha': 1,
            }
        ) + [
            Line2D(
                [0], [0],
                color='black',
                ls='-',
                label='P < .05',
            ),
            Line2D(
                [0], [0],
                color='black',
                ls='--',
                label='P >= .05',
            ),
        ],
        loc='center'
    )

    ax = axes[0]

    ax.set_xlim(0, 1.005)
    xticks = [0, .2, .4, .6, .8, 1]
    ax.set_xticks(xticks)
    ax.set_xticklabels(hnelib.plot.stringify_numbers_without_ugly_zeros(xticks))

    # ax.set_ylim(0, .15)
    yticks = [0, .05, .1, .15]
    ax.set_yticks(yticks)
    ax.set_yticklabels(hnelib.plot.stringify_numbers_without_ugly_zeros(yticks))

    ax.set_ylabel('out-of-field faculty fraction')

    if rank_type == 'production':
        xlabel = 'production rank'
    elif rank_type == 'prestige':
        xlabel = 'prestige rank'

    ax.set_xlabel(xlabel)

    hnelib.plot.add_gridlines_on_ticks(ax)


def plot_self_hire_logits(rank_type='prestige'):
    plot_variable_logit(variable='self-hires', rank_type=rank_type)


def plot_self_hire_attrition_logits(rank_type='employing-institution'):
    plot_variable_logit(variable='self-hires-attrition', rank_type=rank_type)


def plot_new_self_hire_logits(rank_type='prestige'):
    plot_variable_logit(variable='new-self-hires', rank_type=rank_type)


def plot_non_us_logits(rank_type='prestige'):
    plot_variable_logit(variable='non-us', rank_type=rank_type)


def plot_gender_logits(rank_type='prestige'):
    plot_variable_logit(variable='gender', rank_type=rank_type)


def plot_interdisciplinarity_logits(rank_type='prestige'):
    plot_variable_logit(variable='interdisciplinarity', rank_type=rank_type)

def plot_career_length_logits(rank_type='prestige'):
    plot_variable_logit(variable='career-length', rank_type=rank_type)

def plot_logit_multiplot(rank_type='prestige'):
    # fig, axes = plt.subplots(
    #     1, 5,
    #     figsize=(16, 4),
    #     gridspec_kw={
    #         'wspace': .25,
    #     },
    # )

    # fig, axes = plt.subplots(
    #     1, 6,
    #     figsize=(16, 4),
    #     gridspec_kw={
    #         'wspace': .25,
    #     },
    # )

    fig, axes = plt.subplots(
        3, 2,
        figsize=(10, 14),
    )

    # non_us_ax = axes[0]
    # non_us_new_hires_ax = axes[1]
    # gender_ax = axes[2]
    # new_hire_gender_ax = axes[3]
    # self_hires_ax = axes[4]
    # new_self_hires_ax = axes[5]

    non_us_ax = axes[0, 0]
    non_us_new_hires_ax = axes[0, 1]
    gender_ax = axes[1, 0]
    new_hire_gender_ax = axes[1, 1]
    self_hires_ax = axes[2, 0]
    new_self_hires_ax = axes[2, 1]

    plot_variable_logit(
        ax=non_us_ax,
        variable='non-us',
        rank_type=rank_type,
        draw_legend=False,
    )

    plot_variable_logit(
        ax=non_us_new_hires_ax,
        variable='non-us-new-hires',
        rank_type=rank_type,
        draw_legend=False,
        annotate_prestige_direction=False,
    )

    plot_variable_logit(
        ax=gender_ax,
        variable='gender',
        rank_type=rank_type,
        draw_legend=False,
        annotate_prestige_direction=False,
    )

    plot_variable_logit(
        ax=new_hire_gender_ax,
        variable='gender-new-hires',
        rank_type=rank_type,
        draw_legend=True,
        annotate_prestige_direction=False,
    )

    plot_variable_logit(
        ax=self_hires_ax,
        variable='self-hires',
        rank_type=rank_type,
        draw_legend=False,
        annotate_prestige_direction=False,
    )

    plot_variable_logit(
        ax=new_self_hires_ax,
        variable='self-hires-new-hires',
        rank_type=rank_type,
        draw_legend=False,
        annotate_prestige_direction=False,
    )

    axes[0, 0].set_title('all faculty')
    axes[0, 1].set_title('new faculty')

    for ax in axes[:,1]:
        ax.set_ylabel('')
        ax.set_yticks([])

    for ax in axes.flatten()[:-2]:
        ax.set_xlabel('')
        ax.set_xticks([])

    hnelib.plot.finalize(axes.flatten())

def plot_logit_slopes_tadpoleplot():
    fig, axes = plt.subplots(
        1, 3,
        figsize=(hnelib.plot.WIDTHS['2-col'], 2.025),
        # figsize=(14, 4),
        tight_layout=True,
    )

    stats_to_dfs = {
        'self-hires': {
            'all': 'self-hires/by-new-hire/by-rank/existing-hire-logits',
            'new-hires': 'self-hires/by-new-hire/by-rank/logits',
            'ax': axes[0],
            'ymin': -.2,
            'ymax': .2,
            'ystep': .1,
            'title': 'x = self-hires',
            'subfigure': 'A',
        },
        'non-us': {
            'all': 'non-us/by-new-hire/by-rank/existing-hire-logits',
            'new-hires': 'non-us/by-new-hire/by-rank/logits',
            'ax': axes[1],
            'ymin': -.2,
            'ymax': .2,
            'ystep': .1,
            'title': 'x = non-U.S. trained faculty',
            'subfigure': 'B',
        },
        'gender': {
            'all': 'gender/by-new-hire/by-rank/existing-hire-logits',
            'new-hires': 'gender/by-new-hire/by-rank/logits',
            'ax': axes[2],
            'ymin': -.1,
            'ymax': .1,
            'ystep': .05,
            'title': 'x = representation of women',
            'subfigure': 'C',
        },
    }

    plot_data = []

    for val_type, info in stats_to_dfs.items():
        all_df = usfhn.stats.runner.get(info['all'], rank_type='prestige')[
            [
                'TaxonomyLevel',
                'TaxonomyValue',
                'Percentile',
                'Percentile-P',
            ]
        ].drop_duplicates().rename(columns={
            'Percentile': 'OldSlope',
            'Percentile-P': 'Old-P',
        })

        new_df = usfhn.stats.runner.get(info['new-hires'], rank_type='prestige')[
            [
                'TaxonomyLevel',
                'TaxonomyValue',
                'Percentile',
                'Percentile-P',
            ]
        ].drop_duplicates().rename(columns={
            'Percentile': 'NewSlope',
            'Percentile-P': 'New-P',
        })

        df = all_df.merge(
            new_df,
            on=[
                'TaxonomyLevel',
                'TaxonomyValue',
            ],
        )

        df['NewSlope'] /= 10
        df['OldSlope'] /= 10

        df['NewSlope'] *= -1
        df['OldSlope'] *= -1

        df['NewNotSignificant'] = df['New-P'] > .05
        df['OldNotSignificant'] = df['Old-P'] > .05

        data_df = usfhn.views.filter_to_academia_and_domain(df)
        data_df = usfhn.plot_utils.annotate_color(data_df)

        data_df = data_df[
            [
                'TaxonomyLevel',
                'TaxonomyValue',
                'OldSlope',
                'NewSlope',
                'Old-P',
                'New-P',
                'Color',
            ]
        ].rename(columns={
            'OldSlope': 'ExistingFacultySlope',
            'Old-P': 'ExistingFaculty-P',
            'NewSlope': 'NewFacultySlope',
            'New-P': 'NewFaculty-P',
        })

        element_id = 0
        for i, row in data_df.iterrows():
            for col in data_df.columns:
                plot_data.append({
                    'Subfigure': info['subfigure'],
                    'Element': element_id,
                    'Attribute': col,
                    'Value': row[col],
                })

        start_is, end_is, lines, ulabels = usfhn.standard_plots.plot_connected_two_values_by_taxonomy_level(
            df,
            open_column='NewSlope',
            closed_column='OldSlope',
            open_label='existing faculty',
            closed_label='new faculty',
            closed_marker_col='NewNotSignificant',
            open_marker_col='OldNotSignificant',
            y_label='Prestige',
            y_min=info['ymin'],
            y_max=info['ymax'],
            y_step=info['ystep'],
            ax=info['ax'],
            add_gridlines=False,
            add_taxonomy_lines=False,
            skip_fields=True,
        )

        if val_type == 'self-hires':
            annotation_row = df[
                df['TaxonomyValue'] == 'Academia'
            ].iloc[0]

            start = start_is['Academia']
            end = end_is['Academia']
            second_line = lines[1]

        info['ax'].set_title(info['title'], fontsize=hnelib.plot.FONTSIZES['title'])

    arrow_pad = .01

    labels = ['new faculty', 'existing faculty']
    xs = [start, end]
    y_starts = [
        annotation_row['NewSlope'] - arrow_pad,
        annotation_row['OldSlope'] - (2 * arrow_pad),
    ]
    y_ends = [-.03, -.03]

    annotation_kwargs = {
        'ha': 'center',
        'va': 'top',
        'color': PLOT_VARS['colors']['dark_gray'],
        'rotation': 90,
        'annotation_clip': False,
        'fontsize': hnelib.plot.FONTSIZES['annotation'],
    }

    for label, x, y_start, y_end in zip(labels, xs, y_starts, y_ends):
        axes[0].annotate(
            "",
            (x, y_start),
            xytext=(x, y_end),
            arrowprops=hnelib.plot.BASIC_ARROW_PROPS,
        )

        axes[0].annotate(
            label,
            (x, y_end),
            **annotation_kwargs,
        )

    for val_type, info in stats_to_dfs.items():
        ax = info['ax']
        raw_ys = np.arange(info['ymin'], info['ymax'] * 1.01, info['ystep'])
        raw_ys = [round(y, 2) for y in raw_ys]

        if val_type == 'self-hires':
            ys = [y for y in raw_ys if y not in [0, -.1]]
            x1 = second_line
        else:
            x1 = ax.get_xlim()[0]
            ys = raw_ys


        for y in [0, -.1]:
            if y not in raw_ys:
                continue

            color = PLOT_VARS['colors']['dark_gray'] if y else 'black'
            alpha = .5 if y else 1

            ax.plot(
                [x1, ax.get_xlim()[1]],
                [y, y],
                lw=.5,
                alpha=alpha,
                zorder=1,
                color=color,
            )

        hnelib.plot.add_gridlines(ax, ys=ys)
        yticks = [round(y, 2) for y in ax.get_yticks()]
        ax.set_yticklabels(hnelib.plot.stringify_numbers_without_ugly_zeros(yticks))

    for i, ax in enumerate(axes):
        if i:
            ax.set_ylabel('')
        else:
            # ax.set_ylabel('Δ log odds of x for a one-decile\nincrease in prestige')
            ax.set_ylabel('change in log odds of x for a\none-decile increase in prestige')

    plot_utils.add_umbrella_legend(
        axes[-1],
        get_umbrella_legend_handles_kwargs={
            'style': 'scatter',
            'include_academia': True,
            'fade_markerfacecolors': False,
        },
        legend_kwargs={
            'loc': 'center left',
            'bbox_to_anchor': (1, .5),
            'bbox_transform': axes[-1].transAxes,
            'fontsize': hnelib.plot.FONTSIZES['annotation'],
        },
        extra_legend_handles=[
            Line2D(
                [0], [0],
                color='none',
                marker='X',
                markerfacecolor='black',
                markeredgecolor='white',
                markersize=5,
                label='P > .05',
            ),
        ],
    )

    hnelib.plot.finalize(
        axes,
        [-.025, -.025, -.025],
        plot_label_y_pad=1.2,
    )

    return pd.DataFrame(plot_data)


def plot_variable_logit(
    ax=None,
    variable='gender',
    rank_type='prestige',
    draw_legend=True,
    annotate_prestige_direction=True,
):
    variable_type_info = {
        'non-us': {
            'dataset': 'non-us/by-institution/by-rank/logits',
            'ylabel': '% non-U.S. doctorates',
            'ylim': [0, .255],
            'yticks': [0, .05, .1, .15, .2, .25],
            'yticklabels': [0, 5, 10, 15, 20, 25],
            'break_height': .02,
        },
        'non-us-new-hires': {
            'dataset': 'non-us/by-new-hire/by-rank/logits',
            'ylabel': '% non-U.S. doctorates among new faculty',
            'ylim': [0, .255],
            'yticks': [0, .05, .1, .15, .2, .25],
            'yticklabels': [0, 5, 10, 15, 20, 25],
            'break_height': .02,
        },
        'self-hires': {
            'dataset': 'self-hires/by-rank/logits',
            'ylabel': '% self-hires',
            'ylim': [0, .3],
            'yticks': [0, .05, .1, .15, .2, .25, .3],
            'yticklabels': [0, 5, 10, 15, 20, 25, 30],
            'break_height': .015,
        },
        'self-hires-new-hires': {
            'dataset': 'self-hires/by-new-hire/by-rank/logits',
            'ylabel': '% self-hires among new faculty',
            'ylim': [0, .3],
            'yticks': [0, .05, .1, .15, .2, .25, .3],
            'yticklabels': [0, 5, 10, 15, 20, 25, 30],
            'break_height': .02,
        },
        'self-hires-attrition': {
            'dataset': 'attrition/by-rank/institution/by-self-hire/logits',
            'ylabel': 'self-hire attrition risk',
            'ylim': [0, .1],
            'yticks': [0, .025, .05, .075, .1],
            'yticklabels': [0, .025, .05, .075, .1],
            'break_height': .01,
        },
        'gender': {
            'dataset': 'gender/by-rank/logits',
            'ylabel': '% women',
            'ylim': [0, .76],
            'yticks': [0, .25, .5, .75],
            'yticklabels': [0, 25, 50, 75],
            'break_height': .04,
        },
        'gender-new-hires': {
            'dataset': 'gender/by-new-hire/by-rank/logits',
            'ylabel': '% women among new faculty',
            'ylim': [0, .76],
            'yticks': [0, .25, .5, .75],
            'yticklabels': [0, 25, 50, 75],
            'break_height': .02,
        },
        'interdisciplinarity': {
            'dataset': 'interdisciplinarity/by-rank/logits',
            'ylabel': '% interdisciplinary faculty',
            'ylim': [0, .2],
            'yticks': [0, .05, .1, .15, .2],
            'yticklabels': [0, 5, 10, 15, 20],
            'break_height': .04,
        },
        'career-length': {
            'dataset': 'careers/length/by-rank/logits',
            'ylabel': 'mean career age (years since doctorate)',
            'ylim': [0, 1],
            'yticks': [0, .2, .4, .6, .8, 1],
            'yticklabels': [0, 10, 20, 30, 40, 50],
            'break_height': .04,
        },
    }

    info = variable_type_info[variable]

    print(rank_type)

    df = usfhn.stats.runner.get(info['dataset'], rank_type=rank_type)

    ax = plot_institution_rank_logits(df, rank_type=rank_type, ax=ax)

    ax.set_ylim(info['ylim'])
    ax.set_yticks(info['yticks'])
    ax.set_yticklabels(hnelib.plot.stringify_numbers_without_ugly_zeros(info['yticklabels']))

    ax.set_ylabel(info['ylabel'])

    if annotate_prestige_direction:
        usfhn.plot_utils.add_gridlines_and_annotate_rank_direction(
            ax,
            rank_type=rank_type,
            break_height=info['break_height'],
        )
    else:
        hnelib.plot.add_gridlines_on_ticks(ax)

    if draw_legend:
        plot_utils.add_umbrella_legend(
            ax,
            get_umbrella_legend_handles_kwargs={
                'style': 'line',
                'include_academia': True,
            },
            legend_kwargs={
                'loc': 'center left',
                'bbox_to_anchor': (1, .5),
                'bbox_transform': ax.transAxes,
            },
            extra_legend_handles=[
                Line2D(
                    [0], [0],
                    color='black',
                    ls='-',
                    label='P < .05',
                ),
                Line2D(
                    [0], [0],
                    color='black',
                    ls='--',
                    label='P >= .05',
                ),
            ],
        )


def plot_institution_rank_logits(df, rank_type='prestige', ax=None):
    if not ax:
        fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)

    df = usfhn.views.filter_to_academia_and_domain(df)
    
    Xs = pd.DataFrame({'Percentile': np.arange(0, 1.01, .01)})

    for (level, value), rows in df.groupby(['TaxonomyLevel', 'TaxonomyValue']):
        logit_model = rows.iloc[0]
        Ys = hnelib.model.get_logit_predictions(
            logit_model=logit_model,
            df_to_predict=Xs,
        )

        ls = '--' if logit_model['Percentile-P'] > .05 else '-'

        ax.plot(
            Xs['Percentile'],
            Ys,
            color=PLOT_VARS['colors']['umbrellas'][value],
            lw=2,
            ls=ls,
            zorder=3,
        )

    ax.set_xlim(0, 1.002)
    xticks = [0, .2, .4, .6, .8, 1]
    xticklabels = [0, 20, 40, 60, 80, 100]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    ymin, ymax = ax.get_ylim()

    if rank_type == 'production':
        xlabel = 'production rank'
    elif rank_type == 'prestige':
        xlabel = 'prestige rank'
    elif rank_type == 'employing-institution':
        xlabel = 'prestige rank'

    ax.set_xlabel(xlabel)

    return ax


def plot_taxonomy_interdisciplinarity_over_time():
    df = usfhn.stats.runner.get('interdisciplinarity/by-year/taxonomy')

    usfhn.standard_plots.plot_domain_level_values_over_time(
        df,
        'OutOfFieldFraction',
        'fraction of out of field hires',
        ylim=[0, .075],
        yticks=[0, .025, .05, .075],
    )


def plot_degree_years_domain_level():
    df = usfhn.stats.runner.get('demographics/degree-years')

    df = df[
        df['TaxonomyLevel'].isin(['Umbrella', 'Academia'])
    ]

    df = df[
        df['DegreeYear'] > 1960
    ]

    # print(sorted(list(df['DegreeYear'].unique())))
    # import sys; sys.exit()

    # df = df[
    #     df['DegreeYear'].notna
    # ]

    df['Faculty'] = df.groupby([
        'TaxonomyLevel',
        'TaxonomyValue',
        'DegreeYear',
    ])['PersonId'].transform('nunique')

    df['TotalFaculty'] = df.groupby([
        'TaxonomyLevel',
        'TaxonomyValue',
    ])['PersonId'].transform('nunique')

    df['FacultyFraction'] = df['Faculty'] / df['TotalFaculty']
    df['FacultyPercent'] = df['FacultyFraction'] * 100

    df['MeanDegreeYear'] = df.groupby([
        'TaxonomyLevel',
        'TaxonomyValue',
    ])['DegreeYear'].transform('mean')

    df = df[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'Faculty',
            'FacultyPercent',
            'MeanDegreeYear',
            'DegreeYear',
        ]
    ].drop_duplicates()

    umbrellas = usfhn.views.get_umbrellas()

    fig, axes = plt.subplots(2, 5, figsize=(14, 6), tight_layout=True)

    layout = plot_utils.get_academia_umbrella_grid()

    bins = np.arange(0, .6, .01)

    visible_axes = []

    for i, layout_row in enumerate(layout):
        for j, taxonomy_value in enumerate(layout_row):
            ax = axes[i, j]

            if not taxonomy_value:
                hnelib.plot.hide_axis(ax)
                continue

            visible_axes.append(ax)

            _df = df[
                df['TaxonomyValue'] == taxonomy_value
            ]

            color = PLOT_VARS['colors']['umbrellas'][taxonomy_value]

            ax.bar(
                _df['DegreeYear'],
                _df['FacultyPercent'],
                facecolor=hnelib.plot.set_alpha_on_colors(color),
                edgecolor=color,
            )

            mean = round(_df.iloc[0]['MeanDegreeYear'])

            mean_df = _df[
                _df['DegreeYear'] == mean
            ]

            ax.bar(
                mean_df['DegreeYear'],
                mean_df['FacultyPercent'],
                facecolor=PLOT_VARS['colors']['dark_gray'],
                edgecolor=color,
            )

            mean_row = mean_df.iloc[0]
            if taxonomy_value == 'Academia':
                ax.annotate(
                    "mean",
                    xy=(mean_row['DegreeYear'], mean_row['FacultyPercent']),
                    xytext=(mean_row['DegreeYear'], mean_row['FacultyPercent'] * 1.25),
                    ha='right',
                    va='bottom',
                    arrowprops=PLOT_VARS['arrowprops'],
                )

            ax.set_title(
                usfhn.plot_utils.clean_taxonomy_string(taxonomy_value),
                color=color,
            )

            ax.set_xlabel('degree year')
            ax.set_ylabel('% of faculty')

    hnelib.plot.set_lims_to_max(visible_axes, x=False)
    for ax in visible_axes:
        ax.set_yticks([0, 1, 2, 3])
        ax.set_xticks([1960, 1980, 2000, 2020])


def plot_new_hires_by_gender_over_time():
    df = usfhn.stats.runner.get('gender/by-year/new-hire/df')
    df = df[
        df['Gender'] == 'Female'
    ]

    usfhn.standard_plots.plot_domain_level_values_over_time(
        df,
        column='GenderFraction',
        ylabel='fraction female',
        ylim=[0, 1],
        # yticks=[.25],
    )


def plot_new_hires_by_gini_over_time():
    df = usfhn.stats.runner.get('ginis/by-year/new-hire')

    usfhn.standard_plots.plot_domain_level_values_over_time(
        df,
        column='GiniCoefficient',
        ylabel='Gini coefficient',
        ylim=[0, 1],
        # yticks=[.25],
    )


def plot_new_hires_by_self_hire_over_time():
    df = usfhn.stats.runner.get('new-hire/by-year/self-hires')

    df = df[
        df['SelfHire']
    ]

    usfhn.standard_plots.plot_domain_level_values_over_time(
        df,
        column='HireTypeFraction',
        ylabel='fraction self hires',
        ylim=[0, 1],
        # yticks=[.25],
    )


def plot_new_hires_by_non_us_over_time():
    df = usfhn.stats.runner.get('new-hire/by-year/non-us')

    usfhn.standard_plots.plot_domain_level_values_over_time(
        df,
        column='NonUSFraction',
        ylabel='fraction non-U.S. PhDs',
        ylim=[0, .25],
    )



def plot_new_hires_by_interdisciplinarity_over_time():
    df = usfhn.stats.runner.get('new-hire/by-year/interdisciplinarity')

    usfhn.standard_plots.plot_domain_level_values_over_time(
        df,
        column='OutOfFieldFraction',
        ylabel='fraction out of field',
        ylim=[0, .15],
    )


def plot_new_hires_by_steepness_over_time():
    df = usfhn.stats.runner.get('new-hire/by-year/steepness')

    df = df[
        df['MovementType'] == 'Downward'
    ]

    usfhn.standard_plots.plot_domain_level_values_over_time(
        df,
        column='MovementFraction',
        ylabel='steepness',
        ylim=[0, 1],
    )

def plot_rank_change_over_time_by_academia_and_domain(rank_type='prestige'):
    df = usfhn.stats.runner.get('ranks/by-year/hierarchy-stats', rank_type=rank_type)

    df = df[
        df['MovementType'] == 'Downward'
    ]

    usfhn.standard_plots.plot_domain_level_values_over_time(
        df,
        column='MovementFraction',
        ylabel='steepness',
        ylim=[0, 1],
    )

def plot_multiple_geodesics():
    df = usfhn.stats.runner.get('geodesics/df')

    df = usfhn.views.filter_to_academia_and_domain(df)
    df = usfhn.plot_utils.annotate_color(df)

    fig, ax = plt.subplots(figsize=(hnelib.plot.WIDTHS['1-col'], 2.5), tight_layout=True)

    dfs = []
    for level, rows in df.groupby('TaxonomyLevel'):
        rows = rows.copy()

        for value, _rows in rows.groupby('TaxonomyValue'):
            _df = plot_basic_geodesics(
                ax,
                df=_rows,
                level=level,
                value=value,
                color=_rows.iloc[0]['Color'],
                annotate_rank_direction=False,
            )
            dfs.append(_df)

    usfhn.plot_utils.add_gridlines_and_annotate_rank_direction(
        ax,
        rank_type='prestige',
        break_height=.075,
    )

    ax.set_xticklabels([0, 20, 40, 60, 80, 100])

    legend = plot_utils.add_umbrella_legend(
        ax,
        get_umbrella_legend_handles_kwargs={
            'style': 'none',
            'include_academia': True,
        },
        legend_kwargs={
            'fontsize': hnelib.plot.FONTSIZES['legend'],
            'loc': 'center left',
            'bbox_to_anchor': (.9, .5),
            'bbox_transform': ax.transAxes,
        },
    )

    hnelib.plot.finalize(ax)

    data_df = pd.concat(dfs)
    data_df = usfhn.plot_utils.annotate_color(data_df)
    data_df['Line'] = data_df['TaxonomyValue']

    plot_data = []
    element_id = 0
    for i, row in data_df.iterrows():
        for col in data_df.columns:
            plot_data.append({
                'Element': element_id,
                'Attribute': col,
                'Value': row[col],
            })

        element_id += 1

    return pd.DataFrame(plot_data)

def plot_self_hires_binned_prestige():
    df = usfhn.stats.runner.get('attrition/by-self-hires/prestige-deciles', rank_type='employing-institution')

    df = usfhn.views.filter_to_academia_and_domain(df)

    df = usfhn.views.annotate_umbrella_color(df, 'Umbrella')

    df = df[
        df['Decile'] < 10
    ]

    fig, ax = plt.subplots(1, figsize=(8.5, 4), tight_layout=True)
    for value, rows in df.groupby('TaxonomyValue'):
        rows = rows.sort_values(by='Decile')
        ax.plot(
            rows['Decile'],
            rows['Ratio'],
            color=rows.iloc[0]['UmbrellaColor'],
        )

    ax.set_xlabel('prestige decile')
    ax.set_ylabel(r"$\frac{\mathrm{self-hire\ attrition\ risk}}{\mathrm{non-self-hire\ attrition\ risk}}$".replace("-", u"\u2010"))

    plot_utils.add_umbrella_legend(
        ax,
        get_umbrella_legend_handles_kwargs={
            'style': 'scatter',
            'include_academia': True,
            'fade_markerfacecolors': False,
        },
        legend_kwargs={
            'loc': 'center left',
            'bbox_to_anchor': (1, .5),
            'bbox_transform': ax.transAxes,
        },
    )


def plot_basic_geodesics(
    ax=None,
    df=pd.DataFrame(),
    level='Academia',
    value='Academia',
    color=PLOT_VARS['colors']['umbrellas']['Academia'],
    annotate_rank_direction=True,
):
    if not ax:
        fig, ax = plt.subplots(figsize=(6, 6), tight_layout=True)

    if df.empty:
        df = usfhn.stats.runner.get('geodesics/df')
    else:
        df = df.copy()

    df = usfhn.views.filter_by_taxonomy(df, level, value)

    rolling_rows = []
    # window = .025
    window = .05
    for i in np.arange(0, 1.01, .01):
        rows = df[
            (df['Percentile'] < i + window)
            &
            (df['Percentile'] > i - window)
        ].copy()

        rolling_rows.append({
            'Percentile': i,
            'MeanFractionalPathLength': rows['FractionalPathLength'].mean(),
            'TaxonomyLevel': level,
            'TaxonomyValue': value,
        })

    df = pd.DataFrame(rolling_rows)

    df = df.sort_values(by='Percentile')

    ax.plot(
        df['Percentile'],
        df['MeanFractionalPathLength'],
        lw=1,
        color=color,
        zorder=2,
    )

    ticks = [0, .2, .4, .6, .8, 1]
    lim = [0, 1]

    ax.set_xlim(lim)
    ax.set_xticks(ticks)
    ax.set_xticklabels(hnelib.plot.stringify_numbers_without_ugly_zeros(ticks))

    ax.set_ylim(lim)
    ax.set_yticks(ticks)
    ax.set_yticklabels(hnelib.plot.stringify_numbers_without_ugly_zeros(ticks))

    ax.set_xlabel('university prestige')
    ax.set_ylabel(r"$\frac{\mathrm{mean\ geodesic\ distance}}{\mathrm{diameter}}$")

    if annotate_rank_direction:
        usfhn.plot_utils.add_gridlines_and_annotate_rank_direction(
            ax,
            rank_type='prestige',
            break_height=.075,
            x_gridlines_to_break=[.2, .4],
        )

    hnelib.plot.finalize(ax)

    return df


def plot_risk_vs_ranks_logit_slopes_with_p_values(rank_type='production'):
    df = usfhn.stats.runner.get('attritions/by-rank/institution/logits', rank_type=rank_type)

    df['NoFaceColor'] = df['B1-P'] > .05

    fig, axes, legend_ax = plot_univariate_value_across_taxonomy_levels(
        df,
        'B1',
        f'{rank_type} prestige to attrition risk logit slope',
        y_min=0,
        y_max=1,
        y_step=.2,
        extra_legend_handles=[
            Line2D(
                [], [],
                color='none',
                marker='o',
                markerfacecolor=hnelib.plot.set_alpha_on_colors('black'),
                markeredgecolor='black',
                label='P < .05',
            ),
            Line2D(
                [], [],
                color='none',
                marker='o',
                markerfacecolor='w',
                markeredgecolor='black',
                label='P >= .05',
            ),
        ],
    )

def plot_up_vs_down_attrition_risk_ratio_by_time(rank_type='prestige'):
    df = usfhn.stats.runner.get('attrition/risk/by-time/rank-change', rank_type=rank_type)

    groupby_cols = [
        'TaxonomyLevel',
        'TaxonomyValue',
        'Year',
    ]

    up_df = df.copy()[
        df['MovementType'] == 'Upward'
    ][
        groupby_cols + ['AttritionRisk']
    ].drop_duplicates().rename(columns={
        'AttritionRisk': 'UpwardAttritionRisk'
    })

    down_df = df.copy()[
        df['MovementType'] == 'Downward'
    ][
        groupby_cols + ['AttritionRisk']
    ].drop_duplicates().rename(columns={
        'AttritionRisk': 'DownwardAttritionRisk'
    })

    df = up_df.merge(
        down_df,
        on=groupby_cols,
    )

    df['RiskRatio'] = df['UpwardAttritionRisk'] / df['DownwardAttritionRisk']

    axes = usfhn.standard_plots.plot_domain_level_values_over_time(
        df,
        column='RiskRatio',
        ylabel=f'up-hierarchy hire risk of attrition / down-hierarchy hire risk of attrition',
        ylim=[.25, 1.75],
        yticks=[.25, .5, .75, 1, 1.25, 1.5, 1.75],
    )

    axes[0].axhline(
        1,
        color='black',
        zorder=2,
        lw=1,
    )


def plot_up_vs_down_attrition_by_time(rank_type='prestige'):
    df = usfhn.stats.runner.get('attrition/risk/by-time/rank-change', rank_type=rank_type)

    groupby_cols = [
        'TaxonomyLevel',
        'TaxonomyValue',
        'Year',
    ]

    up_df = df.copy()[
        df['MovementType'] == 'Upward'
    ][
        groupby_cols + ['AttritionRisk']
    ].drop_duplicates()

    down_df = df.copy()[
        df['MovementType'] == 'Downward'
    ][
        groupby_cols + ['AttritionRisk']
    ].drop_duplicates()

    usfhn.standard_plots.compare_two_values_at_the_domain_level_over_time(
        up_df,
        down_df,
        column='AttritionRisk',
        ylabel='attrition risk',
        title1='Up-hierarchy hire risk of attrition',
        title2='down-hierarchy hire risk of attrition',
        ylim=[0, .12],
        yticks=[0, .03, .06, .09, .12],
    )


def plot_non_us_multiplot():
    fig, axes = plt.subplots(
        1, 3, figsize=(hnelib.plot.WIDTHS['2-col'], 35/12),
        tight_layout=True,
        gridspec_kw={
            'width_ratios': [1, 1, .65],
        }
    )

    panel_a_df = plot_degree_breakdown_no_deg_us_deg_non_us_deg(parent_ax=axes[0])
    panel_a_df['Subfigure'] = 'A'

    panel_b_df = plot_non_us_by_continent_bars_grid(axes[1])
    panel_b_df['Subfigure'] = 'B'

    sub_axes, panel_c_df = plot_english_vs_non_english_non_us_attrition_risk_ratio(axes[2])

    hnelib.plot.finalize(
        axes,
        [-.05, -.33, -.4],
        plot_label_y_pad=1.09,
    )

    hnelib.plot.annotate_plot_letters(
        [sub_axes[1]],
        x_pads=[-.4],
        y_pad=1.2,
        labels=['D'],
    )

    hnelib.plot.set_ticklabel_fontsize(sub_axes, hnelib.plot.FONTSIZES['annotation'])
    hnelib.plot.set_label_fontsize(sub_axes, hnelib.plot.FONTSIZES['large'])

    return pd.concat([panel_a_df, panel_b_df, panel_c_df])


def plot_non_us_by_continent_bars_grid(parent_ax):
    df = usfhn.stats.runner.get('non-us/by-continent')
    df = usfhn.views.filter_by_taxonomy(df, 'Umbrella')

    continents = reversed(sorted(list(df['Continent'].unique())))
    continent_ordering = {c: i for i, c in enumerate(continents)}

    df['Ordering'] = df['Continent'].apply(continent_ordering.get)

    countries = usfhn.stats.runner.get('non-us/by-country')[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'CountryName',
            'CountryNonUSFraction',
        ]
    ].drop_duplicates()

    countries = usfhn.institutions.annotate_country_continent(countries)

    countries_to_include = ['United Kingdom', 'Canada']

    countries = countries[
        countries['CountryName'].isin(countries_to_include)
    ]

    countries['Ordering'] = countries['Continent'].apply(continent_ordering.get)

    umbrella_to_ax_label = {}
    for i, umbrella in enumerate(usfhn.views.get_umbrellas()):
        umbrella_to_ax_label[umbrella] = list(string.ascii_uppercase)[i]

    ax_label_to_umbrella = {
        'A': 'Social Sciences',
        'B': 'Mathematics and Computing',
        'C': 'Medicine and Health',
        'D': 'Humanities',
        'E': 'Applied Sciences',
        'F': 'Education',
        'G': 'Natural Sciences',
        'H': 'Engineering',
    }

    umbrella_to_ax_label = {u: l for l, u in ax_label_to_umbrella.items()}

    h_space_row = .1
    w_space_col = .1

    n_rows = 4
    n_cols = 2

    plot_h = (1 - (h_space_row * (n_rows - 1))) / n_rows
    plot_w = (1 - (w_space_col * (n_cols - 1))) / n_cols

    layout = [
        ['G', 'H'],
        ['E', 'F'],
        ['C', 'D'],
        ['A', 'B'],
    ]
    axes = {}

    for row, col in itertools.product(range(n_rows), range(n_cols)):
        label = layout[row][col]

        x_start = (plot_w + w_space_col) * col
        y_start =  (plot_h + h_space_row) * row

        axes[label] = parent_ax.inset_axes((x_start, y_start, plot_w, plot_h))

    # hnelib.plot.hide_axis(parent_ax)
    parent_ax.spines['left'].set_visible(False)
    parent_ax.spines['bottom'].set_visible(False)
    parent_ax.spines['top'].set_visible(False)
    parent_ax.spines['right'].set_visible(False)
    parent_ax.set_xticks([0])
    parent_ax.tick_params(axis='x', color='w')
    parent_ax.set_yticks([])
    parent_ax.set_xlabel('% of non-U.S. doctorates')
    hnelib.plot.finalize([parent_ax])

    df['AxLabel'] = df['TaxonomyValue'].apply(lambda u: umbrella_to_ax_label[u])

    df = usfhn.views.annotate_umbrella_color(df, taxonomization_level='Umbrella')
    df['FadedColor'] = df['UmbrellaColor'].apply(hnelib.plot.set_alpha_on_colors)

    for umbrella, rows in df.groupby('TaxonomyValue'):
        rows = rows.sort_values(by='Ordering')
        row = rows.iloc[0]

        ax = axes[row['AxLabel']]

        ax.barh(
            rows['Ordering'],
            rows['ContinentFraction'],
            facecolor='w',
            edgecolor=row['UmbrellaColor'],
            lw=.5,
            zorder=2,
        )

        ax.barh(
            rows['Ordering'],
            rows['ContinentFraction'],
            facecolor=row['FadedColor'],
            edgecolor=row['UmbrellaColor'],
            lw=.5,
            zorder=3,
        )

        _countries = countries[
            (countries['TaxonomyLevel'] == 'Umbrella')
            &
            (countries['TaxonomyValue'] == umbrella)
        ]

        ax.barh(
            _countries['Ordering'],
            _countries['CountryNonUSFraction'],
            color=row['UmbrellaColor'],
            lw=.5,
            zorder=3,
        )

        # if umbrella == 'Social Sciences':
        #     umbrella_text = 'Social\nSciences'
        # elif umbrella == 'Mathematics and Computing':
        #     umbrella_text = 'Mathematics\n& Computing'
        # else:
        umbrella_text = usfhn.plot_utils.clean_taxonomy_string(umbrella)

        ax.set_title(umbrella_text, color=row['UmbrellaColor'], size=hnelib.plot.FONTSIZES['medium'], pad=3)

    data_df = df.copy()[
        df['TaxonomyLevel'] == 'Umbrella'
    ][
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'Continent',
            'ContinentFraction',
        ]
    ].drop_duplicates()
    data_df = usfhn.plot_utils.annotate_color(data_df)
    countries_data_df = countries.copy()[
        countries['TaxonomyLevel'] == 'Umbrella'
    ]

    element_id = 0
    plot_data = []
    for (level, value), rows in data_df.groupby(['TaxonomyLevel', 'TaxonomyValue']):
        rows = rows.copy()

        for continent, _rows in rows.groupby('Continent'):
            rows[f'{continent} Fraction'] = _rows.iloc[0]['ContinentFraction']

        rows = rows.drop(columns=['Continent', 'ContinentFraction'])

        _countries = countries[
            (countries['TaxonomyLevel'] == level)
            &
            (countries['TaxonomyValue'] == value)
        ]

        for country, _rows in _countries.groupby('CountryName'):
            rows[f'{country} Fraction'] = _rows.iloc[0]['CountryNonUSFraction']

        for col in rows.columns:
            plot_data.append({
                'Element': element_id,
                'Attribute': col,
                'Value': rows.iloc[0][col],
            })

        element_id += 1

    axes_with_ylabels = ['A', 'C', 'E', 'G']
    axes_with_xlabels = ['G', 'H']
    for ax_label in umbrella_to_ax_label.values():
        ax = axes[ax_label]

        yticks = list(continent_ordering.values())
        ax.set_yticks(yticks)
        if ax_label in axes_with_ylabels:
            ax.set_yticklabels(["" for c in continent_ordering])

            for continent in continent_ordering:
                ax.annotate(
                    continent,
                    xy=(-.05, continent_ordering[continent]),
                    ha='right',
                    va='center',
                    annotation_clip=False,
                    fontsize=hnelib.plot.FONTSIZES['annotation'],
                )

        else:
            ax.set_yticklabels(['' for _ in yticks])

        ymin, ymax = ax.get_ylim()
        ax.set_xlim(0, .755)
        xticks = [0, .25, .5, .75]
        xticklabels = [0, 25, 50, 75]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)

        if ax_label != 'A':
            hnelib.plot.add_gridlines_on_ticks(ax, y=False)
        else:
            line_kwargs = {
                'color': PLOT_VARS['colors']['dark_gray'],
                'alpha': .5,
                'lw': .5,
                'zorder': 1,
            }
            ax.plot(
                [.5, .5],
                [ymin, 3],
                **line_kwargs
            )

            ax.plot(
                [.75, .75],
                [ymin, ymax],
                **line_kwargs
            )

        if not ax_label in axes_with_xlabels:
            ax.set_xticks([])

        ax.set_ylim(ymin, ymax)

    ax = axes['A']

    x_start = .15
    x_end = .2

    y_pad = .25
    y_end_pad = 1.5
    text_x_pad = .02

    vertical_alignments = ['bottom', 'top']
    y_starts = [continent_ordering['Europe'], continent_ordering['North America']]

    labels = ['United Kingdom', 'Canada']

    for label, y_start, va in zip(labels, y_starts, vertical_alignments):
        multiplier = -1 if va == 'top' else 1

        y_start += multiplier * y_pad
        y_end = y_start + (multiplier * y_end_pad)

        ax.annotate(
            "",
            (x_start, y_start), 
            xytext=(x_start, y_end),
            arrowprops=hnelib.plot.ZERO_SHRINK_A_ARROW_PROPS,
        )

        ax.annotate(
            "",
            (x_start, y_end), 
            xytext=(x_end, y_end),
            arrowprops=hnelib.plot.HEADLESS_ARROW_PROPS,
        )

        ax.annotate(
            label,
            (x_end + text_x_pad, y_end), 
            ha='left',
            va='center',
            color=PLOT_VARS['colors']['dark_gray'],
            fontsize=hnelib.plot.FONTSIZES['annotation'],
        )

    hnelib.plot.finalize(list(axes.values()))

    return pd.DataFrame(plot_data)


def plot_english_vs_non_english_non_us_attrition_risk_ratio(ax=None, axes=None):
    if ax:
        hnelib.plot.hide_axis(ax)
        vspace = .15
        plot_h = (1 - vspace) / 2

        axes = [
            ax.inset_axes((0, 1 - plot_h, 1, plot_h)),
            ax.inset_axes((0, 0, 1, plot_h)),
        ]
    elif not axes:
        fix, axes = plt.subplots(2, 1, figsize=(5, 8), tight_layout=True)

    plot_kwargs = {
        'value_column': 'Ratio',
        'y_min': np.log(.25),
        'y_max': np.log(1.75),
        'y_step': np.log(.25),
        'show_x_ticks': False,
        'x_for_non_significant': True,
        'legend_ax': None,
    }

    df = usfhn.stats.runner.get('attrition/risk/non-us-by-is-english')

    labels = [
        r"$\frac{\mathrm{Canada\ &\ U.K.}}{\mathrm{U.S.}}$",
        r"$\frac{\mathrm{non-U.S.\ (excluding\ Canada\ &\ U.K.)}}{\mathrm{U.S.}}$",
    ]

    subfigures = ['C', 'D']

    dfs = [
        df[df['IsHighlyProductiveNonUSCountry'] | df['US']],
        df[~df['IsHighlyProductiveNonUSCountry'] | df['US']],
    ]

    ytick_pad_top = .05

    yticks = [np.log2(x) for x in [3/4, 1, 5/4, 6/4, 7/4, 2]]
    yticklabels = ['3:4', '1:1', '5:4', '6:4', '7:4', '2:1']
    ylines = yticks

    y_min = -.5
    y_max = yticks[-1] + ytick_pad_top

    umbrellas = ['Academia'] + usfhn.views.get_umbrellas()

    y_pad = .035
    y_len = .15

    x_len = .45
    x_pad = .05
    x_pad_text = .075

    plot_data = []
    annotations = []
    for i, (ax, label, _df, subfigure) in enumerate(zip(axes, labels, dfs, subfigures)):
        _df = hnelib.pandas.aggregate_df_over_column(
            _df,
            agg_col='US',
            join_cols=[
                'TaxonomyLevel',
                'TaxonomyValue',
            ],
            value_cols=[
                'AttritionRisk',
                'AttritionEvents',
                'Events',
            ],
            agg_value_to_label={
                True: 'US',
                False: 'NonUS',
            }
        )

        _df['Ratio'] = _df['NonUSAttritionRisk'] / _df['USAttritionRisk']
        _df['Ratio'] = _df['Ratio'].apply(np.log2)

        _df = _df[
            (_df['Ratio'] >= y_min)
            &
            (_df['Ratio'] <= y_max)
        ]

        _df = _df.merge(
            usfhn.attrition.compute_attrition_risk_significance(
                _df,
                'NonUSAttritionEvents',
                'NonUSEvents',
                'USAttritionEvents',
                'USEvents',
            ),
            on=[
                'TaxonomyLevel',
                'TaxonomyValue',
            ]
        )

        data_df = _df.copy()[
            [
                'TaxonomyLevel',
                'TaxonomyValue',
                'Ratio',
                'PCorrected',
            ]
        ].rename(columns={
            'Ratio': 'Risk Ratio',
            'PCorrected': 'P',
        })

        j = 0
        for _, row in data_df.iterrows():
            for col in data_df.columns:
                plot_data.append({
                    'Subfigure': subfigure,
                    'Element': j,
                    'Attribute': col,
                    'Value': row[col],
                })

            j += 1

        usfhn.standard_plots.plot_univariate_value_across_taxonomy_levels_single_plot(
            _df,
            **plot_kwargs,
            y_label='relative annual attrition\nby doctoral origin',
            ax=ax,
        )

        ax.set_title(
            label.replace("-", u"\u2010"),
            fontsize=hnelib.plot.FONTSIZES['axis'],
            pad=3,
        )

        ax.set_ylim(y_min, y_max)

        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        ax.spines['bottom'].set_visible(False)

        hnelib.plot.add_gridlines(ax, ys=ylines)

        ax.axhline(
            np.log2(1),
            color='black',
            lw=.75,
            zorder=1,
        )

        if not i:
            fields_df = usfhn.views.filter_by_taxonomy(_df, 'Field')
            fields_df = usfhn.views.annotate_umbrella(fields_df, 'Field')

            sig_row = _df[
                _df['TaxonomyValue'] == 'Academia'
            ].iloc[0]

            y_end = np.mean([np.log2(5/4), np.log2(6/4)])

            annotations.append({
                'label': 'significant',
                'y': sig_row['Ratio'],
                'x': umbrellas.index('Academia'),
                'y-len': y_end - sig_row['Ratio'] - y_pad,
                'x-len': .1,
            })

            x_start = ax.get_xlim()[0] + .15
            x_end = umbrellas.index('Academia') + .5 - .15
            for y in [np.log2(y) for y in [1, 5/4]]:
                ax.plot(
                    [x_start, x_end],
                    [y, y],
                    color='w',
                    zorder=2,
                    lw=2,
                )

            x_end = umbrellas.index('Engineering') - .1

            for y in [np.log2(x) for x in [6/4, 5/4]]:
                ax.plot(
                    [x_start, x_end],
                    [y, y],
                    color='w',
                    zorder=1,
                    lw=2,
                )

            insig_df = fields_df[
                fields_df['Umbrella'] == 'Applied Sciences'
            ]

            insig_row = insig_df[
                insig_df['Ratio'] == max(insig_df['Ratio'])
            ].iloc[0]

            y_end = np.mean([np.log2(7/4), np.log2(2)])

            x_start = umbrellas.index(insig_row['Umbrella'])
            y = insig_row['Ratio']

            annotations.append({
                'label': 'not significant',
                'y': y,
                'x': x_start,
                'y-len': y_end - insig_row['Ratio'] - 2 * y_pad,
                'y-pad': .02,
                'x-len': .5,
            })

            x_end = umbrellas.index('Mathematics and Computing')

            ax.plot(
                [x_start - .5, x_end + .25],
                [np.log2(7/4), np.log2(7/4)],
                color='w',
                zorder=2,
                lw=2,
            )


    for annotation in annotations:
        y_multiplier = annotation.get('y-multiplier', 1)
        y_start = annotation['y'] + (y_multiplier * annotation.get('y-pad', y_pad))
        y_end = y_start + (y_multiplier * annotation['y-len'])

        x_start = annotation['x']
        x_end = x_start + annotation.get('x-len', x_len)
        x_text = x_end + x_pad_text

        axes[0].annotate(
            '',
            xy=(x_start, y_start),
            xytext=(x_start, y_end),
            arrowprops=hnelib.plot.ZERO_SHRINK_A_ARROW_PROPS,
            annotation_clip=False,
        )

        axes[0].annotate(
            '',
            xy=(x_start, y_end),
            xytext=(x_end, y_end),
            arrowprops=hnelib.plot.HEADLESS_ARROW_PROPS,
            annotation_clip=False,
        )

        axes[0].annotate(
            annotation['label'],
            xy=(x_text, y_end),
            va=annotation.get('va', 'center'),
            ha=annotation.get('ha', 'left'),
            annotation_clip=False,
            color=PLOT_VARS['colors']['dark_gray'],
            fontsize=hnelib.plot.FONTSIZES['annotation'],
        )


    hnelib.plot.finalize(axes)

    return axes, pd.DataFrame(plot_data)


def plot_non_us_vs_us_attrition_risk_overall(ax=None):
    df = usfhn.stats.runner.get('attrition/risk/us')[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'US',
            'AttritionRisk',
        ]
    ].drop_duplicates()

    df = hnelib.pandas.aggregate_df_over_column(
        df,
        agg_col='US',
        join_cols=[
            'TaxonomyLevel',
            'TaxonomyValue',
        ],
        value_cols=['AttritionRisk'],
        agg_value_to_label={
            True: 'US',
            False: 'NonUS',
        }
    )

    df['Ratio'] = df['NonUSAttritionRisk'] / df['USAttritionRisk']

    fig, ax, legend_ax = usfhn.standard_plots.plot_univariate_value_across_taxonomy_levels_single_plot(
        df,
        value_column='Ratio',
        y_min=.75,
        y_max=1.75,
        y_step=.25,
        y_label=r"$\frac{\mathrm{non-U.S.\ attrition\ risk}}{\mathrm{U.S.\ attrition\ risk}}$".replace("-", u"\u2010"),
        show_x_ticks=False,
        ax=ax,
    )

    ax.axhline(
        1,
        color='black',
        lw=1.5,
        zorder=1,
    )

    hnelib.plot.finalize(ax)


def plot_non_us_vs_us_attrition_risk_ratio_by_time():
    df = usfhn.stats.runner.get('attrition/risk/by-time/us')

    groupby_cols = [
        'TaxonomyLevel',
        'TaxonomyValue',
        'Year',
    ]

    us_df = df.copy()[
        df['US'] == True
    ][
        groupby_cols + ['AttritionRisk']
    ].drop_duplicates().rename(columns={
        'AttritionRisk': 'USAttritionRisk'
    })

    non_us_df = df.copy()[
        df['US'] == False
    ][
        groupby_cols + ['AttritionRisk']
    ].drop_duplicates().rename(columns={
        'AttritionRisk': 'NonUSAttritionRisk',
    })

    df = us_df.merge(
        non_us_df,
        on=groupby_cols,
    )

    df['RiskRatio'] = df['NonUSAttritionRisk'] / df['USAttritionRisk']

    axes = usfhn.standard_plots.plot_domain_level_values_over_time(
        df,
        column='RiskRatio',
        ylabel=f'non-U.S. PhD risk of attrition / U.S. PhD risk of attrition',
        ylim=[.25, 1.75],
        yticks=[.25, .5, .75, 1, 1.25, 1.5, 1.75],
    )

    axes[0].axhline(
        1,
        color='black',
        zorder=2,
        lw=1,
    )

def plot_attrition_risk_for_one_taxonomy_by_degree_year(taxonomy_level='Academia', taxonomy_value='Academia'):
    df = usfhn.stats.runner.get('attrition/risk/by-time/by-degree-year/taxonomy')

    df = usfhn.views.filter_by_taxonomy(df, level=taxonomy_level, value=taxonomy_value)

    df['DegreeYearDiff'] = df['Year'] - df['DegreeYear']

    fig, ax = plt.subplots(figsize=(10, 6), tight_layout=True)

    df = df[
        df['DegreeYear'] >= 1960
    ]

    # ax.set_prop_cycle('color', palettable.cartocolors.sequential.agSunset_7.mpl_colors)
    ax.set_prop_cycle('color', palettable.scientific.sequential.Bamako_10.mpl_colors)

    for year, rows in df.groupby('Year'):
        rows = rows.sort_values(by='DegreeYearDiff')
        ax.plot(
            rows['DegreeYearDiff'],
            rows['AttritionRisk'],
            lw=.5,
            label=year,
        )

    ax.set_xlabel('degree year - year')
    ax.set_ylabel('attrition risk')

    ax.set_xlim(0, 60)
    ax.set_ylim(0, .35)

    hnelib.plot.add_gridlines_on_ticks(ax)

    ax.legend()

def plot_attrition_risk_for_one_taxonomy_by_degree_year_and_by_gender(taxonomy_level='Academia', taxonomy_value='Academia'):
    df = usfhn.stats.runner.get('attrition/risk/by-time/by-degree-year/gender')

    df = usfhn.views.filter_by_taxonomy(df, level=taxonomy_level, value=taxonomy_value)

    df['DegreeYearDiff'] = df['Year'] - df['DegreeYear']

    fig, ax = plt.subplots(figsize=(10, 6), tight_layout=True)

    df = df[
        df['DegreeYear'] >= 1960
    ]

    df = df[
        df['Gender'].isin(['Male', 'Female'])
    ]

    ax.set_prop_cycle('color', palettable.scientific.sequential.Bamako_10.mpl_colors)

    for (year, gender), rows in df.groupby(['Year', 'Gender']):
        rows = rows.sort_values(by='DegreeYearDiff')
        ax.plot(
            rows['DegreeYearDiff'],
            rows['AttritionRisk'],
            lw=.5,
            label=year,
            ls='-' if gender == 'Female' else '--',
        )

    ax.set_xlabel('degree year - year')
    ax.set_ylabel('attrition risk')

    ax.set_xlim(0, 60)
    ax.set_ylim(0, .35)

    hnelib.plot.add_gridlines_on_ticks(ax)

    ax.legend(handles=[
        Line2D(
            [0], [0],
            color='black',
            ls='-',
            label='women',
        ),
        Line2D(
            [0], [0],
            color='black',
            ls='--',
            label='men',
        ),
    ])

def plot_gini_by_taxonomy_and_career_length():
    df = usfhn.stats.runner.get('careers/length/gini')

    df = df[
        df['TaxonomyLevel'].isin(['Academia', 'Umbrella'])
    ]

    max_year = 30

    df = df[
        df['CareerYear'] <= max_year
    ]

    df = usfhn.views.annotate_umbrella_color(df, taxonomization_level='Umbrella')

    fig, axes = plt.subplots(2, 5, figsize=(14, 6), tight_layout=True)

    umbrellas = usfhn.views.get_umbrellas()

    layout = [
        ['Academia'] + umbrellas[:4],
        [None] + umbrellas[4:],
    ]

    visible_axes = []

    for i, layout_row in enumerate(layout):
        for j, taxonomy_value in enumerate(layout_row):
            ax = axes[i, j]

            if not taxonomy_value:
                hnelib.plot.hide_axis(ax)
                continue

            visible_axes.append(ax)

            rows = df[
                df['TaxonomyValue'] == taxonomy_value
            ]

            rows = rows.sort_values(by='CareerYear')

            # hnelib.plot.plot_connected_scatter(
            #     ax,
            #     rows,
            #     x_column='CareerYear',
            #     y_column='GiniCoefficient',
            #     color=rows.iloc[0]['UmbrellaColor'],
            # )

            ax.plot(
                rows['CareerYear'],
                rows['GiniCoefficient'],
                color=rows.iloc[0]['UmbrellaColor'],
            )

            ax.set_title(taxonomy_value, color=rows.iloc[0]['UmbrellaColor'])

            ax.set_xlabel('years since PhD')
            ax.set_ylabel('Gini coefficient')

            ax.set_xlim(0, max_year)
            # ax.set_ylim(0, 1)
            ax.set_ylim(.2, .7)

            # yticks = [0, .2, .4, .6, .8, 1]
            yticks = [.2, .3, .4, .5, .6, .7]
            ax.set_yticks(yticks)
            ax.set_yticklabels(hnelib.plot.stringify_numbers_without_ugly_zeros(yticks))

            # ax.set_xticks([0, 25, 50])

            hnelib.plot.add_gridlines_on_ticks(ax)


def plot_attrition_risk_by_taxonomy_and_gender():
    df = usfhn.stats.runner.get('attrition/risk/by-time/by-degree-year/gender')

    df = df[
        df['TaxonomyLevel'].isin(['Academia', 'Umbrella'])
    ]

    df['CareerYear'] = df['Year'] - df['DegreeYear']

    df = df[
        df['CareerYear'] <= 50
    ]

    df['TotalEvents'] = df.groupby([
        'TaxonomyLevel',
        'TaxonomyValue',
        'CareerYear',
        'Gender',
    ])['Events'].transform('sum')

    df['TotalAttritionEvents'] = df.groupby([
        'TaxonomyLevel',
        'TaxonomyValue',
        'CareerYear',
        'Gender',
    ])['AttritionEvents'].transform('sum')

    df = df.drop(columns=['Year', 'AttritionEvents', 'Events', 'AttritionRisk', 'DegreeYear']).drop_duplicates()
    df['TotalAttritionRisk'] = df['TotalAttritionEvents'] / df['TotalEvents']

    df = df[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'CareerYear',
            'Gender',
            'TotalEvents',
            'TotalAttritionEvents',
            'TotalAttritionRisk',
        ]
    ].drop_duplicates().rename(columns={
        'TotalEvents': 'Events',
        'TotalAttritionEvents': 'AttritionEvents',
        'TotalAttritionRisk': 'AttritionRisk',
    })

    df = df[
        df['Gender'].isin(['Male', 'Female'])
    ]

    fig, ax = plt.subplots(figsize=(10, 6), tight_layout=True)

    df = usfhn.views.annotate_umbrella_color(df, taxonomization_level='Umbrella')

    for (gender, umbrella), rows in df.groupby(['Gender', 'Umbrella']):
        rows = rows.sort_values(by='CareerYear')
        ax.plot(
            rows['CareerYear'],
            rows['AttritionRisk'],
            lw=.5,
            ls='-' if gender == 'Female' else '--',
            color=rows.iloc[0]['UmbrellaColor']
        )


    ax.set_xlabel('years since PhD')
    ax.set_ylabel('attrition risk')

    # ax.set_xlim(1960, 2020)
    ax.set_xlim(0, 50)
    ax.set_ylim(0, .25)

    hnelib.plot.add_gridlines_on_ticks(ax)

    ax.legend(handles=plot_utils.get_umbrella_legend_handles(style='line', include_academia=True) + [
    # ax.legend(handles=[
        Line2D(
            [0], [0],
            color='black',
            ls='-',
            label='women',
        ),
        Line2D(
            [0], [0],
            color='black',
            ls='--',
            label='men',
        ),
    ])

    df = df.drop(columns=['Umbrella', 'UmbrellaColor'])


def plot_attrition_risk_by_taxonomy_with_balanced_hiring():
    career_lengths_df = usfhn.stats.runner.get('careers/length/gender')

    male_careers = career_lengths_df.copy()[
        career_lengths_df['Gender'] == 'Male'
    ].rename(columns={
        'MeanCareerLength': 'MaleMeanCareerLength',
        'MedianCareerLength': 'MaleMedianCareerLength',
    }).drop(columns=['Gender'])

    female_careers = career_lengths_df.copy()[
        career_lengths_df['Gender'] == 'Female'
    ].rename(columns={
        'MeanCareerLength': 'FemaleMeanCareerLength',
        'MedianCareerLength': 'FemaleMedianCareerLength',
    }).drop(columns=['Gender'])

    career_lengths_df = male_careers.merge(
        female_careers,
        on=[
            'TaxonomyLevel',
            'TaxonomyValue',
        ]
    )

    df = usfhn.stats.runner.get('attrition/gender-balanced-hiring')

    df = df.merge(
        career_lengths_df,
        on=[
            'TaxonomyLevel',
            'TaxonomyValue',
        ]
    )

    df = df[
        df['TaxonomyLevel'].isin(['Academia', 'Umbrella'])
    ]

    max_year = 30

    df = df[
        df['CareerYear'] <= max_year
    ]

    df = usfhn.views.annotate_umbrella_color(df, taxonomization_level='Umbrella')

    fig, axes = plt.subplots(2, 5, figsize=(14, 6), tight_layout=True)

    umbrellas = usfhn.views.get_umbrellas()

    layout = [
        ['Academia'] + umbrellas[:4],
        [None] + umbrellas[4:],
    ]

    visible_axes = []

    for i, layout_row in enumerate(layout):
        for j, taxonomy_value in enumerate(layout_row):
            ax = axes[i, j]

            if not taxonomy_value:
                hnelib.plot.hide_axis(ax)
                continue

            visible_axes.append(ax)

            rows = df[
                df['TaxonomyValue'] == taxonomy_value
            ]

            rows = rows.sort_values(by='CareerYear')
            ax.plot(
                rows['CareerYear'],
                rows['FemaleAttritionRisk'],
                lw=1,
                ls='-',
                color=rows.iloc[0]['UmbrellaColor'],
            )

            ax.plot(
                rows['CareerYear'],
                rows['MaleAttritionRisk'],
                lw=1,
                ls='--',
                color=rows.iloc[0]['UmbrellaColor'],
            )

            ax.plot(
                rows['CareerYear'],
                rows['FractionFemaleRetained'],
                lw=1,
                ls=':',
                color=rows.iloc[0]['UmbrellaColor'],
            )

            ax.scatter(
                [rows['FemaleMeanCareerLength'].iloc[0]],
                [.015],
                marker='o',
                edgecolor=rows.iloc[0]['UmbrellaColor'],
                facecolor=hnelib.plot.set_alpha_on_colors(rows.iloc[0]['UmbrellaColor']),
            )

            ax.scatter(
                [rows['MaleMeanCareerLength'].iloc[0]],
                [.015],
                marker='X',
                edgecolor=rows.iloc[0]['UmbrellaColor'],
                facecolor=hnelib.plot.set_alpha_on_colors(rows.iloc[0]['UmbrellaColor']),
            )

            ax.set_title(taxonomy_value, color=rows.iloc[0]['UmbrellaColor'])


            ax.set_xlabel('years since PhD')
            ax.set_ylabel('attrition risk')

            ax.set_xlim(0, max_year)
            ax.set_ylim(0, .50)

            yticks = [0, .1, .2, .3, .4, .5]
            ax.set_yticks(yticks)
            ax.set_yticklabels(hnelib.plot.stringify_numbers_without_ugly_zeros(yticks))

            # ax.set_xticks([0, 25, 50])

            hnelib.plot.add_gridlines_on_ticks(ax)

    axes[1, 0].legend(
        handles=[
            Line2D(
                [0], [0],
                color='black',
                ls='-',
                label='women',
            ),
            Line2D(
                [0], [0],
                color='black',
                ls='--',
                label='men',
            ),
            Line2D(
                [0], [0],
                color='black',
                ls=':',
                label='fraction female under balanced hiring',
            ),
            Line2D(
                [], [],
                color='none',
                marker='o',
                markerfacecolor=hnelib.plot.set_alpha_on_colors('black'),
                markeredgecolor='black',
                label='mean career age (women)',
            ),
            Line2D(
                [], [],
                color='none',
                marker='X',
                markerfacecolor='w',
                markeredgecolor='black',
                label='mean career age (men)',
            ),
        ],
        loc='center',
    )


def plot_ratio_of_in_field_to_out_of_field():
    df = usfhn.closedness.get_closednesses()
    df = views.filter_exploded_df(df)
    df = df[
        ~df['TaxonomyLevel'].isin(['Academia', 'Taxonomy'])
    ]

    df['InField/OutOfField'] = df['USPhDInField'] / df['USPhDOutOfField']

    df = df[
        [
            'TaxonomyLevel',
            'TaxonomyValue',
            'InField/OutOfField',
        ]
    ].drop_duplicates()


    fig, ax = plt.subplots(1, figsize=(4, 6))

    x_pad = .5

    umbrellas = sorted(list(df[df['TaxonomyLevel'] == 'Umbrella']['TaxonomyValue'].unique()))
    umbrella_colors = [PLOT_VARS['colors']['umbrellas'].get(u) for u in umbrellas]

    fields_df = df[
        df['TaxonomyLevel'] == 'Field'
    ].rename(columns={
        'TaxonomyValue': 'Field',
        'InField/OutOfField': 'Field-InField/OutOfField',
    }).drop(columns=[
        'TaxonomyLevel'
    ])

    areas_df = df[
        df['TaxonomyLevel'] == 'Area'
    ].rename(columns={
        'TaxonomyValue': 'Area',
        'InField/OutOfField': 'Area-InField/OutOfField',
    }).drop(columns=[
        'TaxonomyLevel'
    ])

    umbrellas_df = df[
        df['TaxonomyLevel'] == 'Umbrella'
    ].rename(columns={
        'TaxonomyValue': 'Umbrella',
        'InField/OutOfField': 'Umbrella-InField/OutOfField',
    }).drop(columns=[
        'TaxonomyLevel'
    ])

    df = views.get_taxonomization().merge(
        fields_df,
        on='Field'
    ).merge(
        areas_df,
        on='Area',
    ).merge(
        umbrellas_df,
        on='Umbrella',
    )

    df = views.annotate_umbrella_color(df, taxonomization_level='Umbrella', taxonomization_column='Umbrella')
    df['N-Fields'] = df.groupby('Area')['Field'].transform('nunique')

    jitters = {
        'Field': {a: random.random() / 4 for a in df['Field'].unique()},
        'Area': {a: random.random() / 4 for a in df['Area'].unique()},
    }

    taxonomization_level_pad = 2

    # taxonomization_ordering = ['Field', 'Area', 'Umbrella']
    taxonomization_ordering = ['Field', 'Area']
    line_xs = []
    line_ys = []
    line_colors = []
    for i, level in enumerate(taxonomization_ordering):
        # x_start = i * (len(umbrellas) + taxonomization_level_pad)
        x_start = i * taxonomization_level_pad

        val_col = f'{level}-InField/OutOfField'
        for umbrella, rows in df.groupby('Umbrella'):
            if level == 'Umbrella':
                continue

            if level == 'Field':
                rows = rows[
                    rows['N-Fields'] > 1
                ]

            rows = rows.drop_duplicates(subset=[level, val_col])

            xs = []
            for _, r in rows.iterrows():
                xs.append(x_start + jitters[level][r[level]])

            ax.scatter(
                xs,
                rows[val_col],
                color=rows['UmbrellaColor'],
                s=10,
                alpha=.5,
                zorder=2,
            )

            # if level == 'Umbrella':
            if level != 'Field':
                continue

            upper_level = taxonomization_ordering[i + 1]

            upper_val_col = f'{upper_level}-InField/OutOfField'
            upper_x = x_start + taxonomization_level_pad

            upper_xs = []
            for _, r in rows.iterrows():
                upper_xs.append(x_start + taxonomization_level_pad + jitters[upper_level][r[upper_level]])

            for val, u_val, x, upper_x in zip(rows[val_col], rows[upper_val_col], xs, upper_xs):
                line_xs.append([x, upper_x])
                line_ys.append([val, u_val])
                line_colors.append(rows['UmbrellaColor'].iloc[0])

    ax.set_xticks([])

    # hnelib.plot.add_gridlines(
    #     ax,
    #     xs=[
    #         len(umbrellas) + (taxonomization_level_pad / 2),
    #         (2 * len(umbrellas)) + (taxonomization_level_pad) + (taxonomization_level_pad / 2),
    #     ],
    #     lw=.5,
    #     alpha=.2
    # )

    for line_x, line_y, line_color in zip(line_xs, line_ys, line_colors):
        ax.plot(
            line_x,
            line_y,
            color=line_color,
            lw=.5,
            alpha=.25,
        )


################################################################################
#
#
#
# Main
#
#
#
################################################################################
PLOTS = {
    ################################################################################
    # Paper figures
    ################################################################################
    'paper-figures': {
        'non-us': plot_non_us_multiplot,
        'production': plot_academia_production,
        'ginis': plot_gini_multi_plot,
        'gender-multiplot': plot_gender_multi_plot,
        'self-hires': plot_self_hire_multiplot,
        'rank': plot_rank_change_multi_plot,
        'supplement': {
            'faculty-without-degrees': plot_non_doctorates_by_taxonomy,
            'prestige-tadpole-plot': plot_logit_slopes_tadpoleplot,
            'career-length-prestige-self-hires-logits': plot_career_length_prestige_self_hires_logits,
            'geodesics': plot_multiple_geodesics,
        },
    },
    'poster': {
        'attritions-multiplot': plot_attritions_multiplot,
    },
    'misc-paper-figures': {
        'self-hires-binned-prestige': plot_self_hires_binned_prestige,
        'self-hire-attrition': plot_self_hire_attrition_logits,
        'field-level-exclusions-by-domain': plot_field_level_exclusions_by_domain,
        'faculty-ranks-over-time': plot_faculty_ranks_over_time,
        'demographic-change-multiplot': plot_demographic_change_multiplot,
    },
    'paper-figure-variations': {
        'non-us': plot_degree_breakdown,
        'self-hiring-monoplot': plot_self_hiring_by_gender,
    },
    'old-paper-figures': {
        'self-hiring': plot_self_hiring_multi_plot,
        'changes': plot_change_in_things,
        'permeability': plot_permeability_multi_plot,
        'variations': {
            'permeability-horizontal': plot_permeability_multi_plot_horizontal,
        },
    },
    ################################################################################
    # General
    ################################################################################
    'fraction-female-by-taxonomy': plot_taxonomy_gender_ratios,
    'institutions-by-taxonomy': plot_taxonomy_field_sizes,
    ################################################################################
    # Closedness
    ################################################################################
    'closedness': {
        'by-taxonomy': plot_closedness_by_taxonomy,
        'non-us-by-taxonomy': plot_non_us_closedness_by_taxonomy,
        'us-phd-out-of-field-by-taxonomy': plot_us_phd_out_of_field_closedness_by_taxonomy,
        'us-phd-in-field-by-taxonomy': plot_us_phd_in_field_closedness_by_taxonomy,
        'in-field-us-vs-non-us': plot_us_phd_in_field_vs_non_us,
        'closedness-to-field-size': plot_closedness_to_field_size,
        'umbrella-out-of-field-to-non-us': plot_umbrella_out_of_field_to_non_us,
        'ratio-of-in-field-to-out-of-field': plot_ratio_of_in_field_to_out_of_field,
    },
    ################################################################################
    # Production
    ################################################################################
    'production': {
        'gini-subsampling': plot_gini_subsampling,
        'compare-overall-to-new-hire-lorenz': plot_lorenz_comparison,
    },
    ################################################################################
    # Gender
    ################################################################################
    'gender': {
        'gender': plot_gender,
        'gender-fractions-by-taxonomy': plot_gender_fractions,
    },
    ################################################################################
    # Production versus Employment
    ################################################################################
    'production-vs-employment': {
        'histogram': histogram_production_per_employment,
        'percents': plot_production_versus_employment,
        'integers': {
            'do': plot_production_versus_employment,
            'kwargs': {
                'style': 'integers',
            },
        },
    },
    ################################################################################
    # Self Hiring
    ################################################################################
    'self-hires': {
        'by-taxonomy': plot_self_hiring_by_taxonomy,
        'institutional-self-hiring-vs-prestige': plot_institutional_self_hiring_versus_prestige,
        'institutional-self-hiring-vs-size': plot_institutional_self_hiring_versus_size,
        'academia-null-model': {
            'do': plot_self_hiring_null_model,
            'subdirs': ['null-models'],
        },
        'null-model-umbrella-comparison': plot_self_hiring_null_model_umbrella,
        'top-vs-bottom': plot_self_hiring_at_top_vs_bottom_by_taxonomy,
        'bottom-vs-rest': plot_self_hiring_at_bottom_vs_rest_by_taxonomy,
        'risk-ratio': plot_self_hire_risk_ratios,
        'career-age': {
            'normal': plot_self_hires_career_age,
        },
    },
    ################################################################################
    # Steepness
    ################################################################################
    'steepness': {
        'up-down-ratio-by-taxonomy': plot_up_down_ratio_by_taxonomy,
        'vs-gini': plot_steepness_vs_gini,
        'vs-gender-ratio': plot_steepness_vs_gender_ratio,
        'vs-rank-change': plot_hierarchy_steepness_vs_mean_rank_change,
        'vs-self-hire-rate-by-taxonomy': plot_hierarchy_steepness_vs_self_hiring_rate,
    },
    ################################################################################
    # Rank
    ################################################################################
    'rank': {
        'rank-change-by-taxonomy': plot_rank_change_by_taxonomy,
        'institutions-in-top-rank': {
            'do': plot_institutions_in_top_x,
        },
        'rank-change': {
            'do': one_plot_rank_change,
            'include_academia': True,
        },
        'rank-change-at-taxonomization-level': {
            'do': plot_rank_change_taxonomy_level,
            'include_academia': True,
            'kwargs': {
                'taxonomization_level': 'Umbrella',
            },
        },
        'rank-change-by-umbrella': {
            'do': plot_rank_change_umbrella_group,
            'taxonomization_levels_to_include': ['Area'],
            'kwargs': {
                'taxonomization_level': 'Area',
            },
        },
        'hierarchy-line-arcs':  lineplot_hierarchy_arcs,
        'gender-difference-vs-fraction-female': plot_rank_change_gender_difference_vs_fraction_female,
        'beeswarm': plot_rank_change_beeswarm,
        'kde': {
            'do': plot_rank_change_taxonomy,
            'subdirs': ['kde'],
        },
        'change': {
            'by-year': {
                'academia-and-domain': plot_rank_change_over_time_by_academia_and_domain,
            },
        },
    },
    ################################################################################
    # Level-to-Level calculations
    ################################################################################
    'level-to-level': {
        'mean-rank-difference': plot_level_to_level_prestige_rank_changes,
        'prestige-correlations': {
            'do': plot_level_to_level_prestige_correlations,
        },
    },
    ################################################################################
    # Changes over time
    ################################################################################
    "by-year": {
        'overall': {
            'gini': plot_gini_over_time_alone,
            'gender': plot_gender_over_time_alone,
            'self-hires': plot_self_hires_over_time_alone,
            'hierarchy': plot_rank_change_over_time_by_academia_and_domain,
            'non-us': plot_non_us_taxonomy_by_time,
        },
        "gini-over-time": plot_gini_over_time,
        # "compare-junior-senior": {
        #     'do': compare_junior_senior_changes_over_time,
        #     'expander': hnelib.runner.Runner.get_expander(
        #         prefixes={'measurement': [
        #             'ginis',
        #             'gender',
        #             'self-hires',
        #         ]},
        #     )
        # },
        'faculty-ranks': {
            'faculty-ranks-over-time': plot_faculty_ranks_over_time,
            'raw-faculty-ranks-over-time': {
                'do': plot_faculty_ranks_over_time,
                'kwargs': {'use_raw_data': True},
            },
        },
    },
    ################################################################################
    # Non-US vs prestige
    ################################################################################
    "non-us": {
        "institution-makeup": plot_non_us_institution_makeup,
        "institution-makeup-domains": plot_non_us_institution_makeup_domain_level,
        "vs-prestige": {
            'domain-and-academia': plot_non_us_vs_prestige,
            'domain-and-academia-no-scatter': {
                'do': plot_non_us_vs_prestige,
                'kwargs': {'scatter': False},
            },
            'correlation-across-levels': plot_non_us_vs_prestige_correlation_across_levels,
        },
        'by-country': {
            'basic': plot_non_us_by_country,
            'academia-basic': {
                'do': plot_non_us_by_country,
                'kwargs': {
                    'academia_only': True,
                }
            },
            'by-continent': plot_non_us_by_continent,
            'separate-by-domain': plot_non_us_by_continent_separate_domains,
            'compare-to-academia-fraction': plot_non_us_by_country_compare_to_academia_fraction,
        },
        'by-gender': {
            'across-levels': plot_non_us_by_gender_across_levels,
        },
        'career-age': {
            'normal': plot_non_us_career_age,
            'by-continent': plot_non_us_career_age_by_continent,
        },
    },
    ################################################################################
    # Careers
    ################################################################################
    "careers": {
        "career-trajectory": plot_career_trajectory,
        'moves': {
            'moves-over-time-by-gender': {
                'do': moves_over_time_by_gender,
                # 'expander': Runner.get_expander(
                #     prefixes={'gender': ['All', 'Female', 'Male']},
                # ),
            },
            'institution-risk': plot_institution_mcm_risk,
            'hierarchy-changes': {
                'differences': {
                    'do': plot_hierarchy_changes_from_mcms,
                    # 'expander': Runner.get_expander(prefixes={
                    #     'movement_type': ['Upward', 'Downward', 'Self-Hire']
                    # })
                }
            },
        },
        'age': {
            'by-taxonomy': plot_career_length_by_taxonomy,
            'by-time': {
                'by-taxonomy': plot_career_length_by_taxonomy_and_time,
                'by-taxonomy-median': {
                    'do': plot_career_length_by_taxonomy_and_time,
                    'kwargs': {'measure': 'median'},
                },
            },
            'gini': {
                'by-domain': plot_gini_by_taxonomy_and_career_length,
            },
        },
    },
    'demographics': {
        'by-year': {
            'taxonomy-size': plot_taxonomy_size_by_year,
            'taxonomy-size-field-level': plot_taxonomy_size_by_year_field_level,
            # good plot
            'taxonomy-size-percent-change': plot_taxonomy_size_percent_change,
            'domain-level-changes': plot_domain_level_career_length_changes,
        },
        'career-length-change': plot_career_length_change,
    },
    'attrition': {
        'risk-taxonomy': plot_attrition_risk_by_taxonomy,
        'risk-taxonomy-over-time': plot_attrition_risk_over_time,
        'compare-exit-retirement': plot_exit_vs_retirement_attrition_rates,
        'compare-exit-retirement-risk-ratio': plot_exit_vs_retirement_attrition_rates_risk_ratio,
        'self-hires': {
            'by-time': plot_self_hire_attrition_by_time,
            'sh-non-sh-risk-ratio': plot_self_hire_non_self_hire_attrition_risk_ratio_by_time,
            'compare-exit-retirement': plot_self_hire_exit_vs_retirement_attrition_rates,
            'compare-exit-retirement-alt': plot_self_hire_exit_vs_retirement_attrition_rates_other_risk_ratio,
            'compare-exit-retirement-quad': plot_self_hire_exit_vs_retirement_attrition_rates_quad,
        },
        'gender': {
            'by-time': {
                'do': plot_gender_attrition_risk_over_time,
                # 'expander': Runner.get_expander(prefixes={'gender': ['Male', 'Female']})
            },
            'f-m-risk-ratio': plot_gender_attrition_risk_ratio_over_time,
            'field-level-by-gender-risk': plot_field_level_by_gender_attrition_risk,
            'compare-exit-retirement': plot_gender_exit_vs_retirement_attrition_rates,
            'compare-exit-retirement-alt': plot_gender_exit_vs_retirement_attrition_rates_other_risk_ratio,
            'compare-exit-retirement-quad': plot_gender_exit_vs_retirement_attrition_rates_quad,
        },
        'ranks': {
            'academia-and-domain': {
                'do': plot_attrition_by_rank_academia_and_domain,
                # 'expander': Runner.get_expander(
                #     prefixes={'rank_type': ['production', 'employing-institution', 'doctoral-institution']},
                # ),
            },
            'gendered-academia-and-domain': {
                'do': plot_attrition_by_rank_academia_and_domain_by_gender,
                # 'expander': Runner.get_expander(
                #     prefixes={'rank_type': ['production', 'employing-institution', 'doctoral-institution']},
                # ),
            },
            'risk-vs-ranks-logit-slopes-with-p-values': {
                'do': plot_risk_vs_ranks_logit_slopes_with_p_values,
                # 'expander': Runner.get_expander(
                #     prefixes={'rank_type': ['employing-institution', 'doctoral-institution', 'production']},
                # ),
            },
            'directional-risk': {
                'up-vs-down-risk-ratio-by-time': plot_up_vs_down_attrition_risk_ratio_by_time,
                'up-vs-down-by-time': plot_up_vs_down_attrition_by_time,
            },
        },
        'degree-year': {
            'academia-by-year': plot_attrition_risk_for_one_taxonomy_by_degree_year,
            'gender': {
                'academia-by-year': plot_attrition_risk_for_one_taxonomy_by_degree_year_and_by_gender,
                'taxonomy': plot_attrition_risk_by_taxonomy_and_gender,
                'balanced-hiring-plot': plot_attrition_risk_by_taxonomy_with_balanced_hiring,
            },
        },
        'non-us': {
            'overall': plot_non_us_vs_us_attrition_risk_overall,
            'by-time': plot_non_us_vs_us_attrition_risk_ratio_by_time,
        },
    },
    'interdisciplinarity': {
        'taxonomy-over-time': plot_taxonomy_interdisciplinarity_over_time,
        'ranks-vs-in-or-out-of-field-logits': {
            'do': plot_in_or_out_of_field_vs_ranks_logit_slopes_with_p_values,
            # 'expander': Runner.get_expander(
            #     prefixes={'rank_type': ['production', 'prestige']},
            # ),
        },
    },
    'prestige-logits': {
        'gender': plot_gender_logits,
        'self-hires': plot_self_hire_logits,
        'self-hire-attrition': plot_self_hire_attrition_logits,
        'new-self-hires': plot_new_self_hire_logits,
        'non-us': plot_non_us_logits,
        'interdisciplinarity': plot_interdisciplinarity_logits,
        'career-length': plot_career_length_logits,
        'prestige-logits-multiplot': plot_logit_multiplot,
    },
    'degree-years': {
        'domain-level': plot_degree_years_domain_level,
    },
    'new-hire': {
        'gender': plot_new_hires_by_gender_over_time,
        'gini': plot_new_hires_by_gini_over_time,
        'self-hires': plot_new_hires_by_self_hire_over_time,
        'non-us': plot_new_hires_by_non_us_over_time,
        'interdisciplinarity': plot_new_hires_by_interdisciplinarity_over_time,
        'steepness': plot_new_hires_by_steepness_over_time,
    },
    'geodesics': {
        'basic-plot': plot_basic_geodesics,
    },
}

SUBMISSION_PLOTS = {
    'fig-1': plot_non_us_multiplot,
    'fig-2': plot_academia_production,
    'fig-3': plot_gini_multi_plot,
    'fig-4': plot_gender_multi_plot,
    'fig-5': plot_self_hire_multiplot,
    'fig-6': plot_rank_change_multi_plot,
    'ed-fig-1': plot_non_doctorates_by_taxonomy,
    'ed-fig-2': plot_logit_slopes_tadpoleplot,
    'ed-fig-3': plot_career_length_prestige_self_hires_logits,
    'ed-fig-4':  plot_multiple_geodesics,
}



runner = hnelib.runner.PlotRunner(
    collection=PLOTS,
    directory=usfhn.datasets.CURRENT_DATASET.results_path,
)


def plot_submissions_figures():
    for path in usfhn.constants.FIGURES_NATURE_SUB_3_PATH.rglob('*'):
        if path.is_file():
            path.unlink()

    runner = hnelib.runner.PlotRunner(
        collection=SUBMISSION_PLOTS,
        directory=usfhn.constants.FIGURES_NATURE_SUB_3_PATH,
    )

    runner.run_all(save_kwargs={'suffix': '.eps'})

def save_plot_data_for_submission():
    usfhn.constants.FIGURES_DATA_NATURE_SUB_3_PATH.mkdir(exist_ok=True, parents=True)
    for name, function in SUBMISSION_PLOTS.items():
        print(name)
        path = usfhn.constants.FIGURES_DATA_NATURE_SUB_3_PATH.joinpath(name + '.csv')

        df = function()

        df.to_csv(path, index=False)


def plot_to_paper_figures(
    plots={k: v for k, v in PLOTS.items() if k == 'paper-figures'},
    out_directory=constants.PAPER_PATH.joinpath('figures'),
    save_kwargs={
        'dpi': 400,
    },
):
    sub_runner = hnelib.runner.PlotRunner(
        collection=plots,
        directory=out_directory,
    )
    sub_runner.run_all(save_kwargs=save_kwargs)


def plot_pdfs_to_paper_figures():
    plot_to_paper_figures(save_kwargs={
        'suffix': '.pdf',
        'dpi': 400,
    })


def run_all_plots(plots={k: v for k, v in PLOTS.items() if k == 'paper-figures'}):
    global runner
    runner.run_all()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='plotter')
    parser.add_argument('--show', '-s', default=False, action='store_true', help='show the plot')
    parser.add_argument('--plotter_name', '-p', type=str, help='which thing to plot')
    parser.add_argument('--dataset', '-d', type=str, default='primary', help='dataset')
    parser.add_argument('--all_plots', default=False, action='store_true', help='run all plots for the dataset')
    parser.add_argument('--all_datasets', default=False, action='store_true', help='run datasets')
    parser.add_argument('--run_expansions', default=False, action='store_true', help='run plot expansions')
                        
    args = parser.parse_args()

    plots_to_run = []
    if args.all_plots:
        plots_to_run = PLOTS
    elif args.plotter_name:
        plots_to_run.append(args.plotter_name)

    datasets_to_run = []
    if args.all_datasets:
        datasets_to_run = list(usfhn.datasets.DATASETS.keys())
    elif args.dataset:
        datasets_to_run = [args.dataset]

    for dataset_name in datasets_to_run:
        for plotter_name in plots_to_run:
            run_plot(
                plotter_name,
                show=args.show,
                dataset_name=dataset_name,
                run_expansions=args.run_expansions,
            )
