import pandas as pd
import numpy as np

from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import Patch, Arc
from matplotlib.path import Path as MPath
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.axes
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.patches as patches
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.tri as tri

import hnelib.plot

import usfhn.plot_utils
import usfhn.views

# functions to maybe migrate from usfhn.plots:
# - plot_changes_in_things (along with draw_lines_for_changes_plot)
# - plot_relationship_across_taxonomy_levels
# - plot_ranked_attrition?

def plot_univariate_value_across_taxonomy_levels(
    df,
    value_column,
    title,
    y_min,
    y_max,
    y_step,
    y_label=None,
    show_x_ticks=False,
    filter_kwargs={},
    figsize=(10, 5),
    y_pad_percent=0,
    extra_legend_handles=[],
):
    df = usfhn.views.filter_exploded_df(df, **filter_kwargs)

    fig, axes = plt.subplots(1, 4, figsize=figsize, gridspec_kw={'width_ratios': [1, 1, .3, .5]})
    axes = axes.flatten()
    legend_ax = axes[-1]
    axes = axes[:-1]

    x_pad = .5

    umbrellas = sorted(list(df[df['TaxonomyLevel'] == 'Umbrella']['TaxonomyValue'].unique()))
    umbrella_colors = [usfhn.plot_utils.PLOT_VARS['colors']['umbrellas'].get(u) for u in umbrellas]
    
    # for ax, level in zip(axes, ['Field', 'Area', 'Umbrella']):
    for ax, level in zip(axes, ['Field', 'Umbrella']):
        level_df = df[df['TaxonomyLevel'] == level].copy().rename(columns={
            'TaxonomyValue': level,
        }).drop_duplicates(subset=[level])

        level_df = usfhn.views.annotate_umbrella_color(level_df, level)

        ax.set_title(level.lower())
        for umbrella, rows in level_df.groupby('Umbrella'):
            umbrella_index = umbrellas.index(umbrella)

            facecolors = []
            for i, row in rows.iterrows():
                if 'NoFaceColor' in rows.columns and row['NoFaceColor']:
                    facecolor = 'w' 
                else:
                    facecolor = hnelib.plot.set_alpha_on_colors(row['UmbrellaColor'])
                facecolors.append(facecolor)

            ax.scatter(
                [umbrella_index for i in range(len(rows))],
                rows[value_column],
                facecolors=facecolors,
                edgecolors=rows['UmbrellaColor'],
                zorder=2,
            )

    academia_row = df[df['TaxonomyLevel'] == 'Academia'].iloc[0]
    academia_color = usfhn.plot_utils.PLOT_VARS['colors']['Academia']
    if 'NoFaceColor' in df.columns and academia_row['NoFaceColor']:
        academia_facecolor = 'w' 
    else:
        academia_facecolor = hnelib.plot.set_alpha_on_colors(academia_color)

    academia_ax = axes[-1]
    academia_ax.scatter(
        [0],
        [academia_row[value_column]],
        zorder=2,
        facecolors=academia_facecolor,
        edgecolors=academia_color,
    )

    x_pad = .5
    academia_ax.set_title('academia')
    academia_ax.set_xlim(-x_pad, x_pad)

    legend_handles = usfhn.plot_utils.get_umbrella_legend_handles() + [
        Line2D(
            [], [],
            color='none',
            marker='o',
            markerfacecolor=academia_facecolor,
            markeredgecolor=academia_color,
            label='Academia',
        ),
    ]

    if extra_legend_handles:
        legend_handles += extra_legend_handles

    legend_ax.legend(handles=legend_handles, loc='center', prop={'size': 12})
    legend_ax.axis('off')

    padded_y_max = y_max * 1.005
    y_ticks = np.arange(min(y_min, y_max), max(y_min, y_max) * 1.005, y_step)

    padded_y_min = y_min
    if y_pad_percent:
        total_y = padded_y_max - y_min
        padded_y_min = y_min - (total_y * y_pad_percent)

    for i, ax in enumerate(axes):
        ax.set_ylim(padded_y_min, padded_y_max)

        if i != len(axes) - 1:
            ax.set_xlim(-x_pad, len(umbrellas) - x_pad)
            rectangle_height = ax.get_ylim()[1] - ax.get_ylim()[0]

            # for i, umbrella in enumerate(umbrellas):
            #     ax.add_patch(
            #         mpatches.Rectangle(
            #             xy=(i - x_pad, ax.get_ylim()[0]),
            #             width=1,
            #             height=rectangle_height,
            #             color=PLOT_VARS['colors']['umbrellas'][umbrella],
            #             alpha=.1,
            #             zorder=0,
            #             linewidth=.25,
            #         )
            #     )

        hnelib.plot.add_gridlines(ax, ys=[tick for tick in y_ticks if round(tick, 4) != padded_y_min])

        ax.set_yticks([])

        if i == len(axes) - 1:
            xticklabels = ['academia']
            colors = ['black']
        else:
            xticklabels = umbrellas
            colors = umbrella_colors

        if show_x_ticks:
            ax.set_xticks([i for i in range(len(xticklabels))])
            ax.set_xticklabels(xticklabels)
            for tick, color in zip(ax.get_xticklabels(), colors):
                tick.set_color(color)
                tick.set_rotation(90)
        else:
            ax.set_xticks([])

    y_label = y_label if y_label else title
    axes[0].set_ylabel(y_label)
    axes[0].set_yticks(y_ticks)

    plt.suptitle(f'{title}\nby taxonomic level', size=36)
    plt.tight_layout()

    return fig, axes, legend_ax


def plot_univariate_value_across_taxonomy_levels_single_plot(
    df,
    value_column,
    y_min=None,
    y_max=None,
    y_step=None,
    y_label=None,
    show_x_ticks=False,
    x_for_non_significant=False,
    filter_kwargs={},
    figsize=(7, 4),
    y_pad_percent=0,
    extra_legend_handles=[],
    add_gridlines=True,
    fig=None,
    ax=None,
    legend_ax=None,
):
    df = usfhn.views.filter_exploded_df(df, **filter_kwargs)

    big_umbrella_scatter = 20
    small_umbrella_scatter = 15
    big_field_scatter = 12
    small_field_scatter = 8
    lw = .65

    if not ax:
        fig, (ax, legend_ax) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [5, 2]})

    x_pad = .5

    umbrellas = ['Academia'] + usfhn.views.get_umbrellas()
    umbrella_colors = [usfhn.plot_utils.PLOT_VARS['colors']['umbrellas'].get(u) for u in umbrellas]

    fields_df = usfhn.views.filter_by_taxonomy(df, 'Field')
    fields_df = usfhn.views.annotate_umbrella_color(fields_df, taxonomization_level='Field')

    for umbrella_index, umbrella in enumerate(umbrellas):
        if umbrella == 'Academia':
            color = usfhn.plot_utils.PLOT_VARS['colors']['umbrellas'][umbrella]
            facecolor = hnelib.plot.set_alpha_on_colors(color)
            row = usfhn.views.filter_by_taxonomy(df, 'Academia').iloc[0]
        else:
            color = usfhn.plot_utils.PLOT_VARS['colors']['dark_gray']
            facecolor = color
            row = usfhn.views.filter_by_taxonomy(df, 'Umbrella', umbrella).iloc[0]

        if x_for_non_significant:
            if row['Significant']:
                ax.scatter(
                    [umbrella_index],
                    [row[value_column]],
                    facecolor=facecolor,
                    edgecolor=color,
                    zorder=5,
                    s=big_umbrella_scatter,
                    clip_on=False,
                    lw=lw,
                )
            else:
                ax.scatter(
                    [umbrella_index],
                    [row[value_column]],
                    # color=hnelib.plot.set_alpha_on_colors(color, .75),
                    color=color,
                    marker='x',
                    s=small_umbrella_scatter,
                    zorder=4,
                    clip_on=False,
                    lw=lw,
                )
        else:
            ax.scatter(
                [umbrella_index],
                [row[value_column]],
                facecolor=facecolor,
                edgecolor=color,
                zorder=3,
                s=big_umbrella_scatter,
                clip_on=False,
                lw=lw,
            )

        if umbrella != 'Academia':
            rows = fields_df[
                fields_df['Umbrella'] == umbrella
            ]

            if x_for_non_significant:
                sig_rows = rows[
                    rows['Significant']
                ]

                ax.scatter(
                    [umbrella_index for i in range(len(sig_rows))],
                    sig_rows[value_column],
                    facecolors=sig_rows['FadedUmbrellaColor'],
                    edgecolors=sig_rows['UmbrellaColor'],
                    s=big_field_scatter,
                    zorder=3,
                    clip_on=False,
                    lw=lw,
                )

                insig_rows = rows[
                    ~rows['Significant']
                ]

                ax.scatter(
                    [umbrella_index for i in range(len(insig_rows))],
                    insig_rows[value_column],
                    color=insig_rows['FadedUmbrellaColor'],
                    marker='x',
                    s=small_field_scatter,
                    zorder=2,
                    clip_on=False,
                    lw=lw,
                )
            else:
                ax.scatter(
                    [umbrella_index for i in range(len(rows))],
                    rows[value_column],
                    facecolors=rows['FadedUmbrellaColor'],
                    edgecolors=rows['UmbrellaColor'],
                    s=big_field_scatter,
                    zorder=2,
                    clip_on=False,
                    lw=lw,
                )

    edgecolor = 'black'
    facecolor = hnelib.plot.set_alpha_on_colors(edgecolor)
    if legend_ax:
        legend_handles = usfhn.plot_utils.get_umbrella_legend_handles(include_academia=True) + [
            Line2D(
                [], [],
                color='none',
                marker='o',
                markerfacecolor=edgecolor,
                markeredgecolor=edgecolor,
                markersize=4,
                markeredgewidth=.5,
                label='domain',
            ),
            Line2D(
                [], [],
                color='none',
                marker='o',
                markerfacecolor=facecolor,
                markeredgecolor=edgecolor,
                markersize=4,
                markeredgewidth=.65,
                label='field',
            ),
        ]

        if extra_legend_handles:
            legend_handles += extra_legend_handles

        hnelib.plot.hide_axis(legend_ax)
        legend = legend_ax.legend(
            handles=legend_handles,
            loc='center',
            prop={'size': hnelib.plot.FONTSIZES['annotation']},
        )

        usfhn.plot_utils.set_umbrella_legend_text_colors(legend)

    if y_min and y_max and y_step:
        padded_y_max = y_max * 1.005
        y_ticks = np.arange(min(y_min, y_max), max(y_min, y_max) * 1.005, y_step)

        padded_y_min = y_min
        if y_pad_percent:
            total_y = padded_y_max - y_min
            padded_y_min = y_min - (total_y * y_pad_percent)

        ax.set_yticks(y_ticks)
        ax.set_yticklabels(hnelib.plot.stringify_numbers_without_ugly_zeros(y_ticks))

        ax.set_ylim(padded_y_min, padded_y_max)

        if add_gridlines:
            hnelib.plot.add_gridlines(ax, ys=[tick for tick in y_ticks if round(tick, 4) != padded_y_min])

    ax.set_xlim(-x_pad, len(umbrellas) - x_pad)

    ax.set_xticks([])

    if y_label:
        ax.set_ylabel(y_label, fontsize=hnelib.plot.FONTSIZES['axis'])

    plt.tight_layout()

    return fig, ax, legend_ax

def plot_domain_level_values_over_time(
    df,
    column,
    ylabel,
    ylim=[],
    yticks=[],
    ax=None,
    legend_ax=None,
    gridlines=True,
    include_academia=True,
    s=12,
):
    df = df[
        df['TaxonomyLevel'].isin(['Academia', 'Umbrella'])
    ]

    if not include_academia:
        df = df[
            df['TaxonomyLevel'] != 'Academia'
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

    if not ax:
        fig, axes = plt.subplots(1, 2, figsize=(7, 4), gridspec_kw={'width_ratios': [5, 2]})
        ax = axes[0]
        legend_ax = axes[1]
    else:
        axes = [ax, legend_ax]

    umbrellas = usfhn.views.get_umbrellas()

    if include_academia:
        umbrellas.append('Academia')

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
            s=s,
        )

    ax.set_xlabel('year')
    ax.set_ylabel(ylabel)

    if ylim:
        ax.set_ylim(ylim)

    if yticks:
        ax.set_yticks(yticks)
        ax.set_yticklabels(hnelib.plot.stringify_numbers_without_ugly_zeros(yticks))

    if gridlines:
        hnelib.plot.add_gridlines_on_ticks(ax, x=False)

    if legend_ax:
        legend = usfhn.plot_utils.add_umbrella_legend(
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

        hnelib.plot.hide_axis(legend_ax)

    return axes


def compare_two_values_at_the_domain_level_over_time(
    df1,
    df2,
    column,
    ylabel,
    title1,
    title2,
    ylim=[],
    yticks=[],
):
    fig, axes = plt.subplots(
        1, 3,
        figsize=(9, 4),
        tight_layout=True,
        gridspec_kw={'width_ratios': [4, 4, 1]},
    )

    plot_domain_level_values_over_time(
        df1,
        column,
        ylabel,
        ylim=ylim,
        yticks=yticks,
        ax=axes[0],
        gridlines=False,
    )
    
    axes[0].set_title(title1)

    plot_domain_level_values_over_time(
        df2,
        column,
        ylabel,
        ylim=ylim,
        yticks=yticks,
        ax=axes[1],
        gridlines=False,
    )

    axes[1].set_title(title2)

    if ylim:
        for ax in axes[:2]:
            ax.set_ylim(ylim)
    else:
        hnelib.plot.set_lims_to_max(axes[:2], x=False)

    if yticks:
        for ax in axes[:2]:
            ax.set_yticks(yticks)
            ax.set_yticklabels(hnelib.plot.stringify_numbers_without_ugly_zeros(yticks))

    for ax in axes[:2]:
        hnelib.plot.add_gridlines_on_ticks(ax, x=False)

    hnelib.plot.hide_axis(axes[2])
    axes[2].legend(
        handles=usfhn.plot_utils.get_umbrella_legend_handles(
            style='line',
            include_academia=True,
            extra_kwargs={
                'alpha': 1,
            }
        ),
        loc='center'
    )

def compare_four_values_at_the_domain_level_over_time(
    dfs,
    column,
    ylabel,
    titles,
    ylim=[],
    yticks=[],
    logscale=False,
):
    figure_mosaic = """
    ABE
    CDE
    """
    fig, axes = plt.subplot_mosaic(
        mosaic=figure_mosaic,
        figsize=(9, 6),
        tight_layout=True,
        gridspec_kw={
            'width_ratios': [4, 4, 1]
        }
    )

    legend_ax = axes['E']
    axes = [axes['A'], axes['B'], axes['C'], axes['D']]

    for ax, df, title in zip(axes, dfs, titles):
        plot_domain_level_values_over_time(
            df,
            column,
            ylabel,
            ylim=ylim,
            yticks=yticks,
            ax=ax,
            gridlines=False,
        )
        
        ax.set_title(title)

    if ylim:
        for ax in axes:
            ax.set_ylim(ylim)
    else:
        hnelib.plot.set_lims_to_max(axes, x=False)

    if yticks:
        for ax in axes:
            ax.set_yticks(yticks)
            ax.set_yticklabels(hnelib.plot.stringify_numbers_without_ugly_zeros(yticks))

    for ax in axes:
        hnelib.plot.add_gridlines_on_ticks(ax, x=False)

    hnelib.plot.hide_axis(legend_ax)
    legend_ax.legend(
        handles=usfhn.plot_utils.get_umbrella_legend_handles(
            style='line',
            include_academia=True,
            extra_kwargs={
                'alpha': 1,
            }
        ),
        loc='center'
    )

def plot_connected_two_values_by_taxonomy_level(
    df,
    open_column,
    closed_column,
    open_label,
    closed_label,
    y_label,
    y_min,
    y_max,
    y_step,
    open_marker_col=None,
    closed_marker_col=None,
    include_academia=True,
    ax=None,
    legend_ax=None,
    add_gridlines=True,
    add_taxonomy_lines=True,
    skip_fields=False,
):
    if not ax:
        fig, axes = plt.subplots(1, 2, figsize=(7, 4), tight_layout=True, gridspec_kw={'width_ratios': [4, 2]})
        ax = axes[0]
        legend_ax = axes[1]

    umbrellas = usfhn.views.get_umbrellas()

    if include_academia:
        umbrellas = ['Academia'] + umbrellas

    # how do we space things?
    # 0 1 2
    # |s e|
    #   label
    space = 2
    lines = [i * space for i in range(len(umbrellas) + 1)]
    start_is = {u: umbrellas.index(u) * space + .5 for u in umbrellas}
    end_is = {u: umbrellas.index(u) * space + 1.5 for u in umbrellas}
    umbrella_to_label_position = {u: np.mean([start_is[u], end_is[u]]) for u in umbrellas}

    draw_lines_for_change_plot(
        ax,
        df,
        umbrellas,
        start_is,
        end_is,
        open_column,
        closed_column,
        open_marker_col=open_marker_col,
        closed_marker_col=closed_marker_col,
        skip_fields=skip_fields,
    )

    legend_handles = usfhn.plot_utils.get_umbrella_legend_handles(
        style='line',
        include_academia=include_academia,
    ) + [
        Line2D(
            [], [],
            color='none',
            marker='o',
            markerfacecolor='black',
            markeredgecolor='black',
            label=open_label,
        ),
        Line2D(
            [], [],
            color='none',
            marker='o',
            markerfacecolor='w',
            markeredgecolor='black',
            label=closed_label,
        ),
    ]

    ax.set_ylim(y_min, y_max)
    ax.set_yticks(np.arange(y_min, y_max + y_step, y_step))
    ax.set_ylabel(y_label)
    ax.set_xticks([])

    if add_gridlines:
        hnelib.plot.add_gridlines_on_ticks(ax, x=False)

    ax.set_xlim(lines[0], lines[-1])
    if add_taxonomy_lines:
        hnelib.plot.add_gridlines(ax, xs=lines)

    if legend_ax:
        hnelib.plot.hide_axis(legend_ax)
        legend_ax.legend(
            handles=legend_handles,
            loc='center'
        )

    return start_is, end_is, lines, umbrella_to_label_position


def plot_connected_two_values_by_taxonomy_level_horizontal(
    df,
    open_column,
    closed_column,
    open_label,
    closed_label,
    x_label,
    x_min,
    x_max,
    x_step,
    include_academia=True,
    ax=None,
    legend_ax=None,
    add_gridlines=True,
    add_taxonomy_lines=True,
):
    if not ax:
        fig, axes = plt.subplots(2, 1, figsize=(4, 7), tight_layout=True, gridspec_kw={'height_ratios': [4, 2]})
        ax = axes[0]
        legend_ax = axes[1]

    umbrellas = usfhn.views.get_umbrellas()

    if include_academia:
        umbrellas = ['Academia'] + umbrellas

    umbrellas = list(reversed(umbrellas))

    # how do we space things?
    # 0 1 2
    # |s e|
    #   label
    space = 2
    lines = [i * space for i in range(len(umbrellas) + 1)]
    start_is = {u: umbrellas.index(u) * space + .5 for u in umbrellas}
    end_is = {u: umbrellas.index(u) * space + 1.5 for u in umbrellas}
    umbrella_to_label_position = {u: np.mean([start_is[u], end_is[u]]) for u in umbrellas}

    draw_lines_for_change_plot(
        ax,
        df,
        umbrellas,
        start_is,
        end_is,
        open_column,
        closed_column,
        horizontal=True,
    )

    legend_handles = usfhn.plot_utils.get_umbrella_legend_handles(
        style='line',
        include_academia=include_academia,
    ) + [
        Line2D(
            [], [],
            color='none',
            marker='o',
            markerfacecolor='black',
            markeredgecolor='black',
            label=open_label,
        ),
        Line2D(
            [], [],
            color='none',
            marker='o',
            markerfacecolor='w',
            markeredgecolor='black',
            label=closed_label,
        ),
    ]

    ax.set_xlim(x_min, x_max)
    ax.set_xticks(np.arange(x_min, x_max + x_step, x_step))
    ax.set_xlabel(x_label)
    ax.set_yticks([])

    if add_gridlines:
        hnelib.plot.add_gridlines_on_ticks(ax, y=False)

    if add_taxonomy_lines:
        hnelib.plot.add_gridlines(ax, ys=lines)

    if legend_ax:
        hnelib.plot.hide_axis(legend_ax)
        legend_ax.legend(
            handles=legend_handles,
            loc='center'
        )

    return start_is, end_is, umbrella_to_label_position

def draw_lines_for_change_plot(
    ax,
    df,
    umbrellas,
    start_is,
    end_is,
    start_col,
    end_col,
    open_marker_col=None,
    closed_marker_col=None,
    horizontal=False,
    skip_fields=False,
):
    umbrella_size = 25
    field_size = 12
    small_field_size = 8

    umbrella_df = df[
        df['TaxonomyLevel'] == 'Umbrella'
    ]

    academia_df = df[
        df['TaxonomyLevel'] == 'Academia'
    ]

    if not academia_df.empty:
        umbrella_df = pd.concat([academia_df, umbrella_df])

    field_df = df[
        df['TaxonomyLevel'] == 'Field'
    ]
    field_df = usfhn.views.annotate_umbrella_color(field_df, taxonomization_level='Field')

    if horizontal:
        start_is, end_is = end_is, start_is

    lw = .65
    umbrella_line_width = .75 if skip_fields else 1.5

    for umbrella in umbrellas:
        start = start_is[umbrella]
        end = end_is[umbrella]

        umbrella_row = umbrella_df[
            (umbrella_df['TaxonomyValue'] == umbrella)
        ].iloc[0]

        if umbrella == 'Academia':
            color = usfhn.plot_utils.PLOT_VARS['colors']['umbrellas']['Academia']
        else:
            if skip_fields:
                color = usfhn.plot_utils.PLOT_VARS['colors']['umbrellas'][umbrella]
            else:
                color = usfhn.plot_utils.PLOT_VARS['colors']['dark_gray']

        xs = [start, end]
        ys = [umbrella_row[start_col], umbrella_row[end_col]]

        if horizontal:
            xs, ys = ys, xs

        ax.plot(
            xs,
            ys,
            color=color,
            lw=umbrella_line_width,
            zorder=3,
        )

        xs = [start]
        ys = [umbrella_row[start_col]]

        if horizontal:
            xs, ys = ys, xs

        if closed_marker_col and umbrella_row[closed_marker_col]:
            ax.scatter(
                xs,
                ys,
                edgecolor='white',
                facecolor=color,
                zorder=5,
                marker='X',
                lw=lw,
                s=umbrella_size,
            )
        else:
            ax.scatter(
                xs,
                ys,
                color=color,
                zorder=4,
                lw=lw,
                s=umbrella_size,
            )


        xs = [end]
        ys = [umbrella_row[end_col]]

        if horizontal:
            xs, ys = ys, xs

        if open_marker_col and umbrella_row[open_marker_col]:
            ax.scatter(
                xs,
                ys,
                facecolor='white',
                edgecolor=color,
                zorder=5,
                marker='X',
                lw=lw,
                s=umbrella_size,
            )
        else:
            ax.scatter(
                xs,
                ys,
                facecolor='white',
                edgecolor=color,
                zorder=4,
                lw=lw,
                s=umbrella_size,
            )


        if skip_fields:
            continue

        field_rows = field_df[
            field_df['Umbrella'] == umbrella
        ]

        for _, row in field_rows.iterrows():
            xs = [start, end]
            ys = [row[start_col], row[end_col]]

            if horizontal:
                xs, ys = ys, xs

            ax.plot(
                xs,
                ys,
                color=row['UmbrellaColor'],
                lw=lw,
                alpha=.75,
                zorder=1,
            )

            xs = [start]
            ys = [row[start_col]]

            if horizontal:
                xs, ys = ys, xs

            ax.scatter(
                xs,
                ys,
                color=row['UmbrellaColor'],
                zorder=2,
                lw=lw,
                s=small_field_size,
            )

            xs = [end]
            ys = [row[end_col]]

            if horizontal:
                xs, ys = ys, xs

            ax.scatter(
                xs,
                ys,
                facecolor='white',
                edgecolor=row['UmbrellaColor'],
                zorder=2,
                lw=lw,
                s=small_field_size,
            )
