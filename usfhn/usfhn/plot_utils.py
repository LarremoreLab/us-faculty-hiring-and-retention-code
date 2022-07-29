import pandas as pd
import matplotlib.axes
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde, pearsonr, spearmanr

import hnelib.plot

import usfhn.views

PLOT_VARS = {
    'figures': {
        'page-width': 17,
    },
    'colors': {
        'color_1': '#45A1F8',
        'color_2': '#FF6437', 
        'gender': {
            'All': '#5E5E5E',
            'Male': '#4B917D',
            'Female': '#FF6437',
            'Unknown': '#509BF5',
        },
        'dark_gray': '#5E5E5E',
        'categories': {
            'Humanities': '#9BC53D',
            'Social Sciences': '#5BC0EB',
            'STEM': '#E55934',
        },
        'department-level': {
            'black': '#0e6101',
            'red': '#4c4c4c', 
            'green': '#a30005'
        },
        'academia': '#45A1F8',
        'Academia': '#45A1F8',
        'umbrellas': {
            'Applied Sciences': '#1b9e77',
            'Education': '#d95f02',
            'Engineering': '#7570b3',
            'Mathematics and Computing': '#e7298a',
            'Mathematics & Computing': '#e7298a',
            'Humanities': '#66a61e',
            'Medicine and Health': '#e6ab02',
            'Medicine & Health': '#e6ab02',
            'Natural Sciences': '#a6761d',
            'Social Sciences': '#666666',
            # academia's not an umbrella but it's nice to be able to call it by this
            'Academia': '#45A1F8',
            # extra umbrellas
            'Public Administration and Policy': '#1C4D7C',
            'Public Administration & Policy': '#1C4D7C',
            'Journalism, Media, Communication': '#A62A17',
        },
        'ranks': {
            'Assistant Professor': '#e41a1c',
            'Associate Professor': '#377eb8',
            'Professor': '#4daf4a',
        },
    },
    'arrowprops': {
        'arrowstyle': '->',
        'connectionstyle': 'arc3',
        'color': '#5E5E5E', # dark gray
        'lw': .5,
    },
    'abbreviations': {
        'Academia': 'Academia',
        'Applied Sciences': 'App Sc.',
        'Education': 'Edu.',
        'Engineering': 'Eng.',
        'Mathematics and Computing': 'Comp. & Math.',
        'Humanities': 'Hum.',
        'Medicine and Health': 'Med. & Hlth.',
        'Natural Sciences': 'Nat. Sc.',
        'Social Sciences': 'Soc. Sc.',
    },
    'text': {
        'annotations': {
            'labelsize': 12,
        },
        'axis': {
            'fontsize': 16,
        },
    },
    'artists': {
        'title': {
            'functions': [
                (matplotlib.axes.Axes, 'set_title')
            ],
            'kwargs': {
                'pad': 10,
                'fontsize': 30,
                'size': 30,
            },
        },
        'axis_labels': {
            'functions': [
                (matplotlib.axes.Axes, 'set_xlabel'),
                (matplotlib.axes.Axes, 'set_ylabel'),
            ],
            'kwargs': {
                'size': 24,
            },
        },
        'legend': {
            'functions': [
                (matplotlib.axes.Axes, 'legend'),
            ],
            'kwargs': {
                'prop': {
                    'size': 20,
                },
            },
        },
    },
    'subplots_set_methods': [
        {
            'method': 'tick_params',
            'kwargs': {
                'labelsize': 16,
            },
        },
    ],
}

GENDER_TO_STRING = {
    'Female': 'women',
    'Male': 'men',
    'All': 'everyone'
}

MOVEMENT_TO_STRING = {
    'Self-Hire': 'self-hire',
    'Upward': 'violation',
    'Downward': 'downward',
}

FIELD_ABBREVIATIONS = {
    # 'Accounting': '',
    # 'Agricultural Economics': '',
    # 'Agronomy': '',
    # 'Animal Sciences': '',
    # 'Anthropology': '',
    # 'Architecture': '',
    # 'Astronomy': '',
    # 'Biochemistry': '',
    # 'Biostatistics': '',
    # 'Cell Biology': '',
    # 'Chemistry': '',
    # 'Computer Science': '',
    # 'Counselor Education': '',
    # 'Ecology': '',
    # 'Educational Psychology': '',
    # 'Entomology': '',
    # 'Environmental Sciences': '',
    # 'Epidemiology': '',
    # 'Finance': '',
    # 'Food Science': '',
    # 'Geography': '',
    # 'Geology': '',
    # 'History': '',
    # 'Information Science': '',
    # 'Linguistics': '',
    # 'Management': '',
    # 'Marketing': '',
    # 'Mathematics': '',
    # 'Microbiology': '',
    # 'Molecular Biology': '',
    # 'Natural Resources': '',
    # 'Neuroscience': '',
    # 'Nursing': '',
    # 'Nutrition Sciences': '',
    # 'Pathology': '',
    # 'Pharmacology': '',
    # 'Pharmacy': '',
    # 'Philosophy': '',
    # 'Physiology': '',
    # 'Plant Pathology': '',
    # 'Political Science': '',
    # 'Public Health': '',
    # 'Religious Studies': '',
    # 'Social Work': '',
    # 'Sociology': '',
    # 'Soil Science': '',
    # 'Special Education': '',
    # 'Statistics': '',
    # 'Theological Studies': '',
    'Aerospace Engineering': 'Aerospace Eng',
    'Agricultural Engineering': 'Agricultural Eng',
    'Art History and Criticism': 'Art Hist & Crit',
    'Atmospheric Sciences and Meteorology': 'Atmospheric Sciences',
    'Biological Sciences, General': 'Biology',
    'Biological Sciences': 'Biology',
    'Biomedical Engineering': 'Biomedical Eng',
    'Chemical Engineering': 'Chemical Eng',
    'Civil Engineering': 'Civil Eng',
    'Classics and Classical Languages': 'Classics',
    'Communication Disorders and Sciences': 'Comm Disorders & Sci',
    'Computer Engineering': 'Computer Eng',
    'Criminal Justice and Criminology': 'Criminology',
    'Curriculum and Instruction': 'Curriculum',
    'Economics, General': 'Economics',
    'Economics': 'Economics',
    'Education Administration': 'Education Admin',
    'Education, General': 'Education',
    'Education': 'Education',
    'Electrical Engineering': 'Electrical Eng',
    'English Language and Literature': 'English',
    'Environmental Engineering': 'Environmental Eng',
    'Exercise Science, Kinesiology, Rehab, Health': 'Exercise Science',
    'Forestry and Forest Resources': 'Forestry',
    'Health, Physical Education, Recreation': 'Health & Recreation',
    'Human Development and Family Sciences, General': 'Family Sciences',
    'Human Development and Family Sciences': 'Family Sciences',
    'Industrial Engineering': 'Industrial Eng',
    'Materials Engineering': 'Materials Eng',
    'Mechanical Engineering': 'Mechanical Eng',
    'Music, General': 'Music',
    'Music': 'Music',
    'Physics, General': 'Physics',
    'Physics': 'Physics',
    'Psychology, General': 'Psychology',
    'Psychology': 'Psychology',
    'Spanish Language and Literature': 'Spanish',
    'Theatre Literature, History and Criticism': 'Theatre Lit Hist & Crit',
    'Urban and Regional Planning': 'Urban Planning',
    'Veterinary Medical Sciences': 'Veterinary Sciences',
}



def get_umbrella_legend_handles(
    style='scatter',
    include_academia=False,
    fade_markerfacecolors=True,
    extra_kwargs={}
):
    legend_handles = []

    umbrellas = usfhn.views.get_umbrellas()
    umbrellas = [u for u in umbrellas if u != 'Academia']

    if include_academia:
        umbrellas = ['Academia'] + umbrellas

    for umbrella in umbrellas:
        color = PLOT_VARS['colors']['umbrellas'].get(umbrella)
        kwargs = {
            'label': clean_taxonomy_string(umbrella),
            **extra_kwargs,
        }

        if style == 'scatter':
            handle = mlines.Line2D(
                [], [],
                color='none',
                marker='o',
                markerfacecolor=hnelib.plot.set_alpha_on_colors(color) if fade_markerfacecolors else color,
                markeredgecolor=color,
                markersize=4,
                markeredgewidth=.5,
                **kwargs,
            )
        elif style == 'line':
            handle = mlines.Line2D(
                [0], [0],
                color=color,
                **kwargs,
            )
        elif style == 'none':
            handle = mlines.Line2D(
                [0], [0],
                color='none',
                **kwargs,
            )
        else:
            handle = mpatches.Patch(color=color, **kwargs)

        label = handle.get_label()
        legend_handles.append(handle)

    return legend_handles


def set_umbrella_legend_text_colors(legend):
    for text in legend.get_texts():
        if text.get_text() in PLOT_VARS['colors']['umbrellas']:
            plt.setp(text, color=PLOT_VARS['colors']['umbrellas'][text.get_text()])


def add_umbrella_legend(
    ax,
    extra_legend_handles=[],
    get_umbrella_legend_handles_kwargs={},
    legend_kwargs={},
):
    legend_handles = get_umbrella_legend_handles(**get_umbrella_legend_handles_kwargs)
    legend_handles += extra_legend_handles

    legend = ax.legend(
        handles=legend_handles,
        **legend_kwargs,
    )

    set_umbrella_legend_text_colors(legend)

    return legend


def clean_taxonomy_string(string):
    if not isinstance(string, str):
        return ''

    string = string.replace(' and ', ' & ')

    string = string.replace(', General', '')

    return string


def main_text_taxonomy_string(string):
    if not isinstance(string, str):
        return ''

    if string == 'Theatre Literature, History and Criticism':
        string = 'Theatre'
    elif string == 'English Language and Literature':
        string = 'English'
    elif string == 'Classics and Classical Languages':
        string = 'Classics'
    elif string == 'Art History and Criticism':
        string = 'Art History'
    elif string == 'Spanish Language and Literature':
        string = 'Spanish'
    elif string == 'Urban and Regional Planning':
        string = 'Urban Planning'

    return clean_taxonomy_string(string)


def format_umbrella_string(string):
    string = clean_taxonomy_string(string)
    string = string.replace(' ', '\n')
    return string


def break_long_string_with_newline(string, threshold=19):
    if len(string) > threshold:
        words = string.split()
        new_string = ''
        while words:
            word = words.pop(0)
            spacer = ' ' if len(new_string) + len(word) + 1 <= threshold else '\n'
            new_string += spacer + word

        string = new_string.strip()

    return string


def annotate_rectangle_center(ax, text, x, y, width, height, annotate_kwargs={}):
    mid_x = x + (width / 2)
    mid_y = y + (height / 2)
    ax.annotate(
        text,
        xy=(mid_x, mid_y),
        ha='center',
        va='center',
        **annotate_kwargs,
    )


def get_coords_at_axis_fraction(ax, xy_loc=(.1, .9)):
    x_fraction, y_fraction = xy_loc
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    x = x_min + x_fraction * (x_max - x_min)
    y = y_min + y_fraction * (y_max - y_min)
    return (x, y)



################################################################################
# Null model stuff
################################################################################
def scatter_with_fill_by_null_hierarchy_threshold(ax, df, x_column, y_column, threshold=1):
    unfilled_df = df[df['MoreHierarchicalCount'] > threshold]
    ax.scatter(
        unfilled_df[x_column],
        unfilled_df[y_column],
        facecolors='none',
        edgecolors=unfilled_df['UmbrellaColor'],
        alpha=.5,
        zorder=2,
    )

    filled_df = df[df['MoreHierarchicalCount'] <= threshold]
    ax.scatter(
        filled_df[x_column],
        filled_df[y_column],
        facecolors=filled_df['UmbrellaColor'],
        edgecolors=filled_df['UmbrellaColor'],
        alpha=.5,
        zorder=2,
    )


def get_legend_handles_for_scatter_filled_by_threshold(threshold=1):
    return [
        Line2D(
            [0], [0],
            marker='o',
            color='none',
            markerfacecolor=PLOT_VARS['colors']['dark_gray'],
            markeredgecolor=PLOT_VARS['colors']['dark_gray'],
            # markersize=10,
            label=f'null < empirical, > {100 - threshold}/100',
        ),
        Line2D(
            [0], [0],
            marker='o',
            color='none',
            # markersize=10,
            markerfacecolor='none',
            markeredgecolor=PLOT_VARS['colors']['dark_gray'],
            label=f'null < empirical, <= {100 - threshold}/100',
        )
    ]


def get_academia_umbrella_grid():
    umbrellas = usfhn.views.get_umbrellas()
    return [
        ['Academia'] + umbrellas[:4],
        [None] + umbrellas[4:],
    ]

    1

def tick_years(ax):
    ax.set_xticks([2012, 2014, 2016, 2018, 2020])


def add_gridlines_and_annotate_rank_direction(
    ax,
    rank_type='production',
    x_gridlines_to_break=[],
    annotation_arrow_x=.01,
    annotation_text_x=.12,
    break_height=.0075,
    fontsize=hnelib.plot.FONTSIZES['annotation'],
):
    if not x_gridlines_to_break:
        x_gridlines_to_break = [.2, .4] if rank_type == 'production' else [.2]

    text = 'more faculty produced' if rank_type == 'production' else "more prestigious"

    hnelib.plot.add_gridlines(
        ax,
        xs=[x for x in ax.get_xticks() if x not in x_gridlines_to_break],
        ys=ax.get_yticks(),
    )

    annotation_y = break_height / 2

    ax.annotate(
        "",
        xy=(annotation_arrow_x, annotation_y),
        xytext=(annotation_text_x, annotation_y),
        arrowprops=hnelib.plot.BASIC_ARROW_PROPS,
    )

    ax.annotate(
        text,
        xy=(annotation_text_x, annotation_y),
        ha='left',
        va='center',
        fontsize=fontsize,
    )

    for x in x_gridlines_to_break:
        ax.plot(
            [x, x],
            [break_height, ax.get_ylim()[1]],
            lw=.5,
            alpha=.5,
            color=PLOT_VARS['colors']['dark_gray'],
        )

def annotate_color(df, add_umbrella=False):
    """
    expects df to have columns:
    - TaxonomyLevel
    - TaxonomyValue
    """
    taxonomy = usfhn.views.get_taxonomization()

    cols = list(df.columns)

    new_dfs = []
    for level, rows in df.groupby('TaxonomyLevel'):
        rows = rows.copy()

        if level == 'Academia':
            rows['Umbrella'] = 'Academia'
        elif level == 'Umbrella':
            rows['Umbrella'] = rows['TaxonomyValue']
        else:
            _tx_df = taxonomy.copy()[
                [
                    level,
                    'Umbrella',
                ]
            ].drop_duplicates().rename(columns={
                level: 'TaxonomyValue',
            })

            rows = rows.merge(
                _tx_df,
                on='TaxonomyValue',
            )

        rows['Color'] = rows['Umbrella'].apply(PLOT_VARS['colors']['umbrellas'].get)
        new_dfs.append(rows)

    new_cols = ['Color']

    if add_umbrella:
        new_cols.append('Umbrella')

    new_df = pd.concat(new_dfs)
    new_df = new_df[
        cols + new_cols
    ]

    return new_df
