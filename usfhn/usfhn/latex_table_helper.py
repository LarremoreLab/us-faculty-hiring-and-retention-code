import numpy as np


def style_and_save_df_to_path(df, path, column_alignments=[], escape_strings=True):
    """
    adds some lines that I like to tables

    also does some nice little alignment magic on columns
    """
    if not column_alignments:
        column_alignments = ['l' if dtype == np.object else 'r' for c, dtype in df.dtypes.items()]

    if escape_strings:
        for col, dtype in df.dtypes.items():
            if dtype == 'object':
                df[col] = df[col].apply(escape_string)

        df = df.rename(columns={c: escape_string(c) for c in df.columns})

    header = "|" + "|".join(column_alignments) + "|"

    table = df.style.hide(axis='index').to_latex(
        column_format=header,
        hrules=True,
    )

    table_lines = table.splitlines()

    for i, line in enumerate(table_lines):
        if line in ['\\toprule', '\midrule', '\\bottomrule']:
            table_lines[i] = f"{line}\n\\hline"

    table = "\n".join(table_lines)

    path.write_text(table)


def escape_string(string):
    string = string.replace('%', '\%')
    string = string.replace('&', '\&')
    string = string.replace('#', '\#')

    return string
