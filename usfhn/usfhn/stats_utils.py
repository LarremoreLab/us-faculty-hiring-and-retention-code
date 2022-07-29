import pandas as pd
from statsmodels.stats.multitest import multipletests

def correct_multiple_hypotheses(
    df,
    alpha=.05,
    p_col='P',
    corrected_p_col='PCorrected',
    method='fdr_bh',
    significance_col='Significant',
    add_significant_col=False,
):
    corrected_dfs = []
    for _, rows in df.groupby('TaxonomyLevel'):
        rows = rows.copy()

        rows = rows[
            [
                'TaxonomyLevel',
                'TaxonomyValue',
                p_col,
            ]
        ].drop_duplicates()

        p_vals = list(rows[p_col])

        rows = rows.drop(columns=[p_col])

        rejects, corrected_p_vals, _, _ = multipletests(
            pvals=p_vals,
            alpha=alpha,
            method=method,
        )

        rows[corrected_p_col] = corrected_p_vals

        if add_significant_col:
            rows[significance_col] = [not r for r in rejects]

        corrected_dfs.append(rows)

    if add_significant_col and significance_col in df.columns:
        df = df.drop(columns=[significance_col])

    if corrected_p_col in df.columns:
        df = df.drop(columns=[corrected_p_col])

    df = df.merge(
        pd.concat(corrected_dfs),
        on=[
            'TaxonomyLevel',
            'TaxonomyValue',
        ],
    )

    return df
