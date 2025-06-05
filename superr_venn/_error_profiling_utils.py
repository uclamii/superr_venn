import pandas as pd


def compute_sets_from_df(df, id=None, columns=None, incorrect_value=0):

    sets = []
    universe = set()

    if id is None:
        # assume index is the id
        df = df.reset_index()
        id = df.columns[0]
    universe = set(df[id])

    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

        incorrect_mask = df[col] == incorrect_value
        incorrect = df[incorrect_mask][id]
        sets.append(set(incorrect))
    return sets, universe
