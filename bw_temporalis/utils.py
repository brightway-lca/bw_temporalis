import importlib.metadata
from typing import Union

import bw2data as bd
import pandas as pd


def get_version_tuple() -> tuple:
    def as_integer(x: str) -> Union[int, str]:
        try:
            return int(x)
        except ValueError:
            return x

    return tuple(
        as_integer(v)
        for v in importlib.metadata.version("bw_temporalis").strip().split(".")
    )


def supplement_dataframe(df: pd.DataFrame, database_label: str, columns: list[str] | None = ["name", "location", "unit", "categories"]):
    database_df = bd.Database(database_label).nodes_to_dataframe()
    database_df['activity'] = database_df['flow'] = database_df['id']
    df = df.join(
        database_df.copy().drop(
            columns=[c for c in database_df.columns if c not in (["flow"] + columns)]
        ).set_index('flow'),
        on='flow',
        validate='m:1',
        rsuffix="_flow"
    )
    df = df.join(
        database_df.copy().drop(
            columns=[c for c in database_df.columns if c not in (["activity"] + columns)]
        ).set_index('activity'),
        on='activity',
        validate='m:1',
        rsuffix="_activity"
    )
    return df.reset_index(drop=True)
