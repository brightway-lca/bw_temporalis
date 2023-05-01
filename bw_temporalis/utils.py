import importlib.metadata
import math
from typing import Union

import bw2data as bd
import pandas as pd
from tqdm import tqdm
import numpy as np
import numpy.typing as npt

from .temporal_distribution import TemporalDistribution


class IncongruentDistribution(Exception):
    """The sum of `TemporalDistribution` values is different than the exchange"""

    pass


def _array(start: int, end: int, steps: int, mode: str, param: str | None) -> npt.NDArray[int]:
    if mode == "uniform":
        return np.linspace(start, end, steps).astype("datetime64[s]")
    elif mode == "triangular":
        pass
    elif mode == "normal":
        pass


def easy_datetime_array(start: str, end: str, total: float, steps: int, mode: str | None = "Uniform", param: str | None = None) -> TemporalDistribution:
    if mode not in ("uniform", "triangular", "normal"):
        raise ValueError("Mode must be one of uniform, triangular, or normal")

    start = np.array(start, dtype="datetime64[s]").astype(int)
    end = np.array(end, dtype="datetime64[s]").astype(int)
    amount = np.ones(steps) / steps * total
    date = _array(start, end, steps, mode, param)
    return TemporalDistribution(date=date, amount=amount)


def check_database_exchanges(database_label: str) -> None:
    """Check the sum of an exchange ``TemporalDistribution.amount`` is close to its ``amount`` value.

    Raises
    ------
    IncongruentDistribution
        If the two values are more than 1 percent different

    Parameters
    ----------
    database_label : str
        Name of database to check

    """
    MESSAGE = """
    Temporal distribution in exchange differs from `amount`:
    Input:
        {inp}
        id: {inp_id}
    Output
        {outp}
        id: {outp_id}
    Exchange amount: {exc_amount:.4e}
    Temporal distribution amount: {td_amount:.4e}
    """
    for ds in tqdm(bd.Database(database_label)):
        for exc in ds.exchanges():
            # Can't really evaluate callable temporal distributions, they are
            # responsible for summing to the exchange amount on their own.
            if isinstance(exc.get("temporal_distribution"), TemporalDistribution):
                a, b = exc["amount"], exc["temporal_distribution"].amount.sum()
                if not math.isclose(a, b, rel_tol=0.01):
                    raise IncongruentDistribution(
                        MESSAGE.format(
                            inp=exc.input,
                            inp_id=exc.input.id,
                            outp=exc.output,
                            outp_id=exc.output.id,
                            exc_amount=a,
                            td_amount=b,
                        )
                    )


def supplement_dataframe(
    df: pd.DataFrame,
    database_label: str,
    columns: list[str] | None = ["name", "location", "unit", "categories"],
) -> pd.DataFrame:
    database_df = bd.Database(database_label).nodes_to_dataframe()
    database_df["activity"] = database_df["flow"] = database_df["id"]
    df = df.join(
        database_df.copy()
        .drop(columns=[c for c in database_df.columns if c not in (["flow"] + columns)])
        .set_index("flow"),
        on="flow",
        validate="m:1",
        rsuffix="_flow",
    )
    df = df.join(
        database_df.copy()
        .drop(
            columns=[
                c for c in database_df.columns if c not in (["activity"] + columns)
            ]
        )
        .set_index("activity"),
        on="activity",
        validate="m:1",
        rsuffix="_activity",
    )
    return df.reset_index(drop=True)


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
