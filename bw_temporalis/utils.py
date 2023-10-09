import importlib.metadata
import math
import warnings
from numbers import Number
from typing import Union

import bw2data as bd
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import norm, triang
from tqdm import tqdm

from .temporal_distribution import TemporalDistribution


class IncongruentDistribution(Exception):
    """The sum of `TemporalDistribution` values is different than the exchange"""

    pass


def normalized_data_array(
    steps: int, kind: str, param: float | None
) -> npt.NDArray[int]:
    if kind == "uniform":
        return np.ones(steps)
    elif kind == "triangular":
        # Zero probability at bounds, so add sacrificial bounds
        try:
            c = float(param) if param is not None else 0.5
        except ValueError:
            raise ValueError(
                f"Couldn't convert triangular mode parameter `c` '{param}' to number"
            )
        if not 0 <= c <= 1:
            raise ValueError(f"`c` must be in (0, 1); got {c}")
        return triang.pdf(
            np.linspace(0, 1, steps),
            c=c,
        )
    elif kind == "normal":
        if not (isinstance(param, Number) and param > 0):
            raise ValueError(
                "Numerical standard deviation (`param`) greater than zero required for Normal distribution"
            )
        return norm.pdf(np.linspace(-0.5, 0.5, steps), scale=param)
    else:
        raise ValueError(f"Unrecognized array kind {kind}")


def easy_datetime_distribution(
    start: str,
    end: str,
    steps: int | None = 50,
    kind: str | None = "uniform",
    param: float | None = None,
) -> TemporalDistribution:
    """Generate a datetime `TemporalDistribution` with a few input parameters.

    Can generate distributions whose `amount` values are uniformly,
    triangularly, or normally distributed. Please build more complicated
    distributions manually.

    Only the `amount` values are distributed, the resulting distribution
    `date` values are uniformly spaced from `start` to `end`.

    For triangular distributions, `param` is the mode (optional), and should
    be given in the same reference system as `start` and `stop`. The `param`
    value should be in the same format as `start` and `end`, e.g.
    "2023-01-01".

    For normal distributions, `param` is the standard deviation in relation
    to a standardized distribution with mu = 0. `param` is not used for the uniform distribution.

    Raises
    ------
    ValueError
        If the input parameters prevent construction of valide `TemporalDistribution`.

    Parameters
    ----------
    start : str
        Datetime marking the start (inclusive) of the distribution, e.g. "now", "2023-02-01", "2023-03-02T12:34:56"
    end : str
        Datetime marking the end (inclusive) of the distribution, e.g. "now", "2023-02-01", "2023-03-02T12:34:56"
    steps : int, optional
        Number of values in discrete distribution. Normally not more than 50 or 100.
    kind : str, optional
        Distribution type. Must be one of "uniform", "triangular", or "normal"
    param : float, optional
        Input parameter to define triangular or normal distribution

    Returns
    -------
    A `TemporalDistribution` instance.

    """
    # Check carefully as new users will do interesting things
    if not isinstance(steps, int) or not steps > 1:
        raise ValueError(
            f"`steps` must be a positive number greater than one; got {steps}"
        )

    start = np.array(start, dtype="datetime64[s]").astype(int)
    end = np.array(end, dtype="datetime64[s]").astype(int)

    if start >= end:
        raise ValueError(f"Start value is later than end: {start}, {end}")

    if kind == "triangular" and steps < 3:
        raise ValueError("Triangular distribution must have at least three steps")
    elif kind == "triangular" and param is not None:
        # Normalize to (0, 1) interval
        param = (np.array(param, dtype="datetime64[s]").astype(int) - start) / (
            end - start
        )
        if not 0 <= param <= 1:
            raise ValueError("Triangular mode is outside (start, end) bounds")

    date = np.linspace(start, end, steps).astype("datetime64[s]")
    amount = normalized_data_array(steps, kind, param)

    # Could get NaN or Inf with strange `param` values
    mask = np.isreal(amount)
    amount, date = amount[mask], date[mask]

    # Normalize after removing possible NaN/Inf
    amount *= 1 / amount.sum()
    return TemporalDistribution(date=date, amount=amount)


def easy_timedelta_distribution(
    start: int,
    end: int,
    resolution: str,
    steps: int | None = 50,
    kind: str | None = "uniform",
    param: float | None = None,
) -> TemporalDistribution:
    """Generate a timedelta `TemporalDistribution` with a few input parameters.

    Can generate distributions whose `amount` values are uniformly, triangularly, or normally distributed. Please build more complicated distributions manually.

    Only the `amount` values are distributed, the resulting distribution `date` values are uniformly spaced from `start` to `end`.

    For triangular distributions, `param` is the mode (optional). For lognormal distributions, `param` is the standard deviation (required). `param` is not used for the uniform distribution.

    Raises
    ------
    ValueError
        If the input parameters prevent construction of valide `TemporalDistribution`.

    Parameters
    ----------
    start : int
        Start (inclusive) of the distribution in `resolution` units
    end : int
        End (inclusive) of the distribution in `resolution` units
    resolution : str
        Resolution of the created `timedelta64` array. One of `Y` (year), `M` (month), `D` (day), `h` (hour), `m` (minute), `s` (second)
    steps : int, optional
        Number of values in discrete distribution. Normally not more than 50 or 100.
    kind : str, optional
        Distribution type. Must be one of "uniform", "triangular", or "normal"
    param : float, optional
        Input parameter to define triangular or normal distribution

    Returns
    -------
    A `TemporalDistribution` instance.

    """
    # Check carefully as new users will do interesting things
    if not isinstance(steps, int) or not steps > 1:
        raise ValueError(
            f"`steps` must be a positive number greater than one; got {steps}"
        )
    if resolution not in "YMDhms":
        raise ValueError(f"Invalid temporal resolution {resolution}")
    if start >= end:
        raise ValueError(f"Start value is later than end: {start}, {end}")

    if kind == "triangular" and steps < 3:
        raise ValueError("Triangular distribution must have at least three steps")
    elif kind == "triangular" and param is not None:
        # Normalize to (0, 1) interval
        param = (param - start) / (end - start)
        if not 0 <= param <= 1:
            raise ValueError("Triangular mode is outside (start, end) bounds")

    if steps > (end - start + 1):
        MESSAGE = f"""More steps than discrete possibilities ({steps} versus {end - start + 1}).
    Values will be duplicated due to rounding."""
        warnings.warn(MESSAGE)

    date = np.array(np.linspace(start, end, steps), dtype=f"timedelta64[{resolution}]")
    amount = normalized_data_array(steps, kind, param)

    # Could get NaN or Inf with strange `param` values
    mask = np.isreal(amount)
    amount, date = amount[mask], date[mask]

    # Normalize after removing possible NaN/Inf
    amount *= 1 / amount.sum()
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
    Temporal distribution in exchange doesn't sum to one:
    Input:
        {inp}
        id: {inp_id}
    Output
        {outp}
        id: {outp_id}
    Exchange amount: {exc_amount:.4e}
    Temporal distribution sum: {td_amount:.4e}
    """
    for ds in tqdm(bd.Database(database_label)):
        for exc in ds.exchanges():
            # Can't really evaluate callable temporal distributions, they are
            # responsible for summing to the exchange amount on their own.
            if isinstance(exc.get("temporal_distribution"), TemporalDistribution):
                a, b = exc["amount"], exc["temporal_distribution"].amount.sum()
                if not math.isclose(1, b, rel_tol=0.01):
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
