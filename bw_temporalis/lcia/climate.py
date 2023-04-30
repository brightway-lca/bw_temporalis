import numpy as np
import pandas as pd


def characterize_co2(
    series,
    period: int | None = 100,
    cumulative: bool | None = False,
) -> pd.DataFrame:
    """
    Calculate the cumulative or marginal radiative forcing (CRF) from CO2 for each year in a given period.

    If `cumulative` is True, the cumulative CRF is calculated. If `cumulative` is False, the marginal CRF is calculated.
    Takes a single row of the TimeSeries Pandas DataFrame (corresponding to a set of (`date`/`amount`/`flow`/`activity`).
    For each year in the given period, the CRF is calculated.
    Units are watts/square meter/kilogram of CO2.

    Returns
    -------
    A TimeSeries dataframe with the following columns:
    - date: datetime64[s]
    - amount: float
    - flow: str
    - activity: str

    Notes
    -----
    See also the relevant scientific publication on CRF: https://doi.org/10.5194/acp-13-2793-2013
    See also the relevant scientific publication on the numerical calculation of CRF: http://pubs.acs.org/doi/abs/10.1021/acs.est.5b01118

    See Also
    --------
    characterize_methane: The same function for CH4
    """

    # functional variables and units (from publications listed in docstring)
    RE = 1.76e-15  # Radiative forcing (W/m2/kg)
    alpha_0, alpha_1, alpha_2, alpha_3 = 0.2173, 0.2240, 0.2824, 0.2763
    tau_1, tau_2, tau_3 = 394.4, 36.54, 4.304
    decay_term = lambda year, alpha, tau: alpha * tau * (1 - np.exp(-year / tau))

    date_beginning: np.datetime64 = series["date"].to_numpy()
    date_characterized: np.ndarray = date_beginning + np.arange(
        start=0, stop=period, dtype="timedelta64[Y]"
    ).astype("timedelta64[s]")

    decay_multipliers: np.ndarray = np.array(
        [
            RE
            * (
                alpha_0 * year
                + decay_term(year, alpha_1, tau_1)
                + decay_term(year, alpha_2, tau_2)
                + decay_term(year, alpha_3, tau_3)
            )
            for year in range(period)
        ]
    )

    forcing = pd.Series(data=series.amount * decay_multipliers, dtype="float64")
    if not cumulative:
        forcing = forcing.diff(periods=1).fillna(0)

    return pd.DataFrame(
        {
            "date": pd.Series(data=date_characterized, dtype="datetime64[s]"),
            "amount": forcing,
            "flow": series.flow,
            "activity": series.activity,
        }
    )


def characterize_methane(series, period: int = 100, cumulative=False) -> pd.DataFrame:
    """
    Calculate the cumulative or marginal radiative forcing (CRF) from CH4 for each year in a given period.

    If `cumulative` is True, the cumulative CRF is calculated. If `cumulative` is False, the marginal CRF is calculated.
    Takes a single row of the TimeSeries Pandas DataFrame (corresponding to a set of (`date`/`amount`/`flow`/`activity`).
    For earch year in the given period, the CRF is calculated.
    Units are watts/square meter/kilogram of CH4.

    Parameters
    ----------
    series : array-like
        A single row of the TimeSeries dataframe.
    period : int, optional
        Time period for calculation (number of years), by default 100
    cumulative : bool, optional
        Should the RF amounts be summed over time?

    Returns
    -------
    A TimeSeries dataframe with the following columns:
    - date: datetime64[s]
    - amount: float
    - flow: str
    - activity: str

    Notes
    -----
    See also the relevant scientific publication on CRF: https://doi.org/10.5194/acp-13-2793-2013
    See also the relevant scientific publication on the numerical calculation of CRF: http://pubs.acs.org/doi/abs/10.1021/acs.est.5b01118

    See Also
    --------
    characterize_co2: The same function for CO2
    """

    # functional variables and units (from publications listed in docstring)
    f1 = 0.5  # Unitless
    f2 = 0.15  # Unitless
    alpha = 1.27e-13  # Radiative forcing (W/m2/kg)
    tau = 12.4  # Lifetime (years)

    date_beginning: np.datetime64 = series["date"].to_numpy()
    date_characterized: np.ndarray = date_beginning + np.arange(
        start=0, stop=period, dtype="timedelta64[Y]"
    ).astype("timedelta64[s]")

    decay_multipliers: list = np.array(
        [
            (1 + f1 + f2) * alpha * tau * (1 - np.exp(-year / tau))
            for year in range(period)
        ]
    )

    forcing = pd.Series(data=series.amount * decay_multipliers, dtype="float64")
    if not cumulative:
        forcing = forcing.diff(periods=1).fillna(0)

    return pd.DataFrame(
        {
            "date": pd.Series(data=date_characterized, dtype="datetime64[s]"),
            "amount": forcing,
            "flow": series.flow,
            "activity": series.activity,
        }
    )
