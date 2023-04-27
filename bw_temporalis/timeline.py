from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .temporal_distribution import TemporalDistribution


@dataclass
class FlowTD:
    """
    Class for storing a temporal distribution associated with a flow and activity.

    Attributes
    ----------
    distribution : TemporalDistribution
    flow : int
    activity : int

    See Also
    --------
    bw_temporalis.temporal_distribution.TemporalDistribution: A container for a series of values spread over time.
    """

    distribution: TemporalDistribution
    flow: int
    activity: int


class Timeline:
    """
    Sum and group elements over time.
    Timeline calculations produce a list of [(datetime, amount)] tuples.

    Attributes
    ----------
    self.data : list[FlowTD]
    """

    def __init__(self, data: list[FlowTD] | None = None):
        self.data = data or []

    def add_flow_temporal_distribution(
        self, td: TemporalDistribution, flow: int, activity: int
    ) -> None:
        """
        Append a TemporalDistribution object to the Timeline.data object.

        Parameters
        ----------
        td : TemporalDistribution
            Temporal distribution to add.
        flow : int
            Associated flow.
        activity : int
            Associated activity.

        See Also
        --------
        bw_temporalis.temporal_distribution.TemporalDistribution: A container for a series of values spread over time.
        """
        self.data.append(
            FlowTD(distribution=td.nonzero(), flow=flow, activity=activity)
        )

    def build_dataframe(self) -> None:
        """
        Build a Pandas DataFrame from the Timeline.data object and store it as a Timeline.pd object.

        Returns
        -------
        None, creates class attribute Pandas DataFrame `df` with the following columns:
        - times: datetime64[D]
        - values: float64
        - flows: int
        - activities: int
        """
        times = np.hstack([o.distribution.times for o in self.data])
        values = np.hstack([o.distribution.values for o in self.data])
        flows = np.hstack(
            [o.flow * np.ones(len(o.distribution)) for o in self.data]
        )
        activities = np.hstack(
            [o.activity * np.ones(len(o.distribution)) for o in self.data]
        )

        self.df = pd.DataFrame(
            {
                "times": pd.Series(data=times.astype("datetime64[s]"), dtype="datetime64[s]"),
                "values": pd.Series(data=values, dtype="float64"),
                "flows": pd.Series(data=flows, dtype="int"),
                "activities": pd.Series(data=activities, dtype="int"),
            }
        )
        self.df.sort_values(by="times", ascending=True, inplace=True)

    def characterize_dataframe(
        self,
        characterization_function: Callable,
        activities: set[int] | None = None,
        flows: set[int] | None = None,
        cumsum: bool | None = True
    ) -> None:
        """
        Applies a characterization function to a Timeline Pandas DataFrame.

        An input Timeline of the form

        | times | values | flows | activities |
        |-------|--------|-------|------------|
        | 101   | 33     | 1     | 2          |
        | 312   | 21     | 4     | 2          |

        is transformed into

        | times | values | flows | activities |
        |-------|--------|-------|------------|
        | 101   | 33     | 1     | 2          |
        | 102   | 32     | 1     | 2          |
        | 103   | 31     | 1     | 2          |
        | 312   | 21     | 4     | 2          |
        | 313   | 20     | 4     | 2          |
        | 314   | 19     | 4     | 2          |

        Each row of the input Timeline corresponds to a single day (`times`) and the associated value (`values`).
        The `characterization_function` is applied to each row of the input Timeline for a given `period` of days.
        The new rows are appended to the Timeline Pandas DataFrame.

        Parameters
        ----------
        characterization_function : Callable
            Characterization function to apply to the values Timeline Pandas DataFrame.
        period : int
            Period in days.
        activity : int
        flow : int
        """
        df = self.df.copy()
        if activities:
            df = df.loc[self.df["activities"].isin(activities)]
        if flows:
            df = df.loc[self.df["flows"].isin(flows)]
        df.reset_index(drop=True, inplace=True)
        result_df = pd.concat([characterization_function(row) for _, row in df.iterrows()])
        if 'times' in result_df.columns:
            result_df.sort_values(by="times", ascending=True, inplace=True)
        if cumsum and "values" in result_df:
            result_df["values_sum"] = result_df["values"].cumsum()
        return result_df

    def sum_days_to_years(self) -> None:
        """
        Sums the day-resolution `values` of the Timeline Pandas DataFrame to years.

        An input Timeline of the form

        | times | values | flows | activities |
        |-------|--------|-------|------------|
        | 101   | 33     | 1     | 2          |
        | 102   | 32     | 1     | 2          |
        | 103   | 31     | 1     | 2          |
        | 412   | 21     | 4     | 2          |
        | 413   | 20     | 4     | 2          |
        | 514   | 19     | 4     | 2          |

        is transformed into

        | year | values | flows | activities |
        |-------|--------|-------|------------|
        | 1     | 96     | 1     | 2          |
        | 2     | 60     | 4     | 2          |

        Returns
        -------
        None, modifies the Timeline Pandas DataFrame `df` in place.
        """

        _df_grouped = df.groupby([df['times'].dt.year]).agg(
            {
                'values': 'sum',
                'flows': 'first',
                'activities': 'first'
            }
        ).reset_index()

        _df_grouped.rename(columns={'times': 'year'})
        self.df = _df_grouped
