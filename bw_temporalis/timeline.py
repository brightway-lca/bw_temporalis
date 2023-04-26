from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from bw_temporalis.temporal_distribution import TemporalDistribution


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
    self.characterized : TODO
    self.finalized : bool
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
        self.df = pd.DataFrame(
            {
                "times": pd.Series(data=collection_times_new, dtype="datetime64[s]"),
                "values": pd.Series(data=collection_values_new, dtype="float64"),
                "flows": pd.Series(data=collection_flows_new, dtype="int"),
                "activities": pd.Series(data=collection_activities_new, dtype="int"),
            }
        )
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

    def convert_dataframe_to_years(self):
        raise NotImplementedError
