from dataclasses import dataclass
from typing import Callable, List

import bw2data as bd
import numpy as np
import pandas as pd

from .temporal_distribution import TemporalDistribution


class EmptyTimeline(Exception):
    """Operation on empty timeline"""

    pass


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


@dataclass
class NodeTD:
    """
    Class for storing a temporal distribution associated only with an activity.

    Attributes
    ----------
    distribution : TemporalDistribution
    flow : int. Only included for compatibility with `FlowTD`. Always -1.
    activity : int
    num_flows : int. Number of biosphere flow edges from this node.
    num_flows_td : int. Number of biosphere flow edges from this node with temporal distributions.

    See Also
    --------
    bw_temporalis.temporal_distribution.TemporalDistribution: A container for a series of values spread over time.
    """

    distribution: TemporalDistribution
    flow: int
    activity: int
    num_flows: int
    num_flows_td: int


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

    def add_node_temporal_distribution(
        self, td: TemporalDistribution, activity: int, num_flows: int, num_flows_td: int
    ) -> None:
        """
        Append a TemporalDistribution object to the Timeline.data object.

        Parameters
        ----------
        td : TemporalDistribution
            Temporal distribution to add.
        activity : int
            Associated activity.
        num_flows : int
            Number of biosphere flow edges from this node.
        num_flows_td : int
            Number of biosphere flow edges from this node with temporal distributions.

        See Also
        --------
        bw_temporalis.temporal_distribution.TemporalDistribution: A container for a series of values spread over time.
        """
        self.data.append(
            NodeTD(
                distribution=td.nonzero(),
                flow=-1,
                activity=activity,
                num_flows=num_flows,
                num_flows_td=num_flows_td,
            )
        )

    def __len__(self):
        return len(self.data)

    def build_dataframe(self) -> None:
        """
        Build a Pandas DataFrame from the Timeline.data object and store it as a Timeline.pd object.

        Returns
        -------
        None, creates class attribute Pandas DataFrame `df` with the following columns:
        - date: datetime64[s]
        - amount: float64
        - flow: int
        - activity: int
        """
        if not len(self.data):
            raise EmptyTimeline("No `FlowTD` elements present")

        date = np.hstack([o.distribution.date for o in self.data])

        # Not really testable; `TemporalDistribution` will raise an error with an
        # empty array. But our users are creative...
        if not len(date):
            raise EmptyTimeline(
                "This timeline is empty; element: {}".format(
                    [len(x) for x in self.data]
                )
            )

        amount = np.hstack([o.distribution.amount for o in self.data])
        flow = np.hstack([o.flow * np.ones(len(o.distribution)) for o in self.data])
        activity = np.hstack(
            [o.activity * np.ones(len(o.distribution)) for o in self.data]
        )

        self.df = pd.DataFrame(
            {
                "date": pd.Series(
                    data=date.astype("datetime64[s]"), dtype="datetime64[s]"
                ),
                "amount": pd.Series(data=amount, dtype="float64"),
                "flow": pd.Series(data=flow, dtype="int"),
                "activity": pd.Series(data=activity, dtype="int"),
            }
        )
        self.df.sort_values(by="date", ascending=True, inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        return self.df

    def characterize_dataframe(
        self,
        characterization_function: Callable,
        flow: set[int] | None = None,
        activity: set[int] | None = None,
        cumsum: bool | None = True,
    ) -> pd.DataFrame:
        """
        Applies a characterization function to a Timeline Pandas DataFrame.

        The characterization function is expected to take a row from the input Timeline of the form

        | date | amount | flow | activity |
        |-------|-------|------|----------|
        | 101   | 33    | 1    | 2        |
        | 312   | 21    | 4    | 2        |

        and transform it for a given time period. The output for a very simple function could look like:

        | date | amount | flow | activity |
        |------|--------|------|----------|
        | 101  | 33     | 1    | 2        |
        | 102  | 31     | 1    | 2        |
        | 103  | 31     | 1    | 2        |
        | 312  | 21     | 4    | 2        |
        | 313  | 20     | 4    | 2        |
        | 314  | 19     | 4    | 2        |

        Each row of the input Timeline corresponds to a single day (`date`) and the associated value (`amount`).
        The `characterization_function` is applied to each row of the input Timeline for a given `period` of days.
        The new rows are appended to the Timeline Pandas DataFrame.

        Parameters
        ----------
        characterization_function : Callable
            Characterization function to apply to the values Timeline Pandas DataFrame.
        period : int
            Period in days.
        flow : int
        activity : int

        Returns
        -------
        A Pandas DataFrame with the following columns:
        - date: datetime64[s]
        - amount: float64
        - flow: int
        - activity: int

        """
        if not hasattr(self, "df"):
            raise ValueError("Call `.build_dataframe()` first")

        df = self.df.copy()
        if activity:
            df = df.loc[self.df["activity"].isin(activity)]
        if flow:
            df = df.loc[self.df["flow"].isin(flow)]
        df.reset_index(drop=True, inplace=True)
        result_df = pd.DataFrame(
            [characterization_function(row) for row in df.itertuples(index=False)]
        )
        if "date" in result_df.columns:
            result_df = (
                result_df.explode(["amount", "date"])
                .astype({"date": "datetime64[s]", "amount": "float64"})
                .sort_values(by=["date", "amount"])
                .reset_index(drop=True)
            )
        if cumsum and "amount" in result_df:
            result_df["amount_sum"] = result_df["amount"].cumsum()
        return result_df

    def sum_days_to_years(self) -> pd.DataFrame:
        """
        Sums the day-resolution `amount` of the Timeline Pandas DataFrame to years.

        An input Timeline of the form

        | date | amount | flow | activity |
        |------|--------|------|----------|
        | 101  | 33     | 1    | 2        |
        | 102  | 32     | 1    | 2        |
        | 103  | 31     | 1    | 2        |
        | 412  | 21     | 4    | 2        |
        | 413  | 20     | 4    | 2        |
        | 514  | 19     | 4    | 2        |

        is transformed into

        | year | amount | flow | activity |
        |------|--------|------|----------|
        | 1    | 96     | 1    | 2        |
        | 2    | 60     | 4    | 2        |

        Returns
        -------
        A Pandas DataFrame with the following columns:
        - year: int
        - amount: float64
        - flow: int
        - activity: int
        """
        if not hasattr(self, "df"):
            raise ValueError("Call `.build_dataframe()` first")

        result_df = (
            self.df.groupby([self.df["date"].dt.year])
            .agg({"amount": "sum", "flow": "first", "activity": "first"})
            .reset_index()
        )

        result_df.rename(columns={"date": "year"})

        return result_df

    def add_metadata_to_dataframe(
        self,
        database_labels: list[str],
        fields: List[str] = ["name", "unit", "location", "categories"],
    ) -> pd.DataFrame:
        """
        Add additional columns with metadata to the dataframe. Returns a new dataframe.

        Parameters
        ----------
        database_labels : list[str]
            List of all databases to load and add metadata from
        fields : list[str]
            Metadata fields to add.

        """
        if not hasattr(self, "df"):
            raise ValueError("Call `.build_dataframe()` first")

        db = pd.concat(
            [bd.Database(label).nodes_to_dataframe() for label in database_labels]
        )
        db.drop(
            axis=1,
            columns=[label for label in db.columns if label not in ["id"] + fields],
            inplace=True,
        )
        process_db = db.rename(
            columns={field: "{}_{}".format("activity", field) for field in fields},
        )
        df = self.df.merge(
            process_db, how="left", left_on="activity", right_on="id", validate="m:1"
        )
        df.drop(
            axis=1,
            columns=["id"],
            inplace=True,
        )
        flow_db = db.rename(
            columns={field: "{}_{}".format("flow", field) for field in fields},
        )
        df = df.merge(
            flow_db, how="left", left_on="flow", right_on="id", validate="m:1"
        )
        df.drop(
            axis=1,
            columns=["id"],
            inplace=True,
        )
        return df
