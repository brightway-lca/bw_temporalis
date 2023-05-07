import numpy as np
import pandas as pd
import pytest

from bw_temporalis.temporal_distribution import TemporalDistribution
from bw_temporalis.timeline import EmptyTimeline, Timeline


def test_empty_timeline_build_dataframe_missing():
    tl = Timeline()
    with pytest.raises(EmptyTimeline):
        tl.build_dataframe()


@pytest.mark.skip("Empty `TemporalDistribution` will raise an error")
def test_empty_timeline_build_dataframe_blank_tds():
    empty_temp_dist = TemporalDistribution(
        date=np.array([], dtype="datetime64[D]"), amount=np.array([])
    )
    tl = Timeline()
    tl.add_flow_temporal_distribution(empty_temp_dist, 7, 11)
    tl.add_flow_temporal_distribution(empty_temp_dist, 3, 4)
    with pytest.raises(EmptyTimeline):
        tl.build_dataframe()


def define_dataframes() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    df_input:

    | date       | amount | flow | activity |
    |------------|--------|------|----------|
    | 20-12-2020 | 10     | 1    | 2        |
    | 15-12-2020 | 20     | 1    | 2        |
    | 25-05-2022 | 50     | 3    | 4        |

    df_expected_characterize:

    | date       | amount | flow | activity |
    |------------|--------|------|----------|
    | 20-12-2020 | 10     | 1    | 2        |
    | 21-12-2020 | 9      | 1    | 2        |
    | 15-12-2020 | 20     | 1    | 2        |
    | 16-12-2020 | 19     | 1    | 2        |
    | 25-05-2022 | 50     | 3    | 4        |
    | 26-05-2022 | 49     | 3    | 4        |

    df_expected_sum_days_to_years:

    | date | amount | flow | activity |
    |------|--------|------|----------|
    | 2020 | 39     | 1    | 2        |
    | 2022 | 99     | 1    | 4        |
    """

    df_input = pd.DataFrame(
        data={
            "date": pd.Series(
                data=[
                    "20-12-2020",
                    "21-12-2020",
                    "15-12-2020",
                    "16-12-2020",
                    "25-05-2022",
                    "26-05-2022",
                ],
                dtype="datetime64[s]",
            ),
            "amount": pd.Series(
                data=[10.0, 9.0, 20.0, 19.0, 50.0, 49.0], dtype="float64"
            ),
            "flow": pd.Series(data=[1, 1, 1, 1, 3, 3], dtype="int"),
            "activity": pd.Series(data=[2, 2, 2, 2, 4, 4], dtype="int"),
        }
    )

    df_expected_characterize = pd.DataFrame(
        data={
            "date": pd.Series(
                data=[
                    "20-12-2020",
                    "21-12-2020",
                    "15-12-2020",
                    "16-12-2020",
                    "25-05-2022",
                    "26-05-2022",
                ],
                dtype="datetime64[s]",
            ),
            "amount": pd.Series(
                data=[10.0, 9.0, 20.0, 19.0, 50.0, 49.0], dtype="float64"
            ),
            "flow": pd.Series(data=[1, 1, 1, 1, 3, 3], dtype="int"),
            "activity": pd.Series(data=[2, 2, 2, 2, 4, 4], dtype="int"),
        }
    )

    df_expected_sum_days_to_years = pd.DataFrame(
        data={
            "date": pd.Series(data=["01-01-2020", "01-01-2022"], dtype="datetime64[s]"),
            "amount": pd.Series(data=[39.0, 99.0], dtype="float64"),
            "flow": pd.Series(data=[1, 1], dtype="int"),
            "activity": pd.Series(data=[2, 4], dtype="int"),
        }
    )

    return (df_input, df_expected_characterize, df_expected_sum_days_to_years)


def function_characterization_test(series: pd.Series, period: int = 2) -> pd.DataFrame:
    date_beginning: np.datetime64 = series["date"].to_numpy()
    dates_characterized: np.ndarray = date_beginning + np.arange(
        start=0, stop=period, dtype="timedelta64[Y]"
    ).astype("timedelta64[s]")

    amount_beginning: float = series["amount"]
    amount_characterized: np.ndarray = amount_beginning - np.arange(
        start=0, stop=period, dtype="int"
    )

    return pd.DataFrame(
        {
            "date": pd.Series(data=dates_characterized.astype("datetime64[s]")),
            "amount": pd.Series(data=amount_characterized, dtype="float64"),
            "flow": series.flow,
            "activity": series.activity,
        }
    )


@pytest.mark.skip("Not yet")
def test_characterization():
    pass
