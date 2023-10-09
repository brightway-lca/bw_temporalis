import bw2data as bd
import numpy as np
import pandas as pd
import pytest
from bw2calc import LCA
from bw2data.tests import bw2test

from bw_temporalis import TemporalisLCA, easy_timedelta_distribution
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


@bw2test
def test_add_metadata_to_dataframe():
    bd.projects.set_current("__test_fixture__")

    db = bd.Database("db")
    db.write(
        {
            ("db", "CO2"): {
                "type": "emission",
                "name": "carbon dioxide",
                "categories": ("somewhere", "somehow"),
                "weird": "strange",
            },
            ("db", "CH4"): {
                "type": "emission",
                "name": "methane",
            },
            ("db", "A"): {
                "name": "Functional Unit",
                "location": "somewhere",
                "unit": "something",
                "exchanges": [
                    {
                        "amount": 5,
                        "input": ("db", "B"),
                        "temporal_distribution": easy_timedelta_distribution(
                            0, 4, resolution="Y", steps=5
                        ),
                        "type": "technosphere",
                    },
                ],
            },
            ("db", "B"): {
                "exchanges": [
                    {"amount": 2, "input": ("db", "C"), "type": "technosphere"},
                    {"amount": 4, "input": ("db", "D"), "type": "technosphere"},
                    {
                        "amount": 8,
                        "input": ("db", "CO2"),
                        "type": "biosphere",
                        "temporal_distribution": easy_timedelta_distribution(
                            10, 17, steps=4, resolution="Y"
                        ),
                    },
                ],
                "name": "B",
                "location": "Germany",
                "unit": "Pfenning",
            },
            ("db", "C"): {
                "name": "C",
                "exchanges": [
                    {
                        "amount": 0.5,
                        "input": ("db", "CH4"),
                        "type": "biosphere",
                    },
                ],
            },
            ("db", "D"): {
                "name": "D",
                "location": "Belgium",
                "unit": "Bier",
                "exchanges": [
                    {
                        "amount": 2,
                        "input": ("db", "CO2"),
                        "type": "biosphere",
                        "temporal_distribution": easy_timedelta_distribution(
                            -8, -5, steps=4, resolution="Y"
                        ),
                    },
                ],
            },
        }
    )
    bd.Method(("m",)).write([(("db", "CO2"), 1), (("db", "CH4"), 25)])

    lca = LCA({("db", "A"): 2}, ("m",))
    lca.lci()
    lca.lcia()

    assert lca.score == 410

    tlca = TemporalisLCA(
        lca_object=lca,
        starting_datetime="2023-01-01",
    )
    tl = tlca.build_timeline()
    tl.build_dataframe()
    df = tl.add_metadata_to_dataframe(
        database_labels=["db"],
        fields=["name", "unit", "weird", "categories"],
    )

    co2 = bd.get_node(code="CO2").id
    ch4 = bd.get_node(code="CH4").id
    A = bd.get_node(code="A").id
    B = bd.get_node(code="B").id

    assert "activity_location" not in df.columns
    assert "flow_location" not in df.columns

    assert df[df.flow == co2].flow_name.unique() == "carbon dioxide"
    assert df[df.flow == co2].flow_categories.unique()[0] == ("somewhere", "somehow")
    assert len(df[df.flow == co2].flow_categories.unique()) == 1
    assert np.isnan(df[df.flow == co2].flow_unit.unique()[0])
    assert df[df.flow == co2].flow_weird.unique() == ["strange"]

    assert df[df.flow == ch4].flow_name.unique() == "methane"
    assert np.isnan(df[df.flow == ch4].flow_categories.unique()[0])
    assert np.isnan(df[df.flow == ch4].flow_unit.unique()[0])
    assert np.isnan(df[df.flow == ch4].flow_weird.unique()[0])

    assert not (df.activity == A).sum()

    assert df[df.activity == B].activity_name.unique() == "B"
    assert df[df.activity == B].activity_unit.unique() == "Pfenning"
    assert np.isnan(df[df.activity == B].activity_categories.unique()[0])
    assert np.isnan(df[df.activity == B].activity_weird.unique()[0])
