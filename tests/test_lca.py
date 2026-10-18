import bw2data as bd
import numpy as np
import pandas as pd
import pytest
from bw2calc import LCA
from bw2data.tests import bw2test
from bw_graph_tools.testing import flow_equal_dict, node_equal_dict

from bw_temporalis import TemporalDistribution as TD
from bw_temporalis import TemporalisLCA, easy_timedelta_distribution
from bw_temporalis.lca import MultipleTechnosphereExchanges

NAMES = ("CO2", "CH4", "A", "B", "C", "D")


@pytest.fixture
@bw2test
def basic_db():
    bd.projects.set_current("__test_fixture__")

    db = bd.Database("db")
    db.write(
        {
            ("db", "CO2"): {
                "type": "emission",
                "name": "carbon dioxide",
            },
            ("db", "CH4"): {
                "type": "emission",
                "name": "methane",
            },
            ("db", "A"): {
                "name": "Functional Unit",
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
            },
            ("db", "C"): {
                "exchanges": [
                    {
                        "amount": 0.5,
                        "input": ("db", "CH4"),
                        "type": "biosphere",
                    },
                ],
                "name": "C",
            },
            ("db", "D"): {
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
                "name": "D",
            },
        }
    )
    bd.Method(("m",)).write([(("db", "CO2"), 1), (("db", "CH4"), 25)])
    return db


def test_temporalis_lca(basic_db):
    ids = {code: bd.get_id(("db", code)) for code in NAMES}

    lca = LCA({("db", "A"): 2}, ("m",))
    lca.lci()
    lca.lcia()

    assert lca.score == 410

    tlca = TemporalisLCA(
        lca_object=lca,
        starting_datetime="2023-01-01",
    )

    expected_flows = [
        {
            "flow_datapackage_id": ids["CH4"],
            "flow_index": 1,
            "activity_unique_id": 2,
            "activity_id": ids["C"],
            "activity_index": 2,
            "amount": 10,
            "score": 250,
        },
        {
            "flow_datapackage_id": ids["CO2"],
            "flow_index": 0,
            "activity_unique_id": 1,
            "activity_id": ids["B"],
            "activity_index": 1,
            "amount": 80,
            "score": 80,
        },
        {
            "flow_datapackage_id": ids["CO2"],
            "flow_index": 0,
            "activity_unique_id": 3,
            "activity_id": ids["D"],
            "activity_index": 3,
            "amount": 80,
            "score": 80,
        },
    ]
    expected_flows.sort(key=lambda x: x["score"], reverse=True)

    assert len(tlca.flows) == 3
    for a, b in zip(tlca.flows, expected_flows):
        flow_equal_dict(a, b)

    expected_nodes = [
        {
            "unique_id": -1,
            "activity_datapackage_id": -1,
            "activity_index": -1,
            "reference_product_datapackage_id": -1,
            "reference_product_index": -1,
            "reference_product_production_amount": 1,
            "supply_amount": 1,
            "cumulative_score": 410,
            "direct_emissions_score": 0,
        },
        {
            "unique_id": 0,
            "activity_datapackage_id": ids["A"],
            "activity_index": 0,
            "reference_product_datapackage_id": ids["A"],
            "reference_product_index": 0,
            "reference_product_production_amount": 1,
            "supply_amount": 2,
            "cumulative_score": 410,
            "direct_emissions_score": 0,
        },
        {
            "unique_id": 1,
            "activity_datapackage_id": ids["B"],
            "activity_index": 1,
            "reference_product_datapackage_id": ids["B"],
            "reference_product_index": 1,
            "reference_product_production_amount": 1,
            "supply_amount": 10,
            "cumulative_score": 410,
            "direct_emissions_score": 80,
        },
        {
            "unique_id": 2,
            "activity_datapackage_id": ids["C"],
            "activity_index": 2,
            "reference_product_datapackage_id": ids["C"],
            "reference_product_index": 2,
            "reference_product_production_amount": 1,
            "supply_amount": 20,
            "cumulative_score": 250,
            "direct_emissions_score": 250,
        },
        {
            "unique_id": 3,
            "activity_datapackage_id": ids["D"],
            "activity_index": 3,
            "reference_product_datapackage_id": ids["D"],
            "reference_product_index": 3,
            "reference_product_production_amount": 1,
            "supply_amount": 40,
            "cumulative_score": 80,
            "direct_emissions_score": 80,
        },
    ]

    for a in expected_nodes:
        node_equal_dict(tlca.nodes[a["unique_id"]], a)

    tl = tlca.build_timeline()
    given_df = tl.build_dataframe()

    start = (
        TD(
            np.array([np.datetime64("2023-01-01")]),
            np.array([1]),
        )
        * 2
    )
    a_td = easy_timedelta_distribution(0, 4, resolution="Y", steps=5) * 5
    b_td = easy_timedelta_distribution(10, 17, steps=4, resolution="Y") * 8
    c_td = a_td * 1
    d_td = easy_timedelta_distribution(-8, -5, steps=4, resolution="Y") * 8
    assert a_td.amount.sum() == 5
    assert b_td.amount.sum() == 8
    assert c_td.amount.sum() == 5
    assert d_td.amount.sum() == 8

    co2 = start * (a_td * b_td + a_td * d_td)
    assert isinstance(co2, TD)
    assert co2.amount.sum() == 80 + 80

    ch4 = c_td
    assert isinstance(ch4, TD)
    assert ch4.amount.sum() == 5

    expected_df = pd.DataFrame(
        {
            "date": np.hstack(
                [
                    (start * (a_td * b_td)).date,
                    (start * (a_td * d_td)).date,
                    (start * a_td).date,
                ]
            ),
            "amount": np.hstack(
                [
                    (start * (a_td * b_td)).amount,
                    (start * (a_td * d_td)).amount,
                    (start * a_td).amount,
                ]
            ),
            "flow": np.hstack(
                [
                    np.full_like((start * (a_td * b_td)).amount, ids["CO2"], dtype=int),
                    np.full_like((start * (a_td * d_td)).amount, ids["CO2"], dtype=int),
                    np.full_like((start * a_td).amount, ids["CH4"], dtype=int),
                ]
            ).astype(int),
            "activity": np.hstack(
                [
                    np.full_like((start * (a_td * b_td)).amount, ids["B"], dtype=int),
                    np.full_like((start * (a_td * d_td)).amount, ids["D"], dtype=int),
                    np.full_like((start * a_td).amount, ids["C"], dtype=int),
                ]
            ).astype(int),
        }
    )
    expected_df.sort_values(by="date", ascending=True, inplace=True)
    expected_df.reset_index(drop=True, inplace=True)

    pd.testing.assert_frame_equal(given_df, expected_df)


@bw2test
def test_temporalis_static_indices_correct_lookup():
    bd.projects.set_current("__test_fixture__")

    db = bd.Database("db")
    db.write(
        {
            ("db", "CO2"): {
                "type": "emission",
                "name": "carbon dioxide",
            },
            ("db", "CH4"): {
                "type": "emission",
                "name": "methane",
            },
            ("db", "A"): {
                "name": "Functional Unit",
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
            },
            ("db", "C"): {
                "exchanges": [
                    {
                        "amount": 0.5,
                        "input": ("db", "CH4"),
                        "type": "biosphere",
                    },
                ]
            },
            ("db", "D"): {
                "exchanges": [
                    {
                        "amount": 2,
                        "input": ("db", "CO2"),
                        "type": "biosphere",
                        "temporal_distribution": easy_timedelta_distribution(
                            -8, -5, steps=4, resolution="Y"
                        ),
                    },
                ]
            },
        }
    )
    bd.Method(("m",)).write([(("db", "CO2"), 1), (("db", "CH4"), 25)])

    lca = LCA({("db", "A"): 2}, ("m",))
    lca.lci()
    lca.lcia()

    assert lca.score == 410

    # Database IDs, not matrix indices; these must be translated by ``TemporalisLCA``
    static_indices = {x.id for x in db if x["type"] in bd.labels.process_node_types}
    assert static_indices == {bd.get_id(("db", code)) for code in "ABCD"}

    tlca = TemporalisLCA(
        lca_object=lca,
        starting_datetime="2023-01-01",
        static_activity_indices=static_indices,
    )
    # No timeline because everything is static
    assert not tlca.build_timeline()


@pytest.mark.xfail
def test_temporalis_lca_node_timeline(basic_db):
    lca = LCA({("db", "A"): 2}, ("m",))
    lca.lci()
    lca.lcia()

    assert lca.score == 410

    tlca = TemporalisLCA(
        lca_object=lca,
        starting_datetime="2023-01-01",
    )

    tl = tlca.build_timeline(node_timeline=True)
    given_df = tl.build_dataframe()

    print(given_df)

    start = (
        TD(
            np.array([np.datetime64("2023-01-01")]),
            np.array([1]),
        )
        * 2
    )
    a_td = easy_timedelta_distribution(0, 4, resolution="Y", steps=5) * 5
    b_td = easy_timedelta_distribution(10, 17, steps=4, resolution="Y") * 8
    c_td = a_td * 1
    d_td = easy_timedelta_distribution(-8, -5, steps=4, resolution="Y") * 8
    assert a_td.amount.sum() == 5
    assert b_td.amount.sum() == 8
    assert c_td.amount.sum() == 5
    assert d_td.amount.sum() == 8

    co2 = start * (a_td * b_td + a_td * d_td)
    assert isinstance(co2, TD)
    assert co2.amount.sum() == 80 + 80

    ch4 = c_td
    assert isinstance(ch4, TD)
    assert ch4.amount.sum() == 5

    expected_df = pd.DataFrame(
        {
            "date": np.hstack(
                [
                    (start * (a_td * b_td)).date,
                    (start * (a_td * d_td)).date,
                    (start * a_td).date,
                ]
            ),
            "amount": np.hstack(
                [
                    (start * (a_td * b_td)).amount,
                    (start * (a_td * d_td)).amount,
                    (start * a_td).amount,
                ]
            ),
            "flow": -1,
            "activity": np.hstack(
                [
                    4 * np.ones_like((start * (a_td * b_td)).amount),
                    6 * np.ones_like((start * (a_td * d_td)).amount),
                    5 * np.ones_like((start * a_td).amount),
                ]
            ).astype(int),
        }
    )
    expected_df.sort_values(by="date", ascending=True, inplace=True)
    expected_df.reset_index(drop=True, inplace=True)

    pd.testing.assert_frame_equal(given_df, expected_df)


def test_temporalis_lca_draw_from_matrix(basic_db):
    ids = {code: bd.get_id(("db", code)) for code in NAMES}

    lca = LCA({("db", "A"): 2}, ("m",))
    lca.lci()
    lca.lcia()

    assert lca.score == 410

    A = bd.get_node(code="A").id
    B = bd.get_node(code="B").id

    assert lca.technosphere_matrix[lca.dicts.product[B], lca.dicts.activity[A]] == -5
    lca.technosphere_matrix[lca.dicts.product[B], lca.dicts.activity[A]] = -10

    tlca = TemporalisLCA(
        lca_object=lca,
        starting_datetime="2023-01-01",
    )

    expected_flows = [
        {
            "flow_datapackage_id": ids["CH4"],
            "flow_index": 1,
            "activity_unique_id": 2,
            "activity_id": ids["C"],
            "activity_index": 2,
            "amount": 20,
            "score": 500,
        },
        {
            "flow_datapackage_id": ids["CO2"],
            "flow_index": 0,
            "activity_unique_id": 1,
            "activity_id": ids["B"],
            "activity_index": 1,
            "amount": 160,
            "score": 160,
        },
        {
            "flow_datapackage_id": ids["CO2"],
            "flow_index": 0,
            "activity_unique_id": 3,
            "activity_id": ids["D"],
            "activity_index": 3,
            "amount": 160,
            "score": 160,
        },
    ]
    expected_flows.sort(key=lambda x: x["score"], reverse=True)

    assert len(tlca.flows) == 3
    for a, b in zip(tlca.flows, expected_flows):
        flow_equal_dict(a, b)

    expected_nodes = [
        {
            "unique_id": -1,
            "activity_datapackage_id": -1,
            "activity_index": -1,
            "reference_product_datapackage_id": -1,
            "reference_product_index": -1,
            "reference_product_production_amount": 1,
            "supply_amount": 1,
            "cumulative_score": 410,
            "direct_emissions_score": 0,
        },
        {
            "unique_id": 0,
            "activity_datapackage_id": ids["A"],
            "activity_index": 0,
            "reference_product_datapackage_id": ids["A"],
            "reference_product_index": 0,
            "reference_product_production_amount": 1,
            "supply_amount": 2,
            "cumulative_score": 820,
            "direct_emissions_score": 0,
        },
        {
            "unique_id": 1,
            "activity_datapackage_id": ids["B"],
            "activity_index": 1,
            "reference_product_datapackage_id": ids["B"],
            "reference_product_index": 1,
            "reference_product_production_amount": 1,
            "supply_amount": 20,
            "cumulative_score": 820,
            "direct_emissions_score": 160,
        },
        {
            "unique_id": 2,
            "activity_datapackage_id": ids["C"],
            "activity_index": 2,
            "reference_product_datapackage_id": ids["C"],
            "reference_product_index": 2,
            "reference_product_production_amount": 1,
            "supply_amount": 40,
            "cumulative_score": 500,
            "direct_emissions_score": 500,
        },
        {
            "unique_id": 3,
            "activity_datapackage_id": ids["D"],
            "activity_index": 3,
            "reference_product_datapackage_id": ids["D"],
            "reference_product_index": 3,
            "reference_product_production_amount": 1,
            "supply_amount": 80,
            "cumulative_score": 160,
            "direct_emissions_score": 160,
        },
    ]

    for a in expected_nodes:
        node_equal_dict(tlca.nodes[a["unique_id"]], a)


def test_lca_provide_static_activity_indices(basic_db):
    lca = LCA({("db", "A"): 2}, ("m",))
    lca.lci()
    lca.lcia()

    TemporalisLCA(
        lca_object=lca,
        static_activity_indices=[
            x.id for x in basic_db if x["type"] in bd.labels.process_node_types
        ][:2],
    )
    with pytest.raises(TypeError):
        TemporalisLCA(lca_object=lca, static_activity_indices=1001)

def test_temporalis_lca_json_temporal_distribution(basic_db):
    # bw2data stores edge TDs as JSON strings when ``bw_temporalis`` is imported first
    for exc in bd.get_node(code="B").biosphere():
        exc["temporal_distribution"] = exc["temporal_distribution"].to_json()
        exc.save()
    assert isinstance(
        next(iter(bd.get_node(code="B").biosphere()))["temporal_distribution"], str
    )

    lca = LCA({("db", "A"): 2}, ("m",))
    lca.lci()
    lca.lcia()

    tlca = TemporalisLCA(lca_object=lca, starting_datetime="2023-01-01")
    df = tlca.build_timeline().build_dataframe()
    assert np.isclose(df["amount"].sum(), 80 + 80 + 10)


def test_multiple_technosphere_exchanges_error(basic_db):
    EXPECTED = "Found 2 exchanges for link between (db|C|C) and (db|B|B)"
    # add a second exchange of C to activity B
    b = bd.get_activity(("db", "B"))
    b.new_edge(
        input=bd.get_activity(("db", "C")),
        type="technosphere",
        amount=1.5,
    ).save()
    # normal LCA
    lca = LCA({("db", "B"): 1}, method=("m",))
    lca.lci()
    lca.lcia()
    #TemporalisLCA
    dlca = TemporalisLCA(lca)
    with pytest.raises(MultipleTechnosphereExchanges) as exc:
        dlca.build_timeline()
    assert str(exc.value) == EXPECTED