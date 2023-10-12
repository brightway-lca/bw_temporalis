import uuid

import bw2calc as bc
import bw2data as bd
import bw_processing as bwp
import numpy as np
import pytest
from bw2data.tests import bw2test

from bw_temporalis import TemporalisLCA


def safety_razor(
    consumer: bd.Node,
    previous_producer: bd.Node,
    new_producer: bd.Node,
) -> bwp.Datapackage:
    amount = sum(
        exc["amount"]
        for exc in consumer.technosphere()
        if exc.input == previous_producer
    )

    datapackage = bwp.create_datapackage()
    datapackage.add_persistent_vector(
        # This function would need to be adapted for biosphere edges
        matrix="technosphere_matrix",
        name=uuid.uuid4().hex,
        data_array=np.array([0, amount], dtype=float),
        indices_array=np.array(
            [(previous_producer.id, consumer.id), (new_producer.id, consumer.id)],
            dtype=bwp.INDICES_DTYPE,
        ),
        flip_array=np.array([False, True], dtype=bool),
    )
    return datapackage


@pytest.fixture
@bw2test
def patched_matrix():
    db = bd.Database("f")
    db.write(
        {
            ("f", "0"): {},
            ("f", "1"): {
                "exchanges": [
                    {
                        "type": "technosphere",
                        "input": ("f", "2"),
                        "amount": 2,
                    },
                    {
                        "type": "production",
                        "input": ("f", "1"),
                        "amount": 1,
                    },
                ]
            },
            ("f", "2"): {
                "exchanges": [
                    {
                        "type": "technosphere",
                        "input": ("f", "3"),
                        "amount": 4,
                    },
                    {
                        "type": "production",
                        "input": ("f", "2"),
                        "amount": 1,
                    },
                    {
                        "type": "biosphere",
                        "amount": 1,
                        "input": ("f", "0"),
                    },
                ]
            },
            ("f", "3"): {
                "exchanges": [
                    {
                        "type": "production",
                        "input": ("f", "3"),
                        "amount": 1,
                    },
                    {
                        "type": "biosphere",
                        "amount": 10,
                        "input": ("f", "0"),
                    },
                ]
            },
        }
    )
    f1 = bd.get_node(code="1")
    f2 = bd.get_node(code="2")
    f3 = bd.get_node(code="3")
    dp = safety_razor(
        consumer=f1,
        previous_producer=f2,
        new_producer=f3,
    )
    bd.Method(("m",)).write(
        [
            (("f", "0"), 1),
        ]
    )
    fu, data_objs, remapping = bd.prepare_lca_inputs(demand={f1: 1}, method=("m",))
    return fu, data_objs + [dp]


@pytest.fixture
def lca(patched_matrix):
    lca = bc.LCA(demand=patched_matrix[0], data_objs=patched_matrix[1])
    lca.lci()
    lca.lcia()
    return lca


def test_pmlca_without_modification(patched_matrix):
    lca = bc.LCA({bd.get_node(code="1"): 1}, ("m",))
    lca.lci()
    lca.lcia()
    assert lca.score == 82


def test_pmlca_basic(lca):
    assert lca.score == 20


def test_pmlca_values(lca):
    tlca = TemporalisLCA(
        lca_object=lca,
        starting_datetime="2023-01-01",
    )
    tl = tlca.build_timeline()
    assert len(tl.data) == 1
    ftd = tl.data[0]
    assert ftd.flow == bd.get_node(code="0").id
    assert ftd.activity == bd.get_node(code="3").id
    assert ftd.distribution.amount.sum() == 20


def test_pmlca_dataframe(lca):
    tlca = TemporalisLCA(
        lca_object=lca,
        starting_datetime="2023-01-01",
    )
    tl = tlca.build_timeline()
    df = tl.build_dataframe()
    assert str(df.date[0]) == "2023-01-01 00:00:00"
    assert df.amount[0] == 20
    assert df.flow[0] == bd.get_node(code="0").id
    assert df.activity[0] == bd.get_node(code="3").id
