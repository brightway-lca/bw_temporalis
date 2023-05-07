import bw2data as bd
import pytest
from bw2calc import LCA
from bw2data.tests import bw2test

from bw_temporalis import TemporalisLCA, easy_timedelta_distribution


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
                        "temporal distribution": easy_timedelta_distribution(
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
                        "temporal distribution": easy_timedelta_distribution(
                            10, 17, steps=4, resolution="Y"
                        ),
                    },
                ],
                "name": "B",
            },
            ("db", "C"): {"exchanges": []},
            ("db", "D"): {
                "exchanges": [
                    {
                        "amount": 2,
                        "input": ("db", "CO2"),
                        "type": "biosphere",
                        "temporal distribution": easy_timedelta_distribution(
                            -8, -5, steps=4, resolution="Y"
                        ),
                    },
                ]
            },
        }
    )
    bd.Method(("m",)).write([(("db", "CO2"), 1), (("db", "CH4"), 25)])
    return db


def test_temporalis_lca(basic_db):
    lca = LCA({("db", "A"): 10}, ("m",))
    lca.lci()
    lca.lcia()

    print(lca.score)

    tlca = TemporalisLCA(
        lca_object=lca,
        starting_datetime="2023-01-01",
    )
    tl = tlca.build_timeline()
    tl.build_dataframe()
    print(tl.df)
    # assert False
