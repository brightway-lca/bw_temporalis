import json
from collections.abc import Mapping
from numbers import Number
from typing import Any, Union

import bw2calc as bc
import bw2data as bd
import numpy as np
from bw2data.tests import bw2test

import bw_temporalis as bwt


class Half(bwt.TDAware):
    # Make sure that we control multiplication
    _mul_comes_first = True

    def __init__(self, **kwargs: Any):
        pass

    def __mul__(
        self, other: Union[bwt.TemporalDistribution, Number]
    ) -> Union[bwt.TemporalDistribution, "Half"]:
        if isinstance(other, Number):
            return other / 2
        elif isinstance(other, bwt.TemporalDistribution):
            return bwt.TemporalDistribution(date=other.date, amount=other.amount / 2)
        else:
            raise ValueError("Can't multiply `Half` and {}".format(type(other)))

    def to_json(self) -> str:
        return json.dumps({"__loader__": "Half"})

    @classmethod
    def from_json(cls, json_obj):
        if isinstance(json_obj, Mapping):
            data = json_obj
        elif isinstance(json_obj, str):
            data = json.loads(json_obj)
        else:
            raise ValueError(f"Can't understand `from_json` input object {json_obj}")
        return cls(**data)


@bw2test
def test_dynamic_function():
    bwt.loader_registry["Half"] = Half.from_json

    bd.Database("example").write(
        {
            ("example", "CO2"): {
                "type": "emission",
                "name": "carbon dioxide",
                "temporalis code": "co2",
            },
            ("example", "a"): {
                "name": "First one",
                "exchanges": [
                    {
                        "amount": 50,
                        "input": ("example", "b"),
                        "temporal_distribution": bwt.easy_timedelta_distribution(
                            start=0,
                            end=10,
                            resolution="Y",
                            steps=11,
                        ),
                        "type": "technosphere",
                    },
                ],
            },
            ("example", "b"): {
                "name": "Second one",
                "exchanges": [
                    {
                        "amount": 1,
                        "input": ("example", "CO2"),
                        "type": "biosphere",
                        "temporal_distribution": Half(),
                    },
                ],
                "type": "process",
            },
        }
    )
    bd.Method(("GWP", "example")).write(
        [
            (("example", "CO2"), 1),
        ]
    )
    lca = bc.LCA({("example", "a"): 1}, ("GWP", "example"))
    lca.lci()
    lca.lcia()
    lca.score
    assert lca.score == 50

    tlca = bwt.TemporalisLCA(lca, cutoff=0.1)
    tl = tlca.build_timeline()
    tl.build_dataframe()
    assert len(tl.df) == 11
    assert np.allclose(tl.df.amount.sum(), 25)
