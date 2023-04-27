import bw2data as bd
import numpy as np
import pytest
from bw2data.tests import bw2test

from bw_temporalis.temporal_distribution import TemporalDistribution as TD


@pytest.fixture
@bw2test
def eol():
    bd.projects.set_current("__temporalis test fixture__")
    bd.Database("temporalis-example").write(
        {
            ("temporalis-example", "CO2"): {
                "type": "emission",
                "name": "carbon dioxide",
                "temporalis code": "co2",
            },
            ("temporalis-example", "CH4"): {
                "type": "emission",
                "name": "methane",
                "temporalis code": "ch4",
            },
            ("temporalis-example", "Functional Unit"): {
                "name": "Functional Unit",
                "exchanges": [
                    {
                        "amount": 5,
                        "input": ("temporalis-example", "EOL"),
                        "temporal_distribution": TD(
                            np.array([0, 1, 2, 3, 4], dtype="timedelta64[Y]"),
                            np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
                        ),
                        "type": "technosphere",
                    },
                ],
            },
            ("temporalis-example", "EOL"): {
                "exchanges": [
                    {
                        "amount": 0.8,
                        "input": ("temporalis-example", "Waste"),
                        "type": "technosphere",
                    },
                    {
                        "amount": 0.2,
                        "input": ("temporalis-example", "Landfill"),
                        "type": "technosphere",
                    },
                    {
                        "amount": 1,
                        "input": ("temporalis-example", "Use"),
                        "type": "technosphere",
                    },
                ],
                "name": "EOL",
                "type": "process",
            },
            ("temporalis-example", "Use"): {
                "exchanges": [
                    {
                        "amount": 1,
                        "input": ("temporalis-example", "Production"),
                        "temporal_distribution": TD(
                            np.array([4], dtype="timedelta64[M]"), np.array([1.0])
                        ),
                        "type": "technosphere",
                    },
                ],
                "name": "Use",
            },
            ("temporalis-example", "Production"): {
                "exchanges": [
                    {
                        "amount": 1,
                        "input": ("temporalis-example", "Transport"),
                        "temporal_distribution": TD(
                            np.array([200], dtype="timedelta64[D]"), np.array([1.0])
                        ),
                        "type": "technosphere",
                    },
                ],
                "name": "Production",
                "type": "process",
            },
            ("temporalis-example", "Transport"): {
                "exchanges": [
                    {
                        "amount": 1,
                        "input": ("temporalis-example", "Sawmill"),
                        "type": "technosphere",
                    },
                    {
                        "amount": 0.1,
                        "input": ("temporalis-example", "CO2"),
                        "type": "biosphere",
                    },
                ],
                "name": "Production",
                "type": "process",
            },
            ("temporalis-example", "Sawmill"): {
                "exchanges": [
                    {
                        "amount": 1.2,
                        "input": ("temporalis-example", "Forest"),
                        "temporal_distribution": TD(
                            np.array([14], dtype="timedelta64[M]"), np.array([1.2])
                        ),
                        "type": "technosphere",
                    },
                    {
                        "amount": 0.1,
                        "input": ("temporalis-example", "CO2"),
                        "type": "biosphere",
                    },
                ],
                "name": "Sawmill",
                "type": "process",
            },
            ("temporalis-example", "Forest"): {
                "exchanges": [
                    {
                        "amount": -0.2 * 6,
                        "input": ("temporalis-example", "CO2"),
                        "temporal_distribution": TD(
                            np.array([-4, -3, 0, 1, 2, 5], dtype="timedelta64[Y]"),
                            np.array([-0.2] * 6),
                        ),
                        "type": "biosphere",
                    },
                    {
                        "amount": 1.5,
                        "input": ("temporalis-example", "Thinning"),
                        "temporal_distribution": TD(
                            np.array([-3, 0, 1], dtype="timedelta64[Y]"),
                            np.array([0.5] * 3),
                        ),
                        "type": "technosphere",
                    },
                ],
                "name": "Forest",
            },
            ("temporalis-example", "Thinning"): {
                "exchanges": [
                    {
                        "amount": 1,
                        "input": ("temporalis-example", "Thinning"),
                        "type": "production",
                    },
                    {
                        "amount": 1,
                        "input": ("temporalis-example", "Avoided impact - thinnings"),
                        "type": "production",
                    },
                ],
                "name": "Thinning",
                "type": "process",
            },
            ("temporalis-example", "Landfill"): {
                "exchanges": [
                    {
                        "amount": 0.1,
                        "input": ("temporalis-example", "CH4"),
                        "temporal_distribution": TD(
                            np.array([10, 20, 40, 60], dtype="timedelta64[M]"),
                            np.array([0.025] * 4),
                        ),
                        "type": "biosphere",
                    },
                ],
                "name": "Landfill",
                "type": "process",
            },
            ("temporalis-example", "Waste"): {
                "exchanges": [
                    {
                        "amount": 1,
                        "input": ("temporalis-example", "Waste"),
                        "type": "production",
                    },
                    {
                        "amount": 1,
                        "input": ("temporalis-example", "Avoided impact - waste"),
                        "type": "production",
                    },
                ],
                "name": "Waste",
                "type": "process",
            },
            ("temporalis-example", "Avoided impact - waste"): {
                "exchanges": [
                    {
                        "amount": -0.6,
                        "input": ("temporalis-example", "CO2"),
                        "type": "biosphere",
                    },
                    {
                        "amount": 1,
                        "input": ("temporalis-example", "Avoided impact - waste"),
                        "type": "production",
                    },
                ],
                "name": "Avoided impact - waste",
                "type": "process",
            },
            ("temporalis-example", "Avoided impact - thinnings"): {
                "exchanges": [
                    {
                        "amount": -0.2,
                        "input": ("temporalis-example", "CO2"),
                        "type": "biosphere",
                    },
                    {
                        "amount": 1,
                        "input": ("temporalis-example", "Avoided impact - thinnings"),
                        "type": "production",
                    },
                ],
                "name": "Avoided impact - thinnings",
                "type": "process",
            },
        }
    )
    bd.Method(("GWP", "example")).write(
        [
            (("temporalis-example", "CO2"), 1),
            (("temporalis-example", "CH4"), 25),
        ]
    )
