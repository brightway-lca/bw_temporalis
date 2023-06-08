import json

import numpy as np
import pytest

from bw_temporalis import easy_datetime_distribution, easy_timedelta_distribution
from bw_temporalis.temporal_distribution import TemporalDistribution as TD


def test_serialization_absolute_easy():
    td = easy_datetime_distribution(start="2023-01-01", end="2023-01-04", steps=4)
    expected = json.dumps(
        {
            "__loader__": "bw_temporalis.TemporalDistribution",
            "date_dtype": "datetime64[s]",
            "date": [1672531200, 1672617600, 1672704000, 1672790400],
            "amount": [0.25, 0.25, 0.25, 0.25],
        }
    )
    assert td.to_json() == expected


def test_serialization_absolute():
    td = TD(
        date=np.array(
            ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"],
            dtype="datetime64[D]",
        ),
        amount=np.array([1.1, 2, 3, 4.4]),
    )
    expected = json.dumps(
        {
            "__loader__": "bw_temporalis.TemporalDistribution",
            "date_dtype": "datetime64[s]",
            "date": [1672531200, 1672617600, 1672704000, 1672790400],
            "amount": [1.1, 2.0, 3.0, 4.4],
        }
    )
    assert td.to_json() == expected


def test_serialization_relative_easy():
    td = easy_timedelta_distribution(start=0, end=6, steps=4, resolution="h")
    expected = json.dumps(
        {
            "__loader__": "bw_temporalis.TemporalDistribution",
            "date_dtype": "timedelta64[s]",
            "date": [0, 60 * 60 * 2, 60 * 60 * 4, 60 * 60 * 6],
            "amount": [0.25, 0.25, 0.25, 0.25],
        }
    )
    assert td.to_json() == expected


def test_serialization_relative():
    td = TD(
        date=np.array([0, 2, 4, 6], dtype="timedelta64[h]"),
        amount=np.array([1.1, 2, 3, 4.4]),
    )
    expected = json.dumps(
        {
            "__loader__": "bw_temporalis.TemporalDistribution",
            "date_dtype": "timedelta64[s]",
            "date": [0, 60 * 60 * 2, 60 * 60 * 4, 60 * 60 * 6],
            "amount": [1.1, 2.0, 3.0, 4.4],
        }
    )
    assert td.to_json() == expected


def test_deserialization_absolute():
    reference = easy_datetime_distribution(
        start="2023-01-01", end="2023-01-04", steps=4
    )
    given = TD.from_json(
        json.dumps(
            {
                "__loader__": "bw_temporalis.TemporalDistribution",
                "date_dtype": "datetime64[s]",
                "date": [1672531200, 1672617600, 1672704000, 1672790400],
                "amount": [0.25, 0.25, 0.25, 0.25],
            }
        )
    )
    assert np.allclose(reference.date.astype(int), given.date.astype(int))
    assert np.allclose(reference.amount, given.amount)
    assert str(given.date.dtype) == "datetime64[s]"


def test_deserialization_relative():
    reference = easy_timedelta_distribution(start=0, end=6, steps=4, resolution="h")
    given = TD.from_json(
        json.dumps(
            {
                "__loader__": "bw_temporalis.TemporalDistribution",
                "date_dtype": "timedelta64[s]",
                "date": [0, 60 * 60 * 2, 60 * 60 * 4, 60 * 60 * 6],
                "amount": [0.25, 0.25, 0.25, 0.25],
            }
        )
    )
    assert np.allclose(reference.date.astype(int), given.date.astype(int))
    assert np.allclose(reference.amount, given.amount)
    assert str(given.date.dtype) == "timedelta64[s]"


def test_deserialization_already_loaded():
    reference = easy_timedelta_distribution(start=0, end=6, steps=4, resolution="h")
    given = TD.from_json(
        {
            "__loader__": "bw_temporalis.TemporalDistribution",
            "date_dtype": "timedelta64[s]",
            "date": [0, 60 * 60 * 2, 60 * 60 * 4, 60 * 60 * 6],
            "amount": [0.25, 0.25, 0.25, 0.25],
        }
    )
    assert np.allclose(reference.date.astype(int), given.date.astype(int))
    assert np.allclose(reference.amount, given.amount)
    assert str(given.date.dtype) == "timedelta64[s]"


def test_deserialization_wrong_type():
    with pytest.raises(ValueError):
        TD.from_json({"a", "b", "c"})
