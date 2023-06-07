import json

import numpy as np
import pytest

from bw_temporalis import (
    FixedTimeOfYear,
    TemporalDistribution,
    easy_datetime_distribution,
    easy_timedelta_distribution,
)


def test_ftoy_negative_error():
    with pytest.raises(ValueError):
        FixedTimeOfYear(easy_timedelta_distribution(-10, 10, resolution="D", steps=10))


def test_ftoy_multiplication_simple():
    ftoy = FixedTimeOfYear(
        temporal_distribution=easy_timedelta_distribution(
            0, 60, resolution="D", steps=4
        )
    )
    atd = easy_datetime_distribution(start="2020-02-01", end="2020-08-01", steps=4)
    result = atd * ftoy
    expected = TemporalDistribution(
        date=np.array(
            [
                "2019-01-01",
                "2019-01-21",
                "2019-02-10",
                "2019-03-02",
                "2020-01-01",
                "2020-01-21",
                "2020-02-10",
                "2020-03-01",
            ],
            dtype="datetime64[s]",
        ),
        amount=np.array(
            [1 / 16, 1 / 16, 1 / 16, 1 / 16, 3 / 16, 3 / 16, 3 / 16, 3 / 16]
        ),
    )
    assert np.array_equal(result.date, expected.date)
    assert np.array_equal(result.amount, expected.amount)


def test_ftoy_multiplication_allow_overlap():
    ftoy = FixedTimeOfYear(
        temporal_distribution=easy_timedelta_distribution(
            0, 60, resolution="D", steps=4
        ),
        allow_overlap=True,
    )
    atd = easy_datetime_distribution(start="2020-02-01", end="2020-08-01", steps=4)
    result = atd * ftoy
    expected = TemporalDistribution(
        date=np.array(
            [
                "2020-01-01",
                "2020-01-21",
                "2020-02-10",
                "2020-03-01",
            ],
            dtype="datetime64[s]",
        ),
        amount=np.array([1 / 4, 1 / 4, 1 / 4, 1 / 4]),
    )
    assert np.array_equal(result.date, expected.date)
    assert np.array_equal(result.amount, expected.amount)


def test_ftoy_serialization():
    ftoy = FixedTimeOfYear(
        temporal_distribution=easy_timedelta_distribution(
            start=0, end=6, steps=4, resolution="h"
        )
    )
    expected = json.dumps(
        {
            "__loader__": "bw_temporalis.example_functions.FixedTimeOfYear.from_json",
            "date_dtype": "timedelta64[s]",
            "date": [0, 60 * 60 * 2, 60 * 60 * 4, 60 * 60 * 6],
            "amount": [0.25, 0.25, 0.25, 0.25],
            "allow_overlap": False,
        }
    )
    assert ftoy.to_json() == expected


def test_ftoy_serialization_overlap():
    ftoy = FixedTimeOfYear(
        temporal_distribution=easy_timedelta_distribution(
            start=0, end=6, steps=4, resolution="h"
        ),
        allow_overlap=True,
    )
    expected = json.dumps(
        {
            "__loader__": "bw_temporalis.example_functions.FixedTimeOfYear.from_json",
            "date_dtype": "timedelta64[s]",
            "date": [0, 60 * 60 * 2, 60 * 60 * 4, 60 * 60 * 6],
            "amount": [0.25, 0.25, 0.25, 0.25],
            "allow_overlap": True,
        }
    )
    assert ftoy.to_json() == expected


def test_ftoy_deserialization_string():
    reference = FixedTimeOfYear(
        temporal_distribution=easy_timedelta_distribution(
            start=0, end=6, steps=4, resolution="h"
        )
    )
    given = FixedTimeOfYear.from_json(
        json.dumps(
            {
                "__loader__": "bw_temporalis.example_functions.FixedTimeOfYear.from_json",
                "date_dtype": "timedelta64[s]",
                "date": [0, 60 * 60 * 2, 60 * 60 * 4, 60 * 60 * 6],
                "amount": [0.25, 0.25, 0.25, 0.25],
                "allow_overlap": False,
            }
        )
    )
    assert np.allclose(reference.td.date.astype(int), given.td.date.astype(int))
    assert np.allclose(reference.td.amount, given.td.amount)
    assert not given.allow_overlap
    assert str(given.td.date.dtype) == "timedelta64[s]"


def test_ftoy_deserialization_dict():
    reference = FixedTimeOfYear(
        temporal_distribution=easy_timedelta_distribution(
            start=0, end=6, steps=4, resolution="h"
        )
    )
    given = FixedTimeOfYear.from_json(
        {
            "__loader__": "bw_temporalis.example_functions.FixedTimeOfYear.from_json",
            "date_dtype": "timedelta64[s]",
            "date": [0, 60 * 60 * 2, 60 * 60 * 4, 60 * 60 * 6],
            "amount": [0.25, 0.25, 0.25, 0.25],
            "allow_overlap": False,
        }
    )
    assert np.allclose(reference.td.date.astype(int), given.td.date.astype(int))
    assert np.allclose(reference.td.amount, given.td.amount)
    assert not given.allow_overlap
    assert str(given.td.date.dtype) == "timedelta64[s]"
