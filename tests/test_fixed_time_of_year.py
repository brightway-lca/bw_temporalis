import json

import numpy as np
import pytest

from bw_temporalis import (
    FixedTimeOfYearTD,
    TemporalDistribution,
    easy_datetime_distribution,
    easy_timedelta_distribution,
)


def test_ftoy_negative_error():
    td = easy_timedelta_distribution(-10, 10, resolution="D", steps=10)
    with pytest.raises(ValueError):
        FixedTimeOfYearTD(td.date, td.amount)


def test_ftoy_past_one_year_error():
    td = easy_timedelta_distribution(350, 370, resolution="D", steps=10)
    with pytest.raises(ValueError):
        FixedTimeOfYearTD(td.date, td.amount)


def test_ftoy_multiplication_simple():
    td = easy_timedelta_distribution(0, 60, resolution="D", steps=4)
    ftoy = FixedTimeOfYearTD(td.date, td.amount)
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


def test_ftoy_multiplication_td_spans_one_year():
    td = easy_timedelta_distribution(0, 60, resolution="D", steps=4)
    ftoy = FixedTimeOfYearTD(td.date, td.amount)
    atd = easy_datetime_distribution(start="2020-03-01", end="2021-09-01", steps=6)
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
                "2021-01-01",
                "2021-01-21",
                "2021-02-10",
                "2021-03-02",
            ],
            dtype="datetime64[s]",
        ),
        amount=np.array(
            [
                1 / 24,
                1 / 24,
                1 / 24,
                1 / 24,
                3 / 24,
                3 / 24,
                3 / 24,
                3 / 24,
                2 / 24,
                2 / 24,
                2 / 24,
                2 / 24,
            ]
        ),
    )
    assert np.array_equal(result.date, expected.date)
    assert np.array_equal(result.amount, expected.amount)


def test_ftoy_multiplication_allow_overlap():
    td = easy_timedelta_distribution(0, 60, resolution="D", steps=4)
    ftoy = FixedTimeOfYearTD(
        date=td.date,
        amount=td.amount,
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
    td = easy_timedelta_distribution(start=0, end=6, steps=4, resolution="h")
    ftoy = FixedTimeOfYearTD(date=td.date, amount=td.amount)
    expected = json.dumps(
        {
            "__loader__": "bw_temporalis.FixedTimeOfYearTD",
            "date_dtype": "timedelta64[s]",
            "date": [0, 60 * 60 * 2, 60 * 60 * 4, 60 * 60 * 6],
            "amount": [0.25, 0.25, 0.25, 0.25],
            "allow_overlap": False,
        }
    )
    assert ftoy.to_json() == expected


def test_ftoy_serialization_overlap():
    td = easy_timedelta_distribution(start=0, end=6, steps=4, resolution="h")
    ftoy = FixedTimeOfYearTD(
        date=td.date,
        amount=td.amount,
        allow_overlap=True,
    )
    expected = json.dumps(
        {
            "__loader__": "bw_temporalis.FixedTimeOfYearTD",
            "date_dtype": "timedelta64[s]",
            "date": [0, 60 * 60 * 2, 60 * 60 * 4, 60 * 60 * 6],
            "amount": [0.25, 0.25, 0.25, 0.25],
            "allow_overlap": True,
        }
    )
    assert ftoy.to_json() == expected


def test_ftoy_deserialization_string():
    td = easy_timedelta_distribution(start=0, end=6, steps=4, resolution="h")
    reference = FixedTimeOfYearTD(
        date=td.date,
        amount=td.amount,
    )
    given = FixedTimeOfYearTD.from_json(
        json.dumps(
            {
                "__loader__": "bw_temporalis.FixedTimeOfYearTD",
                "date_dtype": "timedelta64[s]",
                "date": [0, 60 * 60 * 2, 60 * 60 * 4, 60 * 60 * 6],
                "amount": [0.25, 0.25, 0.25, 0.25],
                "allow_overlap": False,
            }
        )
    )
    assert np.allclose(reference.date.astype(int), given.date.astype(int))
    assert np.allclose(reference.amount, given.amount)
    assert not given.allow_overlap
    assert str(given.date.dtype) == "timedelta64[s]"


def test_ftoy_deserialization_dict():
    td = easy_timedelta_distribution(start=0, end=6, steps=4, resolution="h")
    reference = FixedTimeOfYearTD(
        date=td.date,
        amount=td.amount,
    )
    given = FixedTimeOfYearTD.from_json(
        {
            "__loader__": "bw_temporalis.FixedTimeOfYearTD",
            "date_dtype": "timedelta64[s]",
            "date": [0, 60 * 60 * 2, 60 * 60 * 4, 60 * 60 * 6],
            "amount": [0.25, 0.25, 0.25, 0.25],
            "allow_overlap": False,
        }
    )
    assert np.allclose(reference.date.astype(int), given.date.astype(int))
    assert np.allclose(reference.amount, given.amount)
    assert not given.allow_overlap
    assert str(given.date.dtype) == "timedelta64[s]"
