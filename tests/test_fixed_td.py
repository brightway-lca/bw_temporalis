import json

import numpy as np
import pytest

from bw_temporalis import (
    FixedTD,
    FixedTimeOfYearTD,
    TDAware,
    easy_datetime_distribution,
    easy_timedelta_distribution,
)


@pytest.fixture
def ftd():
    return FixedTD(
        date=np.array(
            ["2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01", "2024-01-01"],
            dtype="datetime64[s]",
        ),
        amount=np.array([0.2] * 5),
    )


def test_ftd_serialization(ftd):
    expected = json.dumps(
        {
            "__loader__": "bw_temporalis.FixedTD",
            "date_dtype": "datetime64[s]",
            "date": [1577836800, 1609459200, 1640995200, 1672531200, 1704067200],
            "amount": [0.2] * 5,
        }
    )
    assert ftd.to_json() == expected


def test_ftd_addition_error(ftd):
    with pytest.raises(ValueError):
        ftd + 1


def test_ftd_multiply_error(ftd):
    with pytest.raises(ValueError):
        ftd * "w00t"


def test_ftd_multiply_number(ftd):
    result = ftd * 8
    assert np.array_equal(ftd.date, result.date)
    assert np.array_equal(result.amount, ftd.amount * 8)


def test_ftd_multiply_dynamic_function(ftd):
    class W00t(TDAware):
        def __mul__(self, other):
            return 42

    assert ftd * W00t() == 42


def test_ftd_multiply_fixed_td(ftd):
    ftd.amount *= 0.5

    other_td = easy_datetime_distribution(start="2030-01-01", end="2034-01-01", steps=5)
    other = FixedTD(date=other_td.date, amount=other_td.amount * 5)
    result = ftd * other

    assert np.array_equal(other.date, result.date)
    assert np.array_equal(result.amount, other.amount * ftd.amount.sum())


def test_ftd_fixed_time_of_year(ftd):
    td = easy_timedelta_distribution(start=1, end=2, steps=2, resolution="D")
    other = FixedTimeOfYearTD(
        date=td.date,
        amount=td.amount,
    )
    result = ftd * other
    expected = np.array(
        [
            "2019-01-02",
            "2019-01-03",
            "2020-01-02",
            "2020-01-03",
            "2021-01-02",
            "2021-01-03",
            "2022-01-02",
            "2022-01-03",
            "2023-01-02",
            "2023-01-03",
        ],
        dtype="datetime64[s]",
    )

    assert np.array_equal(np.array([0.1] * 10), result.amount)
    assert np.array_equal(expected.astype(int), result.date.astype(int))


def test_ftd_multiply_absolute_td(ftd):
    td = easy_datetime_distribution(start="1020-01-01", end="1024-01-01", steps=5)
    td.amount *= 10
    result = ftd * td
    assert np.array_equal(ftd.date, result.date)
    assert np.array_equal(ftd.amount * 10, result.amount)


def test_ftd_multiply_relative_td(ftd):
    td = easy_timedelta_distribution(start=1, end=2, steps=2, resolution="D")
    result = ftd * td
    expected = np.array(
        [
            "2020-01-02",
            "2020-01-03",
            "2021-01-02",
            "2021-01-03",
            "2022-01-02",
            "2022-01-03",
            "2023-01-02",
            "2023-01-03",
            "2024-01-02",
            "2024-01-03",
        ],
        dtype="datetime64[s]",
    )

    assert np.array_equal(np.array([0.1] * 10), result.amount)
    assert np.array_equal(expected.astype(int), result.date.astype(int))
