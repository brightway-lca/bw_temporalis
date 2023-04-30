import numpy as np
import pytest

from bw_temporalis.convolution import temporal_convolution_datetime_timedelta as tcdt
from bw_temporalis.convolution import temporal_convolution_timedelta_timedelta as tctt


def test_conversion_datetime_to_int_and_back():
    a = np.array(["2022-01-01", "2023-06-08"], dtype="datetime64[s]")
    b = a.copy()
    assert np.array_equal(a.astype(np.int64).astype("datetime64[s]"), b)


def test_conversion_datetime_to_int_and_back_with_modification():
    a = np.array(["2022-01-01", "2023-06-08"], dtype="datetime64[s]")
    b = a.copy() + np.array([1], dtype="timedelta64[D]")
    a_shifted = (a.astype(np.int64) + 24 * 60 * 60).astype("datetime64[s]")
    assert np.array_equal(a_shifted, b)


def test_tcdt_very_early_dates():
    a = np.array(["2022-06-04", "2022-06-05"], dtype="datetime64[s]") - np.array(
        4000, dtype="timedelta64[Y]"
    ).astype("timedelta64[s]")
    b = np.array([7, 8])
    c = np.array([-2, -1], dtype="timedelta64[D]").astype("timedelta64[s]")
    d = np.array([-2, -4], dtype=float)
    date, amount = tcdt(
        first_date=a,
        first_amount=b,
        second_date=c,
        second_amount=d,
    )
    expected_date = np.array(
        [
            "-1978-06-02",
            "-1978-06-03",
            "-1978-06-04",
        ],
        dtype="datetime64[s]",
    )
    expected_amount = np.array(
        [
            -14,
            -44,
            -32,
        ]
    )
    assert np.array_equal(date, expected_date)
    assert np.array_equal(amount, expected_amount)


def test_tcdt():
    a = np.array(
        ["2022-06-04", "2022-06-10", "2022-06-11", "2022-06-12"], dtype="datetime64[s]"
    )
    b = np.array([7, 8, 9, 10])
    c = np.array([-2, -1, 0, 5, 10], dtype="timedelta64[D]").astype("timedelta64[s]")
    d = np.array([-2, -4, 6, 0, 10], dtype=float)
    date, amount = tcdt(
        first_date=a,
        first_amount=b,
        second_date=c,
        second_amount=d,
    )
    expected_date = np.array(
        [
            "2022-06-02",
            "2022-06-03",
            "2022-06-04",
            "2022-06-08",
            "2022-06-09",
            "2022-06-10",
            "2022-06-11",
            "2022-06-12",
            "2022-06-14",
            "2022-06-20",
            "2022-06-21",
            "2022-06-22",
        ],
        dtype="datetime64[s]",
    )
    expected_amount = np.array(
        [
            -14,
            -28,
            42,
            -16,
            -32 - 18,
            48 - 36 - 20,
            54 - 40,
            60,
            70,
            80,
            90,
            100,
        ]
    )
    assert np.array_equal(date, expected_date)
    assert np.array_equal(amount, expected_amount)


def test_tcdt_dt_length_one():
    a = np.array(["2022-06-04"], dtype="datetime64[s]")
    b = np.array([0.5])
    c = np.array([-2, -1, 0, 5, 10], dtype="timedelta64[D]").astype("timedelta64[s]")
    d = np.array([-2, -4, 6, 0, 10], dtype=float)
    date, amount = tcdt(
        first_date=a,
        first_amount=b,
        second_date=c,
        second_amount=d,
    )
    expected_date = np.array(
        ["2022-06-02", "2022-06-03", "2022-06-04", "2022-06-14"], dtype="datetime64[s]"
    )
    expected_amount = np.array([-1, -2, 3, 5], dtype=float)
    assert np.array_equal(date, expected_date)
    assert np.array_equal(amount, expected_amount)


def test_tcdt_td_length_one():
    a = np.array(
        ["2022-06-04", "2022-06-10", "2022-06-11", "2022-06-12"], dtype="datetime64[s]"
    )
    b = np.array([7, 8, 9, 10])
    c = np.array([-1], dtype="timedelta64[D]").astype("timedelta64[s]")
    d = np.array([-2], dtype=float)
    date, amount = tcdt(
        first_date=a,
        first_amount=b,
        second_date=c,
        second_amount=d,
    )
    expected_date = np.array(
        [
            "2022-06-03",
            "2022-06-09",
            "2022-06-10",
            "2022-06-11",
        ],
        dtype="datetime64[s]",
    )
    expected_amount = np.array(
        [
            -14,
            -16,
            -18,
            -20,
        ]
    )
    assert np.array_equal(date, expected_date)
    assert np.array_equal(amount, expected_amount)


def test_tcdt_wrong_temporal_resolution_first():
    a = np.array(["2022-01-01", "2023-06-08"], dtype="datetime64[D]")
    b = np.linspace(0, 0.5, 2)
    c = np.arange(5, dtype="timedelta64[s]")
    d = np.linspace(0, 0.5, 5)
    with pytest.raises(ValueError) as exc:
        tcdt(
            first_date=a,
            first_amount=b,
            second_date=c,
            second_amount=d,
        )
        assert (
            str(exc.value)
            == "`first_date` must have dtype `datetime64[s]`, but got `datetime64[D]`"
        )


def test_tcdt_wrong_temporal_resolution_second():
    a = np.array(["2022-01-01", "2023-06-08"], dtype="datetime64[s]")
    b = np.linspace(0, 0.5, 2)
    c = np.arange(5, dtype="timedelta64[D]")
    d = np.linspace(0, 0.5, 5)
    with pytest.raises(ValueError) as exc:
        tcdt(
            first_date=a,
            first_amount=b,
            second_date=c,
            second_amount=d,
        )
        assert (
            str(exc.value)
            == "`second_date` must have dtype `timedelta64[s]`, but got `timedelta64[D]`"
        )


def test_tctt_wrong_temporal_resolution_first():
    a = np.array([10, 11, 12], dtype="timedelta64[D]")
    b = np.linspace(0, 0.5, 2)
    c = np.arange(5, dtype="timedelta64[s]")
    d = np.linspace(0, 0.5, 5)
    with pytest.raises(ValueError) as exc:
        tctt(
            first_date=a,
            first_amount=b,
            second_date=c,
            second_amount=d,
        )
        assert (
            str(exc.value)
            == "`first_date` must have dtype `timedelta64[s]`, but got `timedelta64[D]`"
        )


def test_tctt_wrong_temporal_resolution_second():
    a = np.array([10, 11, 12], dtype="timedelta64[s]")
    b = np.linspace(0, 0.5, 2)
    c = np.arange(5, dtype="timedelta64[D]")
    d = np.linspace(0, 0.5, 5)
    with pytest.raises(ValueError) as exc:
        tctt(
            first_date=a,
            first_amount=b,
            second_date=c,
            second_amount=d,
        )
        assert (
            str(exc.value)
            == "`second_date` must have dtype `timedelta64[s]`, but got `timedelta64[D]`"
        )


def test_tctt():
    a = np.array([4, 10, 11, 12], dtype="timedelta64[D]").astype("timedelta64[s]")
    b = np.array([7, 8, 9, 10])
    c = np.array([-2, -1, 0, 5, 10], dtype="timedelta64[D]").astype("timedelta64[s]")
    d = np.array([-2, -4, 6, 0, 10], dtype=float)
    date, amount = tctt(
        first_date=a,
        first_amount=b,
        second_date=c,
        second_amount=d,
    )
    expected_date = np.array(
        [2, 3, 4, 8, 9, 10, 11, 12, 14, 20, 21, 22], dtype="timedelta64[D]"
    ).astype("timedelta64[s]")
    expected_amount = np.array(
        [
            -14,
            -28,
            42,
            -16,
            -32 - 18,
            48 - 36 - 20,
            54 - 40,
            60,
            70,
            80,
            90,
            100,
        ]
    )
    assert np.array_equal(date, expected_date)
    assert np.array_equal(amount, expected_amount)


def test_tctt_only_one():
    a = np.array([4], dtype="timedelta64[D]").astype("timedelta64[s]")
    b = np.array([7])
    c = np.array([-2, -1, 0, 5, 10], dtype="timedelta64[D]").astype("timedelta64[s]")
    d = np.array([-2, -4, 6, 0, 10], dtype=float)
    date, amount = tctt(
        first_date=a,
        first_amount=b,
        second_date=c,
        second_amount=d,
    )
    expected_date = np.array([2, 3, 4, 14], dtype="timedelta64[D]").astype(
        "timedelta64[s]"
    )
    expected_amount = np.array(
        [
            -14,
            -28,
            42,
            70,
        ]
    )
    assert np.array_equal(date, expected_date)
    assert np.array_equal(amount, expected_amount)


def test_tctt_negative_deltas():
    a = np.array([-4, 10, 11, -12], dtype="timedelta64[D]").astype("timedelta64[s]")
    b = np.array([7, 8, 9, 10])
    c = np.array([-2, -1, 0, 5, 10], dtype="timedelta64[D]").astype("timedelta64[s]")
    d = np.array([-2, -4, 6, 0, 10], dtype=float)
    date, amount = tctt(
        first_date=a,
        first_amount=b,
        second_date=c,
        second_amount=d,
    )
    expected_date = np.array(
        [-14, -13, -12, -6, -5, -4, -2, 6, 8, 9, 10, 11, 20, 21], dtype="timedelta64[D]"
    ).astype("timedelta64[s]")
    expected_amount = np.array(
        [-20, -40, 60, -14, -28, 42, 100, 70, -16, -50, 12, 54, 80, 90]
    )
    assert np.array_equal(date, expected_date)
    assert np.array_equal(amount, expected_amount)
