import numpy as np
import pytest

from bw_temporalis import easy_datetime_distribution, easy_timedelta_distribution
from bw_temporalis.temporal_distribution import TemporalDistribution as TD


@pytest.fixture
def simple():
    return TD(np.arange(0, 5, dtype="timedelta64[D]"), np.ones(5) * 2)


def test_init():
    with pytest.raises(ValueError):
        TD(None, None)
    with pytest.raises(ValueError):
        TD(np.arange(5), np.array([2, 2, 2, 2]))
    with pytest.raises(ValueError):
        TD(np.arange(5), np.ones(5) * 2)


def test_mul_wrong_type(simple):
    first = TD(np.arange(0, 5, dtype="datetime64[D]"), np.ones(5) * 2)
    second = TD(np.array((-1, 0, 1), dtype="datetime64[D]"), np.ones(3).astype(float))
    with pytest.raises(ValueError):
        first * second
    with pytest.raises(ValueError):
        second * first


def test_mul_td(simple):
    td2 = TD(np.array((-1, 0, 1), dtype="timedelta64[D]"), np.ones(3).astype(float))

    multiplied = simple * td2

    assert np.array_equal(
        np.arange(-1, 6, dtype="timedelta64[D]"),
        multiplied.date,
    )

    assert simple.amount.sum() * td2.amount.sum() == multiplied.amount.sum()

    assert np.allclose(np.array((2.0, 4.0, 6.0, 6.0, 6.0, 4.0, 2.0)), multiplied.amount)


def test_div_td_error(simple):
    td2 = TD(np.array((-1, 0, 1), dtype="datetime64[D]"), np.ones(3).astype(float))
    with pytest.raises(ValueError):
        td2 / 2


def test_div_td_error_two(simple):
    td2 = TD(np.array((-1, 0, 1), dtype="timedelta64[D]"), np.ones(3).astype(float))
    with pytest.raises(ValueError):
        td2 / "w00t"


def test_div_int(simple):
    """check possible division between td and int"""
    divided = simple / 2.0
    assert np.allclose(divided.amount, 1)
    assert np.array_equal(
        np.arange(0, 5, dtype="timedelta64[D]"),
        divided.date,
    )


def test_mul_number_td(simple):
    simple *= 5
    assert np.array_equal(
        simple.date,
        np.arange(0, 5, dtype="timedelta64[D]"),
    )
    assert np.allclose(simple.amount, np.ones(5) * 10)


def test_mul_number_dt(simple):
    a = TD(np.arange(0, 5, dtype="datetime64[D]"), np.ones(5))
    a *= 5
    assert np.array_equal(
        a.date,
        np.arange(0, 5, dtype="datetime64[D]"),
    )
    assert np.allclose(a.amount, np.ones(5) * 5)


def test_nonzero():
    td = TD(
        np.array([0, 0, 1, 1, 2], dtype="timedelta64[D]"), np.array([0, 0, 1, 2, 3])
    ).nonzero()
    assert np.array_equal(td.amount, np.array([1, 2, 3]))
    assert np.array_equal(
        td.date,
        np.array([1, 1, 2], dtype="timedelta64[D]"),
    )


def test_add_error_two_datetime(simple):
    a = TD(np.arange(0, 5, dtype="datetime64[D]"), np.ones(5) * 2)
    b = TD(np.arange(0, 5, dtype="datetime64[D]"), np.ones(5) * 2)
    with pytest.raises(ValueError):
        a + b


def test_add_error_wrong_type(simple):
    a = TD(np.arange(0, 5, dtype="datetime64[D]"), np.ones(5) * 2)
    with pytest.raises(ValueError):
        a + "w00t"


def test_add_timedelta_error(simple):
    a = TD(np.arange(0, 3, dtype="datetime64[D]"), np.ones(3))
    with pytest.raises(ValueError):
        simple + a
    with pytest.raises(ValueError):
        a + simple


def test_add_number(simple):
    simple += 5
    assert np.array_equal(
        simple.date,
        np.arange(0, 5, dtype="timedelta64[D]"),
    )
    assert np.allclose(simple.amount, np.ones(5) * 7)
    simple += 0.5
    assert np.array_equal(
        simple.date,
        np.arange(0, 5, dtype="timedelta64[D]"),
    )
    assert np.allclose(simple.amount, np.ones(5) * 7.5)


def test_add_two_tds(simple):
    td2 = TD(np.array((-1, 0, 1), dtype="timedelta64[D]"), np.ones(3).astype(float))
    added = simple + td2
    assert np.array_equal(
        np.array([-1, 0, 1, 2, 3, 4], dtype="timedelta64[D]"),
        added.date,
    )
    assert added.amount.sum() == (10 + 3)
    assert np.array_equal(added.amount, [1, 3, 3, 2, 2, 2])


def test_add_td_to_dt(simple):
    td2 = TD(np.array((-1, 0, 1, 2, 3), dtype="datetime64[D]"), np.ones(5))
    added = simple + td2
    expected = np.array([-1, 1, 3, 5, 7], dtype="datetime64[D]")
    assert np.array_equal(expected, added.date)
    assert np.array_equal(added.amount, np.ones(5) * 3)


def test_add_td_to_dt_error(simple):
    td2 = TD(np.array((-1, 0, 1), dtype="datetime64[D]"), np.ones(3).astype(float))
    with pytest.raises(ValueError):
        simple + td2


def test_str(simple):
    assert str(simple)


def test_repr(simple):
    assert repr(simple)


def test_simplify_timedelta():
    td = easy_timedelta_distribution(
        start=0, end=500, steps=1500, resolution="D"
    ).simplify()
    assert td.date.dtype == np.dtype("timedelta64[s]")
    assert np.allclose(td.amount.sum(), 1)
    assert len(td) <= 1000
    assert td.date.min() >= np.array(0, dtype="timedelta64[D]")
    assert td.date.max() <= np.array(500, dtype="timedelta64[D]")


def test_simplify_timedelta_num_clusters():
    td = easy_timedelta_distribution(
        start=0, end=500, steps=1250, resolution="D"
    ).simplify(num_clusters=25)
    assert td.date.dtype == np.dtype("timedelta64[s]")
    assert np.allclose(td.amount.sum(), 1)
    assert len(td) <= 25
    assert td.date.min() >= np.array(0, dtype="timedelta64[D]")
    assert td.date.max() <= np.array(500, dtype="timedelta64[D]")


def test_simplify_timedelta_skip():
    td = easy_timedelta_distribution(
        start=0, end=500, steps=25, resolution="D"
    ).simplify()
    assert td.simplify() is td


def test_simplify_datetime():
    td = easy_datetime_distribution(
        start="2023-01-01", end="2023-12-31", steps=1500
    ).simplify()
    assert td.date.dtype == np.dtype("datetime64[s]")
    assert np.allclose(td.amount.sum(), 1)
    assert len(td) <= 1000
    assert td.date.min() >= np.array("2023-01-01", dtype="datetime64[s]")
    assert td.date.max() <= np.array("2023-12-31", dtype="datetime64[s]")
