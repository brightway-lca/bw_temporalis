import bw2data as bd
import numpy as np
import pytest
from bw2data.tests import bw2test

from bw_temporalis import (
    IncongruentDistribution,
    TemporalDistribution,
    check_database_exchanges,
    easy_datetime_distribution,
    easy_timedelta_distribution,
)
from bw_temporalis.utils import normalized_data_array


@pytest.fixture
@bw2test
def db():
    db = bd.Database("test")
    db.register()
    a = db.new_node(code="b", name="a", location="c", unit="d")
    a.save()
    b = db.new_node(code="f", name="e", location="g", unit="h")
    b.save()
    return a, b


def test_check_database_exchanges(db):
    a, b = db
    db = bd.Database("test")
    a.new_exchange(
        input=b,
        type="technosphere",
        amount=4.5,
        temporal_distribution=TemporalDistribution(
            date=np.array([1, 2, 3], dtype="timedelta64[D]"),
            amount=np.array([1.5, 1.5, 1.5]) / 4.5,
        ),
    ).save()
    assert check_database_exchanges("test") is None


def test_check_database_exchanges_small_difference_ok(db):
    a, b = db
    a.new_exchange(
        input=b,
        type="technosphere",
        amount=4.5,
        temporal_distribution=TemporalDistribution(
            date=np.array([1, 2, 3], dtype="timedelta64[D]"),
            amount=np.array([1.5, 1.5, 1.5]) / 4.5 * 0.995,
        ),
    ).save()
    assert check_database_exchanges("test") is None


def test_check_database_exchanges_error(db):
    EXPECTED = """
    Temporal distribution in exchange doesn't sum to one:
    Input:
        'e' (h, g, None)
        id: 2
    Output
        'a' (d, c, None)
        id: 1
    Exchange amount: 4.5000e+00
    Temporal distribution sum: 4.4100e+00
    """

    a, b = db
    a.new_exchange(
        input=b,
        type="technosphere",
        amount=4.5,
        temporal_distribution=TemporalDistribution(
            date=np.array([1, 2, 3], dtype="timedelta64[D]"),
            amount=np.array([1.5, 1.5, 1.5]) / 4.5 * 0.98,
        ),
    ).save()
    with pytest.raises(IncongruentDistribution) as exc:
        check_database_exchanges("test")
        assert str(exc.value) == EXPECTED


def test_normalized_data_array_invalid_kind():
    with pytest.raises(ValueError):
        normalized_data_array(10, "foo", None)


def test_normalized_data_array_normal_no_param():
    with pytest.raises(ValueError):
        normalized_data_array(10, "normal", None)


def test_normalized_data_array_normal_invalid_param():
    with pytest.raises(ValueError):
        normalized_data_array(10, "normal", 0)
    with pytest.raises(ValueError):
        normalized_data_array(10, "normal", -0.2)


def test_normalized_data_array_normal():
    arr = normalized_data_array(10, "normal", 0.2)
    expected = np.array(
        [
            0.0876415,
            0.30121448,
            0.76032691,
            1.40955938,
            1.91922054,
            1.91922054,
            1.40955938,
            0.76032691,
            0.30121448,
            0.0876415,
        ]
    )
    assert np.allclose(arr, expected)
    assert arr.shape == (10,)

    arr = normalized_data_array(5, "normal", 0.5)
    expected = np.array([0.48394145, 0.70413065, 0.79788456, 0.70413065, 0.48394145])
    assert np.allclose(arr, expected)
    assert arr.shape == (5,)


def test_normalized_data_array_uniform():
    arr = normalized_data_array(3, "uniform", None)
    assert np.array_equal(arr, [1, 1, 1])


def test_normalized_data_array_triangular_invalid_param():
    with pytest.raises(ValueError):
        normalized_data_array(10, "trianguarl", "foo")
    with pytest.raises(ValueError):
        normalized_data_array(10, "trianguarl", 1.2)
    with pytest.raises(ValueError):
        normalized_data_array(10, "trianguarl", -0.2)


def test_normalized_data_array_triangular():
    arr = normalized_data_array(5, "triangular", None)
    expected = np.array([0, 1, 2, 1, 0])
    assert np.allclose(arr, expected)
    assert arr.shape == (5,)

    arr = normalized_data_array(5, "triangular", 0.2)
    expected = np.array([0.0, 1.875, 1.25, 0.625, 0.0])
    assert np.allclose(arr, expected)
    assert arr.shape == (5,)


def test_easy_datetime_distribution_invalid_steps():
    with pytest.raises(ValueError):
        easy_datetime_distribution("2023-01-01", "now", steps="foo")
    with pytest.raises(ValueError):
        easy_datetime_distribution("2023-01-01", "now", steps=1)


def test_easy_datetime_distribution_invalid_start():
    with pytest.raises(ValueError):
        easy_datetime_distribution("now", "2023-01-01")


def test_easy_datetime_distribution_triangular_invalid_mode():
    with pytest.raises(ValueError):
        easy_datetime_distribution(
            start="2023-01-01",
            end="2023-01-05",
            steps=5,
            kind="triangular",
            param="2023-01-06",
        )
    with pytest.raises(ValueError):
        easy_datetime_distribution(
            start="2023-01-01",
            end="2023-01-05",
            steps=5,
            kind="triangular",
            param="2022-12-31",
        )


def test_easy_datetime_distribution_triangular_too_few_steps():
    with pytest.raises(ValueError, match="must have at least three steps"):
        easy_datetime_distribution(
            start="2023-01-01",
            end="2023-01-05",
            steps=2,
            kind="triangular",
        )


def test_easy_datetime_distribution_triangular_mode_at_bounds():
    td = easy_datetime_distribution(
        start="2023-01-01",
        end="2023-01-05",
        steps=5,
        kind="triangular",
        param="2023-01-01",
    )
    amount_expected = np.array([2.0, 1.5, 1.0, 0.5, 0.0])
    amount_expected *= 1 / amount_expected.sum()
    date_expected = np.array(
        ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"],
        "datetime64[D]",
    ).astype("datetime64[s]")

    assert date_expected.dtype == td.date.dtype
    assert np.allclose(td.date.astype(int), date_expected.astype(int))
    assert np.allclose(td.amount, amount_expected)


def test_easy_datetime_distribution_triangular_specify_mode():
    td = easy_datetime_distribution(
        start="2023-01-01",
        end="2023-01-05",
        steps=5,
        kind="triangular",
        param="2023-01-02",
    )
    amount_expected = np.array([0.0, 1.875, 1.25, 0.625, 0.0])
    amount_expected *= 1 / amount_expected.sum()
    date_expected = np.array(
        ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"],
        "datetime64[D]",
    ).astype("datetime64[s]")

    assert date_expected.dtype == td.date.dtype
    assert np.allclose(td.date.astype(int), date_expected.astype(int))
    assert np.allclose(td.amount, amount_expected)


def test_easy_datetime_distribution_triangular_default_mode():
    td = easy_datetime_distribution(
        start="2023-01-01", end="2023-01-05", steps=5, kind="triangular"
    )
    amount_expected = np.array([0, 1, 2, 1, 0]).astype(float)
    amount_expected *= 1 / amount_expected.sum()
    date_expected = np.array(
        ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"],
        "datetime64[D]",
    ).astype("datetime64[s]")

    assert date_expected.dtype == td.date.dtype
    assert np.allclose(td.date.astype(int), date_expected.astype(int))
    assert np.allclose(td.amount, amount_expected)


def test_easy_datetime_distribution_uniform():
    td = easy_datetime_distribution(start="2023-01-01", end="2023-01-05", steps=5)
    date_expected = np.array(
        ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"],
        "datetime64[D]",
    ).astype("datetime64[s]")
    amount_expected = np.ones(5) / 5

    assert date_expected.dtype == td.date.dtype
    assert np.allclose(td.date.astype(int), date_expected.astype(int))
    assert np.allclose(td.amount, amount_expected)


def test_easy_datetime_distribution_normal():
    td = easy_datetime_distribution(
        start="2023-01-01",
        end="2023-01-05",
        steps=5,
        kind="normal",
        param=0.5,
    )
    amount_expected = np.array(
        [0.48394145, 0.70413065, 0.79788456, 0.70413065, 0.48394145]
    )
    amount_expected *= 1 / amount_expected.sum()
    date_expected = np.array(
        ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"],
        "datetime64[D]",
    ).astype("datetime64[s]")

    assert date_expected.dtype == td.date.dtype
    assert np.allclose(td.date.astype(int), date_expected.astype(int))
    assert np.allclose(td.amount, amount_expected)


def test_easy_datetime_distribution_default_steps():
    td = easy_datetime_distribution(start="2023-01-01", end="2023-05-01")
    assert td.date.shape == (50,)
    assert td.amount.shape == (50,)


def test_easy_timedelta_distribution_invalid_steps():
    with pytest.raises(ValueError):
        easy_timedelta_distribution(-10, 10, resolution="D", steps="foo")
    with pytest.raises(ValueError):
        easy_timedelta_distribution(-10, 10, resolution="D", steps=1)


def test_easy_timedelta_distribution_invalid_start():
    with pytest.raises(ValueError):
        easy_timedelta_distribution(10, -10, resolution="D")


def test_easy_timedelta_distribution_invalid_resolution():
    with pytest.raises(ValueError):
        easy_timedelta_distribution(-10, 10, resolution="Z")


def test_easy_timedelta_distribution_resolutions():
    def f(resolution, constant):
        td = easy_timedelta_distribution(-10, 10, resolution=resolution, steps=21)
        # We should be doing the conversion ourselves, but then we would need to get
        # the rounding right. This is easier for now.
        expected = (
            np.linspace(-10, 10, 21)
            .astype(f"timedelta64[{resolution}]")
            .astype("timedelta64[s]")
            .astype(int)
        )
        assert np.array_equal(td.date.astype(int), expected)

    f("s", 1)
    f("m", 60)
    f("h", 60 * 60)
    f("D", 60 * 60 * 24)
    f("M", 60 * 60 * 24 * 30)
    f("Y", 60 * 60 * 24 * 365)


def test_easy_timedelta_distribution_steps_warning():
    with pytest.warns(UserWarning, match=r"(20 versus 10)"):
        easy_timedelta_distribution(start=1, end=10, resolution="m", steps=20)


def test_easy_timedelta_distribution_uniform():
    td = easy_timedelta_distribution(start=-10, end=10, resolution="m", steps=21)
    date_expected = np.linspace(-10, 10, 21, dtype="timedelta64[m]").astype(
        "timedelta64[s]"
    )
    amount_expected = np.ones(21).astype(float) / 21

    assert date_expected.dtype == td.date.dtype
    assert np.allclose(td.date.astype(int), date_expected.astype(int))
    assert np.allclose(td.amount, amount_expected)


def test_easy_timedelta_distribution_triangular_default_mode():
    td = easy_timedelta_distribution(
        start=-10, end=10, resolution="m", steps=5, kind="triangular"
    )
    date_expected = np.linspace(-10, 10, 5, dtype="timedelta64[m]").astype(
        "timedelta64[s]"
    )
    amount_expected = np.array([0, 1, 2, 1, 0]).astype(float)
    amount_expected *= 1 / amount_expected.sum()

    assert date_expected.dtype == td.date.dtype
    assert np.allclose(td.date.astype(int), date_expected.astype(int))
    assert np.allclose(td.amount, amount_expected)


def test_easy_timedelta_distribution_triangular_mode_at_bounds():
    td = easy_timedelta_distribution(
        start=-10,
        end=10,
        resolution="m",
        steps=5,
        kind="triangular",
        param=10,
    )
    date_expected = np.linspace(-10, 10, 5, dtype="timedelta64[m]").astype(
        "timedelta64[s]"
    )
    amount_expected = np.array([0.0, 1.7, 3.4, 5.1, 6.8]).astype(float)
    amount_expected *= 1 / amount_expected.sum()

    assert date_expected.dtype == td.date.dtype
    assert np.allclose(td.date.astype(int), date_expected.astype(int))
    assert np.allclose(td.amount, amount_expected)


def test_easy_timedelta_distribution_too_few_steps():
    with pytest.raises(ValueError, match="must have at least three steps"):
        easy_timedelta_distribution(
            start=-10,
            end=10,
            resolution="m",
            steps=2,
            kind="triangular",
        )


def test_easy_timedelta_distribution_triangular_invalid_mode():
    with pytest.raises(ValueError):
        easy_timedelta_distribution(
            start=-10,
            end=10,
            resolution="m",
            steps=5,
            param=-11,
            kind="triangular",
        )
        easy_timedelta_distribution(
            start=-10,
            end=10,
            resolution="m",
            steps=5,
            param=11,
            kind="triangular",
        )


def test_easy_timedelta_distribution_triangular_custom_mode():
    td = easy_timedelta_distribution(
        start=-10,
        end=10,
        resolution="m",
        steps=5,
        param=-6,
        kind="triangular",
    )
    date_expected = np.linspace(-10, 10, 5, dtype="timedelta64[m]").astype(
        "timedelta64[s]"
    )
    amount_expected = np.array([0.0, 1.875, 1.25, 0.625, 0.0]).astype(float)
    amount_expected *= 1 / amount_expected.sum()

    assert date_expected.dtype == td.date.dtype
    assert np.allclose(td.date.astype(int), date_expected.astype(int))
    assert np.allclose(td.amount, amount_expected)


def test_easy_timedelta_distribution_normal():
    td = easy_timedelta_distribution(
        start=-10, end=10, resolution="m", steps=5, param=0.2, kind="normal"
    )
    date_expected = np.linspace(-10, 10, 5, dtype="timedelta64[m]").astype(
        "timedelta64[s]"
    )
    amount_expected = np.array(
        [0.37280396, 3.8847065, 8.48497908, 3.8847065, 0.37280396]
    ).astype(float)
    amount_expected *= 1 / amount_expected.sum()

    assert date_expected.dtype == td.date.dtype
    assert np.allclose(td.date.astype(int), date_expected.astype(int))
    assert np.allclose(td.amount, amount_expected)
