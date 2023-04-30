import bw2data as bd
import numpy as np
import pytest
from bw2data.tests import bw2test

from bw_temporalis import (
    IncongruentDistribution,
    TemporalDistribution,
    check_database_exchanges,
)


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
            amount=np.array([1.5, 1.5, 1.5]),
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
            amount=np.array([1.5, 1.5, 1.5]) * 0.995,
        ),
    ).save()
    assert check_database_exchanges("test") is None


def test_check_database_exchanges_error(db):
    EXPECTED = """
    Temporal distribution in exchange differs from `amount`:
    Input:
        'e' (h, g, None)
        id: 2
    Output
        'a' (d, c, None)
        id: 1
    Exchange amount: 4.5000e+00
    Temporal distribution amount: 4.4100e+00
    """

    a, b = db
    a.new_exchange(
        input=b,
        type="technosphere",
        amount=4.5,
        temporal_distribution=TemporalDistribution(
            date=np.array([1, 2, 3], dtype="timedelta64[D]"),
            amount=np.array([1.5, 1.5, 1.5]) * 0.98,
        ),
    ).save()
    with pytest.raises(IncongruentDistribution) as exc:
        check_database_exchanges("test")
        assert str(exc.value) == EXPECTED
