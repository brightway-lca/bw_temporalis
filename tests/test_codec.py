import json

import numpy as np
import pytest

import bw_temporalis
from bw_temporalis import (
    FixedTD,
    FixedTimeOfYearTD,
    TDAware,
    TemporalDistribution,
    load_temporal_distribution,
)
from bw_temporalis.codec import register, tag

codec = pytest.importorskip("bw2data.codec", reason="bw2data without bw2data.codec")


def _same(first, second):
    assert type(first) is type(second)
    assert np.array_equal(first.date, second.date)
    assert np.array_equal(first.amount, second.amount)


TDS = [
    TemporalDistribution(
        date=np.array([-2, 0, 5], dtype="timedelta64[Y]"), amount=np.array([0.2, 0.5, 0.3])
    ),
    FixedTD(
        date=np.array(["2020-01-01", "2041-06-30"], dtype="datetime64[D]"),
        amount=np.array([0.5, 0.5]),
    ),
    FixedTimeOfYearTD(
        date=np.array([10, 20], dtype="timedelta64[D]"),
        amount=np.array([0.25, 0.75]),
        allow_overlap=True,
    ),
]


@pytest.mark.parametrize("td", TDS, ids=lambda td: type(td).__name__)
def test_round_trip_through_bw2data_codec(td):
    register()
    stored = codec.dumps({"temporal_distribution": td, "amount": 1.0})
    raw = json.loads(stored)["temporal_distribution"]
    assert raw["__bw__"] == f"bw_temporalis.{type(td).__name__}"
    # The payload is what ``to_json()`` writes, so JSON strings stored by older versions
    # can be converted without loading them
    assert raw["v"] == json.loads(td.to_json())

    loaded = codec.loads(stored)["temporal_distribution"]
    _same(loaded, td)
    if isinstance(td, FixedTimeOfYearTD):
        assert loaded.allow_overlap is True


def test_entry_point_is_registered():
    from importlib.metadata import entry_points

    (entry_point,) = [
        ep for ep in entry_points(group="bw2data.codec") if ep.name == "bw_temporalis"
    ]
    assert entry_point.load() is register


def test_register_twice():
    register()
    register()
    _same(codec.loads(codec.dumps(TDS[0])), TDS[0])


class Doubler(TDAware):
    def __init__(self, factor=2, **kwargs):
        self.factor = factor

    def to_json(self) -> str:
        return json.dumps({"__loader__": "tests.Doubler", "factor": self.factor})

    @classmethod
    def from_json(cls, json_obj):
        data = json.loads(json_obj) if isinstance(json_obj, str) else json_obj
        return cls(factor=data["factor"])


def test_user_subclass_registers_itself():
    # The module name depends on how pytest imports the tests
    expected = f"{Doubler.__module__}.Doubler"
    assert tag(Doubler) == expected
    stored = codec.dumps({"temporal_distribution": Doubler(3)})
    assert json.loads(stored)["temporal_distribution"]["__bw__"] == expected
    loaded = codec.loads(stored)["temporal_distribution"]
    assert type(loaded) is Doubler and loaded.factor == 3


def test_user_subclass_of_temporal_distribution_registers_itself():
    class Mine(TemporalDistribution):
        pass

    td = Mine(date=np.array([0, 1], dtype="timedelta64[Y]"), amount=np.array([0.5, 0.5]))
    _same(codec.loads(codec.dumps(td)), td)


def test_class_defined_again_replaces_the_earlier_one():
    def define():
        class Again(TDAware):
            def to_json(self):
                return json.dumps({"__loader__": "Again"})

            @classmethod
            def from_json(cls, json_obj):
                return cls()

        return Again

    first, second = define(), define()
    assert type(codec.loads(codec.dumps(second()))) is second
    with pytest.raises(TypeError):
        codec.dumps(first())


def test_class_without_json_methods_is_not_registered():
    class Plain(TDAware):
        pass

    with pytest.raises(TypeError, match="Plain"):
        codec.dumps(Plain())


def test_value_of_undefined_class_is_restored_with_loader_registry(monkeypatch):
    # Stored by a process where the class was defined, read where it isn't
    stored = b'{"td": {"__bw__": "some_module.Gone", "v": {"__loader__": "tests.Doubler", "factor": 5}}}'
    value = codec.loads(stored)["td"]
    assert type(value) is codec.UnknownTaggedValue
    monkeypatch.setitem(bw_temporalis.loader_registry, "tests.Doubler", Doubler.from_json)
    restored = load_temporal_distribution(value)
    assert type(restored) is Doubler and restored.factor == 5


def test_loader_registered_for_the_tag_is_preferred(monkeypatch):
    stored = b'{"td": {"__bw__": "some_module.Gone", "v": {"__loader__": "tests.Other", "factor": 5}}}'
    monkeypatch.setitem(bw_temporalis.loader_registry, "some_module.Gone", Doubler.from_json)
    restored = load_temporal_distribution(codec.loads(stored)["td"])
    assert type(restored) is Doubler and restored.factor == 5


def test_value_of_undefined_subclass_is_restored_as_parent_with_warning():
    # A subclass which inherits ``to_json`` writes the ``__loader__`` of its parent
    payload = json.loads(TDS[1].to_json())
    value = codec.loads(json.dumps({"__bw__": "some_module.MyFixed", "v": payload}))
    assert type(value) is codec.UnknownTaggedValue
    with pytest.warns(UserWarning, match="some_module.MyFixed"):
        restored = load_temporal_distribution(value)
    _same(restored, TDS[1])


def test_value_of_undefined_class_without_loader_names_the_tag():
    value = codec.loads(b'{"__bw__": "some_module.Gone", "v": {"__loader__": "tests.Nope"}}')
    with pytest.raises(KeyError, match="some_module.Gone"):
        load_temporal_distribution(value)


def test_legacy_loader_name_as_tag():
    # ``to_json`` never wrote this name, but it is in ``loader_registry``
    payload = json.loads(TDS[2].to_json()) | {"__loader__": "bw_temporalis.FixedTimeOfYear"}
    value = codec.loads(json.dumps({"__bw__": "bw_temporalis.FixedTimeOfYear", "v": payload}))
    _same(load_temporal_distribution(value), TDS[2])
