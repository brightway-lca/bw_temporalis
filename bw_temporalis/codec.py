"""Store temporal distributions in ``bw2data``'s tagged JSON (``bw2data.codec``).

``bw2data`` stores the data of exchanges as JSON. Values that JSON can't hold, like a
``TemporalDistribution``, are stored as *tagged values*::

    {"__bw__": "bw_temporalis.TemporalDistribution", "v": {"__loader__": ..., "date": ...}}

The value is the dict written by ``to_json()``, and the object is restored with the
class's ``from_json()``. Every subclass of ``TemporalDistributionBase`` and ``TDAware``
which has both methods is registered with ``bw2data.codec`` when the class is defined,
including subclasses defined by users. The tag is ``"bw_temporalis.<class name>"`` for the
classes of this package (the same as the ``__loader__`` written by ``to_json()``), and
``"<module>.<qualified class name>"`` for all others.

A stored value whose class isn't defined in the reading process (e.g. the module defining
it wasn't imported) is kept by ``bw2data`` unchanged, and ``load_temporal_distribution``
restores it with ``loader_registry``, using the ``__loader__`` in the value.

``bw2data`` versions without ``bw2data.codec`` store exchange data as pickles, and nothing
is registered. ``register`` is the ``bw2data.codec`` entry point of this package (see
``pyproject.toml``), which makes ``bw2data`` import this package when it needs it.
"""

import json


def tag(cls: type) -> str:
    """The tag of ``cls`` in ``bw2data.codec``."""
    module = cls.__module__
    if module == "bw_temporalis" or module.startswith("bw_temporalis."):
        return f"bw_temporalis.{cls.__name__}"
    return f"{module}.{cls.__qualname__}"


def to_payload(obj) -> dict:
    return json.loads(obj.to_json())


def register_class(cls: type) -> bool:
    """Register ``cls`` with ``bw2data.codec``, if it has ``to_json`` and ``from_json`` and
    ``bw2data`` has a codec. Returns whether it was registered.

    A class defined again under the same name (e.g. a notebook cell run twice) replaces the
    earlier one."""
    if not (callable(getattr(cls, "to_json", None)) and callable(getattr(cls, "from_json", None))):
        return False
    try:
        from bw2data import codec
    except ImportError:
        return False
    try:
        codec.register(cls, tag(cls), to_payload, cls.from_json, replace=True)
    except TypeError:
        # ``bw2data.codec`` before ``register(..., replace=...)``
        codec.register(cls, tag(cls), to_payload, cls.from_json)
    return True


def register() -> None:
    """Register the temporal distribution classes of this package with ``bw2data.codec``.

    Subclasses register themselves when they are defined; this makes sure the module
    defining them is imported, and registers them again in case ``bw2data`` was imported
    later."""
    from .temporal_distribution import FixedTD, FixedTimeOfYearTD, TemporalDistribution

    for cls in (TemporalDistribution, FixedTD, FixedTimeOfYearTD):
        register_class(cls)
