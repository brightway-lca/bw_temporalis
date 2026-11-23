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
restores it with ``loader_registry``, using the loader for the tag if there is one, and
otherwise the loader for the ``__loader__`` in the value. A subclass which inherits
``to_json`` writes the ``__loader__`` of its parent, and is then restored as the parent
class, with a warning.

``loader_registry`` also has the name ``"bw_temporalis.FixedTimeOfYear"``. ``to_json()``
never wrote it and it is not a registered tag; ``load_temporal_distribution`` restores a
value tagged with it.

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
    earlier one, whose instances can then no longer be stored. This also holds for classes
    made by a function which defines a class each time it is called: they all have the same
    module and qualified name, so only the last one is registered."""
    if not (callable(getattr(cls, "to_json", None)) and callable(getattr(cls, "from_json", None))):
        return False
    try:
        from bw2data import codec
    except ImportError:
        return False
    codec.register(cls, tag(cls), to_payload, cls.from_json, replace=True)
    return True


def register() -> None:
    """Register the temporal distribution classes of this package with ``bw2data.codec``.

    Subclasses register themselves when they are defined, so importing this package is
    what matters here; registering the classes again changes nothing."""
    from .temporal_distribution import FixedTD, FixedTimeOfYearTD, TemporalDistribution

    for cls in (TemporalDistribution, FixedTD, FixedTimeOfYearTD):
        register_class(cls)
