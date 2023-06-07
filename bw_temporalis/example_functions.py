import json
from collections.abc import Mapping
from numbers import Number
from typing import Optional

import numpy as np

from . import TDAware
from .convolution import timedelta_type
from .temporal_distribution import TemporalDistribution


class FixedTimeOfYear(TDAware):
    """Instead of creating a relative shift in time, regardless of the starting datetime or the properties of the system being modeled, create an absolute period in the first available year.

    The fixed time of year is a *relative temporal distribution*, and must already be constructed as such.

    When multiplied by another temporal distribution, the entire period of the `FixedTimeOfYear` must lie before the other distribution, or else the `FixedTimeOfYear` is shifted back a year. In other words, a `FixedTimeOfYear` of March to May could be multiplied by August 2020 and result in March to May 2020, but if multiplied by April 2020 it would result in March to May 2019. To allow partial overlaps and stay in the same year, set `allow_overlap` to `True`. In any case, the start of the `FixedTimeOfYear` must be before the start of the temporal distribution being multiplied or it is shifted back a year.

    """

    def __init__(
        self,
        temporal_distribution: TemporalDistribution,
        allow_overlap: Optional[bool] = False,
    ):
        self.td = temporal_distribution
        if not np.alltrue(self.td.date.astype(int) >= 0):
            raise ValueError("Can't have negative relative datetime64 values")
        self.allow_overlap = allow_overlap

    def __mul__(self, other: TemporalDistribution) -> TemporalDistribution | TDAware:
        if isinstance(other, Number):
            return FixedTimeOfYear(
                temporal_distribution=self.td * other, allow_overlap=self.allow_overlap
            )
        elif isinstance(other, TemporalDistribution):
            if other.base_time_type == timedelta_type:
                raise ValueError(
                    "Can't multiply `FixedTimeOfYear` by a relative distribution"
                )

            previous_year_mask = (
                other.date - other.date.astype("datetime64[Y]").astype("datetime64[s]")
            ) > (self.td.date.min() if self.allow_overlap else self.td.date.max())
            return (
                TemporalDistribution(
                    date=other.date.astype("datetime64[Y]") - 1 + previous_year_mask,
                    amount=other.amount,
                )
                * self.td
            )
        else:
            raise ValueError(
                "Can only be multipled by a number or an instance of `TemporalDistribution`"
            )

    def to_json(self) -> str:
        return json.dumps(
            {
                "__loader__": "bw_temporalis.example_functions.FixedTimeOfYear.from_json",
                "date_dtype": str(self.td.date.dtype),
                "date": self.td.date.astype(int).tolist(),
                "amount": self.td.amount.tolist(),
                "allow_overlap": self.allow_overlap,
            }
        )

    @classmethod
    def from_json(cls, json_obj):
        if isinstance(json_obj, Mapping):
            data = json_obj
        elif isinstance(json_obj, str):
            data = json.loads(json_obj)
        else:
            raise ValueError(f"Can't understand `from_json` input object {json_obj}")
        return cls(
            temporal_distribution=TemporalDistribution.from_json(data),
            allow_overlap=data["allow_overlap"],
        )
