import json
from numbers import Number
from typing import Any, SupportsFloat, Union

import numpy as np
import numpy.typing as npt
from scipy.cluster.vq import kmeans2

from .convolution import (
    consolidate,
    datetime_type,
    temporal_convolution_datetime_timedelta,
    temporal_convolution_timedelta_timedelta,
    timedelta_type,
)

RESOLUTION_LABELS = {
    "Y": "Years",
    "M": "Months",
    "W": "Weeks",
    "D": "Days",
    "h": "Hours",
    "m": "Minutes",
    "s": "Seconds",
}


class TemporalDistribution:
    """A container for a series of amount spread over time.
    Args:
        * *date* (ndarray): 1D array containg temporal info of `amount` with type `timedelta64` or `datetime64` .
        * *amount* (ndarray): 1D array containg amount with type `float`

        Times and amount must have same length and element of `amount` must correspond to the element of `date`
        with the same index.
    """

    def __init__(
        self, date: npt.NDArray[np.datetime64 | np.timedelta64], amount: npt.NDArray
    ):
        if not isinstance(date, np.ndarray) or not isinstance(amount, np.ndarray):
            raise ValueError("Invalid input types")
        elif not date.shape == amount.shape:
            raise ValueError("Shapes of input `date` and `amount` not identical")
        elif not (
            np.issubdtype(date.dtype, np.datetime64)
            or np.issubdtype(date.dtype, np.timedelta64)
        ):
            raise ValueError(f"Incorrect `date` dtype ({date.dtype})")
        elif not len(date):
            raise ValueError("Empty array")

        # Conversion needed for `temporal_convolution` functions
        self.amount = amount.astype(np.float64)

        if np.issubdtype(date.dtype, np.datetime64):
            self.date = date.astype(datetime_type)
            self.base_time_type = datetime_type
        elif np.issubdtype(date.dtype, np.timedelta64):
            self.date = date.astype(timedelta_type)
            self.base_time_type = timedelta_type
        else:
            raise ValueError("`date` must be numpy datetime or timedelta array")

    def __len__(self) -> int:
        return self.amount.shape[0]

    @property
    def total(self) -> float:
        return float(self.amount.sum())

    def __mul__(
        self, other: Union["TemporalDistribution", SupportsFloat]
    ) -> "TemporalDistribution":
        if isinstance(other, TemporalDistribution):
            if (
                self.base_time_type == datetime_type
                and other.base_time_type == datetime_type
            ):
                raise ValueError("Can't multiple two datetime arrays")
            elif self.base_time_type == datetime_type:
                date, amount = temporal_convolution_datetime_timedelta(
                    first_date=self.date,
                    first_amount=self.amount,
                    second_date=other.date,
                    second_amount=other.amount,
                )
            elif other.base_time_type == datetime_type:
                date, amount = temporal_convolution_datetime_timedelta(
                    first_date=other.date,
                    first_amount=other.amount,
                    second_date=self.date,
                    second_amount=self.amount,
                )
            else:
                date, amount = temporal_convolution_timedelta_timedelta(
                    first_date=self.date,
                    first_amount=self.amount,
                    second_date=other.date,
                    second_amount=other.amount,
                )
            return TemporalDistribution(date=date, amount=amount)
        elif isinstance(other, Number):
            return TemporalDistribution(
                date=self.date, amount=self.amount * float(other)
            )
        else:
            raise ValueError(
                "Can't multiply `TemporalDistribution` and {}".format(type(other))
            )

    def __truediv__(self, other: SupportsFloat) -> "TemporalDistribution":
        if self.base_time_type == datetime_type:
            raise ValueError("Can't divide a datetime array")
        elif not isinstance(other, Number):
            raise ValueError("Can only divide time deltas by a number")
        return TemporalDistribution(self.date, self.amount / float(other))

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, TemporalDistribution):
            return False
        return self.amount.sum() < other.amount.sum()

    def __add__(
        self, other: Union["TemporalDistribution", SupportsFloat]
    ) -> "TemporalDistribution":
        if isinstance(other, TemporalDistribution):
            if self.base_time_type == other.base_time_type == datetime_type:
                raise ValueError("Can't add two datetimes")
            elif self.base_time_type == other.base_time_type == timedelta_type:
                date = np.hstack((self.date, other.date))
                amount = np.hstack((self.amount, other.amount))
                # same as in __mul__
                t, v = consolidate(indices=date.astype("int64"), amounts=amount)
                return TemporalDistribution(t.astype(timedelta_type), v)
            else:
                if not len(self) == len(other):
                    raise ValueError("Incompatible dimensions")
                elif self.base_time_type == timedelta_type:
                    # `self` is timedelta, `other` is datetime
                    return TemporalDistribution(
                        other.date + self.date, self.amount + other.amount
                    )
                else:
                    # `self` is datetime, `other` is timedelta
                    return TemporalDistribution(
                        self.date + other.date, self.amount + other.amount
                    )
        elif isinstance(other, Number):
            return TemporalDistribution(self.date, self.amount + float(other))
        else:
            raise ValueError(
                "Can't add TemporalDistribution and {}".format(type(other))
            )

    def __str__(self) -> str:
        return "TemporalDistribution instance with %s amount and total: %.4g" % (
            len(self.amount),
            self.total,
        )

    def __repr__(self) -> str:
        return str(self)

    def graph(
        self, style: str | None = "fivethirtyeight", resolution: str | None = None
    ):
        """Graph the temporal distribution.

        `resolution` is one of `YMWDhms`.

        This isn't too difficult, if you need more customization write your own :)"""
        try:
            from matplotlib import pyplot as plt
        except ImportError:
            raise ImportError("`matplotlib` required for this function")
        plt.style.use(style)
        axis = plt.gca()

        if np.issubdtype(self.date.dtype, np.datetime64):
            axis.set_xlabel("Date")
            date = self.date
        else:
            if resolution is None:
                date = self.date
                axis.set_xlabel("Time (seconds)")
            else:
                date = self.date.astype(f"timedelta64[{resolution}]")
                axis.set_xlabel("Time ({})".format(RESOLUTION_LABELS[resolution]))
        axis.plot(date, self.amount, marker=".", lw=0)
        axis.set_ylabel("Amount")
        plt.tight_layout()
        plt.xticks(rotation=30)

        return axis

    def to_json(self):
        return json.dumps(
            {
                "__loader__": "bw_temporalis.temporal_distribution.TemporalDistribution",
                "date_dtype": str(self.date.dtype),
                "date": self.date.astype(int).tolist(),
                "amount": self.amount.tolist(),
            }
        )

    @classmethod
    def from_json(cls, json_string):
        data = json.loads(json_string)
        return cls(
            date=np.array(data["date"], dtype=data["date_dtype"]),
            amount=np.array(data["amount"], dtype=float),
        )

    def nonzero(self):
        mask = self.amount == 0
        if mask.sum():
            return TemporalDistribution(self.date[~mask], self.amount[~mask])
        else:
            return self

    def simplify(
        self,
        threshhold: int | None = 1000,
        num_clusters: int | None = None,
        iterations: int | None = 30,
    ) -> "TemporalDistribution":
        """Use clustering to simplify a `TemporalDistribution` with more than
        `threshhold` number of points.

        Uses the `kmeans2` implementation of KNN from
        [scipy.cluster.vq](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.vq.kmeans2.html).
        Subclass and override this method to get a custom clustering algorithm.

        This isn't perfect, `kmeans2` produces "lumpy" distributions - to see
        this graph the simplification of a uniform distribution with many
        points.

        Parameters
        ----------

        threshhold : int, optional
            The number of `date` points above which simplification is triggered
        iterations : int, optional
            `iter` parameters to feed to `kmeans2`

        Returns
        -------

        Either `self` (if no simplification) or a new instance of `TemporalDistribution`.

        """
        if len(self) <= threshhold:
            return self

        num_clusters = num_clusters or threshhold

        means, codebook = kmeans2(
            data=self.date.astype(float),
            iter=iterations,
            k=num_clusters,
            minit="points",
        )
        _, amount = consolidate(indices=codebook, amounts=self.amount)
        # Clusters can be "lumpy", even with smooth input date
        # Chances are that less than `threshhold` clusters are generated
        # so we need to remove unused clusters.
        date = np.sort(means[np.unique(codebook)]).astype(self.date.dtype)
        return TemporalDistribution(date=date, amount=amount)
