from numbers import Number

import numpy as np
from bw2speedups import consolidate


class TemporalDistribution:
    """A container for a series of amount spread over time.
    Args:
        * *date* (ndarray): 1D array containg temporal info of `amount` with type `timedelta64` or `datetime64` .
        * *amount* (ndarray): 1D array containg amount with type `float`

        Times and amount must have same length and element of `amount` must correspond to the element of `date`
        with the same index.
    """

    def __init__(self, date, amount):
        if not isinstance(date, np.ndarray) or not isinstance(amount, np.ndarray):
            raise ValueError("Invalid input types")
        if not date.shape == amount.shape:
            raise ValueError("Shapes of input `date` and `amount` not identical")
        if not (
            np.issubdtype(date.dtype, np.datetime64)
            or np.issubdtype(date.dtype, np.timedelta64)
        ):
            raise ValueError("Incorrect `date` dtype")

        # Conversion needed for bw2speedups.consolidate function
        self.amount = amount.astype(np.float64)

        if np.issubdtype(date.dtype, np.datetime64):
            self.date = date.astype("datetime64[D]")
            self.base_time_type = "datetime64[D]"
        elif np.issubdtype(date.dtype, np.timedelta64):
            self.date = date.astype("timedelta64[D]")
            self.base_time_type = "timedelta64[D]"
        else:
            raise ValueError("`date` must be numpy datetime or timedelta array")

    def __len__(self):
        return self.amount.shape[0]

    @property
    def total(self):
        return float(self.amount.sum())

    def nonzero(self):
        mask = self.amount != 0
        return TemporalDistribution(self.date[mask], self.amount[mask])

    def __getitem__(self, index):
        return TemporalDistribution(
            np.array(self.date[index]), np.array(self.amount[index])
        )

    def __mul__(self, other):
        if isinstance(other, TemporalDistribution):
            if (
                self.base_time_type == "datetime64[D]"
                and other.base_time_type == "datetime64[D]"
            ):
                raise ValueError("Can't multiple two datetime arrays")

            date = (self.date.reshape((-1, 1)) + other.date.reshape((1, -1))).ravel()
            amount = (
                self.amount.reshape((-1, 1)) * other.amount.reshape((1, -1))
            ).ravel()

            # Use array view in consolidate
            # http://stackoverflow.com/a/33528073/4929813
            t, v = consolidate(date.view("int64"), amount)
            return TemporalDistribution(t.astype(self.base_time_type), v)
        else:
            try:
                return TemporalDistribution(self.date, self.amount * float(other))
            except:
                raise ValueError(
                    "Can't multiply `TemporalDistribution` and {}".format(type(other))
                )

    def __truediv__(self, other):
        if self.base_time_type == "datetime64[D]":
            raise ValueError("Can't divide a datetime array")
        elif not isinstance(other, Number):
            raise ValueError("Can only divide time deltas by a number")
        return TemporalDistribution(self.date, self.amount / float(other))

    def __lt__(self, other):
        if not isinstance(other, TemporalDistribution):
            return False
        return self.amount.sum() < other.amount.sum()

    def __add__(self, other):
        if isinstance(other, TemporalDistribution):
            if self.base_time_type == other.base_time_type == "datetime64[D]":
                raise ValueError("Can't add two datedate")
            elif self.base_time_type == other.base_time_type == "timedelta64[D]":
                date = np.hstack((self.date, other.date))
                amount = np.hstack((self.amount, other.amount))
                # same as in __mul__
                t, v = consolidate(date.view("int64"), amount)
                return TemporalDistribution(t.astype("timedelta64[D]"), v)
            else:
                if not len(self) == len(other):
                    raise ValueError("Incompatible dimensions")
                elif self.base_time_type == "timedelta64[D]":
                    return TemporalDistribution(
                        other.date + self.date, self.amount + other.amount
                    )
                else:
                    return TemporalDistribution(
                        self.date + other.date, self.amount + other.amount
                    )
        elif isinstance(other, Number):
            return TemporalDistribution(self.date, self.amount + float(other))
        else:
            raise ValueError(
                "Can't add TemporalDistribution and {}".format(type(other))
            )

    def __iter__(self):
        for index in range(len(self)):
            yield (self.date[index], float(self.amount[index]))

    def __str__(self):
        return "TemporalDistribution instance with %s amount and total: %.4g" % (
            len(self.amount),
            self.total,
        )

    def __repr__(self):
        return str(self)

    def cumulative(self):
        """Return new temporal distribution with cumulative amount"""
        return TemporalDistribution(self.date, np.cumsum(self.amount))
