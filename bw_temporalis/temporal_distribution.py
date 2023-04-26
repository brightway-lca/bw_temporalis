from numbers import Number

import numpy as np
from bw2speedups import consolidate


class TemporalDistribution:
    """A container for a series of values spread over time.
    Args:
        * *times* (ndarray): 1D array containg temporal info of `values` with type `timedelta64` or `datetime64` .
        * *values* (ndarray): 1D array containg values with type `float`

        Times and values must have same length and element of `values` must correspond to the element of `times`
        with the same index.
    """

    def __init__(self, times, values):
        if not isinstance(times, np.ndarray) or not isinstance(values, np.ndarray):
            raise ValueError("Invalid input types")
        if not times.shape == values.shape:
            raise ValueError("Shapes of input `times` and `values` not identical")
        if not (
            np.issubdtype(times.dtype, np.datetime64)
            or np.issubdtype(times.dtype, np.timedelta64)
        ):
            raise ValueError("Incorrect `times` dtype")

        # Conversion needed for bw2speedups.consolidate function
        self.values = values.astype(np.float64)

        if np.issubdtype(times.dtype, np.datetime64):
            self.times = times.astype("datetime64[D]")
            self.base_time_type = "datetime64[D]"
        elif np.issubdtype(times.dtype, np.timedelta64):
            self.times = times.astype("timedelta64[D]")
            self.base_time_type = "timedelta64[D]"
        else:
            raise ValueError("`times` must be numpy datetime or timedelta array")

    def __len__(self):
        return self.values.shape[0]

    @property
    def total(self):
        return float(self.values.sum())

    def nonzero(self):
        mask = self.values != 0
        return TemporalDistribution(self.times[mask], self.values[mask])

    def __getitem__(self, index):
        return TemporalDistribution(
            np.array(self.times[index]), np.array(self.values[index])
        )

    def __mul__(self, other):
        if isinstance(other, TemporalDistribution):
            if (
                self.base_time_type == "datetime64[D]"
                and other.base_time_type == "datetime64[D]"
            ):
                raise ValueError("Can't multiple two datetime arrays")

            times = (self.times.reshape((-1, 1)) + other.times.reshape((1, -1))).ravel()
            values = (
                self.values.reshape((-1, 1)) * other.values.reshape((1, -1))
            ).ravel()

            # Use array view in consolidate
            # http://stackoverflow.com/a/33528073/4929813
            t, v = consolidate(times.view("int64"), values)
            return TemporalDistribution(t.astype(self.base_time_type), v)
        else:
            try:
                return TemporalDistribution(self.times, self.values * float(other))
            except:
                raise ValueError(
                    "Can't multiply `TemporalDistribution` and {}".format(type(other))
                )

    def __truediv__(self, other):
        if self.base_time_type == "datetime64[D]":
            raise ValueError("Can't divide a datetime array")
        elif not isinstance(other, Number):
            raise ValueError("Can only divide time deltas by a number")
        return TemporalDistribution(self.times, self.values / float(other))

    def __lt__(self, other):
        if not isinstance(other, TemporalDistribution):
            return False
        return self.values.sum() < other.values.sum()

    def __add__(self, other):
        if isinstance(other, TemporalDistribution):
            if self.base_time_type == other.base_time_type == "datetime64[D]":
                raise ValueError("Can't add two datetimes")
            elif self.base_time_type == other.base_time_type == "timedelta64[D]":
                times = np.hstack((self.times, other.times))
                values = np.hstack((self.values, other.values))
                # same as in __mul__
                t, v = consolidate(times.view("int64"), values)
                return TemporalDistribution(t.astype("timedelta64[D]"), v)
            else:
                if not len(self) == len(other):
                    raise ValueError("Incompatible dimensions")
                elif self.base_time_type == "timedelta64[D]":
                    return TemporalDistribution(
                        other.times + self.times, self.values + other.values
                    )
                else:
                    return TemporalDistribution(
                        self.times + other.times, self.values + other.values
                    )
        elif isinstance(other, Number):
            return TemporalDistribution(self.times, self.values + float(other))
        else:
            raise ValueError(
                "Can't add TemporalDistribution and {}".format(type(other))
            )

    def __iter__(self):
        for index in range(len(self)):
            yield (self.times[index], float(self.values[index]))

    def __str__(self):
        return "TemporalDistribution instance with %s values and total: %.4g" % (
            len(self.values),
            self.total,
        )

    def __repr__(self):
        return (
            "TemporalDistribution instance with %s values (total: %.4g, min: %.4g, max: %.4g"
            % (len(self.values), self.total, self.values.min(), self.values.max())
        )

    def cumulative(self):
        """Return new temporal distribution with cumulative values"""
        return TemporalDistribution(self.times, np.cumsum(self.values))

    # def datetime_to_timedelta(self, dt):
    #     """Convert TD.times of type datetime64 to timedelta64 based on the datetime64 passed
    #     """
    #     assert 'datetime64' in str(self.times.dtype),'TemporalDistribution.times must be numpy.datetime64'
    #     assert isinstance(dt,np.datetime64),'datetime must be numpy.datetime64'
    #     return  TemporalDistribution(self.times - dt, self.values)

    # def timedelta_to_datetime(self, dt):
    #     """Convert TD.times of type timedelta64 to datetime.datetime based on the datetime64 passed
    #     """
    #     #converted to datetime.datetime cause timeline._groupby_sum_by_flow is ~5 slower when using np.datetime64
    #     #while the conversion here to datatime.datetime is almost the same performance wise
    #     assert 'timedelta64' in str(self.times.dtype),'TemporalDistribution.times must be numpy.datetime64'
    #     assert isinstance(dt,np.datetime64),'datetime must be numpy.datetime64'
    #     return  TemporalDistribution((self.times + dt).astype(datetime.datetime) , self.values)
