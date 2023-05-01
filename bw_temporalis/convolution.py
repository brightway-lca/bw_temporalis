import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_array

OFFSET = 31536000000000
datetime_type = np.dtype("datetime64[s]")
timedelta_type = np.dtype("timedelta64[s]")
time_types = {datetime_type, timedelta_type}


def consolidate(
    *, indices: npt.NDArray[np.int64], amounts: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Sum all values in ``amount`` which have the same index in ``indices``"""
    # Sparse matrices don't allow for negative indices, but this can easily happen
    # with two timedelta64 arrays. Instead of checking we just always offset (still fast).
    # Choose one million years as a reasonable offset (native resolution is seconds)
    # 1_000_000 * 60 * 60 * 24 * 365 = 31536000000000
    matrix = csr_array(
        (amounts, (np.zeros_like(indices), indices + OFFSET)),
        shape=(1, indices.max() + OFFSET + 1),
    )
    coo = matrix.tocoo()
    mask = coo.data != 0
    return (coo.col[mask] - OFFSET), coo.data[mask]


def convolve(
    *,
    first_date: npt.NDArray,
    first_amount: npt.NDArray[np.float64],
    second_date: npt.NDArray,
    second_amount: npt.NDArray[np.float64],
    return_dtype: npt.DTypeLike | str,
) -> tuple[npt.NDArray, npt.NDArray[np.float64]]:
    date = (first_date.reshape((-1, 1)) + second_date.reshape((1, -1))).ravel()
    amount = (first_amount.reshape((-1, 1)) * second_amount.reshape((1, -1))).ravel()
    date, amount = consolidate(indices=date.astype(np.int64), amounts=amount)
    return date.astype(return_dtype), amount.astype(np.float64)


def temporal_convolution_datetime_timedelta(
    *,
    first_date: npt.NDArray[datetime_type],
    first_amount: npt.NDArray[np.float64],
    second_date: npt.NDArray[timedelta_type],
    second_amount: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[datetime_type], npt.NDArray[np.float64]]:
    if not (first_date.dtype == datetime_type):
        raise ValueError(
            f"`first_date` must have dtype `datetime64[s]`, but got `{first_date.dtype}`"
        )
    if not (second_date.dtype == timedelta_type):
        raise ValueError(
            f"`second_date` must have dtype `timedelta64[s]`, but got `{second_date.dtype}`"
        )
    return convolve(
        first_date=first_date,
        first_amount=first_amount,
        second_date=second_date,
        second_amount=second_amount,
        return_dtype=datetime_type,
    )


def temporal_convolution_timedelta_timedelta(
    *,
    first_date: npt.NDArray[timedelta_type],
    first_amount: npt.NDArray[np.float64],
    second_date: npt.NDArray[timedelta_type],
    second_amount: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[timedelta_type], npt.NDArray[np.float64]]:
    if not (first_date.dtype == timedelta_type):
        raise ValueError(
            f"`first_date` must have dtype `timedelta64[s]`, but got {first_date.dtype}"
        )
    if not (second_date.dtype == timedelta_type):
        raise ValueError(
            f"`second_date` must have dtype `timedelta64[s]`, but got {second_date.dtype}"
        )
    return convolve(
        first_date=first_date,
        first_amount=first_amount,
        second_date=second_date,
        second_amount=second_amount,
        return_dtype=timedelta_type,
    )
