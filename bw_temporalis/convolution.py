import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_array


OFFSET = 31536000000000
datetime_type = np.dtype("datetime64[s]")
timedelta_type = np.dtype("timedelta64[s]")
time_types = {datetime_type, timedelta_type}



def _convolve(*, first_date, first_amount, second_date, second_amount, return_dtype):
    # Sparse matrices don't allow for negative indices, but this can easily happen
    # with two timedelta64 arrays. Instead of checking we just always offset (still fast).
    # Choose one million years as a reasonable offset (native resolution is seconds)
    # 1_000_000 * 60 * 60 * 24 * 365 = 31536000000000

    date = (
        (first_date.reshape((-1, 1)) + second_date.reshape((1, -1))).ravel()
    ).astype(np.int64) + OFFSET
    amount = (first_amount.reshape((-1, 1)) * second_amount.reshape((1, -1))).ravel()

    matrix = csr_array((amount, (np.zeros_like(date), date)), shape=(1, date.max() + 1))
    coo = matrix.tocoo()
    mask = coo.data != 0
    return (coo.col[mask] - OFFSET).astype(return_dtype), coo.data[mask].astype(np.float64)


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
    return _convolve(
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
    return _convolve(
        first_date=first_date,
        first_amount=first_amount,
        second_date=second_date,
        second_amount=second_amount,
        return_dtype=timedelta_type,
    )
