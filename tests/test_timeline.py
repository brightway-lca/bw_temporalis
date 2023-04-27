
import pandas as pd
import numpy as np
from bw_temporalis.timeline import Timeline

df_input = pd.DataFrame(
    data = {
        'date': pd.Series(data=['20-12-2020', '21-12-2020', '15-12-2020', '16-12-2020', '25-05-2022', '26-05-2022'], dtype="datetime64[s]"),
        'amount': pd.Series(data=[10.0, 9.0, 20.0, 19.0, 50.0, 49.0], dtype="float64"),
        'flow': pd.Series(data=[1, 1, 1, 1, 3, 3], dtype="int"),
        'activity': pd.Series(data=[2, 2, 2, 2, 4, 4], dtype="int")
    }
)

df_expected_characterize = pd.DataFrame(
    data = {
        'date': pd.Series(data=['20-12-2020', '21-12-2020', '15-12-2020', '16-12-2020', '25-05-2022', '26-05-2022'], dtype="datetime64[s]"),
        'amount': pd.Series(data=[10.0, 9.0, 20.0, 19.0, 50.0, 49.0], dtype="float64"),
        'flow': pd.Series(data=[1, 1, 1, 1, 3, 3], dtype="int"),
        'activity': pd.Series(data=[2, 2, 2, 2, 4, 4], dtype="int")
    }
)

df_expected_sum_days_to_years = pd.DataFrame(
    data = {
        'date': pd.Series(data=['01-01-2020', '01-01-2022'], dtype="datetime64[s]"),
        'amount': pd.Series(data=[39.0, 99.0], dtype="float64"),
        'flow': pd.Series(data=[1, 1], dtype="int"),
        'activity': pd.Series(data=[2, 4], dtype="int")
    }
)


def function_characterization_test(series: pd.Series, period: int = 2) -> pd.DataFrame:

    date_beginning: np.datetime64 =  series['date'].to_numpy()
    dates_characterized: np.ndarray = date_beginning + np.arange(start = 0 , stop = period, dtype="timedelta64[Y]").astype("timedelta64[s]")
    
    amount_beginning: float = series['amount']
    amount_characterized: np.ndarray = amount_beginning - np.arange(start = 0 , stop = period, dtype="int")

    return pd.DataFrame({
        'date': pd.Series(data=dates_characterized.astype("datetime64[s]"),
        'amount': pd.Series(data=amount_characterized, dtype="float64"),
        'flow': series.flow,
        'activity': series.activity,
    })

def test_characterization():
    raise ChrisShouldImplementThisError
