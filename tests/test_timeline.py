import pandas as pd
from bw_temporalis.timeline import Timeline

def create_test_dataframe():
    times = [100, 101, 102, 400, 401, 402]
    values = [10, 20, 30, 50, 40, 30]
    flows = [1, 1, 1, 3, 3, 3]
    activities = [2, 2, 2, 4, 4, 4]
    
    data = {"times": times, "values": values, "flows": flows, "activities": activities}
    df = pd.DataFrame(data)
    
    return df

def test_characterization_function(series: pd.Series, period: int = 100) -> pd.DataFrame:

    def linear_characterization(series, period=100):
    dates = (
        series.times.to_numpy() + 
        np.arange(period, dtype="timedelta64[Y]").astype("timedelta64[s]")
    )
    values = np.linspace(1, 0, period) * series['values']
    return pd.DataFrame({
        'times': dates.astype("datetime64[s]"),
        'values': values,
        'flow': series.flows,
        'activity': series.activities,
    })



def test_characterization():
