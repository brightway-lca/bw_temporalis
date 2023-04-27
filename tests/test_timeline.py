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

def test_characterization():
