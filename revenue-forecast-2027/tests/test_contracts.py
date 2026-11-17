import pandas as pd
from revenue_forecast.contracts import CONTRACTS,validate_frame

def test_valid_revenue():
    df=pd.DataFrame({"date":["2026-01-01"],"revenue":[1.0]})
    assert validate_frame(df,CONTRACTS["revenue"])==[]

def test_missing_column():
    assert validate_frame(pd.DataFrame({"date":["2026-01-01"]}),CONTRACTS["revenue"])
