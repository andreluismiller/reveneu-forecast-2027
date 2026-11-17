import pandas as pd
from revenue_forecast.preprocessing import aggregate_sentiment,flatten_columns

def test_sentiment_monthly():
    df=pd.DataFrame({"date":["2026-01-01","2026-01-02"],"commodity":["cobre","cobre"],"news_count":[2,3],"sentiment_weighted":[.1,.3],"price_impact_weighted":[.2,.4]})
    out=flatten_columns(aggregate_sentiment(df));assert out.loc[0,"news_count_cobre"]==5
