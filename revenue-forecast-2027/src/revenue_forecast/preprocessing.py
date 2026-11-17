import pandas as pd

def month_start(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.to_period("M").dt.to_timestamp()

def aggregate_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    x=df.copy();x["date"]=month_start(x["date"])
    agg=x.groupby(["date","commodity"],as_index=False).agg(news_count=("news_count","sum"),sentiment_weighted=("sentiment_weighted","mean"),price_impact_weighted=("price_impact_weighted","mean"))
    return agg.pivot(index="date",columns="commodity",values=["news_count","sentiment_weighted","price_impact_weighted"]).sort_index(axis=1).rename(columns=lambda c:str(c)).reset_index()

def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    out=df.copy()
    if isinstance(out.columns,pd.MultiIndex): out.columns=["_".join(str(v) for v in c if str(v)!="") for c in out.columns]
    return out

def pivot_forecast(df: pd.DataFrame) -> pd.DataFrame:
    x=df.copy();x["date"]=month_start(x["date"])
    return flatten_columns(x.pivot_table(index="date",columns="commodity",values=["price_p10","price_p50","price_p90"],aggfunc="mean").reset_index())

def build_exposure(df: pd.DataFrame,forecast: pd.DataFrame) -> pd.DataFrame:
    x=df.copy();x["date"]=month_start(x["date"]);f=forecast.copy();f["date"]=month_start(f["date"])
    z=x.merge(f,on=["date","commodity"],how="left")
    for q in ["p10","p50","p90"]: z[f"material_cost_{q}"]=z["quantity"]*z[f"price_{q}"]
    return z.groupby("date",as_index=False)[["material_cost_p10","material_cost_p50","material_cost_p90"]].sum(min_count=1)
