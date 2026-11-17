import pandas as pd

def add_calendar(df):
    x=df.copy();x["month"]=x["date"].dt.month;x["quarter"]=x["date"].dt.quarter;x["year"]=x["date"].dt.year
    x["month_sin"]=(2*3.141592653589793*x["month"]/12).map(__import__('math').sin);x["month_cos"]=(2*3.141592653589793*x["month"]/12).map(__import__('math').cos);return x

def add_target_history(df,target="revenue",lags=(1,2,3,6,12),windows=(3,6,12)):
    x=df.sort_values("date").copy()
    for lag in lags:x[f"{target}_lag_{lag}"]=x[target].shift(lag)
    for w in windows:
        x[f"{target}_ma_{w}"]=x[target].shift(1).rolling(w,min_periods=max(2,w//2)).mean();x[f"{target}_std_{w}"]=x[target].shift(1).rolling(w,min_periods=max(2,w//2)).std()
    return x

def add_external_lags(df,columns,lags=(1,2,3)):
    x=df.copy()
    for col in columns:
        for lag in lags:x[f"{col}_lag_{lag}"]=x[col].shift(lag)
    return x
