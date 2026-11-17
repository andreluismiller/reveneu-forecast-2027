import pandas as pd
from .io import read_frame,write_frame
from .settings import Settings
from .preprocessing import month_start,aggregate_sentiment,flatten_columns,pivot_forecast,build_exposure
from .features import add_calendar,add_target_history,add_external_lags

def build(settings: Settings) -> pd.DataFrame:
    revenue=read_frame(settings.revenue_path);operations=read_frame(settings.operations_path);sentiment=read_frame(settings.sentiment_path)
    revenue["date"]=month_start(revenue["date"]);operations["date"]=month_start(operations["date"])
    sent=flatten_columns(aggregate_sentiment(sentiment))
    data=revenue.merge(operations,on="date",how="outer").merge(sent,on="date",how="left")
    if settings.commodity_forecast_path.exists():
        forecast=read_frame(settings.commodity_forecast_path);fp=pivot_forecast(forecast);data=data.merge(fp,on="date",how="left")
        if settings.material_exposure_path.exists(): data=data.merge(build_exposure(read_frame(settings.material_exposure_path),forecast),on="date",how="left")
    data=data.sort_values("date");data=add_calendar(data);data=add_target_history(data)
    external=[c for c in data.columns if c not in {"date","revenue"} and pd.api.types.is_numeric_dtype(data[c])]
    data=add_external_lags(data,external,lags=(1,2,3));write_frame(data,settings.model_dataset_path);return data
