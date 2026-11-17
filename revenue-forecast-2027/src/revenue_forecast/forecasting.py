import joblib,numpy as np,pandas as pd
from .io import read_frame,write_frame
from .features import add_calendar,add_target_history,add_external_lags
from .settings import Settings

def forecast(settings: Settings,year: int):
    bundle=joblib.load(settings.artifact_dir/"model.joblib");history=read_frame(settings.model_dataset_path).sort_values("date");history["date"]=pd.to_datetime(history["date"])
    future_dates=pd.date_range(f"{year}-01-01",f"{year}-12-01",freq="MS");forecast_rows=[];work=history.copy();rng=np.random.default_rng(settings.random_seed);res=np.asarray(bundle["residuals"])
    for date in future_dates:
        row={"date":date,"revenue":np.nan}
        # Usa variáveis externas futuras presentes no dataset; caso ausentes, carrega o último valor conhecido.
        for c in work.columns:
            if c not in row: row[c]=work[c].dropna().iloc[-1] if work[c].notna().any() else np.nan
        work=pd.concat([work,pd.DataFrame([row])],ignore_index=True);base=work[[c for c in work.columns if not c.startswith("revenue_lag_") and not c.startswith("revenue_ma_") and not c.startswith("revenue_std_")]].copy()
        base=add_calendar(base);base=add_target_history(base)
        ext=[c for c in base.columns if c not in {"date","revenue"} and pd.api.types.is_numeric_dtype(base[c]) and not c.endswith(("_lag_1","_lag_2","_lag_3"))]
        base=add_external_lags(base,ext,(1,2,3));x=base.tail(1).reindex(columns=bundle["features"]);p50=float(bundle["model"].predict(x)[0]);sims=p50+rng.choice(res,size=settings.monte_carlo_runs,replace=True)
        p10,p90=np.quantile(sims,[.1,.9]);forecast_rows.append({"date":date,"revenue_p10":float(p10),"revenue_p50":p50,"revenue_p90":float(p90)});work.loc[work.index[-1],"revenue"]=p50
    out=pd.DataFrame(forecast_rows);settings.output_dir.mkdir(parents=True,exist_ok=True);write_frame(out,settings.output_dir/f"forecast_{year}.parquet");write_frame(out,settings.output_dir/f"forecast_{year}.csv");return out
