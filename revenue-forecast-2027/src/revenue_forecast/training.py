import json
from pathlib import Path
import joblib,numpy as np,pandas as pd
from .io import read_frame
from .models import candidates
from .evaluation import walk_forward
from .settings import Settings

def feature_columns(df):
    blocked={"date","revenue"};return [c for c in df.columns if c not in blocked and pd.api.types.is_numeric_dtype(df[c])]

def train(settings: Settings):
    df=read_frame(settings.model_dataset_path).sort_values("date");df["date"]=pd.to_datetime(df["date"]);df=df.dropna(subset=["revenue"]);features=feature_columns(df);X=df[features];y=df["revenue"]
    results={};all_preds={}
    for name,model in candidates(settings.random_seed).items():
        folds,preds=walk_forward(model,X,y,df["date"],settings.cv_min_train_months,settings.cv_test_months,settings.cv_step_months)
        if folds.empty: raise ValueError("Histórico insuficiente para a validação. Reduza CV_MIN_TRAIN_MONTHS ou forneça mais meses.")
        results[name]={"mae_mean":float(folds.mae.mean()),"rmse_mean":float(folds.rmse.mean()),"smape_mean":float(folds.smape.mean()),"folds":folds.to_dict("records")};all_preds[name]=preds
    best=min(results,key=lambda n:results[n]["mae_mean"]);model=candidates(settings.random_seed)[best];model.fit(X,y)
    residuals=(all_preds[best]["actual"]-all_preds[best]["predicted"]).to_numpy();settings.artifact_dir.mkdir(parents=True,exist_ok=True)
    joblib.dump({"model":model,"features":features,"residuals":residuals,"last_date":str(df.date.max().date()),"last_rows":df.tail(12)},settings.artifact_dir/"model.joblib")
    meta={"best_model":best,"features":features,"validation":results,"training_rows":len(df),"training_end":str(df.date.max().date())};(settings.artifact_dir/"model_metadata.json").write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding="utf-8")
    return meta
