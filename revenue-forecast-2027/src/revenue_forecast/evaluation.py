import numpy as np,pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error,mean_squared_error

def metrics(y,p):
    y=np.asarray(y);p=np.asarray(p);den=np.where(np.abs(y)<1e-9,np.nan,np.abs(y))
    return {"mae":float(mean_absolute_error(y,p)),"rmse":float(mean_squared_error(y,p)**.5),"smape":float(np.nanmean(2*np.abs(p-y)/(np.abs(y)+np.abs(p)+1e-9))*100),"mape":float(np.nanmean(np.abs(p-y)/den)*100)}

def splits(n,min_train=36,test=6,step=3):
    end=min_train
    while end+test<=n: yield np.arange(end),np.arange(end,end+test);end+=step

def walk_forward(model,X,y,dates,min_train,test,step):
    rows=[];preds=[]
    for fold,(tr,te) in enumerate(splits(len(X),min_train,test,step),1):
        m=clone(model);m.fit(X.iloc[tr],y.iloc[tr]);p=m.predict(X.iloc[te]);rows.append({"fold":fold,**metrics(y.iloc[te],p)});preds.extend({"date":dates.iloc[i],"actual":float(y.iloc[i]),"predicted":float(v),"fold":fold} for i,v in zip(te,p))
    return pd.DataFrame(rows),pd.DataFrame(preds)
