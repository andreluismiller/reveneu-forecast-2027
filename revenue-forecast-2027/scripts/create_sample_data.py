from pathlib import Path
import numpy as np,pandas as pd
rng=np.random.default_rng(42);out=Path("data/input");out.mkdir(parents=True,exist_ok=True)
dates=pd.date_range("2021-01-01",periods=72,freq="MS");season=1+.12*np.sin(2*np.pi*dates.month/12);volume=rng.integers(2500,3800,len(dates));price=rng.normal(28000,2500,len(dates));revenue=volume*price*season+rng.normal(0,3e6,len(dates))
pd.DataFrame({"date":dates,"revenue":revenue}).to_parquet(out/"revenue_monthly.parquet",index=False)
pd.DataFrame({"date":dates,"volume_units":volume,"kva_produced":rng.normal(350000,35000,len(dates)),"kva_billed":rng.normal(330000,40000,len(dates)),"backlog":rng.normal(180e6,25e6,len(dates)),"avg_sale_price":price,"capacity_utilization":rng.uniform(.7,.95,len(dates))}).to_parquet(out/"operations_monthly.parquet",index=False)
commodities=["cobre","aluminio","aco_carbono","aco_eletrico_silicio","oleo_isolante"]
daily=pd.date_range(dates.min(),dates.max()+pd.offsets.MonthEnd(0),freq="D");rows=[]
for d in daily:
 for c in commodities: rows.append({"date":d,"commodity":c,"news_count":rng.poisson(4),"sentiment_weighted":rng.normal(0,.2),"price_impact_weighted":rng.normal(0,.25)})
pd.DataFrame(rows).to_parquet(out/"sentiment_daily.parquet",index=False)
future=pd.date_range("2027-01-01",periods=12,freq="MS");rows=[];expo=[]
base={"cobre":9000,"aluminio":2500,"aco_carbono":700,"aco_eletrico_silicio":1800,"oleo_isolante":1100}
for d in future:
 for c,b in base.items(): rows.append({"date":d,"commodity":c,"price_p10":b*.9,"price_p50":b,"price_p90":b*1.12});expo.append({"date":d,"commodity":c,"quantity":rng.uniform(50,300),"unit_cost_actual":b})
pd.DataFrame(rows).to_parquet(out/"commodity_forecast.parquet",index=False);pd.DataFrame(expo).to_parquet(out/"material_exposure.parquet",index=False)
print("Dados sintéticos criados. Não usar para decisão real.")
