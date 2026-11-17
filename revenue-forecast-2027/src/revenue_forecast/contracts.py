from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True)
class Contract:
    name: str
    required: tuple[str,...]
    unique_key: tuple[str,...]

CONTRACTS={
 "revenue":Contract("revenue",("date","revenue"),("date",)),
 "operations":Contract("operations",("date","volume_units","backlog","avg_sale_price"),("date",)),
 "sentiment":Contract("sentiment",("date","commodity","news_count","sentiment_weighted","price_impact_weighted"),("date","commodity")),
 "commodity_forecast":Contract("commodity_forecast",("date","commodity","price_p10","price_p50","price_p90"),("date","commodity")),
 "material_exposure":Contract("material_exposure",("date","commodity","quantity"),("date","commodity")),
}

def validate_frame(df: pd.DataFrame,contract: Contract) -> list[str]:
    errors=[];missing=sorted(set(contract.required)-set(df.columns))
    if missing: errors.append(f"{contract.name}: colunas ausentes: {missing}")
    if "date" in df.columns and pd.to_datetime(df["date"],errors="coerce").isna().any(): errors.append(f"{contract.name}: datas inválidas")
    if set(contract.unique_key).issubset(df.columns) and df.duplicated(list(contract.unique_key)).any(): errors.append(f"{contract.name}: chave duplicada {contract.unique_key}")
    return errors
