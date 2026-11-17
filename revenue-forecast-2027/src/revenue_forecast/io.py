from pathlib import Path
import pandas as pd

def read_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    suffix=path.suffix.lower()
    if suffix==".parquet": return pd.read_parquet(path)
    if suffix==".csv": return pd.read_csv(path)
    if suffix in {".xlsx",".xls"}: return pd.read_excel(path,engine="openpyxl")
    raise ValueError(f"Formato não suportado: {suffix}")

def write_frame(df: pd.DataFrame,path: Path) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.suffix.lower()==".parquet": df.to_parquet(path,index=False)
    elif path.suffix.lower()==".csv": df.to_csv(path,index=False,encoding="utf-8-sig")
    else: raise ValueError(f"Formato de saída não suportado: {path.suffix}")
