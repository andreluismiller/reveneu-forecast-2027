import json,pandas as pd
from .settings import Settings

def create(settings: Settings,year: int):
    f=pd.read_parquet(settings.output_dir/f"forecast_{year}.parquet");meta=json.loads((settings.artifact_dir/"model_metadata.json").read_text(encoding="utf-8"));total=f[["revenue_p10","revenue_p50","revenue_p90"]].sum()
    text=f"""# Resumo executivo — previsão de faturamento {year}\n\n## Resultado anual\n- P10: R$ {total.revenue_p10:,.2f}\n- P50: R$ {total.revenue_p50:,.2f}\n- P90: R$ {total.revenue_p90:,.2f}\n\n## Modelo\n- Modelo selecionado: `{meta['best_model']}`\n- Histórico de treino: {meta['training_rows']} meses\n- Fim do treino: {meta['training_end']}\n\n## Uso\nP10 representa cenário adverso, P50 cenário central e P90 cenário favorável. Os intervalos derivam dos resíduos observados na validação temporal; não constituem garantia financeira.\n"""
    path=settings.output_dir/"executive_summary.md";path.write_text(text,encoding="utf-8");return path
