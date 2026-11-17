# Revenue Forecast 2027

Segunda aplicação do programa de inteligência preditiva. Consolida dados financeiros e operacionais com os arquivos Parquet produzidos pela aplicação de notícias e commodities, treina modelos temporais de faturamento e gera previsão mensal de 2027 com cenários P10, P50 e P90.

## Entradas

| Arquivo | Granularidade | Obrigatório | Finalidade |
|---|---|---:|---|
| `data/input/revenue_monthly.parquet` | mensal | sim | target histórico de faturamento |
| `data/input/operations_monthly.parquet` | mensal | sim | carteira, volume, KVA, mix, capacidade |
| `data/input/sentiment_daily.parquet` | diária/commodity | sim | saída Gold da aplicação 1 |
| `data/input/commodity_forecast.parquet` | mensal/commodity | recomendado | preços previstos e cenários |
| `data/input/material_exposure.parquet` | mensal/commodity | recomendado | consumo/exposição dos insumos |

## Contratos mínimos

### revenue_monthly.parquet
`date,revenue`

### operations_monthly.parquet
`date,volume_units,kva_produced,kva_billed,backlog,avg_sale_price,capacity_utilization`

### sentiment_daily.parquet
`date,commodity,news_count,sentiment_weighted,price_impact_weighted`

### commodity_forecast.parquet
`date,commodity,price_p10,price_p50,price_p90`

### material_exposure.parquet
`date,commodity,quantity,unit_cost_actual`

## Codespaces
1. Crie um repositório vazio no GitHub e envie estes arquivos.
2. Abra **Code > Codespaces > Create codespace on main**.
3. O `postCreateCommand` instala o uv e executa `uv sync --all-groups`.
4. Copie `.env.example` para `.env` e ajuste os caminhos.
5. Para segredos reais, prefira **Codespaces secrets**. O `.env` está no `.gitignore`.

## Execução rápida

```bash
cp .env.example .env
uv sync --all-groups
uv run revenue-forecast validate
uv run revenue-forecast prepare
uv run revenue-forecast train
uv run revenue-forecast forecast --year 2027
uv run revenue-forecast report
```

Pipeline completo:

```bash
uv run revenue-forecast run-all --year 2027
```

Gerar dados sintéticos apenas para teste:

```bash
uv run python scripts/create_sample_data.py
uv run revenue-forecast run-all --year 2027
```

## Saídas
- `data/processed/model_dataset.parquet`
- `artifacts/model.joblib`
- `artifacts/model_metadata.json`
- `outputs/forecast_2027.parquet`
- `outputs/forecast_2027.csv`
- `outputs/metrics.json`
- `outputs/executive_summary.md`

## Modelos
- Baseline sazonal: faturamento de 12 meses atrás.
- Elastic Net: baseline regularizado e interpretável.
- HistGradientBoosting: modelo não linear principal.
- Random Forest: candidato robusto para comparação.
- Seleção pelo menor MAE médio em validação walk-forward.
- Intervalos P10/P50/P90 por distribuição empírica dos resíduos fora da amostra.

## Segurança
- `.env` nunca deve ser versionado.
- `.env.example` contém apenas nomes e valores fictícios.
- Em Codespaces, segredos definidos no repositório/organização ficam disponíveis como variáveis de ambiente.
- O código não imprime credenciais em logs.
