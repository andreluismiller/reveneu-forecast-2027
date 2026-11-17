import json
import typer
from .settings import settings
from .logging_config import configure
from .validation import validate_inputs
from .dataset import build
from .training import train
from .forecasting import forecast
from .reporting import create
app=typer.Typer(help="Previsão de faturamento 2027")

@app.command()
def validate():
    configure(settings.log_level);errors=validate_inputs(settings)
    if errors:
        for e in errors: typer.echo(f"ERRO: {e}")
        raise typer.Exit(1)
    typer.echo("Contratos de entrada válidos.")

@app.command()
def prepare():
    df=build(settings);typer.echo(f"Dataset criado: {settings.model_dataset_path} ({len(df)} linhas)")

@app.command()
def train_model():
    meta=train(settings);typer.echo(json.dumps({"best_model":meta["best_model"],"training_rows":meta["training_rows"]},ensure_ascii=False))

@app.command()
def forecast_year(year:int=typer.Option(settings.forecast_year,"--year")):
    df=forecast(settings,year);typer.echo(f"Previsão criada: {len(df)} meses")

@app.command()
def report(year:int=typer.Option(settings.forecast_year,"--year")):
    typer.echo(str(create(settings,year)))

@app.command()
def run_all(year:int=typer.Option(settings.forecast_year,"--year")):
    errors=validate_inputs(settings)
    if errors: raise typer.BadParameter(" | ".join(errors))
    build(settings);train(settings);forecast(settings,year);typer.echo(str(create(settings,year)))

# nomes amigáveis de CLI
app.command("train")(train_model)
app.command("forecast")(forecast_year)
app.command("run-all")(run_all)
