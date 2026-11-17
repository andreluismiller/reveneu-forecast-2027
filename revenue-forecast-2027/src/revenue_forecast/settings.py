from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    app_env: str = "development"
    log_level: str = "INFO"
    random_seed: int = 42
    revenue_path: Path = Path("data/input/revenue_monthly.parquet")
    operations_path: Path = Path("data/input/operations_monthly.parquet")
    sentiment_path: Path = Path("data/input/sentiment_daily.parquet")
    commodity_forecast_path: Path = Path("data/input/commodity_forecast.parquet")
    material_exposure_path: Path = Path("data/input/material_exposure.parquet")
    model_dataset_path: Path = Path("data/processed/model_dataset.parquet")
    artifact_dir: Path = Path("artifacts")
    output_dir: Path = Path("outputs")
    forecast_year: int = 2027
    test_months: int = 12
    cv_min_train_months: int = 36
    cv_test_months: int = 6
    cv_step_months: int = 3
    monte_carlo_runs: int = 5000
    azure_storage_account: str | None = None
    azure_storage_container: str | None = None
    azure_storage_sas_token: str | None = None
    database_url: str | None = None

settings = Settings()
