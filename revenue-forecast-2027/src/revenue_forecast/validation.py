from .contracts import CONTRACTS,validate_frame
from .io import read_frame
from .settings import Settings

def validate_inputs(settings: Settings):
    paths={"revenue":settings.revenue_path,"operations":settings.operations_path,"sentiment":settings.sentiment_path,"commodity_forecast":settings.commodity_forecast_path,"material_exposure":settings.material_exposure_path};errors=[]
    for name,path in paths.items():
        if not path.exists():
            if name in {"commodity_forecast","material_exposure"}: continue
            errors.append(f"{name}: arquivo inexistente: {path}");continue
        errors.extend(validate_frame(read_frame(path),CONTRACTS[name]))
    return errors
