from dataclasses import dataclass


@dataclass
class ModelNameConfig:
    model_name: str = "LinearRegression"
    fine_tuning: bool = False
