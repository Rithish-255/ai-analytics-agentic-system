from dataclasses import dataclass


@dataclass
class ProjectConfig:
    random_seed: int = 42
    prediction_days: int = 7
    train_test_split: float = 0.8
    sales_column: str = "sales"


CONFIG = ProjectConfig()
