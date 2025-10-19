"""
Configuration management for Mobile Price Tracker
"""

import os
from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class APIConfig:
    """API configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    title: str = "Mobile Price Tracker API"
    description: str = "API for predicting mobile phone price ranges"
    version: str = "1.0.0"


@dataclass
class ModelConfig:
    """Model configuration"""
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    n_jobs: int = -1


@dataclass
class DataConfig:
    """Data configuration"""
    raw_data_path: str = "data/raw/mobile_data.csv"
    processed_data_path: str = "data/processed/mobile_dataset.csv"
    models_path: str = "data/models"
    logs_path: str = "logs"


@dataclass
class Config:
    """Main configuration class"""
    api: APIConfig = None
    model: ModelConfig = None
    data: DataConfig = None
    
    def __post_init__(self):
        """Create directories after initialization"""
        if self.api is None:
            self.api = APIConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.data is None:
            self.data = DataConfig()
        
        Path(self.data.models_path).mkdir(parents=True, exist_ok=True)
        Path(self.data.logs_path).mkdir(parents=True, exist_ok=True)


def get_config() -> Config:
    """Get configuration instance"""
    return Config()
