"""
Data preprocessing for Mobile Price Tracker
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from loguru import logger

from ..utils.config import get_config


class MobileDataPreprocessor:
    """Preprocessor for mobile phone data"""
    
    def __init__(self):
        self.config = get_config()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        
    def load_mobile_dataset(self) -> pd.DataFrame:
        """Load the mobile phone dataset"""
        try:
            # Try to load from the existing dataset.csv file
            dataset_path = Path("dataset.csv")
            if dataset_path.exists():
                df = pd.read_csv(dataset_path)
                logger.info(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
                return df
            else:
                # Generate synthetic data if no dataset exists
                logger.warning("No dataset found, generating synthetic data")
                return self._generate_synthetic_data()
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return self._generate_synthetic_data()
    
    def _generate_synthetic_data(self) -> pd.DataFrame:
        """Generate synthetic mobile phone data"""
        np.random.seed(42)
        n_samples = 2000
        
        data = {
            'battery_power': np.random.randint(500, 5000, n_samples),
            'blue': np.random.choice([0, 1], n_samples),
            'clock_speed': np.random.uniform(0.5, 3.0, n_samples),
            'dual_sim': np.random.choice([0, 1], n_samples),
            'fc': np.random.randint(0, 20, n_samples),
            'four_g': np.random.choice([0, 1], n_samples),
            'int_memory': np.random.choice([2, 4, 8, 16, 32, 64, 128, 256], n_samples),
            'm_deep': np.random.uniform(0.1, 1.0, n_samples),
            'mobile_wt': np.random.randint(80, 200, n_samples),
            'n_cores': np.random.choice([1, 2, 4, 6, 8], n_samples),
            'pc': np.random.randint(0, 50, n_samples),
            'px_height': np.random.choice([480, 720, 1080, 1440, 2160], n_samples),
            'px_width': np.random.choice([320, 480, 720, 1080, 1440], n_samples),
            'ram': np.random.choice([256, 512, 1024, 2048, 4096, 8192], n_samples),
            'sc_h': np.random.uniform(5, 20, n_samples),
            'sc_w': np.random.uniform(3, 15, n_samples),
            'talk_time': np.random.randint(2, 30, n_samples),
            'three_g': np.random.choice([0, 1], n_samples),
            'touch_screen': np.random.choice([0, 1], n_samples),
            'wifi': np.random.choice([0, 1], n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Generate price_range based on features
        df['price_range'] = self._generate_price_range(df)
        
        logger.info(f"Generated synthetic dataset with {len(df)} samples")
        return df
    
    def _generate_price_range(self, df: pd.DataFrame) -> pd.Series:
        """Generate realistic price ranges based on features"""
        price_range = np.zeros(len(df))
        
        # Low cost (0): Basic features
        low_cost_mask = (
            (df['ram'] <= 1024) & 
            (df['int_memory'] <= 16) & 
            (df['pc'] <= 8) &
            (df['battery_power'] <= 2000)
        )
        price_range[low_cost_mask] = 0
        
        # Medium cost (1): Moderate features
        medium_cost_mask = (
            (df['ram'] <= 2048) & 
            (df['int_memory'] <= 64) & 
            (df['pc'] <= 16) &
            (df['battery_power'] <= 3000) &
            ~low_cost_mask
        )
        price_range[medium_cost_mask] = 1
        
        # High cost (2): Good features
        high_cost_mask = (
            (df['ram'] <= 4096) & 
            (df['int_memory'] <= 128) & 
            (df['pc'] <= 32) &
            (df['battery_power'] <= 4000) &
            ~low_cost_mask & ~medium_cost_mask
        )
        price_range[high_cost_mask] = 2
        
        # Very high cost (3): Premium features
        very_high_cost_mask = ~low_cost_mask & ~medium_cost_mask & ~high_cost_mask
        price_range[very_high_cost_mask] = 3
        
        return price_range.astype(int)
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer new features from existing ones"""
        df_eng = df.copy()
        
        # Screen area
        df_eng['screen_area'] = df_eng['sc_h'] * df_eng['sc_w']
        
        # Pixel density
        df_eng['pixel_density'] = (df_eng['px_height'] * df_eng['px_width']) / df_eng['screen_area']
        
        # Camera ratio (front/primary)
        df_eng['camera_ratio'] = df_eng['fc'] / (df_eng['pc'] + 1)  # +1 to avoid division by zero
        
        # Memory efficiency (RAM per GB of internal memory)
        df_eng['memory_efficiency'] = df_eng['ram'] / (df_eng['int_memory'] * 1024)
        
        # Connectivity score
        df_eng['connectivity_score'] = (
            df_eng['blue'] + df_eng['four_g'] + df_eng['three_g'] + 
            df_eng['wifi'] + df_eng['dual_sim']
        )
        
        # Performance score
        df_eng['performance_score'] = (
            df_eng['clock_speed'] * df_eng['n_cores'] + 
            df_eng['ram'] / 1000 + 
            df_eng['int_memory']
        )
        
        # Battery efficiency (battery power per gram)
        df_eng['battery_efficiency'] = df_eng['battery_power'] / df_eng['mobile_wt']
        
        # Screen resolution
        df_eng['screen_resolution'] = df_eng['px_height'] * df_eng['px_width']
        
        logger.info(f"Engineered {len(df_eng.columns) - len(df.columns)} new features")
        return df_eng
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for training"""
        # Separate features and target
        target_col = 'price_range'
        feature_cols = [col for col in df.columns if col != target_col]
        
        X = df[feature_cols]
        y = df[target_col]
        
        # Store feature columns for later use
        self.feature_columns = feature_cols
        
        logger.info(f"Prepared training data: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    
    def save_dataset(self, df: pd.DataFrame) -> None:
        """Save processed dataset"""
        output_path = Path(self.config.data.processed_data_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Dataset saved to {output_path}")
    
    def load_dataset(self) -> pd.DataFrame:
        """Load processed dataset"""
        path = Path(self.config.data.processed_data_path)
        if not path.exists():
            raise FileNotFoundError(f"Processed dataset not found at {path}")
        
        df = pd.read_csv(path)
        logger.info(f"Loaded processed dataset: {df.shape[0]} rows, {df.shape[1]} columns")
        return df


def get_preprocessor() -> MobileDataPreprocessor:
    """Get preprocessor instance"""
    return MobileDataPreprocessor()
