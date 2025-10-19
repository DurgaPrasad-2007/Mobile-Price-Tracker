"""
Ensemble model for Mobile Price Tracker
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any, List
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from loguru import logger

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not available")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    logger.warning("CatBoost not available")

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import to_categorical
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available")

from ..utils.config import get_config


class MobilePriceEnsemble:
    """Ensemble model for mobile price prediction"""
    
    def __init__(self):
        self.config = get_config()
        self.models = {}
        self.model_weights = {}
        self.is_trained = False
        self.feature_columns = None
        
    def _create_models(self) -> Dict[str, Any]:
        """Create individual models"""
        models = {}
        
        # Random Forest
        models['random_forest'] = RandomForestClassifier(
            n_estimators=100,
            random_state=self.config.model.random_state,
            n_jobs=self.config.model.n_jobs
        )
        
        # XGBoost
        if XGBOOST_AVAILABLE:
            models['xgboost'] = xgb.XGBClassifier(
                n_estimators=100,
                random_state=self.config.model.random_state,
                n_jobs=self.config.model.n_jobs,
                eval_metric='mlogloss'
            )
        
        # LightGBM
        if LIGHTGBM_AVAILABLE:
            models['lightgbm'] = lgb.LGBMClassifier(
                n_estimators=100,
                random_state=self.config.model.random_state,
                n_jobs=self.config.model.n_jobs,
                verbose=-1
            )
        
        # CatBoost
        if CATBOOST_AVAILABLE:
            models['catboost'] = cb.CatBoostClassifier(
                iterations=100,
                random_state=self.config.model.random_state,
                verbose=False
            )
        
        # Neural Network
        if TENSORFLOW_AVAILABLE:
            models['neural_network'] = self._create_neural_network()
        
        return models
    
    def _create_neural_network(self):
        """Create neural network model"""
        from tensorflow.keras.layers import Input
        
        model = Sequential([
            Input(shape=(28,)),  # Fixed input shape
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(4, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Dict[str, float], pd.DataFrame, pd.Series]:
        """Train all models"""
        logger.info("Starting model training...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config.model.test_size,
            random_state=self.config.model.random_state,
            stratify=y
        )
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        # Create models
        self.models = self._create_models()
        
        # Train each model
        results = {}
        cv = StratifiedKFold(n_splits=self.config.model.cv_folds, shuffle=True, random_state=self.config.model.random_state)
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            
            try:
                if name == 'neural_network' and TENSORFLOW_AVAILABLE:
                    # Special handling for neural network
                    y_train_cat = to_categorical(y_train, num_classes=4)
                    y_test_cat = to_categorical(y_test, num_classes=4)
                    
                    model.fit(X_train, y_train_cat, epochs=50, batch_size=32, verbose=0)
                    
                    # Cross-validation for neural network
                    cv_scores = []
                    for train_idx, val_idx in cv.split(X_train, y_train):
                        X_train_cv, X_val_cv = X_train.iloc[train_idx], X_train.iloc[val_idx]
                        y_train_cv, y_val_cv = y_train.iloc[train_idx], y_train.iloc[val_idx]
                        y_train_cv_cat = to_categorical(y_train_cv, num_classes=4)
                        y_val_cv_cat = to_categorical(y_val_cv, num_classes=4)
                        
                        temp_model = self._create_neural_network()
                        temp_model.fit(X_train_cv, y_train_cv_cat, epochs=30, batch_size=32, verbose=0)
                        val_pred = temp_model.predict(X_val_cv, verbose=0)
                        val_pred_classes = np.argmax(val_pred, axis=1)
                        cv_scores.append(accuracy_score(y_val_cv, val_pred_classes))
                    
                    cv_score = np.mean(cv_scores)
                    
                else:
                    # Standard sklearn models
                    model.fit(X_train, y_train)
                    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
                    cv_score = cv_scores.mean()
                
                results[name] = cv_score
                logger.info(f"{name} CV accuracy: {cv_score:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to train {name}: {e}")
                results[name] = 0.0
        
        # Set model weights based on performance
        self._set_model_weights(results)
        
        self.is_trained = True
        logger.info("Model training completed")
        
        return results, X_test, y_test
    
    def _set_model_weights(self, results: Dict[str, float]) -> None:
        """Set model weights based on performance"""
        total_score = sum(results.values())
        if total_score > 0:
            self.model_weights = {name: score/total_score for name, score in results.items()}
        else:
            # Equal weights if all models failed
            self.model_weights = {name: 1.0/len(results) for name in results.keys()}
        
        logger.info(f"Model weights: {self.model_weights}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using ensemble"""
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")
        
        predictions = []
        weights = []
        
        for name, model in self.models.items():
            try:
                if name == 'neural_network' and TENSORFLOW_AVAILABLE:
                    # Convert DataFrame to numpy array for neural network
                    X_np = X.values if hasattr(X, 'values') else np.array(X)
                    pred_proba = model.predict(X_np, verbose=0)
                    pred_classes = np.argmax(pred_proba, axis=1)
                else:
                    pred_classes = model.predict(X)
                
                # Ensure prediction is a 1D numpy array
                pred_classes = np.array(pred_classes).flatten()
                predictions.append(pred_classes)
                weights.append(self.model_weights.get(name, 0))
                
            except Exception as e:
                logger.error(f"Failed to predict with {name}: {e}")
                continue
        
        if not predictions:
            raise ValueError("No models available for prediction")
        
        # Weighted voting
        # Ensure all predictions are numpy arrays of the same shape
        predictions = [np.array(pred) for pred in predictions]
        predictions = np.array(predictions)
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize weights
        
        # Weighted average for final prediction
        final_predictions = np.average(predictions, axis=0, weights=weights)
        final_predictions = np.round(final_predictions).astype(int)
        
        return final_predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities"""
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")
        
        probabilities = []
        weights = []
        
        for name, model in self.models.items():
            try:
                if name == 'neural_network' and TENSORFLOW_AVAILABLE:
                    # Convert DataFrame to numpy array for neural network
                    X_np = X.values if hasattr(X, 'values') else np.array(X)
                    proba = model.predict(X_np, verbose=0)
                else:
                    proba = model.predict_proba(X)
                
                probabilities.append(proba)
                weights.append(self.model_weights.get(name, 0))
                
            except Exception as e:
                logger.error(f"Failed to get probabilities from {name}: {e}")
                continue
        
        if not probabilities:
            raise ValueError("No models available for prediction")
        
        # Weighted average of probabilities
        # Ensure all probabilities are numpy arrays of the same shape
        probabilities = [np.array(prob) for prob in probabilities]
        probabilities = np.array(probabilities)
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize weights
        
        final_proba = np.average(probabilities, axis=0, weights=weights)
        
        return final_proba
    
    def save_models(self) -> None:
        """Save trained models"""
        models_path = Path(self.config.data.models_path)
        models_path.mkdir(parents=True, exist_ok=True)
        
        for name, model in self.models.items():
            try:
                if name == 'neural_network' and TENSORFLOW_AVAILABLE:
                    model.save(models_path / f"{name}.h5")
                else:
                    joblib.dump(model, models_path / f"{name}.pkl")
                logger.info(f"Saved {name} model")
            except Exception as e:
                logger.error(f"Failed to save {name}: {e}")
        
        # Save metadata
        metadata = {
            'model_weights': self.model_weights,
            'feature_columns': self.feature_columns,
            'is_trained': self.is_trained
        }
        joblib.dump(metadata, models_path / "metadata.pkl")
        
        logger.info("All models saved successfully")
    
    def load_models(self) -> None:
        """Load trained models"""
        models_path = Path(self.config.data.models_path)
        
        if not models_path.exists():
            raise FileNotFoundError(f"Models directory not found: {models_path}")
        
        # Load metadata
        metadata_path = models_path / "metadata.pkl"
        if metadata_path.exists():
            metadata = joblib.load(metadata_path)
            self.model_weights = metadata['model_weights']
            self.feature_columns = metadata['feature_columns']
            self.is_trained = metadata['is_trained']
        
        # Load individual models
        self.models = self._create_models()
        
        for name, model in self.models.items():
            try:
                if name == 'neural_network' and TENSORFLOW_AVAILABLE:
                    from tensorflow.keras.models import load_model
                    self.models[name] = load_model(models_path / f"{name}.h5")
                else:
                    self.models[name] = joblib.load(models_path / f"{name}.pkl")
                logger.info(f"Loaded {name} model")
            except Exception as e:
                logger.error(f"Failed to load {name}: {e}")
                # If loading fails, recreate the model
                if name == 'neural_network' and TENSORFLOW_AVAILABLE:
                    self.models[name] = self._create_neural_network()
                    logger.info(f"Recreated {name} model")
        
        logger.info("All models loaded successfully")


def get_model() -> MobilePriceEnsemble:
    """Get model instance"""
    return MobilePriceEnsemble()
