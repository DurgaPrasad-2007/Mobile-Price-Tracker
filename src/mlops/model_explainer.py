"""
MLOps components for Mobile Price Tracker
"""

import os
import json
import joblib
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not available")

try:
    import dvc.api
    DVC_AVAILABLE = True
except ImportError:
    DVC_AVAILABLE = False
    logger.warning("DVC not available")


class MLflowManager:
    """MLflow experiment tracking manager"""
    
    def __init__(self, experiment_name: str = "mobile-price-tracker"):
        self.experiment_name = experiment_name
        self.experiment_id = None
        
        if MLFLOW_AVAILABLE:
            self._setup_experiment()
    
    def _setup_experiment(self):
        """Setup MLflow experiment"""
        try:
            # Create or get experiment
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                self.experiment_id = mlflow.create_experiment(self.experiment_name)
            else:
                self.experiment_id = experiment.experiment_id
            
            logger.info(f"MLflow experiment '{self.experiment_name}' ready")
        except Exception as e:
            logger.error(f"Failed to setup MLflow experiment: {e}")
    
    def log_model_performance(self, 
                            model_name: str,
                            metrics: Dict[str, float],
                            parameters: Dict[str, Any],
                            model_path: str,
                            run_name: Optional[str] = None):
        """Log model performance to MLflow"""
        if not MLFLOW_AVAILABLE:
            logger.warning("MLflow not available, skipping logging")
            return
        
        try:
            with mlflow.start_run(experiment_id=self.experiment_id, run_name=run_name):
                # Log parameters
                mlflow.log_params(parameters)
                
                # Log metrics
                mlflow.log_metrics(metrics)
                
                # Log model
                mlflow.sklearn.log_model(
                    sk_model=joblib.load(model_path),
                    artifact_path="model",
                    registered_model_name=f"{self.experiment_name}-{model_name}"
                )
                
                logger.info(f"Logged {model_name} performance to MLflow")
        except Exception as e:
            logger.error(f"Failed to log model performance: {e}")


class DVCManager:
    """DVC data versioning manager"""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = repo_path
        
        if DVC_AVAILABLE:
            self._check_dvc_repo()
    
    def _check_dvc_repo(self):
        """Check if DVC is initialized"""
        try:
            dvc.api.get_url("dataset.csv", repo=self.repo_path)
            logger.info("DVC repository detected")
        except Exception:
            logger.warning("DVC not initialized or dataset not tracked")
    
    def add_data(self, data_path: str, message: str = "Add dataset"):
        """Add data to DVC tracking"""
        if not DVC_AVAILABLE:
            logger.warning("DVC not available, skipping data tracking")
            return
        
        try:
            import subprocess
            subprocess.run(["dvc", "add", data_path], check=True)
            subprocess.run(["git", "add", f"{data_path}.dvc"], check=True)
            subprocess.run(["git", "commit", "-m", message], check=True)
            logger.info(f"Added {data_path} to DVC tracking")
        except Exception as e:
            logger.error(f"Failed to add data to DVC: {e}")
    
    def pull_data(self, data_path: str):
        """Pull data from DVC"""
        if not DVC_AVAILABLE:
            logger.warning("DVC not available, skipping data pull")
            return
        
        try:
            import subprocess
            subprocess.run(["dvc", "pull", data_path], check=True)
            logger.info(f"Pulled {data_path} from DVC")
        except Exception as e:
            logger.error(f"Failed to pull data from DVC: {e}")


class ModelExplainer:
    """Model explainability using SHAP and LIME"""
    
    def __init__(self):
        self.explainer = None
        self.feature_names = None
        
        try:
            import shap
            self.shap_available = True
        except ImportError:
            self.shap_available = False
            logger.warning("SHAP not available")
        
        try:
            import lime
            import lime.tabular
            self.lime_available = True
        except ImportError:
            self.lime_available = False
            logger.warning("LIME not available")
    
    def setup_explainer(self, model, X_train: pd.DataFrame, model_type: str = "tree"):
        """Setup explainer for the model"""
        self.feature_names = X_train.columns.tolist()
        
        if self.shap_available:
            try:
                if model_type == "tree":
                    self.explainer = shap.TreeExplainer(model)
                elif model_type == "linear":
                    self.explainer = shap.LinearExplainer(model, X_train)
                else:
                    self.explainer = shap.Explainer(model)
                
                logger.info("SHAP explainer setup successfully")
            except Exception as e:
                logger.error(f"Failed to setup SHAP explainer: {e}")
                self.explainer = None
    
    def explain_prediction(self, X: pd.DataFrame, prediction_idx: int = 0) -> Dict[str, Any]:
        """Explain a single prediction"""
        if not self.explainer:
            return {"error": "Explainer not available"}
        
        try:
            # Get SHAP values
            shap_values = self.explainer.shap_values(X.iloc[[prediction_idx]])
            
            # Handle multi-class case
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # Take first class
            
            # Create explanation
            explanation = {
                "feature_names": self.feature_names,
                "feature_values": X.iloc[prediction_idx].values.tolist(),
                "shap_values": shap_values[0].tolist(),
                "base_value": float(self.explainer.expected_value) if hasattr(self.explainer, 'expected_value') else 0.0
            }
            
            return explanation
            
        except Exception as e:
            logger.error(f"Failed to explain prediction: {e}")
            return {"error": str(e)}
    
    def get_feature_importance(self, X: pd.DataFrame) -> Dict[str, float]:
        """Get feature importance scores"""
        if not self.explainer:
            return {}
        
        try:
            # Get SHAP values for all samples
            shap_values = self.explainer.shap_values(X)
            
            # Handle multi-class case
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # Take first class
            
            # Calculate mean absolute SHAP values
            importance_scores = {}
            for i, feature in enumerate(self.feature_names):
                importance_scores[feature] = float(abs(shap_values[:, i]).mean())
            
            return importance_scores
            
        except Exception as e:
            logger.error(f"Failed to get feature importance: {e}")
            return {}


def get_mlflow_manager() -> MLflowManager:
    """Get MLflow manager instance"""
    return MLflowManager()


def get_dvc_manager() -> DVCManager:
    """Get DVC manager instance"""
    return DVCManager()


def get_model_explainer() -> ModelExplainer:
    """Get model explainer instance"""
    return ModelExplainer()

