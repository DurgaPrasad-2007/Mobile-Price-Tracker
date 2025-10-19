"""
FastAPI endpoints for Mobile Price Tracker
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from loguru import logger
import time
from pathlib import Path

from ..utils.config import get_config
from ..models.ensemble import get_model
from ..data.preprocessing import get_preprocessor
from ..monitoring.metrics import get_metrics_collector


# Pydantic models for API
class MobilePhoneFeatures(BaseModel):
    """Mobile phone features for prediction"""
    battery_power: int = Field(..., ge=100, le=6000, description="Battery capacity in mAh")
    blue: int = Field(..., ge=0, le=1, description="Has Bluetooth (0/1)")
    clock_speed: float = Field(..., ge=0.1, le=5.0, description="Processor speed in GHz")
    dual_sim: int = Field(..., ge=0, le=1, description="Has dual sim support (0/1)")
    fc: int = Field(..., ge=0, le=50, description="Front camera megapixels")
    four_g: int = Field(..., ge=0, le=1, description="Has 4G (0/1)")
    int_memory: int = Field(..., ge=1, le=512, description="Internal memory in GB")
    m_deep: float = Field(..., ge=0.1, le=2.0, description="Mobile depth in cm")
    mobile_wt: int = Field(..., ge=50, le=300, description="Weight in grams")
    n_cores: int = Field(..., ge=1, le=12, description="Processor core count")
    pc: int = Field(..., ge=0, le=100, description="Primary camera megapixels")
    px_height: int = Field(..., ge=240, le=4320, description="Pixel resolution height")
    px_width: int = Field(..., ge=320, le=7680, description="Pixel resolution width")
    ram: int = Field(..., ge=128, le=16384, description="RAM in MB")
    sc_h: float = Field(..., ge=3.0, le=25.0, description="Screen height in cm")
    sc_w: float = Field(..., ge=2.0, le=20.0, description="Screen width in cm")
    talk_time: int = Field(..., ge=1, le=50, description="Talk time in hours")
    three_g: int = Field(..., ge=0, le=1, description="Has 3G (0/1)")
    touch_screen: int = Field(..., ge=0, le=1, description="Has touch screen (0/1)")
    wifi: int = Field(..., ge=0, le=1, description="Has WiFi (0/1)")


class PredictionResponse(BaseModel):
    """Prediction response"""
    price_range: int = Field(..., description="Predicted price range (0=low, 1=medium, 2=high, 3=very high)")
    price_range_label: str = Field(..., description="Price range label")
    confidence: float = Field(..., description="Prediction confidence")
    probabilities: Dict[str, float] = Field(..., description="Probability for each price range")
    processing_time: float = Field(..., description="Processing time in seconds")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: float
    models_loaded: bool
    version: str


class ModelInfoResponse(BaseModel):
    """Model information response"""
    models: Dict[str, Any]
    weights: Dict[str, float]
    feature_count: int
    is_trained: bool


# Initialize FastAPI app
config = get_config()
app = FastAPI(
    title=config.api.title,
    description=config.api.description,
    version=config.api.version
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (for serving the frontend)
static_path = Path(__file__).parent.parent.parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Global variables for model and metrics
model = None
metrics_collector = get_metrics_collector()


def get_model_instance():
    """Get model instance"""
    global model
    if model is None:
        model = get_model()
        try:
            model.load_models()
        except FileNotFoundError:
            logger.warning("Models not found, will need to train first")
    return model


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global model
    logger.info("Starting Mobile Price Tracker API...")
    model = get_model()
    try:
        model.load_models()
        logger.info("Models loaded successfully")
    except FileNotFoundError:
        logger.warning("Models not found - will train on first prediction")
        # Don't fail startup, just mark as not trained
        model.is_trained = False
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        model.is_trained = False


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend HTML"""
    html_path = Path(__file__).parent.parent.parent / "index.html"
    if html_path.exists():
        with open(html_path, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    else:
        return HTMLResponse(content="""
        <!DOCTYPE html>
        <html>
        <head><title>Mobile Price Tracker</title></head>
        <body>
            <h1>Mobile Price Tracker API</h1>
            <p>Version: """ + config.api.version + """</p>
            <p><a href="/docs">API Documentation</a></p>
            <p><a href="/health">Health Check</a></p>
        </body>
        </html>
        """)

@app.get("/api", response_model=Dict[str, str])
async def api_info():
    """API information endpoint"""
    return {
        "message": "Mobile Price Tracker API",
        "version": config.api.version,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    global model
    
    models_loaded = False
    if model is not None:
        models_loaded = model.is_trained
    
    return HealthResponse(
        status="healthy",
        timestamp=time.time(),
        models_loaded=models_loaded,
        version=config.api.version
    )


@app.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get model information"""
    model_instance = get_model_instance()
    
    return ModelInfoResponse(
        models=list(model_instance.models.keys()),
        weights=model_instance.model_weights,
        feature_count=len(model_instance.feature_columns) if model_instance.feature_columns else 0,
        is_trained=model_instance.is_trained
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_price_range(features: MobilePhoneFeatures):
    """Predict mobile phone price range"""
    start_time = time.time()
    
    try:
        model_instance = get_model_instance()
        
        if not model_instance.is_trained:
            raise HTTPException(
                status_code=503, 
                detail="Models not trained. Please train models first."
            )
        
        # Convert features to DataFrame
        feature_dict = features.dict()
        df = pd.DataFrame([feature_dict])
        
        # Ensure all columns are numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Engineer features (same as training)
        from ..data.preprocessing import get_preprocessor
        preprocessor = get_preprocessor()
        df_engineered = preprocessor.engineer_features(df)
        
        # Ensure all engineered features are numeric
        for col in df_engineered.columns:
            df_engineered[col] = pd.to_numeric(df_engineered[col], errors='coerce')
        
        # Use all engineered features for prediction
        # The model expects the same features it was trained with
        
        # Make prediction
        prediction = model_instance.predict(df_engineered)[0]
        probabilities = model_instance.predict_proba(df_engineered)[0]
        
        # Map prediction to label
        price_labels = {
            0: "Low Cost",
            1: "Medium Cost", 
            2: "High Cost",
            3: "Very High Cost"
        }
        
        # Get confidence (max probability)
        confidence = float(np.max(probabilities))
        
        # Format probabilities
        prob_dict = {
            f"range_{i}": float(prob) 
            for i, prob in enumerate(probabilities)
        }
        
        processing_time = time.time() - start_time
        
        # Record metrics
        metrics_collector.record_prediction(
            prediction=prediction,
            processing_time=processing_time,
            confidence=confidence
        )
        
        return PredictionResponse(
            price_range=int(prediction),
            price_range_label=price_labels[prediction],
            confidence=confidence,
            probabilities=prob_dict,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-batch", response_model=List[PredictionResponse])
async def predict_batch(features_list: List[MobilePhoneFeatures]):
    """Predict price ranges for multiple phones"""
    if len(features_list) > 100:
        raise HTTPException(status_code=400, detail="Batch size too large (max 100)")
    
    start_time = time.time()
    
    try:
        model_instance = get_model_instance()
        
        if not model_instance.is_trained:
            raise HTTPException(
                status_code=503, 
                detail="Models not trained. Please train models first."
            )
        
        # Convert features to DataFrame
        feature_dicts = [features.dict() for features in features_list]
        df = pd.DataFrame(feature_dicts)
        
        # Engineer features (same as training)
        from ..data.preprocessing import get_preprocessor
        preprocessor = get_preprocessor()
        df_engineered = preprocessor.engineer_features(df)
        
        # Use all engineered features for prediction
        # The model expects the same features it was trained with
        
        # Make predictions
        predictions = model_instance.predict(df_engineered)
        probabilities = model_instance.predict_proba(df_engineered)
        
        # Map predictions to labels
        price_labels = {
            0: "Low Cost",
            1: "Medium Cost", 
            2: "High Cost",
            3: "Very High Cost"
        }
        
        results = []
        processing_time = time.time() - start_time
        
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            confidence = float(np.max(probs))
            prob_dict = {
                f"range_{j}": float(prob) 
                for j, prob in enumerate(probs)
            }
            
            results.append(PredictionResponse(
                price_range=int(pred),
                price_range_label=price_labels[pred],
                confidence=confidence,
                probabilities=prob_dict,
                processing_time=processing_time / len(features_list)
            ))
        
        # Record batch metrics
        metrics_collector.record_batch_prediction(
            batch_size=len(features_list),
            processing_time=processing_time
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    """Get Prometheus metrics"""
    return metrics_collector.get_metrics()


@app.get("/stats")
async def get_stats():
    """Get prediction statistics"""
    return metrics_collector.get_stats()


@app.post("/train")
async def train_models():
    """Train models if they don't exist"""
    global model
    
    try:
        if model.is_trained:
            return {"message": "Models already trained", "status": "success"}
        
        # Import here to avoid circular imports
        from ..data.preprocessing import get_preprocessor
        
        preprocessor = get_preprocessor()
        model_instance = get_model()
        
        # Load dataset
        df = preprocessor.load_dataset()
        X, y = preprocessor.prepare_training_data(df)
        
        # Train models
        results, X_test, y_test = model_instance.train_models(X, y)
        
        # Save models
        model_instance.save_models()
        
        # Update global model
        model = model_instance
        
        return {
            "message": "Models trained successfully",
            "status": "success",
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Failed to train models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
