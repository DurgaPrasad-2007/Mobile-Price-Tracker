"""
Monitoring and metrics for Mobile Price Tracker
"""

import time
from typing import Dict, Any, List
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from loguru import logger


class MetricsCollector:
    """Collects and exposes metrics for the mobile price tracker"""
    
    def __init__(self):
        # Prometheus metrics
        self.predictions_total = Counter(
            'mobile_price_predictions_total',
            'Total number of predictions made',
            ['price_range']
        )
        
        self.prediction_duration = Histogram(
            'mobile_price_prediction_duration_seconds',
            'Time spent on predictions',
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
        )
        
        self.prediction_confidence = Histogram(
            'mobile_price_prediction_confidence',
            'Prediction confidence scores',
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )
        
        self.batch_predictions_total = Counter(
            'mobile_price_batch_predictions_total',
            'Total number of batch predictions',
            ['batch_size_range']
        )
        
        self.batch_prediction_duration = Histogram(
            'mobile_price_batch_prediction_duration_seconds',
            'Time spent on batch predictions',
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
        )
        
        self.active_requests = Gauge(
            'mobile_price_active_requests',
            'Number of active requests'
        )
        
        self.model_load_status = Gauge(
            'mobile_price_model_loaded',
            'Whether models are loaded (1=loaded, 0=not loaded)'
        )
        
        # Internal tracking
        self._prediction_history = []
        self._batch_history = []
        self._start_time = time.time()
        
        logger.info("Metrics collector initialized")
    
    def record_prediction(self, prediction: int, processing_time: float, confidence: float):
        """Record a single prediction"""
        price_range_labels = {0: 'low', 1: 'medium', 2: 'high', 3: 'very_high'}
        price_range_label = price_range_labels.get(prediction, 'unknown')
        
        # Update Prometheus metrics
        self.predictions_total.labels(price_range=price_range_label).inc()
        self.prediction_duration.observe(processing_time)
        self.prediction_confidence.observe(confidence)
        
        # Store for internal stats
        self._prediction_history.append({
            'prediction': prediction,
            'processing_time': processing_time,
            'confidence': confidence,
            'timestamp': time.time()
        })
        
        # Keep only last 1000 predictions
        if len(self._prediction_history) > 1000:
            self._prediction_history = self._prediction_history[-1000:]
    
    def record_batch_prediction(self, batch_size: int, processing_time: float):
        """Record a batch prediction"""
        # Determine batch size range
        if batch_size <= 10:
            size_range = 'small'
        elif batch_size <= 50:
            size_range = 'medium'
        else:
            size_range = 'large'
        
        # Update Prometheus metrics
        self.batch_predictions_total.labels(batch_size_range=size_range).inc()
        self.batch_prediction_duration.observe(processing_time)
        
        # Store for internal stats
        self._batch_history.append({
            'batch_size': batch_size,
            'processing_time': processing_time,
            'timestamp': time.time()
        })
        
        # Keep only last 100 batch predictions
        if len(self._batch_history) > 100:
            self._batch_history = self._batch_history[-100:]
    
    def set_model_status(self, loaded: bool):
        """Set model load status"""
        self.model_load_status.set(1 if loaded else 0)
    
    def increment_active_requests(self):
        """Increment active requests counter"""
        self.active_requests.inc()
    
    def decrement_active_requests(self):
        """Decrement active requests counter"""
        self.active_requests.dec()
    
    def get_metrics(self) -> str:
        """Get Prometheus metrics"""
        return generate_latest().decode('utf-8')
    
    def get_stats(self) -> Dict[str, Any]:
        """Get internal statistics"""
        current_time = time.time()
        uptime = current_time - self._start_time
        
        # Calculate prediction statistics
        if self._prediction_history:
            predictions = [p['prediction'] for p in self._prediction_history]
            confidences = [p['confidence'] for p in self._prediction_history]
            processing_times = [p['processing_time'] for p in self._prediction_history]
            
            prediction_counts = {}
            for pred in predictions:
                # Convert numpy types to Python int for JSON serialization
                pred_key = int(pred) if hasattr(pred, 'item') else pred
                prediction_counts[pred_key] = prediction_counts.get(pred_key, 0) + 1
            
            stats = {
                'uptime_seconds': float(uptime),
                'total_predictions': int(len(self._prediction_history)),
                'predictions_per_minute': float(len(self._prediction_history) / (uptime / 60)),
                'prediction_distribution': prediction_counts,
                'average_confidence': float(sum(confidences) / len(confidences)),
                'average_processing_time': float(sum(processing_times) / len(processing_times)),
                'min_confidence': float(min(confidences)),
                'max_confidence': float(max(confidences)),
                'min_processing_time': float(min(processing_times)),
                'max_processing_time': float(max(processing_times))
            }
        else:
            stats = {
                'uptime_seconds': float(uptime),
                'total_predictions': 0,
                'predictions_per_minute': 0.0,
                'prediction_distribution': {},
                'average_confidence': 0.0,
                'average_processing_time': 0.0,
                'min_confidence': 0.0,
                'max_confidence': 0.0,
                'min_processing_time': 0.0,
                'max_processing_time': 0.0
            }
        
        # Add batch statistics
        if self._batch_history:
            batch_sizes = [b['batch_size'] for b in self._batch_history]
            batch_times = [b['processing_time'] for b in self._batch_history]
            
            stats.update({
                'total_batch_predictions': int(len(self._batch_history)),
                'average_batch_size': float(sum(batch_sizes) / len(batch_sizes)),
                'average_batch_processing_time': float(sum(batch_times) / len(batch_times)),
                'max_batch_size': int(max(batch_sizes)),
                'min_batch_size': int(min(batch_sizes))
            })
        else:
            stats.update({
                'total_batch_predictions': 0,
                'average_batch_size': 0.0,
                'average_batch_processing_time': 0.0,
                'max_batch_size': 0,
                'min_batch_size': 0
            })
        
        return stats


# Global metrics collector instance
_metrics_collector = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector
