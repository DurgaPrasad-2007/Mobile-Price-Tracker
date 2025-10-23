# Mobile Price Tracker

A mobile phone price prediction system using machine learning to classify phones into price ranges (low/medium/high/very high).

## Overview

This project implements a mobile phone price prediction system that analyzes phone features to predict price ranges. It uses ensemble machine learning models to make predictions and provides explanations for its decisions.

## Features

- Mobile phone price range prediction (0=low cost, 1=medium cost, 2=high cost, 3=very high cost)
- Multiple ML models (XGBoost, LightGBM, CatBoost, Neural Networks)
- Model explainability with SHAP and LIME
- REST API for integration
- **Modern Web Frontend** with clean, responsive UI
- Real-time prediction with confidence scores
- Interactive specification input forms
- Live statistics and monitoring dashboard
- Basic monitoring and logging

## Architecture

```
Data → Features → Models → API → Response
```

## Technology Stack

- Python 3.11
- FastAPI for the API
- XGBoost, LightGBM, CatBoost for ML models
- TensorFlow for neural networks
- SHAP & LIME for explanations
- Docker for deployment

## Getting Started

### Prerequisites
- Python 3.11+
- Poetry (for dependency management)

### Installation
```bash
# Install Poetry (if not installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Run the system
poetry run devrun
```

### Quick Start with Frontend
```bash
# Start the complete system (backend + frontend on port 8000)
poetry run devrun

# Or start in serve mode
poetry run python main.py --mode serve
```

### Access the Application
- **Frontend UI**: http://localhost:8000 - Modern web interface for testing
- **Backend API**: http://localhost:8000/api - REST API endpoints
- **API Documentation**: http://localhost:8000/docs - Interactive API docs
- **Health Check**: http://localhost:8000/health - System status

### What the system does
1. Loads mobile phone dataset with features
2. Engineers features from phone specifications
3. Trains ensemble ML models
4. Provides real-time price range predictions via API

## Dataset Features

The system analyzes the following mobile phone features:
- **battery_power**: Battery Capacity in mAh
- **blue**: Has Bluetooth or not
- **clock_speed**: Processor speed
- **dual_sim**: Has dual sim support or not
- **fc**: Front camera megapixels
- **four_g**: Has 4G or not
- **int_memory**: Internal Memory in GB
- **m_deep**: Mobile depth in cm
- **mobile_wt**: Weight in gm
- **n_cores**: Processor Core Count
- **pc**: Primary Camera megapixels
- **px_height**: Pixel Resolution height
- **px_width**: Pixel Resolution width
- **ram**: RAM in MB
- **sc_h**: Mobile Screen height in cm
- **sc_w**: Mobile Screen width in cm
- **talk_time**: Time a single battery charge will last in hours
- **three_g**: Has 3G or not
- **touch_screen**: Has touch screen or not
- **wifi**: Has WiFi or not
- **price_range**: Target variable (0=low cost, 1=medium cost, 2=high cost, 3=very high cost)

## Data Processing Pipeline

### Feature Engineering
Creates features like:
- Screen area calculation
- Pixel density
- Camera ratio (front/primary)
- Memory efficiency ratios
- Connectivity features aggregation

### Model Training
Trains multiple models and combines them:
- XGBoost (30%)
- LightGBM (30%) 
- CatBoost (20%)
- Neural Network (15%)
- Random Forest (5%)

## API Usage

### Single Phone Price Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "battery_power": 2000,
    "blue": 1,
    "clock_speed": 2.0,
    "dual_sim": 1,
    "fc": 8,
    "four_g": 1,
    "int_memory": 64,
    "m_deep": 0.8,
    "mobile_wt": 150,
    "n_cores": 4,
    "pc": 12,
    "px_height": 1920,
    "px_width": 1080,
    "ram": 4096,
    "sc_h": 15,
    "sc_w": 8,
    "talk_time": 20,
    "three_g": 1,
    "touch_screen": 1,
    "wifi": 1
  }'
```

### Health Check
```bash
curl http://localhost:8000/health
```

## Testing

Run the test suite:
```bash
poetry run pytest tests/ -v
```

Run a simple system test:
```bash
python test_system.py
```

### Model Evaluation Notebook

For comprehensive evaluation and testing, use the Jupyter notebook:

```bash
# Install Jupyter if not already installed
poetry run pip install jupyter

# Run the evaluation notebook
poetry run jupyter notebook mobile_price_tracker_evaluation.ipynb
```

The notebook provides:
- **Dataset Analysis**: Explore the mobile phone dataset and feature distributions
- **Model Evaluation**: Comprehensive performance metrics and visualizations
- **API Testing**: Test all FastAPI endpoints with real requests
- **Performance Testing**: Measure response times and system throughput
- **Sample Predictions**: Test with different phone specifications
- **Model Comparison**: Analyze individual model contributions

### Quick System Test

The `test_system.py` script provides a fast way to verify system functionality:

```bash
python test_system.py
```

This tests:
- ✅ Module imports
- ✅ Data loading and preprocessing
- ✅ Model loading
- ✅ Direct model predictions
- ✅ API endpoint availability

## Development

### Code Quality
```bash
# Format code
poetry run black src/
poetry run isort src/

# Lint code
poetry run flake8 src/

# Type checking
poetry run mypy src/

# Security scan
poetry run safety check
poetry run bandit -r src/
```

### Pre-commit Hooks
```bash
poetry run pre-commit install
```

## Docker Deployment

### Single Container
```bash
docker build -t mobile-price-tracker .
docker run -p 8000:8000 mobile-price-tracker
```

### Multi-Service
```bash
docker-compose up -d
```

## Project Structure

```
mobile-price-tracker/
├── src/                          # Source code
│   ├── data/                     # Data processing
│   ├── models/                   # ML models
│   ├── api/                      # API endpoints
│   ├── monitoring/               # Monitoring
│   ├── mlops/                    # MLOps components
│   └── utils/                    # Utilities
├── config/                       # Configuration files
├── tests/                        # Test suite
├── pyproject.toml               # Poetry configuration
├── Dockerfile                   # Container config
└── main.py                      # Application entry point
```

## Known Issues

- Models need to be trained before first use
- No authentication on API endpoints (TODO)
- Limited input validation (TODO)
- No rate limiting implemented (TODO)
- Hardcoded configuration values (TODO)

## TODO

- [ ] Add proper authentication
- [ ] Implement rate limiting
- [ ] Add more comprehensive tests
- [ ] Improve error handling
- [ ] Add model versioning
- [ ] Implement proper monitoring
- [ ] Add database integration
- [ ] Implement caching

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
