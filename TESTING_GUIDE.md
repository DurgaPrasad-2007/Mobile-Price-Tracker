# Mobile Price Tracker - Testing Guide

## Overview

This guide explains how to test the Mobile Price Tracker system, including unit tests, integration tests, and manual testing procedures.

## Prerequisites

- Python 3.11+
- Poetry installed
- All dependencies installed (`poetry install`)

## Running Tests

### Unit Tests

```bash
# Run all tests
poetry run pytest tests/ -v

# Run specific test file
poetry run pytest tests/test_preprocessing.py -v

# Run with coverage
poetry run pytest tests/ --cov=src --cov-report=html
```

### Integration Tests

```bash
# Run integration tests
poetry run pytest tests/integration/ -v

# Run API tests
poetry run pytest tests/test_api.py -v
```

## Test Structure

```
tests/
├── unit/                    # Unit tests
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_utils.py
├── integration/             # Integration tests
│   ├── test_data_pipeline.py
│   └── test_model_training.py
├── api/                     # API tests
│   ├── test_endpoints.py
│   └── test_validation.py
└── fixtures/                # Test data
    └── sample_data.csv
```

## Manual Testing

### 1. Setup and Training

```bash
# Setup the system
poetry run python main.py --mode setup

# Or run in development mode (includes training)
poetry run devrun
```

### 2. API Testing

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Single Prediction
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

#### Batch Prediction
```bash
curl -X POST "http://localhost:8000/predict-batch" \
  -H "Content-Type: application/json" \
  -d '[
    {
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
    },
    {
      "battery_power": 3000,
      "blue": 1,
      "clock_speed": 2.5,
      "dual_sim": 1,
      "fc": 16,
      "four_g": 1,
      "int_memory": 128,
      "m_deep": 0.7,
      "mobile_wt": 180,
      "n_cores": 8,
      "pc": 24,
      "px_height": 2160,
      "px_width": 1440,
      "ram": 8192,
      "sc_h": 16,
      "sc_w": 9,
      "talk_time": 25,
      "three_g": 1,
      "touch_screen": 1,
      "wifi": 1
    }
  ]'
```

#### Get Metrics
```bash
curl http://localhost:8000/metrics
```

#### Get Statistics
```bash
curl http://localhost:8000/stats
```

### 3. Model Performance Testing

#### Test Different Price Ranges

**Low Cost Phone:**
```json
{
  "battery_power": 1500,
  "blue": 0,
  "clock_speed": 1.0,
  "dual_sim": 0,
  "fc": 2,
  "four_g": 0,
  "int_memory": 8,
  "m_deep": 1.0,
  "mobile_wt": 200,
  "n_cores": 2,
  "pc": 5,
  "px_height": 480,
  "px_width": 320,
  "ram": 512,
  "sc_h": 10,
  "sc_w": 6,
  "talk_time": 10,
  "three_g": 1,
  "touch_screen": 1,
  "wifi": 0
}
```

**Very High Cost Phone:**
```json
{
  "battery_power": 4500,
  "blue": 1,
  "clock_speed": 3.0,
  "dual_sim": 1,
  "fc": 32,
  "four_g": 1,
  "int_memory": 256,
  "m_deep": 0.6,
  "mobile_wt": 120,
  "n_cores": 8,
  "pc": 64,
  "px_height": 2160,
  "px_width": 1440,
  "ram": 8192,
  "sc_h": 18,
  "sc_w": 10,
  "talk_time": 30,
  "three_g": 1,
  "touch_screen": 1,
  "wifi": 1
}
```

## Performance Testing

### Load Testing

```bash
# Install Apache Bench
# Ubuntu/Debian: sudo apt-get install apache2-utils
# macOS: brew install httpd

# Test single endpoint
ab -n 100 -c 10 -H "Content-Type: application/json" \
  -p test_data.json http://localhost:8000/predict

# Test batch endpoint
ab -n 50 -c 5 -H "Content-Type: application/json" \
  -p batch_test_data.json http://localhost:8000/predict-batch
```

### Memory Usage Testing

```bash
# Monitor memory usage
python -m memory_profiler main.py

# Or use htop/top to monitor system resources
```

## Docker Testing

### Build and Run

```bash
# Build image
docker build -t mobile-price-tracker .

# Run container
docker run -p 8000:8000 mobile-price-tracker

# Run with docker-compose
docker-compose up -d
```

### Test in Container

```bash
# Test health endpoint
docker exec -it mobile-price-tracker curl http://localhost:8000/health

# Test prediction
docker exec -it mobile-price-tracker curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d @test_data.json
```

## Monitoring and Observability

### Prometheus Metrics

Access Prometheus at: http://localhost:9090

Key metrics to monitor:
- `mobile_price_predictions_total`
- `mobile_price_prediction_duration_seconds`
- `mobile_price_prediction_confidence`

### Grafana Dashboard

Access Grafana at: http://localhost:3000 (admin/admin)

### Logs

```bash
# View logs
tail -f logs/mobile_price_tracker.log

# Or in Docker
docker logs -f mobile-price-tracker
```

## Troubleshooting

### Common Issues

1. **Models not found error**
   - Run: `poetry run python main.py --mode train`
   - Check if models exist in `data/models/`

2. **Port already in use**
   - Change port in config or kill existing process
   - Use: `lsof -i :8000` to find process using port

3. **Memory issues**
   - Reduce batch size
   - Use smaller models
   - Increase system memory

4. **Slow predictions**
   - Check system resources
   - Optimize feature engineering
   - Use faster models

### Debug Mode

```bash
# Run with debug logging
LOG_LEVEL=DEBUG poetry run python main.py --mode serve

# Or set in environment
export LOG_LEVEL=DEBUG
poetry run python main.py --mode serve
```

## Test Data

Sample test data files are provided in `tests/fixtures/`:
- `sample_data.csv` - Sample mobile phone data
- `test_predictions.json` - Test prediction requests
- `batch_test_data.json` - Batch test data

## Continuous Integration

The project includes GitHub Actions workflows for:
- Code quality checks (black, isort, flake8, mypy)
- Security scanning (safety, bandit)
- Unit and integration tests
- Docker build testing

Run locally:
```bash
# Pre-commit hooks
poetry run pre-commit install
poetry run pre-commit run --all-files

# Security checks
poetry run safety check
poetry run bandit -r src/
```

