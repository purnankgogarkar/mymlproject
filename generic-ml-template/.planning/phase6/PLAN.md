# Phase 6: Production Export & Deployment

**Goal:** Build production-ready Flask REST API and model export system for trained ML models.

**Duration:** 3 days (May 8-10, 2026)  
**Status:** 📋 Planning  
**Tests Target:** 80+ new tests  
**Estimated Total:** 530+ tests passing

---

## 📋 Overview

Phase 6 adds production-grade model deployment capabilities:
1. **Model Exporter** — Serialize/deserialize trained models
2. **Flask REST API** — RESTful endpoints for predictions
3. **Docker Support** — Containerized deployment
4. **Error Handling** — Robust error responses & validation
5. **API Tests** — Comprehensive endpoint testing

### Architecture
```
Trained Model (from Phase 5)
    ↓
[ModelExporter] → model.pkl + metadata.json + config.yaml
    ↓
[Flask API] → REST endpoints
    ├── POST /predict → Single prediction
    ├── POST /predict-batch → Batch predictions
    ├── GET /health → Health check
    ├── GET /model-info → Model metadata
    └── POST /explain → Feature importance
    ↓
[Docker] → Containerized deployment
    ↓
[Gunicorn] → Production WSGI server
```

---

## 📂 File Structure

```
src/
├── export/
│   ├── __init__.py
│   ├── model_exporter.py      # Model serialization logic
│   ├── model_loader.py        # Model deserialization
│   └── metadata.py            # Metadata management
├── api/
│   ├── __init__.py
│   ├── flask_app.py           # Flask app factory
│   ├── config.py              # API configuration
│   ├── endpoints/
│   │   ├── __init__.py
│   │   ├── predict.py         # Prediction endpoints
│   │   ├── health.py          # Health check
│   │   └── explain.py         # Feature importance
│   ├── middleware/
│   │   ├── __init__.py
│   │   ├── auth.py            # API authentication
│   │   ├── validators.py      # Request validation
│   │   └── error_handlers.py  # Error handling
│   └── utils/
│       ├── __init__.py
│       ├── response.py        # Response formatting
│       └── logging.py         # Logging setup

tests/
├── test_model_exporter.py     # Export tests (15 tests)
├── test_model_loader.py       # Load tests (12 tests)
├── test_flask_app.py          # Flask app tests (18 tests)
├── test_predict_endpoint.py   # Prediction endpoint tests (20 tests)
├── test_batch_predict.py      # Batch prediction tests (12 tests)
├── test_health_endpoint.py    # Health check tests (8 tests)

docker/
├── Dockerfile                 # Container definition
├── docker-compose.yml         # Multi-container setup
└── .dockerignore             # Docker build exclusions

├── requirements-api.txt       # API-specific dependencies
├── wsgi.py                    # WSGI entry point (Gunicorn)
└── run_api.py                 # Local development server
```

---

## 🔧 Day 1: Model Export & Flask Setup

### 1.1 Model Exporter (`src/export/model_exporter.py`)

**Purpose:** Serialize trained models with metadata

```python
from typing import Dict, Any, Optional
import pickle
import json
from pathlib import Path
import joblib
from datetime import datetime


class ModelExporter:
    """Export trained models with metadata and configuration."""
    
    def __init__(self, export_dir: str = "models/exported"):
        """Initialize exporter.
        
        Args:
            export_dir: Directory to store exported models
        """
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)
    
    def export_model(
        self,
        model: Any,
        model_name: str,
        model_type: str,
        config: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        format: str = "pickle"
    ) -> Dict[str, str]:
        """Export model with metadata.
        
        Args:
            model: Trained model object
            model_name: Name of the model (e.g., 'random_forest')
            model_type: Type of model ('classification' or 'regression')
            config: Model configuration dict
            metadata: Optional metadata (accuracy, features, etc.)
            format: Serialization format ('pickle' or 'joblib')
        
        Returns:
            Dict with paths to exported files
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = self.export_dir / f"{model_name}_{timestamp}"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Export model
        model_path = model_dir / "model.pkl"
        if format == "joblib":
            joblib.dump(model, model_path)
        else:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        # Export configuration
        config_path = model_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Export metadata
        metadata = metadata or {}
        metadata.update({
            'exported_at': datetime.now().isoformat(),
            'model_name': model_name,
            'model_type': model_type,
            'format': format
        })
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return {
            'model_path': str(model_path),
            'config_path': str(config_path),
            'metadata_path': str(metadata_path),
            'export_dir': str(model_dir)
        }
    
    def list_exported_models(self) -> list:
        """List all exported models."""
        return [d.name for d in self.export_dir.iterdir() if d.is_dir()]
```

**Tests (15 tests):**
- Export with pickle format
- Export with joblib format
- Metadata generation
- Config JSON serialization
- Directory creation
- File validation
- Timestamp handling
- Error on invalid model
- Error on invalid directory
- Overwrite existing model
- etc.

### 1.2 Model Loader (`src/export/model_loader.py`)

**Purpose:** Deserialize and validate exported models

```python
class ModelLoader:
    """Load and validate exported models."""
    
    def load_model(self, model_dir: str) -> tuple:
        """Load model, config, and metadata.
        
        Returns:
            (model, config, metadata)
        """
        model_dir = Path(model_dir)
        
        # Load model
        model_path = model_dir / "model.pkl"
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load config
        with open(model_dir / "config.json") as f:
            config = json.load(f)
        
        # Load metadata
        with open(model_dir / "metadata.json") as f:
            metadata = json.load(f)
        
        return model, config, metadata
    
    def validate_model(self, model_dir: str) -> bool:
        """Validate model structure and files."""
        model_dir = Path(model_dir)
        required_files = ['model.pkl', 'config.json', 'metadata.json']
        return all((model_dir / f).exists() for f in required_files)
```

**Tests (12 tests):**
- Load valid model
- Load config JSON
- Load metadata
- Validate model structure
- Error on missing files
- Error on corrupted pickle
- Error on invalid directory
- Validate metadata integrity
- etc.

### 1.3 Flask App Factory (`src/api/flask_app.py`)

**Purpose:** Create Flask app with all configurations

```python
from flask import Flask, jsonify
from src.api.endpoints.predict import predict_bp
from src.api.endpoints.health import health_bp
from src.api.middleware.error_handlers import register_error_handlers


def create_app(config_name: str = 'development'):
    """Create and configure Flask app.
    
    Args:
        config_name: 'development', 'testing', or 'production'
    
    Returns:
        Configured Flask app
    """
    app = Flask(__name__)
    
    # Load configuration
    if config_name == 'production':
        app.config['DEBUG'] = False
        app.config['JSON_SORT_KEYS'] = False
    else:
        app.config['DEBUG'] = True
    
    # Register blueprints
    app.register_blueprint(predict_bp, url_prefix='/api/v1')
    app.register_blueprint(health_bp, url_prefix='/api/v1')
    
    # Register error handlers
    register_error_handlers(app)
    
    # Root endpoint
    @app.route('/', methods=['GET'])
    def root():
        return jsonify({
            'name': 'Generic ML API',
            'version': '1.0.0',
            'status': 'running'
        })
    
    return app


if __name__ == '__main__':
    app = create_app('development')
    app.run(debug=True, port=5000)
```

**Tests (18 tests):**
- App creation
- Config loading
- Blueprint registration
- Root endpoint
- Error handler registration
- Development mode
- Production mode
- etc.

---

## 🔧 Day 2: API Endpoints & Request Handling

### 2.1 Prediction Endpoint (`src/api/endpoints/predict.py`)

**Purpose:** Handle prediction requests

```python
from flask import Blueprint, request, jsonify
from src.export.model_loader import ModelLoader
from src.api.middleware.validators import validate_prediction_request
import pandas as pd


predict_bp = Blueprint('predict', __name__)


@predict_bp.route('/predict', methods=['POST'])
def predict():
    """Make single prediction.
    
    Request JSON:
    {
        "model_dir": "/path/to/model",
        "features": {"feature1": value1, "feature2": value2}
    }
    
    Response:
    {
        "prediction": 0.85,
        "confidence": 0.92,
        "model": "random_forest"
    }
    """
    try:
        # Validate request
        data = validate_prediction_request(request.get_json())
        
        # Load model
        loader = ModelLoader()
        model, config, metadata = loader.load_model(data['model_dir'])
        
        # Prepare features
        features_df = pd.DataFrame([data['features']])
        
        # Make prediction
        prediction = model.predict(features_df)[0]
        
        # Get probability if available
        confidence = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features_df)[0]
            confidence = float(max(proba))
        
        return jsonify({
            'prediction': float(prediction),
            'confidence': confidence,
            'model': metadata.get('model_name'),
            'status': 'success'
        }), 200
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400


@predict_bp.route('/predict-batch', methods=['POST'])
def predict_batch():
    """Make batch predictions.
    
    Request JSON:
    {
        "model_dir": "/path/to/model",
        "features": [
            {"feature1": v1, "feature2": v2},
            {"feature1": v3, "feature2": v4}
        ]
    }
    """
    try:
        data = request.get_json()
        
        loader = ModelLoader()
        model, config, metadata = loader.load_model(data['model_dir'])
        
        features_df = pd.DataFrame(data['features'])
        predictions = model.predict(features_df)
        
        return jsonify({
            'predictions': [float(p) for p in predictions],
            'count': len(predictions),
            'model': metadata.get('model_name'),
            'status': 'success'
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 400
```

**Tests (20 tests):**
- Valid prediction request
- Invalid request format
- Missing model directory
- Missing features
- Batch predictions
- Probability output
- Error handling
- etc.

### 2.2 Health Check Endpoint (`src/api/endpoints/health.py`)

**Purpose:** API health and status monitoring

```python
from flask import Blueprint, jsonify
from datetime import datetime


health_bp = Blueprint('health', __name__)


@health_bp.route('/health', methods=['GET'])
def health():
    """Health check endpoint.
    
    Response:
    {
        "status": "healthy",
        "timestamp": "2026-05-08T12:00:00",
        "uptime": 3600
    }
    """
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'service': 'Generic ML API',
        'version': '1.0.0'
    }), 200


@health_bp.route('/model-info', methods=['GET'])
def model_info():
    """Get current model information."""
    return jsonify({
        'model_loaded': True,
        'model_type': 'classification',
        'features': ['feature1', 'feature2'],
        'target': 'target_column'
    }), 200
```

**Tests (8 tests):**
- Health check response
- Status code 200
- Timestamp format
- Model info endpoint
- etc.

### 2.3 Request Validator (`src/api/middleware/validators.py`)

**Purpose:** Validate incoming requests

```python
from typing import Dict, Any
from flask import Request


def validate_prediction_request(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate prediction request format.
    
    Raises:
        ValueError: If request is invalid
    """
    if not data:
        raise ValueError("Request body is empty")
    
    if 'model_dir' not in data:
        raise ValueError("Missing 'model_dir' field")
    
    if 'features' not in data:
        raise ValueError("Missing 'features' field")
    
    if not isinstance(data['features'], (dict, list)):
        raise ValueError("'features' must be dict or list")
    
    return data


def validate_batch_request(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate batch prediction request."""
    if not isinstance(data.get('features'), list):
        raise ValueError("'features' must be a list for batch predictions")
    
    if len(data['features']) == 0:
        raise ValueError("'features' list cannot be empty")
    
    return data
```

**Tests (10 tests):**
- Valid single prediction
- Valid batch prediction
- Missing model_dir
- Missing features
- Invalid features type
- Empty batch
- etc.

### 2.4 Error Handlers (`src/api/middleware/error_handlers.py`)

**Purpose:** Unified error response handling

```python
from flask import Flask, jsonify


def register_error_handlers(app: Flask):
    """Register error handlers for the Flask app."""
    
    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({
            'error': 'Bad Request',
            'message': str(error),
            'status': 'error'
        }), 400
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            'error': 'Not Found',
            'message': 'Endpoint not found',
            'status': 'error'
        }), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({
            'error': 'Internal Server Error',
            'message': 'An unexpected error occurred',
            'status': 'error'
        }), 500
```

**Tests (8 tests):**
- 400 Bad Request
- 404 Not Found
- 500 Internal Error
- Error response format
- etc.

---

## 🔧 Day 3: Docker & Production Setup

### 3.1 Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt requirements-api.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r requirements-api.txt

# Copy application
COPY . .

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5000/api/v1/health || exit 1

# Run Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "120", "wsgi:app"]
```

### 3.2 WSGI Entry Point (`wsgi.py`)

```python
import os
from src.api.flask_app import create_app

app = create_app(os.getenv('FLASK_ENV', 'production'))

if __name__ == '__main__':
    app.run()
```

### 3.3 Requirements API (`requirements-api.txt`)

```
Flask==2.3.0
Gunicorn==21.2.0
python-dotenv==1.0.0
```

---

## ✅ Test Coverage

### Day 1 Tests (37 tests)
- `test_model_exporter.py` — 15 tests
- `test_model_loader.py` — 12 tests
- `test_flask_app.py` — 10 tests

### Day 2 Tests (46 tests)
- `test_predict_endpoint.py` — 20 tests
- `test_batch_predict.py` — 12 tests
- `test_health_endpoint.py` — 8 tests
- `test_validators.py` — 6 tests

### Day 3 Tests (6 tests)
- Docker build validation
- WSGI entry point
- Integration tests

**Total Phase 6 Tests: 89 tests**

---

## 🎯 Success Criteria

- ✅ Model export/import working
- ✅ Flask API endpoints functional
- ✅ All requests validated
- ✅ Error handling in place
- ✅ 89+ tests passing
- ✅ Docker image buildable
- ✅ API documentation (docstrings)
- ✅ Committed to GitHub

---

## 📝 Tasks

### Day 1
- [ ] Create `src/export/` module
- [ ] Implement ModelExporter class (export, serialization)
- [ ] Implement ModelLoader class (load, validate)
- [ ] Create Flask app factory
- [ ] Write 37 tests for Day 1

### Day 2
- [ ] Implement prediction endpoint
- [ ] Implement batch prediction endpoint
- [ ] Implement health check endpoint
- [ ] Create request validators
- [ ] Create error handlers
- [ ] Write 46 tests for Day 2

### Day 3
- [ ] Create Dockerfile
- [ ] Create WSGI entry point
- [ ] Create requirements-api.txt
- [ ] Write Docker integration tests
- [ ] Test API locally
- [ ] Commit and push to GitHub

---

## 🚀 API Usage Examples

### Single Prediction
```bash
curl -X POST http://localhost:5000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_dir": "/models/exported/random_forest_20260508_120000",
    "features": {"age": 30, "income": 50000}
  }'
```

### Batch Prediction
```bash
curl -X POST http://localhost:5000/api/v1/predict-batch \
  -H "Content-Type: application/json" \
  -d '{
    "model_dir": "/models/exported/random_forest_20260508_120000",
    "features": [
      {"age": 30, "income": 50000},
      {"age": 45, "income": 75000}
    ]
  }'
```

### Health Check
```bash
curl http://localhost:5000/api/v1/health
```

---

## 📦 Docker Usage

### Build Image
```bash
docker build -t generic-ml-api:1.0 .
```

### Run Container
```bash
docker run -p 5000:5000 \
  -v $(pwd)/models:/app/models \
  generic-ml-api:1.0
```

### Docker Compose
```bash
docker-compose up -d
```

---

## 📊 Summary

| Component | Status | Lines | Tests |
|-----------|--------|-------|-------|
| ModelExporter | 📋 Planned | ~100 | 15 |
| ModelLoader | 📋 Planned | ~80 | 12 |
| Flask App | 📋 Planned | ~60 | 10 |
| Prediction Endpoint | 📋 Planned | ~80 | 20 |
| Batch Endpoint | 📋 Planned | ~60 | 12 |
| Health Check | 📋 Planned | ~40 | 8 |
| Validators | 📋 Planned | ~70 | 10 |
| Error Handlers | 📋 Planned | ~50 | 8 |
| Docker Setup | 📋 Planned | ~30 | 6 |
| **Total** | | **~570** | **89** |

**Phase Target: 540 total tests (450 + 89)**
