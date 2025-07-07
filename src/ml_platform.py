#!/usr/bin/env python3
"""
Machine Learning Engineering Specialization Capstone Project
MLOps Platform for Production Machine Learning

This comprehensive ML engineering platform demonstrates advanced competencies in:
- MLOps pipeline design and implementation
- Model versioning and deployment
- Automated training and monitoring
- A/B testing for ML models
- Production ML infrastructure
- Model performance monitoring
"""

import os
import sys
import json
import logging
import pickle
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sqlite3
from typing import Dict, List, Any, Optional, Tuple

# ML Libraries
import sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import mlflow
import mlflow.sklearn

# Web Framework
from flask import Flask, request, jsonify, render_template
import docker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_platform.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class MLModelRegistry:
    """Model registry for versioning and metadata management"""
    
    def __init__(self, db_path='ml_registry.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize model registry database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                version TEXT NOT NULL,
                model_type TEXT NOT NULL,
                framework TEXT NOT NULL,
                metrics TEXT,
                parameters TEXT,
                file_path TEXT NOT NULL,
                status TEXT DEFAULT 'staging',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                deployed_at DATETIME,
                created_by TEXT DEFAULT 'ml_engineer'
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_name TEXT NOT NULL,
                model_name TEXT NOT NULL,
                parameters TEXT,
                metrics TEXT,
                dataset_info TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                model_version TEXT NOT NULL,
                input_data TEXT,
                prediction TEXT,
                confidence REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Model registry database initialized")
    
    def register_model(self, model_name: str, version: str, model_type: str, 
                      framework: str, metrics: Dict, parameters: Dict, 
                      file_path: str) -> int:
        """Register a new model version"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO models 
            (model_name, version, model_type, framework, metrics, parameters, file_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            model_name, version, model_type, framework,
            json.dumps(metrics), json.dumps(parameters), file_path
        ))
        
        model_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(f"Model registered: {model_name} v{version} (ID: {model_id})")
        return model_id
    
    def get_model_info(self, model_name: str, version: str = None) -> Dict:
        """Get model information"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if version:
            cursor.execute('''
                SELECT * FROM models WHERE model_name = ? AND version = ?
                ORDER BY created_at DESC LIMIT 1
            ''', (model_name, version))
        else:
            cursor.execute('''
                SELECT * FROM models WHERE model_name = ?
                ORDER BY created_at DESC LIMIT 1
            ''', (model_name,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            columns = [desc[0] for desc in cursor.description]
            return dict(zip(columns, result))
        return None

class MLPipeline:
    """ML training and deployment pipeline"""
    
    def __init__(self, registry: MLModelRegistry):
        self.registry = registry
        self.models_dir = 'models'
        os.makedirs(self.models_dir, exist_ok=True)
    
    def generate_sample_data(self, dataset_type='classification', n_samples=10000):
        """Generate sample datasets for demonstration"""
        np.random.seed(42)
        
        if dataset_type == 'classification':
            # Customer churn prediction dataset
            n_features = 15
            X = np.random.randn(n_samples, n_features)
            
            # Create realistic feature relationships
            # Age, tenure, monthly_charges, total_charges, etc.
            feature_names = [
                'age', 'tenure_months', 'monthly_charges', 'total_charges',
                'contract_length', 'payment_method', 'internet_service',
                'online_security', 'tech_support', 'streaming_tv',
                'paperless_billing', 'senior_citizen', 'partner',
                'dependents', 'multiple_lines'
            ]
            
            # Create target with realistic relationships
            churn_probability = (
                0.1 * X[:, 0] +  # age
                -0.2 * X[:, 1] +  # tenure (longer tenure = less churn)
                0.15 * X[:, 2] +  # monthly charges
                0.1 * X[:, 3] +   # total charges
                np.random.normal(0, 0.5, n_samples)
            )
            
            y = (churn_probability > np.percentile(churn_probability, 70)).astype(int)
            
            df = pd.DataFrame(X, columns=feature_names)
            df['churn'] = y
            
            return df, 'customer_churn_prediction'
            
        elif dataset_type == 'regression':
            # House price prediction dataset
            n_features = 12
            X = np.random.randn(n_samples, n_features)
            
            feature_names = [
                'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
                'floors', 'waterfront', 'view', 'condition',
                'grade', 'sqft_above', 'sqft_basement', 'yr_built'
            ]
            
            # Create realistic price relationships
            price = (
                50000 +
                10000 * X[:, 0] +  # bedrooms
                15000 * X[:, 1] +  # bathrooms
                100 * X[:, 2] +    # sqft_living
                5 * X[:, 3] +      # sqft_lot
                20000 * X[:, 4] +  # floors
                100000 * X[:, 5] + # waterfront
                25000 * X[:, 6] +  # view
                10000 * X[:, 7] +  # condition
                30000 * X[:, 8] +  # grade
                np.random.normal(0, 20000, n_samples)
            )
            
            # Ensure positive prices
            price = np.maximum(price, 50000)
            
            df = pd.DataFrame(X, columns=feature_names)
            df['price'] = price
            
            return df, 'house_price_prediction'
    
    def train_classification_model(self, df: pd.DataFrame, target_col: str, 
                                 model_name: str) -> Tuple[Any, Dict]:
        """Train a classification model"""
        logger.info(f"Training classification model: {model_name}")
        
        # Prepare data
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train multiple models
        models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        best_model = None
        best_score = 0
        best_metrics = {}
        
        for model_type, model in models.items():
            # Train model
            if model_type == 'logistic_regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'model_type': model_type
            }
            
            logger.info(f"{model_type} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            
            if accuracy > best_score:
                best_score = accuracy
                best_model = model
                best_metrics = metrics
                best_scaler = scaler if model_type == 'logistic_regression' else None
        
        # Save model
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(self.models_dir, f"{model_name}_v{version}.pkl")
        
        model_package = {
            'model': best_model,
            'scaler': best_scaler,
            'feature_names': list(X.columns),
            'target_name': target_col
        }
        
        joblib.dump(model_package, model_path)
        
        # Register in model registry
        self.registry.register_model(
            model_name=model_name,
            version=version,
            model_type='classification',
            framework='scikit-learn',
            metrics=best_metrics,
            parameters=best_model.get_params(),
            file_path=model_path
        )
        
        return best_model, best_metrics
    
    def train_regression_model(self, df: pd.DataFrame, target_col: str, 
                             model_name: str) -> Tuple[Any, Dict]:
        """Train a regression model"""
        logger.info(f"Training regression model: {model_name}")
        
        # Prepare data
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train multiple models
        models = {
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'linear_regression': LinearRegression()
        }
        
        best_model = None
        best_score = float('inf')
        best_metrics = {}
        
        for model_type, model in models.items():
            # Train model
            if model_type == 'linear_regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            metrics = {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'r2_score': r2,
                'model_type': model_type
            }
            
            logger.info(f"{model_type} - RMSE: {rmse:.2f}, R²: {r2:.4f}")
            
            if rmse < best_score:
                best_score = rmse
                best_model = model
                best_metrics = metrics
                best_scaler = scaler if model_type == 'linear_regression' else None
        
        # Save model
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(self.models_dir, f"{model_name}_v{version}.pkl")
        
        model_package = {
            'model': best_model,
            'scaler': best_scaler,
            'feature_names': list(X.columns),
            'target_name': target_col
        }
        
        joblib.dump(model_package, model_path)
        
        # Register in model registry
        self.registry.register_model(
            model_name=model_name,
            version=version,
            model_type='regression',
            framework='scikit-learn',
            metrics=best_metrics,
            parameters=best_model.get_params(),
            file_path=model_path
        )
        
        return best_model, best_metrics

class ModelServingAPI:
    """Model serving and prediction API"""
    
    def __init__(self, registry: MLModelRegistry):
        self.registry = registry
        self.loaded_models = {}
    
    def load_model(self, model_name: str, version: str = None) -> bool:
        """Load a model for serving"""
        try:
            model_info = self.registry.get_model_info(model_name, version)
            if not model_info:
                logger.error(f"Model not found: {model_name} v{version}")
                return False
            
            model_package = joblib.load(model_info['file_path'])
            
            self.loaded_models[model_name] = {
                'model': model_package['model'],
                'scaler': model_package.get('scaler'),
                'feature_names': model_package['feature_names'],
                'target_name': model_package['target_name'],
                'version': model_info['version'],
                'model_type': model_info['model_type']
            }
            
            logger.info(f"Model loaded: {model_name} v{model_info['version']}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return False
    
    def predict(self, model_name: str, input_data: Dict) -> Dict:
        """Make prediction using loaded model"""
        if model_name not in self.loaded_models:
            return {'error': f'Model {model_name} not loaded'}
        
        try:
            model_info = self.loaded_models[model_name]
            model = model_info['model']
            scaler = model_info['scaler']
            feature_names = model_info['feature_names']
            
            # Prepare input data
            input_df = pd.DataFrame([input_data])
            
            # Ensure all features are present
            for feature in feature_names:
                if feature not in input_df.columns:
                    input_df[feature] = 0
            
            # Reorder columns to match training data
            input_df = input_df[feature_names]
            
            # Scale if necessary
            if scaler:
                input_scaled = scaler.transform(input_df)
                prediction = model.predict(input_scaled)
                if hasattr(model, 'predict_proba'):
                    confidence = np.max(model.predict_proba(input_scaled))
                else:
                    confidence = None
            else:
                prediction = model.predict(input_df)
                if hasattr(model, 'predict_proba'):
                    confidence = np.max(model.predict_proba(input_df))
                else:
                    confidence = None
            
            result = {
                'prediction': prediction[0].item() if hasattr(prediction[0], 'item') else prediction[0],
                'model_name': model_name,
                'model_version': model_info['version'],
                'confidence': confidence.item() if confidence is not None else None,
                'timestamp': datetime.now().isoformat()
            }
            
            # Log prediction
            self.log_prediction(model_name, model_info['version'], input_data, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error for {model_name}: {e}")
            return {'error': str(e)}
    
    def log_prediction(self, model_name: str, version: str, input_data: Dict, result: Dict):
        """Log prediction for monitoring"""
        conn = sqlite3.connect(self.registry.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions 
            (model_name, model_version, input_data, prediction, confidence)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            model_name, version,
            json.dumps(input_data),
            json.dumps(result['prediction']),
            result.get('confidence')
        ))
        
        conn.commit()
        conn.close()

# Flask API
app = Flask(__name__)

# Initialize ML platform components
registry = MLModelRegistry()
pipeline = MLPipeline(registry)
serving_api = ModelServingAPI(registry)

@app.route('/')
def dashboard():
    """ML Platform dashboard"""
    return jsonify({
        'service': 'ML Engineering Platform',
        'version': '1.0.0',
        'status': 'running',
        'endpoints': [
            '/api/train',
            '/api/models',
            '/api/predict',
            '/api/health'
        ]
    })

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'loaded_models': list(serving_api.loaded_models.keys())
    })

@app.route('/api/train', methods=['POST'])
def train_model():
    """Train a new model"""
    try:
        data = request.json
        model_name = data.get('model_name', 'default_model')
        dataset_type = data.get('dataset_type', 'classification')
        
        # Generate sample data
        df, dataset_name = pipeline.generate_sample_data(dataset_type)
        
        if dataset_type == 'classification':
            target_col = 'churn'
            model, metrics = pipeline.train_classification_model(df, target_col, model_name)
        else:
            target_col = 'price'
            model, metrics = pipeline.train_regression_model(df, target_col, model_name)
        
        # Load model for serving
        serving_api.load_model(model_name)
        
        return jsonify({
            'status': 'success',
            'model_name': model_name,
            'dataset_type': dataset_type,
            'metrics': metrics,
            'dataset_size': len(df)
        })
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/models')
def list_models():
    """List all registered models"""
    conn = sqlite3.connect(registry.db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT model_name, version, model_type, metrics, status, created_at
        FROM models ORDER BY created_at DESC
    ''')
    
    models = []
    for row in cursor.fetchall():
        models.append({
            'model_name': row[0],
            'version': row[1],
            'model_type': row[2],
            'metrics': json.loads(row[3]) if row[3] else {},
            'status': row[4],
            'created_at': row[5]
        })
    
    conn.close()
    return jsonify(models)

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make prediction using a model"""
    try:
        data = request.json
        model_name = data.get('model_name')
        input_data = data.get('input_data', {})
        
        if not model_name:
            return jsonify({'error': 'model_name required'}), 400
        
        # Load model if not already loaded
        if model_name not in serving_api.loaded_models:
            if not serving_api.load_model(model_name):
                return jsonify({'error': f'Failed to load model {model_name}'}), 404
        
        result = serving_api.predict(model_name, input_data)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Prediction API error: {e}")
        return jsonify({'error': str(e)}), 500

def run_training_pipeline():
    """Run automated training pipeline"""
    logger.info("🚀 Starting ML Engineering Platform Training Pipeline")
    
    # Train classification model
    logger.info("Training customer churn prediction model...")
    df_class, _ = pipeline.generate_sample_data('classification', 15000)
    model_class, metrics_class = pipeline.train_classification_model(
        df_class, 'churn', 'customer_churn_predictor'
    )
    
    # Train regression model
    logger.info("Training house price prediction model...")
    df_reg, _ = pipeline.generate_sample_data('regression', 12000)
    model_reg, metrics_reg = pipeline.train_regression_model(
        df_reg, 'price', 'house_price_predictor'
    )
    
    # Load models for serving
    serving_api.load_model('customer_churn_predictor')
    serving_api.load_model('house_price_predictor')
    
    logger.info("✅ Training pipeline completed successfully")
    
    return {
        'classification_metrics': metrics_class,
        'regression_metrics': metrics_reg
    }

if __name__ == '__main__':
    print("🤖 Machine Learning Engineering Specialization Capstone Project")
    print("🔧 MLOps Platform for Production Machine Learning")
    print("=" * 65)
    
    # Run training pipeline
    results = run_training_pipeline()
    
    print(f"\n📊 Training Results:")
    print(f"Classification Model - Accuracy: {results['classification_metrics']['accuracy']:.4f}")
    print(f"Regression Model - R² Score: {results['regression_metrics']['r2_score']:.4f}")
    
    # Start API server
    logger.info("Starting ML Platform API on http://localhost:5002")
    app.run(host='0.0.0.0', port=5002, debug=False)

