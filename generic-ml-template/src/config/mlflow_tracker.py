"""
MLflow Experiment Tracking Integration

Track ML experiments with metrics, parameters, models, and artifacts.
Supports local and remote tracking servers.
"""

import os
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional
import warnings

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    warnings.warn("MLflow not installed. Install with: pip install mlflow")


class MLflowTracker:
    """Track ML experiments with MLflow."""
    
    def __init__(self, experiment_name: str, tracking_uri: Optional[str] = None, 
                 tags: Optional[Dict[str, str]] = None):
        """
        Initialize MLflow tracker.
        
        Args:
            experiment_name: Name of experiment
            tracking_uri: MLflow tracking server URI (local or remote)
            tags: Dictionary of tags to apply to all runs
            
        Raises:
            RuntimeError: If MLflow not installed
        """
        if not MLFLOW_AVAILABLE:
            raise RuntimeError("MLflow not installed. Install with: pip install mlflow")
        
        self.experiment_name = experiment_name
        self.tracking_uri = self._normalize_uri(tracking_uri or "./mlruns")
        self.tags = tags or {}
        self.run_id = None
        
        # Set tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Create or get experiment
        try:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        except Exception:
            # Experiment already exists
            self.experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    
    def start_run(self, run_name: Optional[str] = None, 
                 tags: Optional[Dict[str, str]] = None) -> 'MLflowTracker':
        """
        Start a new MLflow run.
        
        Args:
            run_name: Name for this run
            tags: Additional tags for this run
            
        Returns:
            self for method chaining
        """
        # Combine default tags with run-specific tags
        all_tags = self.tags.copy()
        if tags:
            all_tags.update(tags)
        
        # Start run
        mlflow.start_run(experiment_id=self.experiment_id, run_name=run_name)
        self.run_id = mlflow.active_run().info.run_id
        
        # Log tags
        for key, value in all_tags.items():
            mlflow.set_tag(key, value)
        
        return self
    
    def log_params(self, params: Dict[str, Any]) -> 'MLflowTracker':
        """
        Log hyperparameters.
        
        Args:
            params: Dictionary of parameters
            
        Returns:
            self for method chaining
        """
        # Flatten nested dicts
        flat_params = self._flatten_dict(params)
        mlflow.log_params(flat_params)
        return self
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> 'MLflowTracker':
        """
        Log metrics.
        
        Args:
            metrics: Dictionary of metrics
            step: Step/epoch number (optional)
            
        Returns:
            self for method chaining
        """
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value, step=step)
        
        return self
    
    def log_config(self, config: Dict[str, Any], artifact_name: str = "config.json") -> 'MLflowTracker':
        """
        Log configuration as artifact.
        
        Args:
            config: Configuration dictionary
            artifact_name: Name for artifact
            
        Returns:
            self for method chaining
        """
        # Save to temp file
        temp_path = f".mlflow_temp_{self.run_id}.json"
        try:
            with open(temp_path, 'w') as f:
                json.dump(config, f, indent=2)
            mlflow.log_artifact(temp_path, artifact_path="configs")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        return self
    
    def log_model(self, model: Any, model_name: str = "model") -> 'MLflowTracker':
        """
        Log trained model.
        
        Args:
            model: Trained model object
            model_name: Name for model artifact
            
        Returns:
            self for method chaining
        """
        # Save model to temp file
        temp_path = f".mlflow_temp_model_{self.run_id}.pkl"
        try:
            with open(temp_path, 'wb') as f:
                pickle.dump(model, f)
            mlflow.log_artifact(temp_path, artifact_path="models")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        return self
    
    def log_artifact(self, artifact_path: str, artifact_type: str = None) -> 'MLflowTracker':
        """
        Log any artifact file.
        
        Args:
            artifact_path: Path to artifact file
            artifact_type: Type of artifact (e.g., "plots", "data")
            
        Returns:
            self for method chaining
        """
        if os.path.exists(artifact_path):
            mlflow.log_artifact(artifact_path, artifact_path=artifact_type)
        
        return self
    
    def end_run(self) -> 'MLflowTracker':
        """
        End current MLflow run.
        
        Returns:
            self for method chaining
        """
        mlflow.end_run()
        self.run_id = None
        return self
    
    def get_best_run(self, metric: str, mode: str = "max") -> Optional[Dict[str, Any]]:
        """
        Get best run by metric.
        
        Args:
            metric: Metric name to optimize
            mode: 'max' or 'min'
            
        Returns:
            Dictionary with run info or None if no runs
        """
        experiment = mlflow.get_experiment(self.experiment_id)
        if not experiment:
            return None
        
        # Get all runs
        runs = mlflow.search_runs(experiment_ids=[self.experiment_id])
        
        if runs.empty:
            return None
        
        # Find best run
        if mode == "max":
            best_idx = runs[f'metrics.{metric}'].idxmax()
        else:
            best_idx = runs[f'metrics.{metric}'].idxmin()
        
        best_run = runs.loc[best_idx]
        return best_run.to_dict()
    
    def list_runs(self) -> Dict[str, Any]:
        """
        List all runs in experiment.
        
        Returns:
            DataFrame of runs with metrics and params
        """
        return mlflow.search_runs(experiment_ids=[self.experiment_id])
    
    def get_run_info(self, run_id: str) -> Dict[str, Any]:
        """
        Get information about a specific run.
        
        Args:
            run_id: Run ID
            
        Returns:
            Dictionary with run info
        """
        run = mlflow.get_run(run_id)
        if not run:
            return {}
        
        return {
            'run_id': run.info.run_id,
            'experiment_id': run.info.experiment_id,
            'params': run.data.params,
            'metrics': run.data.metrics,
            'tags': run.data.tags,
        }
    
    @staticmethod
    def _normalize_uri(uri: str) -> str:
        """
        Normalize local file paths to file:// URIs for MLflow compatibility.
        Handles Windows paths (C:\\...) and Unix paths (/...).
        """
        if uri.startswith('file://'):
            return uri
        
        # Check if it's a Windows path (has drive letter like C:\)
        if len(uri) > 1 and uri[1] == ':':
            # Convert Windows path to file:// URI
            normalized = uri.replace('\\', '/')
            return f"file:///{normalized}"
        
        # Check if it's an absolute Unix path
        if uri.startswith('/'):
            return f"file://{uri}"
        
        # Relative path - convert to absolute then to file:// URI
        import os
        abs_path = os.path.abspath(uri).replace('\\', '/')
        if abs_path[1] == ':':
            return f"file:///{abs_path}"
        return f"file://{abs_path}"
    
    @staticmethod
    def _flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """
        Flatten nested dictionary.
        
        Args:
            d: Dictionary to flatten
            parent_key: Parent key prefix
            sep: Separator for nested keys
            
        Returns:
            Flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.extend(MLflowTracker._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, (int, float, str, bool)):
                items.append((new_key, v))
            else:
                items.append((new_key, str(v)))
        
        return dict(items)
