"""
Optuna Hyperparameter Tuner

Automated hyperparameter optimization using Optuna.
Supports different samplers, pruning strategies, and parallel trials.
"""

import warnings
from typing import Dict, Any, Tuple, Optional, Callable
import numpy as np

try:
    import optuna
    from optuna.samplers import TPESampler, RandomSampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    warnings.warn("Optuna not installed. Install with: pip install optuna")

from sklearn.model_selection import cross_val_score
from .model_defaults import get_model_defaults, get_tuning_space


class OptunaTuner:
    """Automated hyperparameter optimization using Optuna."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, 
                 model_class: type, problem_type: str = 'classification',
                 cv_folds: int = 5, random_state: int = 42,
                 n_jobs: int = 1, verbose: int = 0):
        """
        Initialize Optuna tuner.
        
        Args:
            X: Feature matrix
            y: Target vector
            model_class: Model class (e.g., RandomForestClassifier)
            problem_type: 'classification' or 'regression'
            cv_folds: Number of cross-validation folds
            random_state: Random seed
            n_jobs: Number of parallel jobs (-1 for all cores)
            verbose: Verbosity level
            
        Raises:
            RuntimeError: If Optuna not installed
        """
        if not OPTUNA_AVAILABLE:
            raise RuntimeError("Optuna not installed. Install with: pip install optuna")
        
        self.X = X
        self.y = y
        self.model_class = model_class
        self.problem_type = problem_type
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        self.study = None
        self.best_params = None
        self.best_score = None
        self.trials_history = []
        
        # Determine scoring metric
        if problem_type == 'classification':
            self.scoring = 'f1_weighted'
        else:
            self.scoring = 'r2'
    
    def tune(self, n_trials: int = 50, model_name: str = None,
            sampler: str = 'tpe', pruning: bool = True,
            direction: str = 'maximize') -> 'OptunaTuner':
        """
        Run hyperparameter optimization.
        
        Args:
            n_trials: Number of trials
            model_name: Name of model (for predefined search space)
            sampler: 'tpe' (default) or 'random'
            pruning: Whether to use pruning
            direction: 'maximize' or 'minimize'
            
        Returns:
            self for method chaining
        """
        # Create sampler
        if sampler == 'tpe':
            sampler_obj = TPESampler(seed=self.random_state)
        elif sampler == 'random':
            sampler_obj = RandomSampler(seed=self.random_state)
        else:
            raise ValueError(f"Unknown sampler: {sampler}")
        
        # Create pruner
        pruner_obj = MedianPruner() if pruning else None
        
        # Create study
        self.study = optuna.create_study(
            sampler=sampler_obj,
            pruner=pruner_obj,
            direction=direction,
        )
        
        # Define objective function
        if model_name and model_name in get_tuning_space.__doc__:
            # Use predefined search space
            search_space = get_tuning_space(model_name)
            objective = self._make_objective_with_space(search_space)
        else:
            # Use generic objective
            objective = self._objective
        
        # Optimize
        self.study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=self.verbose > 0
        )
        
        # Extract results
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        self.trials_history = self.study.trials
        
        return self
    
    def get_best_params(self) -> Dict[str, Any]:
        """Get best hyperparameters found."""
        if self.best_params is None:
            raise RuntimeError("Must call tune() first")
        
        return self.best_params.copy()
    
    def get_best_score(self) -> float:
        """Get best score found."""
        if self.best_score is None:
            raise RuntimeError("Must call tune() first")
        
        return self.best_score
    
    def get_trials_history(self) -> list:
        """Get history of all trials."""
        if not self.trials_history:
            raise RuntimeError("Must call tune() first")
        return self.trials_history.copy()
    
    def get_trial_results(self) -> Dict[str, list]:
        """
        Get trial results as dictionary of lists.
        
        Returns:
            Dictionary with trial numbers, parameters, scores
        """
        if not self.trials_history:
            raise RuntimeError("Must call tune() first")
        
        results = {
            'trial_number': [],
            'score': [],
            'params': [],
        }
        
        for trial in self.trials_history:
            results['trial_number'].append(trial.number)
            results['score'].append(trial.value)
            results['params'].append(trial.params)
        
        return results
    
    def plot_optimization_history(self, save_path: Optional[str] = None):
        """
        Plot optimization history.
        
        Args:
            save_path: Path to save plot (optional)
            
        Returns:
            Plotly figure
        """
        if self.study is None:
            raise RuntimeError("Must call tune() first")
        
        try:
            fig = optuna.visualization.plot_optimization_history(self.study).to_plotly_figure()
            
            if save_path:
                fig.write_html(save_path)
            
            return fig
        except Exception as e:
            warnings.warn(f"Could not plot optimization history: {e}")
            return None
    
    def plot_param_importance(self, save_path: Optional[str] = None):
        """
        Plot parameter importance.
        
        Args:
            save_path: Path to save plot (optional)
            
        Returns:
            Plotly figure
        """
        if self.study is None:
            raise RuntimeError("Must call tune() first")
        
        try:
            fig = optuna.visualization.plot_param_importances(self.study).to_plotly_figure()
            
            if save_path:
                fig.write_html(save_path)
            
            return fig
        except Exception as e:
            warnings.warn(f"Could not plot param importance: {e}")
            return None
    
    def print_results(self) -> None:
        """Print tuning results."""
        if self.best_score is None:
            raise RuntimeError("Must call tune() first")
        
        print("\n" + "="*60)
        print("OPTUNA TUNING RESULTS")
        print("="*60)
        print(f"Best Score: {self.best_score:.6f}")
        print("\nBest Parameters:")
        for key, value in self.best_params.items():
            print(f"  {key}: {value}")
        print("="*60 + "\n")
    
    def _objective(self, trial: optuna.Trial) -> float:
        """
        Generic objective function for hyperparameter optimization.
        Tries common hyperparameters.
        """
        # Common hyperparameters for tree-based models
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        }
        
        # Filter to only valid parameters for this model
        model_init_params = {}
        try:
            model_params = self.model_class().get_params()
            for key in params:
                if key in model_params:
                    model_init_params[key] = params[key]
        except:
            # If we can't inspect params, try all
            model_init_params = params
        
        try:
            model = self.model_class(**model_init_params)
            score = cross_val_score(
                model, self.X, self.y,
                cv=self.cv_folds,
                scoring=self.scoring,
                n_jobs=self.n_jobs
            ).mean()
            
            return score
        except Exception as e:
            # Return worst possible score on error
            if self.verbose > 0:
                print(f"Trial failed: {e}")
            return -1.0 if self.scoring == 'f1_weighted' else -np.inf
    
    def _make_objective_with_space(self, search_space: Dict[str, Tuple]):
        """
        Create objective function with predefined search space.
        
        Args:
            search_space: Dictionary of parameter search spaces
            
        Returns:
            Objective function
        """
        def objective(trial: optuna.Trial) -> float:
            params = {}
            
            # Build parameters from search space
            for key, space_def in search_space.items():
                space_type = space_def[0]
                
                if space_type == 'int':
                    _, min_val, max_val = space_def
                    params[key] = trial.suggest_int(key, min_val, max_val)
                
                elif space_type == 'uniform':
                    _, min_val, max_val = space_def
                    params[key] = trial.suggest_uniform(key, min_val, max_val)
                
                elif space_type == 'loguniform':
                    _, min_val, max_val = space_def
                    params[key] = trial.suggest_loguniform(key, min_val, max_val)
                
                elif space_type == 'categorical':
                    _, choices = space_def[0], space_def[1:]
                    params[key] = trial.suggest_categorical(key, choices)
            
            try:
                model = self.model_class(**params)
                score = cross_val_score(
                    model, self.X, self.y,
                    cv=self.cv_folds,
                    scoring=self.scoring,
                    n_jobs=self.n_jobs
                ).mean()
                
                return score
            except Exception as e:
                if self.verbose > 0:
                    print(f"Trial failed: {e}")
                return -1.0 if self.scoring == 'f1_weighted' else -np.inf
        
        return objective
