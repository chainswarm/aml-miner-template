import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from loguru import logger
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

try:
    import optuna
    from optuna.visualization import plot_optimization_history, plot_param_importances
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not available. Install with: pip install optuna")

from aml_miner.training.train_scorer import prepare_training_data
from aml_miner.training.train_ranker import prepare_ranking_data, compute_ndcg


class HyperparameterTuner:
    
    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_type: str = 'scorer',
        groups: Optional[np.ndarray] = None
    ):
        self.X = X
        self.y = y
        self.model_type = model_type
        self.groups = groups
        self.best_params: Dict = {}
        
        if model_type not in ['scorer', 'ranker', 'cluster']:
            raise ValueError(f"Invalid model_type: {model_type}. Must be 'scorer', 'ranker', or 'cluster'")
        
        if model_type == 'ranker' and groups is None:
            raise ValueError("groups parameter is required for ranker model type")
        
        logger.info(f"Initialized HyperparameterTuner for {model_type} model")
        logger.info(f"Dataset size: {len(X)} samples, {len(X.columns)} features")
    
    def define_search_space(self) -> Dict:
        search_space = {
            'num_leaves': (20, 100),
            'max_depth': (3, 10),
            'learning_rate': (0.01, 0.1),
            'n_estimators': (50, 500),
            'min_child_samples': (10, 50),
            'subsample': (0.6, 1.0),
            'colsample_bytree': (0.6, 1.0),
            'reg_alpha': (0.0, 1.0),
            'reg_lambda': (0.0, 1.0),
        }
        
        if self.model_type == 'ranker':
            search_space['min_gain_to_split'] = (0.0, 1.0)
        
        return search_space
    
    def objective(self, trial) -> float:
        search_space = self.define_search_space()
        
        params = {
            'num_leaves': trial.suggest_int('num_leaves', *search_space['num_leaves']),
            'max_depth': trial.suggest_int('max_depth', *search_space['max_depth']),
            'learning_rate': trial.suggest_float('learning_rate', *search_space['learning_rate'], log=True),
            'n_estimators': trial.suggest_int('n_estimators', *search_space['n_estimators']),
            'min_child_samples': trial.suggest_int('min_child_samples', *search_space['min_child_samples']),
            'subsample': trial.suggest_float('subsample', *search_space['subsample']),
            'colsample_bytree': trial.suggest_float('colsample_bytree', *search_space['colsample_bytree']),
            'reg_alpha': trial.suggest_float('reg_alpha', *search_space['reg_alpha']),
            'reg_lambda': trial.suggest_float('reg_lambda', *search_space['reg_lambda']),
            'verbose': -1,
            'force_row_wise': True,
        }
        
        if self.model_type == 'scorer':
            params['objective'] = 'binary'
            params['metric'] = 'auc'
            return self._cv_scorer(params)
        
        elif self.model_type == 'ranker':
            params['objective'] = 'lambdarank'
            params['metric'] = 'ndcg'
            params['ndcg_eval_at'] = [5]
            params['min_gain_to_split'] = trial.suggest_float('min_gain_to_split', *search_space['min_gain_to_split'])
            return self._cv_ranker(params)
        
        elif self.model_type == 'cluster':
            params['objective'] = 'regression'
            params['metric'] = 'rmse'
            return self._cv_cluster(params)
        
        raise ValueError(f"Unknown model_type: {self.model_type}")
    
    def _cv_scorer(self, params: Dict) -> float:
        cv_scores = []
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for train_idx, val_idx in skf.split(self.X, self.y):
            X_train = self.X.iloc[train_idx]
            y_train = self.y.iloc[train_idx]
            X_val = self.X.iloc[val_idx]
            y_val = self.y.iloc[val_idx]
            
            train_data = lgb.Dataset(X_train, y_train)
            val_data = lgb.Dataset(X_val, y_val, reference=train_data)
            
            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                callbacks=[
                    lgb.early_stopping(50),
                    lgb.log_evaluation(0)
                ]
            )
            
            y_pred = model.predict(X_val)
            score = roc_auc_score(y_val, y_pred)
            cv_scores.append(score)
        
        mean_score = np.mean(cv_scores)
        return mean_score
    
    def _cv_ranker(self, params: Dict) -> float:
        unique_groups = np.unique(self.groups)
        n_groups = len(unique_groups)
        n_folds = min(5, n_groups)
        
        cv_scores = []
        
        fold_size = n_groups // n_folds
        
        for fold in range(n_folds):
            val_start = fold * fold_size
            val_end = (fold + 1) * fold_size if fold < n_folds - 1 else n_groups
            
            val_groups = unique_groups[val_start:val_end]
            train_groups = np.concatenate([unique_groups[:val_start], unique_groups[val_end:]])
            
            train_mask = np.isin(self.groups, train_groups)
            val_mask = np.isin(self.groups, val_groups)
            
            X_train = self.X[train_mask]
            y_train = self.y[train_mask]
            groups_train = self.groups[train_mask]
            
            X_val = self.X[val_mask]
            y_val = self.y[val_mask]
            groups_val = self.groups[val_mask]
            
            train_group_sizes = np.bincount(groups_train)[train_groups]
            val_group_sizes = np.bincount(groups_val)[val_groups]
            
            train_data = lgb.Dataset(X_train, y_train, group=train_group_sizes)
            val_data = lgb.Dataset(X_val, y_val, group=val_group_sizes, reference=train_data)
            
            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                callbacks=[
                    lgb.early_stopping(50),
                    lgb.log_evaluation(0)
                ]
            )
            
            y_pred = model.predict(X_val)
            
            ndcg_scores = []
            for group_id in val_groups:
                group_mask = groups_val == group_id
                y_true_group = y_val[group_mask].values
                y_pred_group = y_pred[group_mask]
                
                if len(y_true_group) > 0:
                    ndcg = compute_ndcg(y_true_group, y_pred_group, k=5)
                    ndcg_scores.append(ndcg)
            
            cv_scores.append(np.mean(ndcg_scores))
        
        mean_score = np.mean(cv_scores)
        return mean_score
    
    def _cv_cluster(self, params: Dict) -> float:
        cv_scores = []
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        y_binary = (self.y > self.y.median()).astype(int)
        
        for train_idx, val_idx in skf.split(self.X, y_binary):
            X_train = self.X.iloc[train_idx]
            y_train = self.y.iloc[train_idx]
            X_val = self.X.iloc[val_idx]
            y_val = self.y.iloc[val_idx]
            
            train_data = lgb.Dataset(X_train, y_train)
            val_data = lgb.Dataset(X_val, y_val, reference=train_data)
            
            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                callbacks=[
                    lgb.early_stopping(50),
                    lgb.log_evaluation(0)
                ]
            )
            
            y_pred = model.predict(X_val)
            rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
            cv_scores.append(-rmse)
        
        mean_score = np.mean(cv_scores)
        return mean_score
    
    def optimize(self, n_trials: int = 100) -> Dict:
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for hyperparameter optimization. Install with: pip install optuna")
        
        logger.info(f"Starting hyperparameter optimization with {n_trials} trials")
        
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)
        
        self.best_params = study.best_params
        
        logger.info(f"Optimization completed!")
        logger.info(f"Best score: {study.best_value:.4f}")
        logger.info(f"Best parameters: {self.best_params}")
        
        return {
            'best_params': self.best_params,
            'best_score': float(study.best_value),
            'n_trials': n_trials,
            'study': study
        }
    
    def save_best_params(self, path: Path):
        if not self.best_params:
            raise ValueError("No optimization has been run yet. Call optimize() first.")
        
        output = {
            'model_type': self.model_type,
            'best_params': self.best_params,
            'search_space': {k: list(v) for k, v in self.define_search_space().items()}
        }
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(output, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Best parameters saved to {path}")


def main():
    parser = argparse.ArgumentParser(
        description='Hyperparameter tuning for AML Miner models'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        required=True,
        help='Directory containing training batch data'
    )
    parser.add_argument(
        '--trials',
        type=int,
        default=100,
        help='Number of optimization trials'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output path for best parameters YAML file'
    )
    parser.add_argument(
        '--model-type',
        choices=['scorer', 'ranker', 'cluster'],
        required=True,
        help='Type of model to tune'
    )
    
    args = parser.parse_args()
    
    logger.info("Hyperparameter Tuning Pipeline")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Number of trials: {args.trials}")
    logger.info(f"Output path: {args.output}")
    
    if not args.data_dir.exists():
        raise ValueError(f"Data directory does not exist: {args.data_dir}")
    
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is required. Install with: pip install optuna")
    
    logger.info("Loading training data")
    
    if args.model_type == 'ranker':
        X, y, groups = prepare_ranking_data(args.data_dir)
        tuner = HyperparameterTuner(X, y, model_type=args.model_type, groups=groups)
    else:
        X, y = prepare_training_data(args.data_dir)
        tuner = HyperparameterTuner(X, y, model_type=args.model_type)
    
    logger.info("Starting optimization")
    result = tuner.optimize(n_trials=args.trials)
    
    tuner.save_best_params(args.output)
    
    study = result['study']
    
    plots_dir = args.output.parent / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        fig = plot_optimization_history(study)
        fig.write_html(str(plots_dir / 'optimization_history.html'))
        logger.info(f"Optimization history plot saved to {plots_dir / 'optimization_history.html'}")
    except Exception as e:
        logger.warning(f"Could not save optimization history plot: {e}")
    
    try:
        fig = plot_param_importances(study)
        fig.write_html(str(plots_dir / 'param_importances.html'))
        logger.info(f"Parameter importances plot saved to {plots_dir / 'param_importances.html'}")
    except Exception as e:
        logger.warning(f"Could not save parameter importances plot: {e}")
    
    logger.info("Hyperparameter tuning completed successfully")


if __name__ == '__main__':
    main()