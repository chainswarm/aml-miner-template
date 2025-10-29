import lightgbm as lgb
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from loguru import logger


class ModelTrainer:
    
    def __init__(self, model_type: str = 'alert_scorer'):
        self.model_type = model_type
        self.model = None
        self.feature_names = None
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        hyperparameters: Dict = None,
        cv_folds: int = 5
    ) -> Tuple[lgb.Booster, Dict]:
        
        logger.info(
            "Starting model training",
            extra={
                "model_type": self.model_type,
                "num_samples": len(X),
                "num_features": len(X.columns),
                "cv_folds": cv_folds
            }
        )
        
        self.feature_names = list(X.columns)
        
        if hyperparameters is None:
            hyperparameters = self._get_default_hyperparameters()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
        
        logger.info(f"Train set: {len(X_train):,} samples")
        logger.info(f"Test set: {len(X_test):,} samples")
        
        train_data = lgb.Dataset(
            X_train,
            label=y_train,
            feature_name=self.feature_names
        )
        
        test_data = lgb.Dataset(
            X_test,
            label=y_test,
            reference=train_data,
            feature_name=self.feature_names
        )
        
        self.model = lgb.train(
            hyperparameters,
            train_data,
            valid_sets=[train_data, test_data],
            valid_names=['train', 'test'],
            callbacks=[
                lgb.early_stopping(50),
                lgb.log_evaluation(10)
            ]
        )
        
        test_metrics = self._evaluate_model(X_test, y_test)
        
        if cv_folds > 1:
            cv_metrics = self._cross_validate(X_train, y_train, hyperparameters, cv_folds)
            test_metrics.update(cv_metrics)
        
        logger.success(
            "Training completed",
            extra={"metrics": test_metrics}
        )
        
        return self.model, test_metrics
    
    def _get_default_hyperparameters(self) -> Dict:
        
        if self.model_type == 'alert_scorer':
            return {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'max_depth': -1,
                'min_data_in_leaf': 20,
                'verbose': -1
            }
        elif self.model_type == 'alert_ranker':
            return {
                'objective': 'lambdarank',
                'metric': 'ndcg',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.1,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'max_depth': -1,
                'verbose': -1
            }
        elif self.model_type == 'cluster_scorer':
            return {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'max_depth': -1,
                'min_data_in_leaf': 20,
                'verbose': -1
            }
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _evaluate_model(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict:
        
        logger.info("Evaluating model on test set")
        
        y_pred_proba = self.model.predict(X_test)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        metrics = {
            'test_auc': float(roc_auc_score(y_test, y_pred_proba)),
            'test_precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'test_recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'test_f1': float(f1_score(y_test, y_pred, zero_division=0)),
            'best_iteration': self.model.best_iteration,
            'num_trees': self.model.num_trees()
        }
        
        logger.info(
            "Test metrics",
            extra={
                "auc": f"{metrics['test_auc']:.4f}",
                "precision": f"{metrics['test_precision']:.4f}",
                "recall": f"{metrics['test_recall']:.4f}",
                "f1": f"{metrics['test_f1']:.4f}"
            }
        )
        
        return metrics
    
    def _cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        hyperparameters: Dict,
        n_folds: int
    ) -> Dict:
        
        logger.info(f"Performing {n_folds}-fold cross-validation")
        
        cv_scores = []
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_seed=42)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_fold_train = X.iloc[train_idx]
            y_fold_train = y.iloc[train_idx]
            X_fold_val = X.iloc[val_idx]
            y_fold_val = y.iloc[val_idx]
            
            fold_train_data = lgb.Dataset(X_fold_train, label=y_fold_train)
            fold_val_data = lgb.Dataset(X_fold_val, label=y_fold_val, reference=fold_train_data)
            
            fold_model = lgb.train(
                hyperparameters,
                fold_train_data,
                valid_sets=[fold_val_data],
                valid_names=['validation'],
                callbacks=[
                    lgb.early_stopping(50),
                    lgb.log_evaluation(0)
                ]
            )
            
            y_val_pred = fold_model.predict(X_fold_val)
            fold_auc = roc_auc_score(y_fold_val, y_val_pred)
            cv_scores.append(fold_auc)
            
            logger.info(f"Fold {fold} AUC: {fold_auc:.4f}")
        
        cv_metrics = {
            'cv_auc_mean': float(np.mean(cv_scores)),
            'cv_auc_std': float(np.std(cv_scores)),
            'cv_scores': [float(s) for s in cv_scores]
        }
        
        logger.info(
            f"Cross-validation: {cv_metrics['cv_auc_mean']:.4f} ± {cv_metrics['cv_auc_std']:.4f}"
        )
        
        return cv_metrics