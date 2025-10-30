from typing import Dict, Any
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from loguru import logger
import joblib
from .base import ModelTrainer


class XGBoostTrainer(ModelTrainer):
    
    def __init__(self, hyperparameters: dict = None):
        self.hyperparameters = hyperparameters or {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'base_score': 0.5,
            'random_state': 42
        }
        self.model = None
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: pd.Series = None
    ) -> Any:
        
        logger.info(f"Training XGBoost with {len(X)} samples, {len(X.columns)} features")
        
        positive_rate = y.mean()
        if positive_rate == 0.0 or positive_rate == 1.0:
            logger.warning(
                f"Imbalanced dataset detected: positive_rate={positive_rate:.2f}. "
                "Using base_score=0.5 for stability"
            )
            params = self.hyperparameters.copy()
            params['base_score'] = 0.5
        else:
            params = self.hyperparameters
        
        self.model = xgb.XGBClassifier(**params)
        
        if sample_weights is not None:
            logger.info("Using sample weights for training")
            self.model.fit(X, y, sample_weight=sample_weights)
        else:
            self.model.fit(X, y)
        
        logger.success("XGBoost training completed")
        return self.model
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        
        if self.model is None:
            raise ValueError("Model not trained. Call train() first")
        
        return self.model.predict_proba(X)[:, 1]
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, float]:
        
        y_pred = self.predict(X)
        
        unique_classes = y.nunique()
        if unique_classes < 2:
            logger.warning(
                f"Only {unique_classes} class present in y_true. "
                "Metrics requiring both classes cannot be calculated"
            )
            metrics = {
                'auc': 0.0,
                'pr_auc': 0.0
            }
            logger.info("Model evaluation: Skipped (single class dataset)")
            return metrics
        
        auc_score = roc_auc_score(y, y_pred)
        
        precision, recall, _ = precision_recall_curve(y, y_pred)
        pr_auc = auc(recall, precision)
        
        metrics = {
            'auc': float(auc_score),
            'pr_auc': float(pr_auc)
        }
        
        logger.info(f"Model evaluation: AUC={metrics['auc']:.4f}, PR-AUC={metrics['pr_auc']:.4f}")
        
        return metrics
    
    def save(self, path: str):
        
        if self.model is None:
            raise ValueError("No model to save")
        
        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        
        self.model = joblib.load(path)
        logger.info(f"Model loaded from {path}")