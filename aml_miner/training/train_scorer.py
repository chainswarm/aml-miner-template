import argparse
import json
from pathlib import Path
from typing import Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from loguru import logger
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from aml_miner.features import FeatureBuilder
from aml_miner.utils.data_loader import BatchDataLoader


def prepare_training_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.Series]:
    loader = BatchDataLoader()
    builder = FeatureBuilder()
    
    all_X = []
    all_y = []
    
    logger.info(f"Loading training data from {data_dir}")
    
    batch_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    
    if not batch_dirs:
        raise ValueError(f"No batch directories found in {data_dir}")
    
    logger.info(f"Found {len(batch_dirs)} batch directories")
    
    for batch_dir in batch_dirs:
        logger.info(f"Loading batch: {batch_dir.name}")
        
        batch = loader.load_batch(batch_dir)
        
        X = builder.build_all_features(
            batch['alerts'],
            batch['features'],
            batch['clusters']
        )
        
        if 'ground_truth' not in batch['alerts'].columns:
            raise ValueError(f"Missing 'ground_truth' column in {batch_dir}")
        
        y = batch['alerts']['ground_truth']
        
        all_X.append(X)
        all_y.append(y)
    
    X_combined = pd.concat(all_X, ignore_index=True)
    y_combined = pd.concat(all_y, ignore_index=True)
    
    logger.info(f"Prepared {len(X_combined)} samples with {len(X_combined.columns)} features")
    logger.info(f"Label distribution: {y_combined.value_counts().to_dict()}")
    
    return X_combined, y_combined


def train_alert_scorer(
    X: pd.DataFrame,
    y: pd.Series,
    config: dict,
    cv_folds: int = 5,
    eval_metric: str = 'auc'
) -> Tuple[lgb.Booster, dict]:
    logger.info("Starting model training")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Train set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    train_data = lgb.Dataset(X_train, y_train)
    test_data = lgb.Dataset(X_test, y_test, reference=train_data)
    
    model = lgb.train(
        config,
        train_data,
        valid_sets=[train_data, test_data],
        valid_names=['train', 'test'],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(10)
        ]
    )
    
    logger.info("Evaluating model on test set")
    
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    metrics = {
        'test_auc': float(roc_auc_score(y_test, y_pred_proba)),
        'test_precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'test_recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'test_f1': float(f1_score(y_test, y_pred, zero_division=0)),
        'best_iteration': model.best_iteration,
        'num_features': model.num_feature(),
    }
    
    logger.info("Test Metrics:")
    logger.info(f"  AUC: {metrics['test_auc']:.4f}")
    logger.info(f"  Precision: {metrics['test_precision']:.4f}")
    logger.info(f"  Recall: {metrics['test_recall']:.4f}")
    logger.info(f"  F1: {metrics['test_f1']:.4f}")
    logger.info(f"  Best iteration: {metrics['best_iteration']}")
    
    if cv_folds > 1:
        logger.info(f"Performing {cv_folds}-fold cross-validation")
        
        cv_scores = []
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_val = y_train.iloc[val_idx]
            
            fold_train_data = lgb.Dataset(X_fold_train, y_fold_train)
            fold_val_data = lgb.Dataset(X_fold_val, y_fold_val, reference=fold_train_data)
            
            fold_model = lgb.train(
                config,
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
            
            logger.info(f"  Fold {fold} AUC: {fold_auc:.4f}")
        
        metrics['cv_auc_mean'] = float(np.mean(cv_scores))
        metrics['cv_auc_std'] = float(np.std(cv_scores))
        metrics['cv_scores'] = [float(s) for s in cv_scores]
        
        logger.info(f"Cross-validation AUC: {metrics['cv_auc_mean']:.4f} ± {metrics['cv_auc_std']:.4f}")
    
    return model, metrics


def main():
    parser = argparse.ArgumentParser(
        description='Train Alert Scorer model'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        required=True,
        help='Directory containing training batch data'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output path for trained model (.txt format)'
    )
    parser.add_argument(
        '--eval-metric',
        default='auc',
        choices=['auc', 'binary_logloss', 'binary_error'],
        help='Metric to optimize during training'
    )
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='Number of cross-validation folds (0 to disable)'
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('aml_miner/config/model_config.yaml'),
        help='Path to model configuration YAML file'
    )
    
    args = parser.parse_args()
    
    logger.info("Alert Scorer Training Pipeline")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output path: {args.output}")
    logger.info(f"Evaluation metric: {args.eval_metric}")
    logger.info(f"CV folds: {args.cv_folds}")
    
    if not args.data_dir.exists():
        raise ValueError(f"Data directory does not exist: {args.data_dir}")
    
    with open(args.config) as f:
        config_data = yaml.safe_load(f)
        config = config_data['alert_scorer']
    
    config['metric'] = args.eval_metric
    
    logger.info("Loading and preparing training data")
    X, y = prepare_training_data(args.data_dir)
    
    logger.info("Training model")
    model, metrics = train_alert_scorer(X, y, config, cv_folds=args.cv_folds)
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    model.save_model(str(args.output))
    logger.info(f"Model saved to {args.output}")
    
    report_path = args.output.with_suffix('.json')
    with open(report_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Training report saved to {report_path}")
    
    logger.info("Training completed successfully")


if __name__ == '__main__':
    main()