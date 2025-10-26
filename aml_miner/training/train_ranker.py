import argparse
import json
from pathlib import Path
from typing import Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from loguru import logger
from sklearn.model_selection import GroupKFold, train_test_split

from aml_miner.features import FeatureBuilder
from aml_miner.utils.data_loader import BatchDataLoader


def compute_ndcg(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    order = np.argsort(y_pred)[::-1][:k]
    
    dcg = np.sum((2 ** y_true[order] - 1) / np.log2(np.arange(2, k + 2)))
    
    ideal_order = np.argsort(y_true)[::-1][:k]
    idcg = np.sum((2 ** y_true[ideal_order] - 1) / np.log2(np.arange(2, k + 2)))
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def prepare_ranking_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
    loader = BatchDataLoader()
    builder = FeatureBuilder()
    
    all_X = []
    all_y = []
    all_groups = []
    
    logger.info(f"Loading ranking training data from {data_dir}")
    
    batch_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    
    if not batch_dirs:
        raise ValueError(f"No batch directories found in {data_dir}")
    
    logger.info(f"Found {len(batch_dirs)} batch directories")
    
    for batch_idx, batch_dir in enumerate(batch_dirs):
        logger.info(f"Loading batch {batch_idx + 1}/{len(batch_dirs)}: {batch_dir.name}")
        
        batch = loader.load_batch(batch_dir)
        
        X = builder.build_all_features(
            batch['alerts'],
            batch['features'],
            batch['clusters']
        )
        
        if 'ground_truth' not in batch['alerts'].columns:
            raise ValueError(f"Missing 'ground_truth' column in {batch_dir}")
        
        ground_truth = batch['alerts']['ground_truth'].values
        
        if 'severity' in batch['alerts'].columns:
            severity = batch['alerts']['severity'].values
            severity_map = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1, 'info': 0}
            severity_scores = np.array([severity_map.get(s.lower(), 1) if isinstance(s, str) else 1 for s in severity])
        else:
            severity_scores = np.ones(len(ground_truth))
        
        relevance = ground_truth.astype(int) * 2 + (severity_scores / 4.0)
        relevance = np.clip(relevance, 0, 4)
        
        y = pd.Series(relevance)
        
        group_size = len(X)
        groups = np.full(group_size, batch_idx)
        
        all_X.append(X)
        all_y.append(y)
        all_groups.append(groups)
    
    X_combined = pd.concat(all_X, ignore_index=True)
    y_combined = pd.concat(all_y, ignore_index=True)
    groups_combined = np.concatenate(all_groups)
    
    logger.info(f"Prepared {len(X_combined)} samples with {len(X_combined.columns)} features")
    logger.info(f"Number of query groups: {len(np.unique(groups_combined))}")
    logger.info(f"Relevance distribution: {pd.Series(y_combined).value_counts().sort_index().to_dict()}")
    
    return X_combined, y_combined, groups_combined


def train_alert_ranker(
    X: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray,
    config: dict,
    ndcg_at: list = None
) -> Tuple[lgb.Booster, dict]:
    if ndcg_at is None:
        ndcg_at = [5, 10, 20]
    
    logger.info("Starting ranking model training")
    
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    
    train_groups = unique_groups[:int(0.8 * n_groups)]
    test_groups = unique_groups[int(0.8 * n_groups):]
    
    train_mask = np.isin(groups, train_groups)
    test_mask = np.isin(groups, test_groups)
    
    X_train = X[train_mask]
    y_train = y[train_mask]
    groups_train = groups[train_mask]
    
    X_test = X[test_mask]
    y_test = y[test_mask]
    groups_test = groups[test_mask]
    
    logger.info(f"Train set: {len(X_train)} samples in {len(train_groups)} groups")
    logger.info(f"Test set: {len(X_test)} samples in {len(test_groups)} groups")
    
    train_group_sizes = np.bincount(groups_train)[train_groups]
    test_group_sizes = np.bincount(groups_test)[test_groups]
    
    train_data = lgb.Dataset(
        X_train,
        y_train,
        group=train_group_sizes
    )
    
    test_data = lgb.Dataset(
        X_test,
        y_test,
        group=test_group_sizes,
        reference=train_data
    )
    
    config['objective'] = 'lambdarank'
    config['metric'] = ['ndcg']
    config['ndcg_eval_at'] = ndcg_at
    
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
    
    logger.info("Evaluating ranking model on test set")
    
    y_pred = model.predict(X_test)
    
    metrics = {
        'best_iteration': model.best_iteration,
        'num_features': model.num_feature(),
    }
    
    unique_test_groups = np.unique(groups_test)
    for k in ndcg_at:
        ndcg_scores = []
        
        for group_id in unique_test_groups:
            group_mask = groups_test == group_id
            y_true_group = y_test[group_mask].values
            y_pred_group = y_pred[group_mask]
            
            if len(y_true_group) > 0:
                ndcg = compute_ndcg(y_true_group, y_pred_group, k)
                ndcg_scores.append(ndcg)
        
        mean_ndcg = np.mean(ndcg_scores)
        metrics[f'test_ndcg@{k}'] = float(mean_ndcg)
        logger.info(f"  NDCG@{k}: {mean_ndcg:.4f}")
    
    return model, metrics


def main():
    parser = argparse.ArgumentParser(
        description='Train Alert Ranker model'
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
        '--ndcg-at',
        type=int,
        nargs='+',
        default=[5, 10, 20],
        help='NDCG evaluation cutoff points'
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('aml_miner/config/model_config.yaml'),
        help='Path to model configuration YAML file'
    )
    
    args = parser.parse_args()
    
    logger.info("Alert Ranker Training Pipeline")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output path: {args.output}")
    logger.info(f"NDCG evaluation at: {args.ndcg_at}")
    
    if not args.data_dir.exists():
        raise ValueError(f"Data directory does not exist: {args.data_dir}")
    
    with open(args.config) as f:
        config_data = yaml.safe_load(f)
        config = config_data['alert_ranker']
    
    logger.info("Loading and preparing ranking training data")
    X, y, groups = prepare_ranking_data(args.data_dir)
    
    logger.info("Training ranking model")
    model, metrics = train_alert_ranker(X, y, groups, config, ndcg_at=args.ndcg_at)
    
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