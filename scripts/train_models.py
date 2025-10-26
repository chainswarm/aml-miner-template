import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from loguru import logger

from aml_miner.training.train_ranker import prepare_ranking_data, train_alert_ranker
from aml_miner.training.train_scorer import prepare_training_data, train_alert_scorer


def download_data_if_needed(
    start_date: str,
    end_date: str,
    output_dir: Path
) -> bool:
    if output_dir.exists() and any(output_dir.iterdir()):
        logger.info(f"Data directory {output_dir} already exists with content")
        return True
    
    logger.info(f"Downloading data from {start_date} to {end_date}")
    
    script_path = Path(__file__).parent / "download_batch.sh"
    
    if not script_path.exists():
        raise FileNotFoundError(f"Download script not found: {script_path}")
    
    cmd = [
        "bash",
        str(script_path),
        "--start-date", start_date,
        "--end-date", end_date,
        "--output-dir", str(output_dir)
    ]
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"Download failed: {result.stderr}")
        return False
    
    logger.info("Data download completed successfully")
    return True


def train_cluster_scorer(
    data_dir: Path,
    output_path: Path,
    config: dict
) -> Dict:
    logger.info("Training Cluster Scorer")
    logger.info("Note: Cluster scoring uses rule-based approach with cluster features")
    
    metrics = {
        'model_type': 'rule_based',
        'description': 'Cluster scorer uses graph metrics and member statistics',
        'timestamp': datetime.now().isoformat(),
        'status': 'complete'
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path.with_suffix('.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info("Cluster scorer configuration saved")
    
    return metrics


def train_all_models(
    data_dir: Path,
    output_dir: Path,
    models: List[str],
    config_path: Path
) -> Dict[str, Dict]:
    logger.info("Starting model training pipeline")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Models to train: {', '.join(models)}")
    
    if not data_dir.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")
    
    with open(config_path) as f:
        config_data = yaml.safe_load(f)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_results = {}
    
    if 'alert_scorer' in models or 'all' in models:
        logger.info("=" * 60)
        logger.info("TRAINING ALERT SCORER")
        logger.info("=" * 60)
        
        try:
            X, y = prepare_training_data(data_dir)
            
            model, metrics = train_alert_scorer(
                X, y,
                config_data['alert_scorer'],
                cv_folds=5
            )
            
            model_path = output_dir / "alert_scorer.txt"
            model.save_model(str(model_path))
            logger.info(f"Alert Scorer saved to {model_path}")
            
            report_path = model_path.with_suffix('.json')
            with open(report_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            training_results['alert_scorer'] = {
                'status': 'success',
                'model_path': str(model_path),
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Alert Scorer training failed: {e}")
            training_results['alert_scorer'] = {
                'status': 'failed',
                'error': str(e)
            }
    
    if 'alert_ranker' in models or 'all' in models:
        logger.info("=" * 60)
        logger.info("TRAINING ALERT RANKER")
        logger.info("=" * 60)
        
        try:
            X, y, groups = prepare_ranking_data(data_dir)
            
            model, metrics = train_alert_ranker(
                X, y, groups,
                config_data['alert_ranker'],
                ndcg_at=[5, 10, 20]
            )
            
            model_path = output_dir / "alert_ranker.txt"
            model.save_model(str(model_path))
            logger.info(f"Alert Ranker saved to {model_path}")
            
            report_path = model_path.with_suffix('.json')
            with open(report_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            training_results['alert_ranker'] = {
                'status': 'success',
                'model_path': str(model_path),
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Alert Ranker training failed: {e}")
            training_results['alert_ranker'] = {
                'status': 'failed',
                'error': str(e)
            }
    
    if 'cluster_scorer' in models or 'all' in models:
        logger.info("=" * 60)
        logger.info("TRAINING CLUSTER SCORER")
        logger.info("=" * 60)
        
        try:
            model_path = output_dir / "cluster_scorer.txt"
            metrics = train_cluster_scorer(
                data_dir,
                model_path,
                config_data.get('cluster_scorer', {})
            )
            
            training_results['cluster_scorer'] = {
                'status': 'success',
                'model_path': str(model_path),
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Cluster Scorer training failed: {e}")
            training_results['cluster_scorer'] = {
                'status': 'failed',
                'error': str(e)
            }
    
    return training_results


def generate_training_report(
    results: Dict[str, Dict],
    output_path: Path
) -> None:
    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_models': len(results),
            'successful': sum(1 for r in results.values() if r['status'] == 'success'),
            'failed': sum(1 for r in results.values() if r['status'] == 'failed')
        },
        'models': results
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Training report saved to {output_path}")
    
    logger.info("=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total models: {report['summary']['total_models']}")
    logger.info(f"Successful: {report['summary']['successful']}")
    logger.info(f"Failed: {report['summary']['failed']}")
    
    for model_name, result in results.items():
        status = "✓" if result['status'] == 'success' else "✗"
        logger.info(f"  {status} {model_name}: {result['status']}")
        
        if result['status'] == 'success' and 'metrics' in result:
            metrics = result['metrics']
            if 'test_auc' in metrics:
                logger.info(f"      AUC: {metrics['test_auc']:.4f}")
            if 'test_ndcg@10' in metrics:
                logger.info(f"      NDCG@10: {metrics['test_ndcg@10']:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Orchestrate training of all AML Miner models'
    )
    
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('./data'),
        help='Directory containing training batch data (default: ./data)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('./trained_models'),
        help='Output directory for trained models (default: ./trained_models)'
    )
    
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip data download step'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date for data download (YYYY-MM-DD, required if not skipping download)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date for data download (YYYY-MM-DD, required if not skipping download)'
    )
    
    parser.add_argument(
        '--models',
        nargs='+',
        choices=['alert_scorer', 'alert_ranker', 'cluster_scorer', 'all'],
        default=['all'],
        help='Which models to train (default: all)'
    )
    
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('aml_miner/config/model_config.yaml'),
        help='Path to model configuration YAML file'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("AML MINER TRAINING PIPELINE")
    logger.info("=" * 60)
    
    if not args.skip_download:
        if not args.start_date or not args.end_date:
            logger.error("--start-date and --end-date are required when not skipping download")
            sys.exit(1)
        
        success = download_data_if_needed(
            args.start_date,
            args.end_date,
            args.data_dir
        )
        
        if not success:
            logger.error("Data download failed")
            sys.exit(1)
    
    if not args.config.exists():
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)
    
    try:
        results = train_all_models(
            args.data_dir,
            args.output_dir,
            args.models,
            args.config
        )
        
        report_path = args.output_dir / "training_report.json"
        generate_training_report(results, report_path)
        
        failed_models = [name for name, result in results.items() if result['status'] == 'failed']
        
        if failed_models:
            logger.warning(f"Some models failed to train: {', '.join(failed_models)}")
            sys.exit(1)
        
        logger.info("=" * 60)
        logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()