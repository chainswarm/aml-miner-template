import argparse
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
from loguru import logger

from alert_scoring.evaluation.quality.integrity import IntegrityValidator
from alert_scoring.evaluation.quality.behavior import BehaviorValidator
from alert_scoring.evaluation.quality.performance import GroundTruthValidator
from alert_scoring.evaluation.quality.utils import compute_final_score, format_validation_report


def load_scores(scores_path: Path) -> pd.DataFrame:
    if not scores_path.exists():
        raise FileNotFoundError(f"Scores file not found: {scores_path}")
    
    scores_df = pd.read_parquet(scores_path)
    logger.info(f"Loaded {len(scores_df)} scores from {scores_path}")
    
    return scores_df


def load_alerts(processing_date: str, input_dir: Path) -> pd.DataFrame:
    alerts_path = input_dir / processing_date / "alerts.parquet"
    
    if not alerts_path.exists():
        raise FileNotFoundError(f"Alerts file not found: {alerts_path}")
    
    alerts_df = pd.read_parquet(alerts_path)
    logger.info(f"Loaded {len(alerts_df)} alerts from {alerts_path}")
    
    return alerts_df


def load_ground_truth(ground_truth_path: Path) -> pd.DataFrame:
    if not ground_truth_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {ground_truth_path}")
    
    gt_df = pd.read_parquet(ground_truth_path)
    logger.info(f"Loaded {len(gt_df)} ground truth labels")
    
    return gt_df


def validate_single_model(scores_df: pd.DataFrame, alerts_df: pd.DataFrame,
                         ground_truth_df: pd.DataFrame, pattern_traps: list = None) -> dict:
    integrity_validator = IntegrityValidator()
    behavior_validator = BehaviorValidator()
    ground_truth_validator = GroundTruthValidator()
    
    logger.info("Running integrity validation...")
    integrity_result = integrity_validator.validate(scores_df, alerts_df)
    
    if not integrity_result['passed']:
        logger.error("Integrity validation failed")
        return {
            'validation_passed': False,
            'integrity_result': integrity_result,
            'reason': 'integrity_check_failed'
        }
    
    logger.info("Running behavior validation...")
    behavior_result = behavior_validator.validate(scores_df, pattern_traps or [])
    
    logger.info("Running ground truth validation...")
    ground_truth_result = ground_truth_validator.validate(scores_df, ground_truth_df)
    
    if not ground_truth_result['passed']:
        logger.warning("Ground truth validation failed (insufficient data)")
    
    final_result = compute_final_score(
        integrity_result,
        behavior_result,
        ground_truth_result
    )
    
    return {
        'validation_passed': integrity_result['passed'] and ground_truth_result['passed'],
        'final_result': final_result,
        'integrity_result': integrity_result,
        'behavior_result': behavior_result,
        'ground_truth_result': ground_truth_result
    }


def compare_models(model_a_path: Path, model_b_path: Path,
                  processing_date: str, input_dir: Path, ground_truth_path: Path) -> dict:
    logger.info("=== Model Comparison: A/B Testing ===")
    logger.info(f"Model A: {model_a_path}")
    logger.info(f"Model B: {model_b_path}")
    
    scores_a_df = load_scores(model_a_path)
    scores_b_df = load_scores(model_b_path)
    
    alerts_df = load_alerts(processing_date, input_dir)
    ground_truth_df = load_ground_truth(ground_truth_path)
    
    logger.info("\n--- Validating Model A ---")
    result_a = validate_single_model(scores_a_df, alerts_df, ground_truth_df)
    
    logger.info("\n--- Validating Model B ---")
    result_b = validate_single_model(scores_b_df, alerts_df, ground_truth_df)
    
    logger.info("\n--- Comparing Performance ---")
    gt_validator = GroundTruthValidator()
    comparison = gt_validator.compare_models(scores_a_df, scores_b_df, ground_truth_df)
    
    comparison_result = {
        'model_a': {
            'path': str(model_a_path),
            'validation': result_a,
            'report': format_validation_report(result_a['final_result']) if result_a.get('final_result') else None
        },
        'model_b': {
            'path': str(model_b_path),
            'validation': result_b,
            'report': format_validation_report(result_b['final_result']) if result_b.get('final_result') else None
        },
        'comparison': comparison,
        'recommendation': None
    }
    
    if comparison['comparison_valid']:
        winner = comparison['winner']
        improvements = comparison['improvements']
        
        comparison_result['recommendation'] = {
            'winner': winner,
            'reason': f"AUC-ROC improvement: {improvements['auc_roc']:.4f}",
            'improvements': improvements
        }
        
        logger.info(f"\n🏆 Winner: {winner.upper()}")
        logger.info(f"   AUC-ROC improvement: {improvements['auc_roc']:.4f}")
        logger.info(f"   AUC-PR improvement: {improvements['auc_pr']:.4f}")
        logger.info(f"   F1 improvement: {improvements['f1']:.4f}")
    
    return comparison_result


def main():
    parser = argparse.ArgumentParser(description='Validate and compare models using A/B testing')
    parser.add_argument('--model-a', required=True, help='Path to model A scores (Parquet)')
    parser.add_argument('--model-b', required=True, help='Path to model B scores (Parquet)')
    parser.add_argument('--processing-date', required=True, help='Processing date (YYYY-MM-DD)')
    parser.add_argument('--input-dir', default='input', help='Input directory with alerts')
    parser.add_argument('--ground-truth', required=True, help='Path to ground truth Parquet file')
    parser.add_argument('--output', required=True, help='Output JSON file for results')
    
    args = parser.parse_args()
    
    try:
        datetime.strptime(args.processing_date, '%Y-%m-%d')
    except ValueError:
        logger.error(f"Invalid date format: {args.processing_date}. Expected YYYY-MM-DD")
        return 1
    
    model_a_path = Path(args.model_a)
    model_b_path = Path(args.model_b)
    ground_truth_path = Path(args.ground_truth)
    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        comparison_result = compare_models(
            model_a_path,
            model_b_path,
            args.processing_date,
            input_dir,
            ground_truth_path
        )
        
        with open(output_path, 'w') as f:
            json.dump(comparison_result, f, indent=2, default=str)
        
        logger.info(f"\nResults saved to {output_path}")
        
        if comparison_result.get('recommendation'):
            winner = comparison_result['recommendation']['winner']
            return 0 if winner == 'model_b' else 2
        
        return 0
        
    except Exception as e:
        logger.exception(f"Error during validation: {e}")
        return 1


if __name__ == "__main__":
    exit(main())