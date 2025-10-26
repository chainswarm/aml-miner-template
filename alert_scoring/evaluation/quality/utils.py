from typing import Dict
from loguru import logger


def compute_final_score(integrity_result: Dict, behavior_result: Dict, ground_truth_result: Dict) -> Dict:
    integrity_score = integrity_result.get('score', 0.0)
    behavior_score = behavior_result.get('score', 0.0)
    ground_truth_score = ground_truth_result.get('score', 0.0)
    
    total_score = integrity_score + behavior_score + ground_truth_score
    
    final_result = {
        'total_score': total_score,
        'breakdown': {
            'integrity': integrity_score,
            'behavior': behavior_score,
            'ground_truth': ground_truth_score
        },
        'all_passed': (
            integrity_result.get('passed', False) and
            behavior_result.get('passed', True) and
            ground_truth_result.get('passed', False)
        ),
        'details': {
            'integrity_checks': integrity_result.get('checks', {}),
            'behavior_traps': behavior_result.get('traps_detected', []),
            'ground_truth_metrics': ground_truth_result.get('metrics', {})
        }
    }
    
    logger.info(f"Final validation score: {total_score:.3f} (integrity={integrity_score:.3f}, behavior={behavior_score:.3f}, ground_truth={ground_truth_score:.3f})")
    
    return final_result


def format_validation_report(validation_result: Dict) -> str:
    report_lines = [
        "=" * 60,
        "VALIDATION REPORT",
        "=" * 60,
        f"Total Score: {validation_result['total_score']:.3f} / 1.0",
        f"Status: {'PASSED' if validation_result['all_passed'] else 'FAILED'}",
        "",
        "Score Breakdown:",
        f"  - Integrity:    {validation_result['breakdown']['integrity']:.3f} / 0.2",
        f"  - Behavior:     {validation_result['breakdown']['behavior']:.3f} / 0.3",
        f"  - Ground Truth: {validation_result['breakdown']['ground_truth']:.3f} / 0.5",
        "",
    ]
    
    integrity_checks = validation_result['details']['integrity_checks']
    if integrity_checks:
        report_lines.append("Integrity Checks:")
        for check, passed in integrity_checks.items():
            status = "✓" if passed else "✗"
            report_lines.append(f"  {status} {check}")
        report_lines.append("")
    
    behavior_traps = validation_result['details']['behavior_traps']
    if behavior_traps:
        report_lines.append("Behavior Issues Detected:")
        for trap in behavior_traps:
            report_lines.append(f"  ⚠ {trap}")
        report_lines.append("")
    
    gt_metrics = validation_result['details']['ground_truth_metrics']
    if gt_metrics:
        report_lines.append("Ground Truth Metrics:")
        report_lines.append(f"  - AUC-ROC:        {gt_metrics.get('auc_roc', 0):.4f}")
        report_lines.append(f"  - AUC-PR:         {gt_metrics.get('auc_pr', 0):.4f}")
        report_lines.append(f"  - Best F1:        {gt_metrics.get('best_f1', 0):.4f}")
        report_lines.append(f"  - Best Threshold: {gt_metrics.get('best_threshold', 0):.4f}")
        report_lines.append(f"  - Samples:        {gt_metrics.get('n_samples', 0)}")
        report_lines.append(f"  - Positives:      {gt_metrics.get('n_positive', 0)}")
        report_lines.append("")
    
    report_lines.append("=" * 60)
    
    return "\n".join(report_lines)