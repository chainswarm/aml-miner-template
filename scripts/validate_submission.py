import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from loguru import logger

from aml_miner.config import Settings
from aml_miner.models import AlertScorerModel, AlertRankerModel, ClusterScorerModel
from aml_miner.utils.data_loader import BatchDataLoader


def test_api_locally(
    host: str = "127.0.0.1",
    port: int = 8000,
    timeout: int = 30
) -> Dict:
    logger.info("Testing API locally")
    
    results = {
        'api_accessible': False,
        'endpoints': {},
        'server_started': False,
        'errors': []
    }
    
    server_process = None
    
    try:
        logger.info("Starting API server")
        server_process = subprocess.Popen(
            [sys.executable, "-m", "aml_miner.api.server"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        base_url = f"http://{host}:{port}"
        
        logger.info(f"Waiting for server to start at {base_url}")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{base_url}/health", timeout=2)
                if response.status_code == 200:
                    results['server_started'] = True
                    logger.info("Server started successfully")
                    break
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(1)
        
        if not results['server_started']:
            results['errors'].append("Server failed to start within timeout")
            return results
        
        results['api_accessible'] = True
        
        endpoints = [
            ('GET', '/health', None),
            ('GET', '/version', None),
            ('GET', '/', None)
        ]
        
        for method, endpoint, data in endpoints:
            url = f"{base_url}{endpoint}"
            logger.info(f"Testing {method} {endpoint}")
            
            try:
                start = time.time()
                
                if method == 'GET':
                    response = requests.get(url, timeout=10)
                elif method == 'POST':
                    response = requests.post(url, json=data, timeout=10)
                
                latency = (time.time() - start) * 1000
                
                results['endpoints'][endpoint] = {
                    'status_code': response.status_code,
                    'success': response.status_code == 200,
                    'latency_ms': round(latency, 2),
                    'response_size_bytes': len(response.content)
                }
                
                logger.info(f"  Status: {response.status_code}, Latency: {latency:.2f}ms")
                
            except Exception as e:
                logger.error(f"  Error: {str(e)}")
                results['endpoints'][endpoint] = {
                    'status_code': 0,
                    'success': False,
                    'error': str(e)
                }
                results['errors'].append(f"{endpoint}: {str(e)}")
        
    except Exception as e:
        logger.error(f"API test failed: {str(e)}")
        results['errors'].append(str(e))
        
    finally:
        if server_process:
            logger.info("Stopping server")
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_process.kill()
                server_process.wait()
            
            logger.info("Server stopped")
    
    return results


def check_determinism(
    batch_path: Path,
    n_iterations: int = 10
) -> Dict:
    logger.info(f"Checking determinism with {n_iterations} iterations")
    
    results = {
        'deterministic': True,
        'iterations': n_iterations,
        'models_tested': {},
        'errors': []
    }
    
    try:
        settings = Settings()
        loader = BatchDataLoader()
        
        batch = loader.load_batch(batch_path)
        
        alerts_df = batch['alerts']
        features_df = batch['features']
        clusters_df = batch['clusters']
        
        if 'alert_scorer' not in results['models_tested']:
            results['models_tested']['alert_scorer'] = {'deterministic': True, 'scores': []}
        
        if settings.ALERT_SCORER_PATH.exists():
            logger.info("Testing Alert Scorer determinism")
            
            scorer = AlertScorerModel()
            scorer.load_model(settings.ALERT_SCORER_PATH)
            
            X = scorer.prepare_features(alerts_df, features_df, clusters_df)
            
            all_scores = []
            
            for i in range(n_iterations):
                scores = scorer.predict(X)
                all_scores.append(scores)
                
                if i > 0:
                    if not np.allclose(all_scores[0], scores, rtol=1e-9, atol=1e-9):
                        results['models_tested']['alert_scorer']['deterministic'] = False
                        results['deterministic'] = False
                        logger.warning(f"Alert Scorer not deterministic at iteration {i}")
            
            results['models_tested']['alert_scorer']['scores'] = [float(s) for s in all_scores[0][:5]]
            logger.info(f"Alert Scorer: {'Deterministic' if results['models_tested']['alert_scorer']['deterministic'] else 'Non-deterministic'}")
        
        if 'alert_ranker' not in results['models_tested']:
            results['models_tested']['alert_ranker'] = {'deterministic': True, 'ranks': []}
        
        if settings.ALERT_RANKER_PATH.exists():
            logger.info("Testing Alert Ranker determinism")
            
            ranker = AlertRankerModel()
            ranker.load_model(settings.ALERT_RANKER_PATH)
            
            X = ranker.prepare_features(alerts_df, features_df, clusters_df)
            alert_ids = alerts_df['alert_id'].tolist() if 'alert_id' in alerts_df.columns else list(range(len(alerts_df)))
            
            all_rankings = []
            
            for i in range(n_iterations):
                ranking = ranker.rank_alerts(X, alert_ids)
                all_rankings.append(ranking)
                
                if i > 0:
                    if not ranking.equals(all_rankings[0]):
                        results['models_tested']['alert_ranker']['deterministic'] = False
                        results['deterministic'] = False
                        logger.warning(f"Alert Ranker not deterministic at iteration {i}")
            
            results['models_tested']['alert_ranker']['ranks'] = all_rankings[0]['rank'].tolist()[:5]
            logger.info(f"Alert Ranker: {'Deterministic' if results['models_tested']['alert_ranker']['deterministic'] else 'Non-deterministic'}")
        
        if 'cluster_scorer' not in results['models_tested']:
            results['models_tested']['cluster_scorer'] = {'deterministic': True, 'scores': []}
        
        if settings.CLUSTER_SCORER_PATH.exists():
            logger.info("Testing Cluster Scorer determinism")
            
            cluster_scorer = ClusterScorerModel()
            cluster_scorer.load_model(settings.CLUSTER_SCORER_PATH)
            
            X = cluster_scorer.prepare_features(alerts_df, features_df, clusters_df)
            
            all_scores = []
            
            for i in range(n_iterations):
                scores = cluster_scorer.predict(X)
                all_scores.append(scores)
                
                if i > 0:
                    if not np.allclose(all_scores[0], scores, rtol=1e-9, atol=1e-9):
                        results['models_tested']['cluster_scorer']['deterministic'] = False
                        results['deterministic'] = False
                        logger.warning(f"Cluster Scorer not deterministic at iteration {i}")
            
            results['models_tested']['cluster_scorer']['scores'] = [float(s) for s in all_scores[0][:5]]
            logger.info(f"Cluster Scorer: {'Deterministic' if results['models_tested']['cluster_scorer']['deterministic'] else 'Non-deterministic'}")
        
    except Exception as e:
        logger.error(f"Determinism check failed: {str(e)}")
        results['errors'].append(str(e))
        results['deterministic'] = False
    
    return results


def measure_performance(batch_path: Path) -> Dict:
    logger.info("Measuring performance")
    
    results = {
        'models': {},
        'errors': []
    }
    
    try:
        settings = Settings()
        loader = BatchDataLoader()
        
        batch = loader.load_batch(batch_path)
        
        alerts_df = batch['alerts']
        features_df = batch['features']
        clusters_df = batch['clusters']
        
        n_alerts = len(alerts_df)
        n_clusters = len(clusters_df)
        
        logger.info(f"Batch size: {n_alerts} alerts, {n_clusters} clusters")
        
        if settings.ALERT_SCORER_PATH.exists():
            logger.info("Measuring Alert Scorer performance")
            
            scorer = AlertScorerModel()
            scorer.load_model(settings.ALERT_SCORER_PATH)
            
            X = scorer.prepare_features(alerts_df, features_df, clusters_df)
            
            start = time.time()
            scores = scorer.predict(X)
            total_time = time.time() - start
            
            results['models']['alert_scorer'] = {
                'total_time_ms': round(total_time * 1000, 2),
                'avg_latency_per_alert_ms': round((total_time * 1000) / n_alerts, 2),
                'throughput_alerts_per_sec': round(n_alerts / total_time, 2),
                'num_alerts': n_alerts
            }
            
            logger.info(f"  Total time: {total_time * 1000:.2f}ms")
            logger.info(f"  Avg latency: {(total_time * 1000) / n_alerts:.2f}ms per alert")
            logger.info(f"  Throughput: {n_alerts / total_time:.2f} alerts/sec")
        
        if settings.ALERT_RANKER_PATH.exists():
            logger.info("Measuring Alert Ranker performance")
            
            ranker = AlertRankerModel()
            ranker.load_model(settings.ALERT_RANKER_PATH)
            
            X = ranker.prepare_features(alerts_df, features_df, clusters_df)
            alert_ids = alerts_df['alert_id'].tolist() if 'alert_id' in alerts_df.columns else list(range(len(alerts_df)))
            
            start = time.time()
            ranking = ranker.rank_alerts(X, alert_ids)
            total_time = time.time() - start
            
            results['models']['alert_ranker'] = {
                'total_time_ms': round(total_time * 1000, 2),
                'avg_latency_per_alert_ms': round((total_time * 1000) / n_alerts, 2),
                'throughput_alerts_per_sec': round(n_alerts / total_time, 2),
                'num_alerts': n_alerts
            }
            
            logger.info(f"  Total time: {total_time * 1000:.2f}ms")
            logger.info(f"  Avg latency: {(total_time * 1000) / n_alerts:.2f}ms per alert")
            logger.info(f"  Throughput: {n_alerts / total_time:.2f} alerts/sec")
        
        if settings.CLUSTER_SCORER_PATH.exists():
            logger.info("Measuring Cluster Scorer performance")
            
            cluster_scorer = ClusterScorerModel()
            cluster_scorer.load_model(settings.CLUSTER_SCORER_PATH)
            
            X = cluster_scorer.prepare_features(alerts_df, features_df, clusters_df)
            
            start = time.time()
            scores = cluster_scorer.predict(X)
            total_time = time.time() - start
            
            results['models']['cluster_scorer'] = {
                'total_time_ms': round(total_time * 1000, 2),
                'avg_latency_per_cluster_ms': round((total_time * 1000) / n_clusters, 2),
                'throughput_clusters_per_sec': round(n_clusters / total_time, 2),
                'num_clusters': n_clusters
            }
            
            logger.info(f"  Total time: {total_time * 1000:.2f}ms")
            logger.info(f"  Avg latency: {(total_time * 1000) / n_clusters:.2f}ms per cluster")
            logger.info(f"  Throughput: {n_clusters / total_time:.2f} clusters/sec")
        
    except Exception as e:
        logger.error(f"Performance measurement failed: {str(e)}")
        results['errors'].append(str(e))
    
    return results


def generate_validation_report(
    api_results: Optional[Dict],
    determinism_results: Optional[Dict],
    performance_results: Optional[Dict],
    output_path: Path
) -> bool:
    logger.info("Generating validation report")
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'validation_summary': {
            'api_test': 'skipped',
            'determinism_test': 'skipped',
            'performance_test': 'skipped',
            'overall_status': 'unknown'
        },
        'api_test': api_results,
        'determinism_test': determinism_results,
        'performance_test': performance_results
    }
    
    all_passed = True
    
    if api_results:
        api_passed = api_results.get('api_accessible', False) and len(api_results.get('errors', [])) == 0
        report['validation_summary']['api_test'] = 'passed' if api_passed else 'failed'
        all_passed = all_passed and api_passed
    
    if determinism_results:
        det_passed = determinism_results.get('deterministic', False)
        report['validation_summary']['determinism_test'] = 'passed' if det_passed else 'failed'
        all_passed = all_passed and det_passed
    
    if performance_results:
        perf_passed = len(performance_results.get('errors', [])) == 0
        report['validation_summary']['performance_test'] = 'passed' if perf_passed else 'failed'
        all_passed = all_passed and perf_passed
    
    report['validation_summary']['overall_status'] = 'passed' if all_passed else 'failed'
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Validation report saved to {output_path}")
    
    logger.info("=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 60)
    
    for test_name, status in report['validation_summary'].items():
        if test_name != 'overall_status':
            logger.info(f"  {test_name}: {status}")
    
    logger.info("=" * 60)
    logger.info(f"Overall Status: {report['validation_summary']['overall_status'].upper()}")
    logger.info("=" * 60)
    
    text_report_path = output_path.with_suffix('.txt')
    with open(text_report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("AML MINER VALIDATION REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Timestamp: {report['timestamp']}\n\n")
        
        f.write("VALIDATION SUMMARY\n")
        f.write("-" * 60 + "\n")
        for test_name, status in report['validation_summary'].items():
            f.write(f"{test_name}: {status}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"Overall Status: {report['validation_summary']['overall_status'].upper()}\n")
        f.write("=" * 60 + "\n")
    
    logger.info(f"Text report saved to {text_report_path}")
    
    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description='Validate AML Miner submission before deployment'
    )
    
    parser.add_argument(
        '--batch-path',
        type=Path,
        help='Path to test batch directory'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('./validation_report.json'),
        help='Path for validation report (default: ./validation_report.json)'
    )
    
    parser.add_argument(
        '--skip-api-test',
        action='store_true',
        help='Skip API testing'
    )
    
    parser.add_argument(
        '--determinism-iterations',
        type=int,
        default=10,
        help='Number of iterations for determinism check (default: 10)'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("AML MINER VALIDATION PIPELINE")
    logger.info("=" * 60)
    
    api_results = None
    determinism_results = None
    performance_results = None
    
    if not args.skip_api_test:
        logger.info("\n[1/3] Testing API...")
        api_results = test_api_locally()
    else:
        logger.info("\n[1/3] Skipping API test")
    
    if args.batch_path:
        if not args.batch_path.exists():
            logger.error(f"Batch path does not exist: {args.batch_path}")
            sys.exit(1)
        
        logger.info("\n[2/3] Checking determinism...")
        determinism_results = check_determinism(
            args.batch_path,
            n_iterations=args.determinism_iterations
        )
        
        logger.info("\n[3/3] Measuring performance...")
        performance_results = measure_performance(args.batch_path)
    else:
        logger.info("\n[2/3] Skipping determinism check (no batch path provided)")
        logger.info("\n[3/3] Skipping performance measurement (no batch path provided)")
    
    logger.info("\nGenerating validation report...")
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    all_passed = generate_validation_report(
        api_results,
        determinism_results,
        performance_results,
        args.output
    )
    
    if not all_passed:
        logger.error("Validation failed")
        sys.exit(1)
    
    logger.info("Validation completed successfully")


if __name__ == '__main__':
    main()