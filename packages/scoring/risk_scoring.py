import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from abc import ABC
from loguru import logger
from clickhouse_connect.driver import Client
from packages.training.feature_extraction import FeatureExtractor
from packages.training.feature_builder import FeatureBuilder
from .model_loader import ModelLoader
from .score_generator import ScoreGenerator
from .score_writer import ScoreWriter


PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()


class RiskScoring(ABC):
    
    def __init__(
        self,
        network: str,
        processing_date: str,
        client: Client,
        window_days: int = 7,
        models_dir: Path = None,
        model_types: List[str] = None
    ):
        self.network = network
        self.processing_date = processing_date
        self.client = client
        self.window_days = window_days
        
        if models_dir is None:
            models_dir = PROJECT_ROOT / 'data' / 'trained_models'
        
        self.models_dir = models_dir
        
        if model_types is None:
            model_types = ['alert_scorer', 'alert_ranker', 'cluster_scorer']
        
        self.model_types = model_types
        
        self.loader = ModelLoader(models_dir)
        self.generator = ScoreGenerator()
        self.writer = ScoreWriter(client)
    
    def run(self):
        
        logger.info(
            "Starting risk scoring workflow",
            extra={
                "network": self.network,
                "processing_date": self.processing_date,
                "window_days": self.window_days,
                "model_types": self.model_types
            }
        )
        
        start_time = time.time()
        
        try:
            logger.info("Cleaning up previous results for this processing date")
            self.writer.cleanup_processing_date(self.processing_date)
            
            logger.info("Extracting data from ClickHouse")
            extractor = FeatureExtractor(self.client)
            data = extractor.extract_training_data(
                start_date=self.processing_date,
                end_date=self.processing_date,
                window_days=self.window_days
            )
            
            input_counts = {
                'alerts': len(data['alerts']),
                'features': len(data['features']),
                'clusters': len(data['clusters'])
            }
            
            logger.info(
                "Extracted data shapes",
                extra={
                    "alerts_shape": data['alerts'].shape,
                    "features_shape": data['features'].shape,
                    "clusters_shape": data['clusters'].shape,
                    "money_flows_shape": data['money_flows'].shape,
                    "address_labels_shape": data['address_labels'].shape
                }
            )
            
            if len(data['alerts']) > 0:
                sample_size = min(3, len(data['alerts']))
                for i in range(sample_size):
                    alert = data['alerts'].iloc[i]
                    logger.info(
                        f"Sample alert {i+1}",
                        extra={
                            "alert_id": alert.get('alert_id'),
                            "volume_usd": float(alert.get('volume_usd', 0)),
                            "severity": alert.get('severity'),
                            "confidence": float(alert.get('alert_confidence_score', 0))
                        }
                    )
            
            if len(data['features']) > 0:
                sample_size = min(3, len(data['features']))
                for i in range(sample_size):
                    feat = data['features'].iloc[i]
                    numeric_cols = data['features'].select_dtypes(include=[np.number]).columns[:5]
                    feat_vals = {col: float(feat[col]) for col in numeric_cols if col in feat.index}
                    logger.info(
                        f"Sample feature row {i+1}",
                        extra={
                            "address": feat.get('address'),
                            "sample_values": feat_vals
                        }
                    )
            
            logger.info("Building inference features")
            builder = FeatureBuilder()
            X = builder.build_inference_features(data)
            
            logger.info(
                "Built inference features",
                extra={
                    "feature_matrix_shape": X.shape,
                    "num_samples": len(X),
                    "num_features": len(X.columns)
                }
            )
            
            sample_size = min(3, len(X))
            for i in range(sample_size):
                row = X.iloc[i]
                logger.info(
                    f"Sample feature vector {i+1}",
                    extra={
                        "min": float(row.min()),
                        "max": float(row.max()),
                        "mean": float(row.mean()),
                        "std": float(row.std()),
                        "num_zeros": int((row == 0).sum()),
                        "num_nonzeros": int((row != 0).sum())
                    }
                )
            
            alert_ids = data['alerts']['alert_id']
            
            metadata = {
                'input_counts_alerts': input_counts['alerts'],
                'input_counts_features': input_counts['features'],
                'input_counts_clusters': input_counts['clusters'],
                'output_counts_alert_scores': 0,
                'output_counts_alert_rankings': 0,
                'output_counts_cluster_scores': 0,
                'latencies_ms_alert_scoring': 0,
                'latencies_ms_alert_ranking': 0,
                'latencies_ms_cluster_scoring': 0,
                'latencies_ms_total': 0,
                'model_versions_alert_scorer': '',
                'model_versions_alert_ranker': '',
                'model_versions_cluster_scorer': '',
                'status': 'PROCESSING',
                'error_message': ''
            }
            
            if 'alert_scorer' in self.model_types:
                self._score_alerts(X, alert_ids, metadata)
            
            if 'alert_ranker' in self.model_types:
                self._rank_alerts(X, alert_ids, metadata)
            
            if 'cluster_scorer' in self.model_types and not data['clusters'].empty:
                self._score_clusters(data, metadata)
            
            total_latency_ms = (time.time() - start_time) * 1000
            metadata['latencies_ms_total'] = int(total_latency_ms)
            metadata['status'] = 'COMPLETED'
            
            logger.info("Updating batch metadata")
            self.writer.update_batch_metadata(self.processing_date, metadata)
            
            logger.success(
                "Risk scoring workflow completed successfully",
                extra={
                    "total_latency_ms": total_latency_ms,
                    "alert_scores": metadata['output_counts_alert_scores'],
                    "alert_rankings": metadata['output_counts_alert_rankings'],
                    "cluster_scores": metadata['output_counts_cluster_scores']
                }
            )
        
        except Exception as e:
            logger.error(f"Risk scoring failed: {e}")
            
            total_latency_ms = (time.time() - start_time) * 1000
            metadata = {
                'input_counts_alerts': 0,
                'input_counts_features': 0,
                'input_counts_clusters': 0,
                'output_counts_alert_scores': 0,
                'output_counts_alert_rankings': 0,
                'output_counts_cluster_scores': 0,
                'latencies_ms_alert_scoring': 0,
                'latencies_ms_alert_ranking': 0,
                'latencies_ms_cluster_scoring': 0,
                'latencies_ms_total': int(total_latency_ms),
                'model_versions_alert_scorer': '',
                'model_versions_alert_ranker': '',
                'model_versions_cluster_scorer': '',
                'status': 'FAILED',
                'error_message': str(e)
            }
            
            self.writer.update_batch_metadata(self.processing_date, metadata)
            raise
    
    def _score_alerts(self, X, alert_ids, metadata: Dict[str, Any]):
        
        logger.info("Scoring alerts")
        
        model = self.loader.load_latest_model(self.network, 'alert_scorer')
        model_path = self.loader.models_dir / self.network
        model_files = sorted(model_path.glob('alert_scorer_*.txt'), reverse=True)
        model_version = model_files[0].stem if model_files else 'unknown'
        
        scores, latency_ms = self.generator.score_alerts(model, X, alert_ids)
        
        self.writer.write_alert_scores(
            self.processing_date,
            scores,
            model_version
        )
        
        metadata['output_counts_alert_scores'] = len(scores)
        metadata['latencies_ms_alert_scoring'] = int(latency_ms)
        metadata['model_versions_alert_scorer'] = model_version
        
        logger.success(f"Alert scoring completed: {len(scores)} alerts scored")
    
    def _rank_alerts(self, X, alert_ids, metadata: Dict[str, Any]):
        
        logger.info("Ranking alerts")
        
        model = self.loader.load_latest_model(self.network, 'alert_ranker')
        model_path = self.loader.models_dir / self.network
        model_files = sorted(model_path.glob('alert_ranker_*.txt'), reverse=True)
        model_version = model_files[0].stem if model_files else 'unknown'
        
        rankings, latency_ms = self.generator.rank_alerts(model, X, alert_ids)
        
        self.writer.write_alert_rankings(
            self.processing_date,
            rankings,
            model_version
        )
        
        metadata['output_counts_alert_rankings'] = len(rankings)
        metadata['latencies_ms_alert_ranking'] = int(latency_ms)
        metadata['model_versions_alert_ranker'] = model_version
        
        logger.success(f"Alert ranking completed: {len(rankings)} alerts ranked")
    
    def _score_clusters(self, data: Dict, metadata: Dict[str, Any]):
        
        logger.info("Scoring clusters")
        
        if data['clusters'].empty:
            logger.warning("No clusters to score")
            return
        
        logger.info(
            "Cluster data before feature building",
            extra={
                "num_clusters": len(data['clusters']),
                "clusters_shape": data['clusters'].shape
            }
        )
        
        model = self.loader.load_latest_model(self.network, 'cluster_scorer')
        model_path = self.loader.models_dir / self.network
        model_files = sorted(model_path.glob('cluster_scorer_*.txt'), reverse=True)
        model_version = model_files[0].stem if model_files else 'unknown'
        
        builder = FeatureBuilder()
        X_clusters = builder.build_cluster_features(data)
        
        logger.info(
            "Built cluster features",
            extra={
                "feature_matrix_shape": X_clusters.shape,
                "num_clusters": len(X_clusters),
                "num_features": len(X_clusters.columns)
            }
        )
        
        sample_size = min(3, len(X_clusters))
        for i in range(sample_size):
            row = X_clusters.iloc[i]
            logger.info(
                f"Sample cluster feature vector {i+1}",
                extra={
                    "min": float(row.min()),
                    "max": float(row.max()),
                    "mean": float(row.mean()),
                    "std": float(row.std()),
                    "num_zeros": int((row == 0).sum()),
                    "num_nonzeros": int((row != 0).sum())
                }
            )
        
        cluster_ids = data['clusters']['cluster_id']
        
        scores, latency_ms = self.generator.score_clusters(model, X_clusters, cluster_ids)
        
        self.writer.write_cluster_scores(
            self.processing_date,
            scores,
            model_version
        )
        
        metadata['output_counts_cluster_scores'] = len(scores)
        metadata['latencies_ms_cluster_scoring'] = int(latency_ms)
        metadata['model_versions_cluster_scorer'] = model_version
        
        logger.success(f"Cluster scoring completed: {len(scores)} clusters scored")