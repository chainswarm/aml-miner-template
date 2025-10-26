from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from decimal import Decimal
from datetime import date


class AlertData(BaseModel):
    window_days: int
    processing_date: date
    alert_id: str
    address: str
    typology_type: str
    pattern_id: Optional[str] = ""
    pattern_type: Optional[str] = ""
    severity: str = "medium"
    suspected_address_type: str = "unknown"
    suspected_address_subtype: Optional[str] = ""
    alert_confidence_score: float
    description: str
    volume_usd: Decimal = Decimal("0")
    evidence_json: str
    risk_indicators: List[str]
    
    class Config:
        json_schema_extra = {
            "example": {
                "window_days": 7,
                "processing_date": "2025-10-26",
                "alert_id": "alert_12345",
                "address": "0xabc123",
                "typology_type": "layering",
                "pattern_id": "pattern_001",
                "pattern_type": "rapid_movement",
                "severity": "high",
                "suspected_address_type": "mixer",
                "suspected_address_subtype": "tornado_cash",
                "alert_confidence_score": 0.87,
                "description": "Suspected layering pattern detected",
                "volume_usd": "150000.00",
                "evidence_json": '{"pattern_matches": ["rapid_tx"], "confidence": 0.87}',
                "risk_indicators": ["high_volume", "rapid_movement"]
            }
        }


class ClusterData(BaseModel):
    window_days: int
    processing_date: date
    cluster_id: str
    cluster_type: str
    primary_address: Optional[str] = ""
    pattern_id: Optional[str] = ""
    primary_alert_id: str
    related_alert_ids: List[str]
    addresses_involved: List[str]
    total_alerts: int
    total_volume_usd: Decimal
    severity_max: str = "medium"
    confidence_avg: float
    earliest_alert_timestamp: int
    latest_alert_timestamp: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "window_days": 7,
                "processing_date": "2025-10-26",
                "cluster_id": "cluster_001",
                "cluster_type": "pattern_based",
                "primary_address": "0xabc123",
                "pattern_id": "pattern_001",
                "primary_alert_id": "alert_001",
                "related_alert_ids": ["alert_001", "alert_002", "alert_003"],
                "addresses_involved": ["0xabc123", "0xdef456"],
                "total_alerts": 5,
                "total_volume_usd": "250000.00",
                "severity_max": "high",
                "confidence_avg": 0.85,
                "earliest_alert_timestamp": 1729900000,
                "latest_alert_timestamp": 1729950000
            }
        }


class MoneyFlowData(BaseModel):
    source_address: str
    dest_address: str
    amount_usd: Decimal
    timestamp: int
    asset: str
    tx_hash: str
    network: str
    block_number: Optional[int] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "source_address": "0xabc123",
                "dest_address": "0xdef456",
                "amount_usd": "10000.00",
                "timestamp": 1729900000,
                "asset": "USDT",
                "tx_hash": "0x123abc...",
                "network": "ethereum",
                "block_number": 18500000
            }
        }


class BatchData(BaseModel):
    alerts: List[AlertData]
    features: List[Dict[str, Any]]
    clusters: List[ClusterData]
    money_flows: Optional[List[MoneyFlowData]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "alerts": [
                    {
                        "window_days": 7,
                        "processing_date": "2025-10-26",
                        "alert_id": "alert_001",
                        "address": "0xabc123",
                        "typology_type": "layering",
                        "severity": "high",
                        "alert_confidence_score": 0.87,
                        "description": "Layering detected",
                        "volume_usd": "100000",
                        "evidence_json": "{}",
                        "risk_indicators": ["high_volume"]
                    }
                ],
                "features": [
                    {
                        "address": "0xabc123",
                        "degree_in": 10,
                        "degree_out": 15,
                        "total_volume_usd": "500000",
                        "pagerank": 0.001234
                    }
                ],
                "clusters": [
                    {
                        "cluster_id": "cluster_001",
                        "cluster_type": "pattern_based",
                        "primary_alert_id": "alert_001",
                        "related_alert_ids": ["alert_001"],
                        "addresses_involved": ["0xabc123"],
                        "total_alerts": 5,
                        "total_volume_usd": "250000",
                        "confidence_avg": 0.85,
                        "earliest_alert_timestamp": 1729900000,
                        "latest_alert_timestamp": 1729950000,
                        "window_days": 7,
                        "processing_date": "2025-10-26"
                    }
                ]
            }
        }


class ScoreResponse(BaseModel):
    alert_id: str
    score: float
    model_version: str
    latency_ms: float
    explain_json: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "alert_id": "alert_12345",
                "score": 0.92,
                "model_version": "1.0.0",
                "latency_ms": 12.5,
                "explain_json": '{"top_features": ["volume_usd", "pagerank"], "feature_contributions": {"volume_usd": 0.45, "pagerank": 0.35}}'
            }
        }


class RankResponse(BaseModel):
    alert_id: str
    rank: int
    model_version: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "alert_id": "alert_12345",
                "rank": 1,
                "model_version": "1.0.0"
            }
        }


class ClusterScore(BaseModel):
    processing_date: str
    network: str
    cluster_id: str
    score: float
    model_version: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "processing_date": "2025-10-26",
                "network": "ethereum",
                "cluster_id": "cluster_001",
                "score": 0.88,
                "model_version": "1.0.0"
            }
        }


class ClusterScoreResponse(BaseModel):
    cluster_id: str
    score: float
    model_version: str
    latency_ms: float
    explain_json: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "cluster_id": "cluster_001",
                "score": 0.88,
                "model_version": "1.0.0",
                "latency_ms": 8.3,
                "explain_json": '{"top_features": ["total_alerts", "severity_max"]}'
            }
        }


class HealthResponse(BaseModel):
    status: str
    models_loaded: Dict[str, bool]
    timestamp: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "models_loaded": {
                    "alert_scorer": True,
                    "alert_ranker": True,
                    "cluster_scorer": True
                },
                "timestamp": "2025-10-26T08:30:00Z"
            }
        }


class VersionResponse(BaseModel):
    api_version: str
    model_versions: Dict[str, str]
    
    class Config:
        json_schema_extra = {
            "example": {
                "api_version": "1.0.0",
                "model_versions": {
                    "alert_scorer": "1.0.0",
                    "alert_ranker": "1.0.0",
                    "cluster_scorer": "1.0.0"
                }
            }
        }


class MetadataResponse(BaseModel):
    processing_date: str
    processed_at: str
    input_counts: Dict[str, int]
    output_counts: Dict[str, int]
    latencies_ms: Dict[str, int]
    model_versions: Dict[str, Optional[str]]
    
    class Config:
        json_schema_extra = {
            "example": {
                "processing_date": "2025-10-26",
                "processed_at": "2025-10-26T08:30:00Z",
                "input_counts": {
                    "alerts": 1000,
                    "features": 1000,
                    "clusters": 50
                },
                "output_counts": {
                    "alert_scores": 1000,
                    "alert_rankings": 1000,
                    "cluster_scores": 50
                },
                "latencies_ms": {
                    "alert_scoring": 5000,
                    "alert_ranking": 2000,
                    "total": 7000
                },
                "model_versions": {
                    "alert_scorer": "1.0.0",
                    "alert_ranker": "1.0.0",
                    "cluster_scorer": "1.0.0"
                }
            }
        }