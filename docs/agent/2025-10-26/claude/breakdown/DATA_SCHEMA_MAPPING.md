# Data Schema Mapping - SOT to Miner Template API
**Date**: 2025-10-26  
**Purpose**: Map SOT database schemas to Parquet files and API request/response formats

---

## Overview

This document maps the **SOT (Source of Truth) database schemas** to:
1. **Parquet files** sent to miners
2. **API request schemas** (FastAPI Pydantic models)
3. **Internal data structures** used by the template

---

## Data Flow

```
┌──────────────────────────────────────────────────────────────┐
│                    VALIDATOR (SOT Database)                   │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ analyzers_alerts (ClickHouse)                          │  │
│  │ analyzers_features (ClickHouse)                        │  │
│  │ analyzers_alert_clusters (ClickHouse)                  │  │
│  │ core_money_flows (ClickHouse)                          │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
                              │
                              │ Export to Parquet
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                        Batch Directory                        │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ alerts.parquet          (12K-50K rows)                 │  │
│  │ features.parquet        (100K-500K rows)               │  │
│  │ clusters.parquet        (1K-5K rows)                   │  │
│  │ money_flows.parquet     (millions of rows)             │  │
│  │ META.json               (metadata, hashes)             │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
                              │
                              │ Load into memory
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                   MINER (aml-miner-template)                  │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ Data Loader: Read Parquet → pandas DataFrames         │  │
│  └────────────────────────────────────────────────────────┘  │
│                              │
│                              ▼
│  ┌────────────────────────────────────────────────────────┐  │
│  │ Feature Builder: Transform to ML features             │  │
│  └────────────────────────────────────────────────────────┘  │
│                              │
│                              ▼
│  ┌────────────────────────────────────────────────────────┐  │
│  │ Models: Score alerts using features                   │  │
│  └────────────────────────────────────────────────────────┘  │
│                              │
│                              ▼
│  ┌────────────────────────────────────────────────────────┐  │
│  │ API Response: Return scores + explanations            │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP Response
                              ▼
┌──────────────────────────────────────────────────────────────┐
│              SUBNET MINER (Bittensor Neuron)                  │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ Proxy: Forward scores to validator                    │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

---

## Schema Mapping: Alerts

### Source: `analyzers_alerts` (ClickHouse)
[`docs/sot/schema/analyzers_alerts.sql`](../../sot/schema/analyzers_alerts.sql:1)

### Parquet File: `alerts.parquet`

```python
# Columns in Parquet file (subset of SQL schema)
{
    "window_days": int16,
    "processing_date": date32,
    "alert_id": string,
    "address": string,
    "typology_type": string,
    "pattern_id": string,
    "pattern_type": string,
    "severity": string,
    "suspected_address_type": string,
    "suspected_address_subtype": string,
    "alert_confidence_score": float32,
    "description": string,
    "volume_usd": decimal128(18),
    "evidence_json": string,
    "risk_indicators": list<string>,
    "_version": uint64
}
```

### API Request Schema: `AlertData` (Pydantic)

```python
from pydantic import BaseModel, Field
from typing import List, Optional
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
                "address": "0xabc123...",
                "typology_type": "layering",
                "severity": "high",
                "alert_confidence_score": 0.87,
                "description": "Suspected layering pattern detected",
                "volume_usd": "150000.00",
                "evidence_json": '{"pattern_matches": [...]}',
                "risk_indicators": ["high_volume", "rapid_movement"]
            }
        }
```

---

## Schema Mapping: Features

### Source: `analyzers_features` (ClickHouse)
[`docs/sot/schema/analyzers_features.sql`](../../sot/schema/analyzers_features.sql:1)

### Parquet File: `features.parquet`

```python
# 160+ columns in Parquet file (full SQL schema)
# Note: This is a WIDE table with many features per address
{
    "window_days": int16,
    "processing_date": date32,
    "address": string,
    
    # Node topology (4 features)
    "degree_in": uint32,
    "degree_out": uint32,
    "degree_total": uint32,
    "unique_counterparties": uint32,
    
    # Volume features (20+ features)
    "total_in_usd": decimal128(18),
    "total_out_usd": decimal128(18),
    "net_flow_usd": decimal128(18),
    "total_volume_usd": decimal128(18),
    "avg_tx_in_usd": decimal128(18),
    "avg_tx_out_usd": decimal128(18),
    # ... 14 more volume features
    
    # Statistical distribution (10+ features)
    "amount_variance": float64,
    "amount_skewness": float64,
    "amount_kurtosis": float64,
    # ... 7 more statistical features
    
    # Transaction counts (3 features)
    "tx_in_count": uint64,
    "tx_out_count": uint64,
    "tx_total_count": uint64,
    
    # Temporal features (15+ features)
    "activity_days": uint32,
    "activity_span_days": uint32,
    "avg_daily_volume_usd": decimal128(18),
    # ... 12 more temporal features
    
    # Flow characteristics (10+ features)
    "reciprocity_ratio": float32,
    "flow_diversity": float32,
    # ... 8 more flow features
    
    # Asset diversity (5 features)
    "unique_assets_in": uint32,
    "unique_assets_out": uint32,
    "dominant_asset_in": string,
    "dominant_asset_out": string,
    "asset_diversity_score": float32,
    
    # Behavioral patterns (15+ features)
    "hourly_activity": list<uint16>,  # 24-element array
    "daily_activity": list<uint16>,   # 7-element array
    # ... 13 more behavioral features
    
    # Graph algorithms (15+ features)
    "pagerank": float32,
    "betweenness": float32,
    "closeness": float32,
    "clustering_coefficient": float32,
    # ... 11 more graph features
    
    # Risk/anomaly (10+ features)
    "behavioral_anomaly_score": float32,
    "graph_anomaly_score": float32,
    "neighborhood_anomaly_score": float32,
    "global_anomaly_score": float32,
    # ... 6 more anomaly features
    
    # Classification (15+ boolean features)
    "is_exchange_like": bool,
    "is_whale": bool,
    "is_mixer_like": bool,
    # ... 12 more classification features
    
    # Metadata (5 features)
    "completeness_score": float32,
    "quality_score": float32,
    "confidence_score": float32,
    "first_activity_timestamp": uint64,
    "last_activity_timestamp": uint64,
    
    "_version": uint64
}
```

### API Request Schema: Features passed as DataFrame

**Note**: Features are NOT sent as individual Pydantic models but as a complete pandas DataFrame to preserve columnar structure. The API receives the entire features table.

```python
# In api/schemas.py
class BatchData(BaseModel):
    alerts: List[AlertData]
    features_df: bytes  # Serialized pandas DataFrame (pickle or parquet bytes)
    clusters: List[ClusterData]
    money_flows: Optional[List[MoneyFlowData]] = None
    
    # Alternative: Use dict representation
    features: List[dict]  # Each dict = {address: str, **feature_values}
```

---

## Schema Mapping: Clusters

### Source: `analyzers_alert_clusters` (ClickHouse)
[`docs/sot/schema/analyzers_alert_clusters.sql`](../../sot/schema/analyzers_alert_clusters.sql:1)

### Parquet File: `clusters.parquet`

```python
{
    "window_days": int16,
    "processing_date": date32,
    "cluster_id": string,
    "cluster_type": string,
    "primary_address": string,
    "pattern_id": string,
    "primary_alert_id": string,
    "related_alert_ids": list<string>,
    "addresses_involved": list<string>,
    "total_alerts": uint32,
    "total_volume_usd": decimal128(18),
    "severity_max": string,
    "confidence_avg": float32,
    "earliest_alert_timestamp": uint64,
    "latest_alert_timestamp": uint64,
    "_version": uint64
}
```

### API Request Schema: `ClusterData` (Pydantic)

```python
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
```

---

## Schema Mapping: Money Flows

### Source: `core_money_flows` (ClickHouse)
Reference table (not in docs/sot/schema but used for graph structure)

### Parquet File: `money_flows.parquet`

```python
{
    "source_address": string,
    "dest_address": string,
    "amount_usd": decimal128(18),
    "timestamp": uint64,
    "asset": string,
    "tx_hash": string,
    "network": string,
    "block_number": uint64
}
```

### API Request Schema: `MoneyFlowData` (Pydantic)

```python
class MoneyFlowData(BaseModel):
    source_address: str
    dest_address: str
    amount_usd: Decimal
    timestamp: int
    asset: str
    tx_hash: str
    network: str
    block_number: Optional[int] = None
```

---

## Complete API Request Schema

```python
# api/schemas.py

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from decimal import Decimal
from datetime import date

class AlertData(BaseModel):
    """Single alert from analyzers_alerts table"""
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

class ClusterData(BaseModel):
    """Single cluster from analyzers_alert_clusters table"""
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

class MoneyFlowData(BaseModel):
    """Single money flow edge"""
    source_address: str
    dest_address: str
    amount_usd: Decimal
    timestamp: int
    asset: str
    tx_hash: str
    network: str
    block_number: Optional[int] = None

class BatchData(BaseModel):
    """
    Complete batch data sent to miner API
    Corresponds to a single day's batch from SOT
    """
    alerts: List[AlertData]              # 12K-50K alerts
    features: List[Dict[str, Any]]       # 100K-500K address features
    clusters: List[ClusterData]          # 1K-5K clusters
    money_flows: Optional[List[MoneyFlowData]] = None  # Optional graph structure
    
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
                        "total_alerts": 5,
                        "total_volume_usd": "250000"
                    }
                ]
            }
        }
```

---

## Data Loading Implementation

```python
# aml_miner/utils/data_loader.py

import pandas as pd
from pathlib import Path
from typing import Dict
from loguru import logger

class BatchDataLoader:
    """Load batch data from Parquet files"""
    
    def load_batch(self, batch_dir: Path) -> Dict[str, pd.DataFrame]:
        """
        Load all Parquet files from batch directory
        
        Args:
            batch_dir: Path to batch directory containing Parquet files
            
        Returns:
            Dictionary with DataFrames: alerts, features, clusters, money_flows
        """
        logger.info(f"Loading batch from {batch_dir}")
        
        # Load Parquet files
        alerts_df = pd.read_parquet(batch_dir / "alerts.parquet")
        features_df = pd.read_parquet(batch_dir / "features.parquet")
        clusters_df = pd.read_parquet(batch_dir / "clusters.parquet")
        
        # Money flows are optional (large file)
        money_flows_path = batch_dir / "money_flows.parquet"
        money_flows_df = None
        if money_flows_path.exists():
            money_flows_df = pd.read_parquet(money_flows_path)
        
        logger.info(f"Loaded {len(alerts_df)} alerts, "
                   f"{len(features_df)} features, "
                   f"{len(clusters_df)} clusters")
        
        return {
            "alerts": alerts_df,
            "features": features_df,
            "clusters": clusters_df,
            "money_flows": money_flows_df
        }
    
    def validate_schema(self, df: pd.DataFrame, expected_schema: Dict[str, type]):
        """Validate DataFrame schema matches expected"""
        for col, dtype in expected_schema.items():
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")
            # Type checking can be added here
```

---

## Feature Engineering Pipeline

```python
# aml_miner/features/feature_builder.py

import pandas as pd
from typing import Dict

class FeatureBuilder:
    """Build ML features from raw batch data"""
    
    def build_features_for_alerts(
        self,
        alerts_df: pd.DataFrame,
        features_df: pd.DataFrame,
        clusters_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Join alerts with their features and cluster information
        
        Args:
            alerts_df: Alerts from analyzers_alerts
            features_df: Features from analyzers_features
            clusters_df: Clusters from analyzers_alert_clusters
            
        Returns:
            DataFrame with one row per alert and all features
        """
        # Join alerts with address features
        alert_features = alerts_df.merge(
            features_df,
            left_on="address",
            right_on="address",
            how="left",
            suffixes=("_alert", "_feature")
        )
        
        # Add cluster information
        # Create cluster lookup
        cluster_lookup = {}
        for _, cluster in clusters_df.iterrows():
            for alert_id in cluster["related_alert_ids"]:
                cluster_lookup[alert_id] = {
                    "cluster_id": cluster["cluster_id"],
                    "cluster_size": cluster["total_alerts"],
                    "cluster_volume": cluster["total_volume_usd"],
                    "cluster_severity": cluster["severity_max"]
                }
        
        # Add cluster features to alerts
        alert_features["cluster_size"] = alert_features["alert_id"].map(
            lambda x: cluster_lookup.get(x, {}).get("cluster_size", 1)
        )
        alert_features["cluster_volume"] = alert_features["alert_id"].map(
            lambda x: cluster_lookup.get(x, {}).get("cluster_volume", 0)
        )
        
        return alert_features
```

---

## Summary

### Key Points

1. **Data Format**: Parquet files (columnar, efficient, preserves types)
2. **Schema Source**: SOT ClickHouse schemas define the structure
3. **API Layer**: Pydantic models validate incoming data
4. **Internal Processing**: pandas DataFrames for ML operations
5. **Feature Count**: 160+ features per address available
6. **Batch Size**: 12K-50K alerts, 100K-500K address features per day

### File Sizes (Approximate)

- `alerts.parquet`: 5-20 MB (12K-50K rows × 16 columns)
- `features.parquet`: 50-200 MB (100K-500K rows × 160 columns)
- `clusters.parquet`: 1-5 MB (1K-5K rows × 16 columns)
- `money_flows.parquet`: 500 MB - 2 GB (millions of rows × 8 columns)

### Performance Considerations

- **Use PyArrow** for fast Parquet I/O
- **Lazy loading** for money_flows (only if needed)
- **Columnar selection** (only load needed features)
- **Memory mapping** for large files
- **Batch processing** for ML inference