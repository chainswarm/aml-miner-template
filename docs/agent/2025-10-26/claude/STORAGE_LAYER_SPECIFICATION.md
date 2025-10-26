# Storage Layer Specification
## Repository Pattern with ClickHouse/DuckDB Support

**Date**: 2025-10-26  
**Purpose**: Define production-ready storage architecture using repository pattern

---

## Directory Structure

```
aml_miner/
└── storage/
    ├── __init__.py              # ClientFactory, connection management, migrations
    ├── utils.py                 # ClickHouse enum conversion, Pydantic helpers
    │
    ├── schema/                  # SQL DDL files
    │   ├── miner_raw_alerts.sql
    │   ├── miner_raw_features.sql
    │   ├── miner_raw_clusters.sql
    │   ├── miner_alert_scores.sql
    │   ├── miner_alert_rankings.sql
    │   ├── miner_cluster_scores.sql
    │   └── miner_batch_metadata.sql
    │
    └── repositories/            # Data access layer
        ├── __init__.py          # Repository exports
        ├── base_repository.py   # ABC base class
        ├── alerts_repository.py
        ├── features_repository.py
        ├── clusters_repository.py
        ├── scores_repository.py
        ├── rankings_repository.py
        └── metadata_repository.py
```

---

## Schema Files

### `storage/schema/miner_raw_alerts.sql`

```sql
-- Raw alerts from SOT
CREATE TABLE IF NOT EXISTS miner_raw_alerts (
    window_days Int32,
    processing_date Date,
    network String,
    alert_id String,
    address String,
    typology_type String,
    pattern_id String DEFAULT '',
    pattern_type String DEFAULT '',
    severity Enum8('LOW' = 1, 'MEDIUM' = 2, 'HIGH' = 3, 'CRITICAL' = 4) DEFAULT 'MEDIUM',
    suspected_address_type String DEFAULT 'unknown',
    suspected_address_subtype String DEFAULT '',
    alert_confidence_score Float64,
    description String,
    volume_usd Decimal(18, 2) DEFAULT 0,
    evidence_json String,
    risk_indicators Array(String),
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY (network, toYYYYMM(processing_date))
ORDER BY (processing_date, network, alert_id)
SETTINGS index_granularity = 8192;

-- Index for fast lookups
CREATE INDEX IF NOT EXISTS idx_alert_id ON miner_raw_alerts(alert_id) TYPE bloom_filter GRANULARITY 1;
```

### `storage/schema/miner_raw_features.sql`

```sql
-- Raw features from SOT
CREATE TABLE IF NOT EXISTS miner_raw_features (
    processing_date Date,
    network String,
    address String,
    feature_name String,
    feature_value Float64,
    feature_metadata String DEFAULT '',
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY (network, toYYYYMM(processing_date))
ORDER BY (processing_date, network, address, feature_name)
SETTINGS index_granularity = 8192;
```

### `storage/schema/miner_raw_clusters.sql`

```sql
-- Raw alert clusters from SOT
CREATE TABLE IF NOT EXISTS miner_raw_clusters (
    window_days Int32,
    processing_date Date,
    network String,
    cluster_id String,
    cluster_type String,
    primary_address String DEFAULT '',
    pattern_id String DEFAULT '',
    primary_alert_id String,
    related_alert_ids Array(String),
    addresses_involved Array(String),
    total_alerts Int32,
    total_volume_usd Decimal(18, 2),
    severity_max Enum8('LOW' = 1, 'MEDIUM' = 2, 'HIGH' = 3, 'CRITICAL' = 4) DEFAULT 'MEDIUM',
    confidence_avg Float64,
    earliest_alert_timestamp Int64,
    latest_alert_timestamp Int64,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY (network, toYYYYMM(processing_date))
ORDER BY (processing_date, network, cluster_id)
SETTINGS index_granularity = 8192;
```

### `storage/schema/miner_alert_scores.sql`

```sql
-- Computed alert scores
CREATE TABLE IF NOT EXISTS miner_alert_scores (
    processing_date Date,
    network String,
    alert_id String,
    score Float64,
    model_version String,
    latency_ms Float64,
    explain_json String DEFAULT '',
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY (network, toYYYYMM(processing_date))
ORDER BY (processing_date, network, alert_id)
SETTINGS index_granularity = 8192;

-- Index for score-based queries
CREATE INDEX IF NOT EXISTS idx_score ON miner_alert_scores(score) TYPE minmax GRANULARITY 4;
```

### `storage/schema/miner_alert_rankings.sql`

```sql
-- Computed alert rankings
CREATE TABLE IF NOT EXISTS miner_alert_rankings (
    processing_date Date,
    network String,
    alert_id String,
    rank Int32,
    model_version String,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY (network, toYYYYMM(processing_date))
ORDER BY (processing_date, network, rank)
SETTINGS index_granularity = 8192;
```

### `storage/schema/miner_cluster_scores.sql`

```sql
-- Computed cluster scores
CREATE TABLE IF NOT EXISTS miner_cluster_scores (
    processing_date Date,
    network String,
    cluster_id String,
    score Float64,
    model_version String,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY (network, toYYYYMM(processing_date))
ORDER BY (processing_date, network, cluster_id)
SETTINGS index_granularity = 8192;
```

### `storage/schema/miner_batch_metadata.sql`

```sql
-- Batch processing metadata
CREATE TABLE IF NOT EXISTS miner_batch_metadata (
    processing_date Date,
    network String,
    processed_at DateTime,
    input_counts Nested(
        alerts Int32,
        features Int32,
        clusters Int32
    ),
    output_counts Nested(
        alert_scores Int32,
        alert_rankings Int32,
        cluster_scores Int32
    ),
    latencies_ms Nested(
        alert_scoring Int32,
        alert_ranking Int32,
        cluster_scoring Int32,
        total Int32
    ),
    model_versions Nested(
        alert_scorer String,
        alert_ranker String,
        cluster_scorer String
    ),
    status Enum8('PROCESSING' = 1, 'COMPLETED' = 2, 'FAILED' = 3) DEFAULT 'PROCESSING',
    error_message String DEFAULT '',
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY (network, toYYYYMM(processing_date))
ORDER BY (processing_date, network)
SETTINGS index_granularity = 8192;
```

---

## Base Repository

### `storage/repositories/base_repository.py`

```python
from abc import ABC, abstractmethod
import time
from clickhouse_connect.driver import Client


class BaseRepository(ABC):
    def __init__(self, client: Client, partition_id: int = None):
        self.client = client
        self.partition_id = partition_id
    
    def _generate_version(self) -> int:
        base_version = int(time.time() * 1000000)
        if self.partition_id is not None:
            return base_version + self.partition_id
        return base_version
    
    @classmethod
    @abstractmethod
    def schema(cls) -> str:
        pass
    
    @classmethod
    @abstractmethod
    def table_name(cls) -> str:
        pass
```

---

## Storage Utilities

### `storage/utils.py`

```python
from typing import List, Dict, Any, Type, TypeVar, Union
from enum import IntEnum
from pydantic import BaseModel

T = TypeVar('T', bound=BaseModel)


def row_to_dict(row: tuple, column_names: List[str]) -> Dict:
    return dict(zip(column_names, row))


def convert_clickhouse_enum(enum_class: Type[IntEnum], value: Any) -> IntEnum:
    if value is None:
        return None
    
    if isinstance(value, enum_class):
        return value
    
    if isinstance(value, int):
        try:
            return enum_class(value)
        except ValueError:
            pass
    
    if isinstance(value, str) and value.isdigit():
        try:
            return enum_class(int(value))
        except ValueError:
            pass
    
    if isinstance(value, str):
        try:
            return enum_class[value.upper()]
        except KeyError:
            pass
    
    available_values = [f"{e.name}({e.value})" for e in enum_class]
    raise ValueError(
        f"Cannot convert '{value}' (type: {type(value).__name__}) to {enum_class.__name__}. "
        f"Available values: {', '.join(available_values)}"
    )


def clickhouse_row_to_pydantic(
    model_class: Type[T],
    row_data: Union[Dict[str, Any], tuple],
    column_names: List[str] = None,
    enum_fields: Dict[str, Type[IntEnum]] = None
) -> T:
    if isinstance(row_data, tuple):
        if not column_names:
            raise ValueError("column_names required when row_data is a tuple")
        row_dict = row_to_dict(row_data, column_names)
    else:
        row_dict = row_data.copy()
    
    if enum_fields:
        for field_name, enum_class in enum_fields.items():
            if field_name in row_dict:
                row_dict[field_name] = convert_clickhouse_enum(enum_class, row_dict[field_name])
    
    return model_class(**row_dict)


def rows_to_pydantic_list(
    model_class: Type[T],
    rows: List[Union[Dict[str, Any], tuple]],
    column_names: List[str] = None,
    enum_fields: Dict[str, Type[IntEnum]] = None
) -> List[T]:
    return [
        clickhouse_row_to_pydantic(model_class, row, column_names, enum_fields)
        for row in rows
    ]
```

---

## Client Factory & Migration

### `storage/__init__.py`

```python
import io
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Iterable
from clickhouse_connect import get_client
from clickhouse_connect.driver import Client
from clickhouse_connect.driver.exceptions import ClickHouseError
from loguru import logger


def get_connection_params(network: str) -> dict:
    connection_params = {
        "host": os.getenv(f"{network.upper()}_CLICKHOUSE_HOST", "localhost"),
        "port": os.getenv(f"{network.upper()}_CLICKHOUSE_PORT", "8123"),
        "database": os.getenv(f"{network.upper()}_CLICKHOUSE_DATABASE", f"miner_{network.lower()}"),
        "user": os.getenv(f"{network.upper()}_CLICKHOUSE_USER", "default"),
        "password": os.getenv(f"{network.upper()}_CLICKHOUSE_PASSWORD", ""),
        "max_execution_time": int(os.getenv(f"{network.upper()}_CLICKHOUSE_MAX_EXECUTION_TIME", "1800")),
        "max_query_size": int(os.getenv(f"{network.upper()}_CLICKHOUSE_MAX_QUERY_SIZE", "5000000")),
    }
    return connection_params


def create_database(connection_params: dict):
    client = get_client(
        host=connection_params['host'],
        port=int(connection_params['port']),
        username=connection_params['user'],
        password=connection_params['password'],
        database='default',
        settings={
            'enable_http_compression': 1,
            'send_progress_in_http_headers': 0,
            'http_headers_progress_interval_ms': 1000,
            'http_zlib_compression_level': 3
        }
    )
    
    client.command(f"CREATE DATABASE IF NOT EXISTS {connection_params['database']}")


def apply_schema(client: Client, schema: str, replacements: dict = None):
    def _split_clickhouse_sql(sql_text: str) -> Iterable[str]:
        cleaned = io.StringIO()
        for line in sql_text.splitlines():
            if line.strip().startswith("--"):
                continue
            parts = line.split("--", 1)
            cleaned.write(parts[0] + "\n")
        
        buf = []
        for ch in cleaned.getvalue():
            if ch == ";":
                stmt = "".join(buf).strip()
                if stmt:
                    yield stmt
                buf = []
            else:
                buf.append(ch)
        
        tail = "".join(buf).strip()
        if tail:
            yield tail
    
    schema_path = Path(__file__).resolve().parent / "schema" / schema
    if not schema_path.exists():
        raise FileNotFoundError(f"schema {schema} does not exist")
    
    raw = schema_path.read_text(encoding="utf-8")
    
    statements = list(_split_clickhouse_sql(raw))
    if not statements:
        return
    
    for stmt in statements:
        client.command(stmt)


class ClientFactory:
    def __init__(self, connection_params: dict):
        self.connection_params = connection_params
        self.client = None
    
    def _get_client(self) -> Client:
        self.client = get_client(
            host=self.connection_params['host'],
            port=int(self.connection_params['port']),
            username=self.connection_params['user'],
            password=self.connection_params['password'],
            database=self.connection_params['database'],
            settings={
                'output_format_parquet_compression_method': 'zstd'
            }
        )
        return self.client
    
    @contextmanager
    def client_context(self) -> Iterator[Client]:
        client = self._get_client()
        try:
            yield client
        except ClickHouseError as e:
            import traceback
            logger.error(
                "ClickHouse error",
                error=e,
                traceback=traceback.format_exc(),
            )
            raise
        finally:
            if client:
                client.close()


class MigrateSchema:
    def __init__(self, client: Client):
        self.client = client
    
    def create_database(self, database_name: str):
        try:
            self.client.command(f"CREATE DATABASE IF NOT EXISTS {database_name}")
        except Exception as e:
            logger.error(f"Error creating database {database_name}: {str(e)}")
            raise
    
    def run_miner_migrations(self):
        miner_schemas = [
            "miner_raw_alerts.sql",
            "miner_raw_features.sql",
            "miner_raw_clusters.sql",
            "miner_alert_scores.sql",
            "miner_alert_rankings.sql",
            "miner_cluster_scores.sql",
            "miner_batch_metadata.sql",
        ]
        
        for schema_file in miner_schemas:
            apply_schema(self.client, schema_file)
            logger.info(f"Executed {schema_file}")
```

---

## Repository Implementations

### `storage/repositories/scores_repository.py`

```python
from typing import List, Optional
import pandas as pd
from pydantic import BaseModel
from loguru import logger

from aml_miner.storage.repositories.base_repository import BaseRepository
from aml_miner.storage.utils import rows_to_pydantic_list


class AlertScore(BaseModel):
    processing_date: str
    network: str
    alert_id: str
    score: float
    model_version: str
    latency_ms: float
    explain_json: str = ""


class ScoresRepository(BaseRepository):
    @classmethod
    def schema(cls) -> str:
        return "miner_alert_scores.sql"
    
    @classmethod
    def table_name(cls) -> str:
        return "miner_alert_scores"
    
    def insert_scores(self, scores_df: pd.DataFrame, processing_date: str, network: str):
        scores_df['processing_date'] = processing_date
        scores_df['network'] = network
        
        self.client.insert_df(
            self.table_name(),
            scores_df
        )
        
        logger.info(f"Inserted {len(scores_df)} scores for {processing_date}/{network}")
    
    def get_scores(self, processing_date: str, network: str) -> List[AlertScore]:
        query = f'''
            SELECT processing_date, network, alert_id, score, model_version, latency_ms, explain_json
            FROM {self.table_name()}
            WHERE processing_date = %(date)s AND network = %(network)s
            ORDER BY score DESC
        '''
        
        result = self.client.query(query, {'date': processing_date, 'network': network})
        
        return rows_to_pydantic_list(
            AlertScore,
            result.result_rows,
            result.column_names
        )
    
    def get_latest_date(self, network: str) -> Optional[str]:
        query = f'''
            SELECT max(processing_date) as latest
            FROM {self.table_name()}
            WHERE network = %(network)s
        '''
        
        result = self.client.query(query, {'network': network})
        
        if result.result_rows and result.result_rows[0][0]:
            return str(result.result_rows[0][0])
        
        return None
    
    def get_available_dates(self, network: str) -> List[str]:
        query = f'''
            SELECT DISTINCT processing_date
            FROM {self.table_name()}
            WHERE network = %(network)s
            ORDER BY processing_date DESC
        '''
        
        result = self.client.query(query, {'network': network})
        
        return [str(row[0]) for row in result.result_rows]
```

---

## Usage Example

### Initialize Storage

```python
from aml_miner.storage import get_connection_params, create_database, ClientFactory, MigrateSchema

# Get connection params for network
network = "ethereum"
connection_params = get_connection_params(network)

# Create database
create_database(connection_params)

# Run migrations
client_factory = ClientFactory(connection_params)
with client_factory.client_context() as client:
    migrator = MigrateSchema(client)
    migrator.run_miner_migrations()
```

### Use Repository

```python
from aml_miner.storage.repositories.scores_repository import ScoresRepository

# Create repository
with client_factory.client_context() as client:
    scores_repo = ScoresRepository(client)
    
    # Insert scores
    scores_repo.insert_scores(scores_df, '2025-10-26', 'ethereum')
    
    # Get scores
    scores = scores_repo.get_scores('2025-10-26', 'ethereum')
    
    # Get latest date
    latest = scores_repo.get_latest_date('ethereum')
```

---

## Summary

**Structure:**
- SQL schemas in `storage/schema/`
- Repository pattern in `storage/repositories/`
- Connection management in `storage/__init__.py`
- Utilities in `storage/utils.py`

**Benefits:**
- ✅ Clean separation of concerns
- ✅ Testable (can mock repositories)
- ✅ Schema versioning via SQL files
- ✅ Type-safe with Pydantic models
- ✅ Production-ready error handling