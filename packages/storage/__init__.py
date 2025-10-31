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
        "host": os.getenv(f"CLICKHOUSE_HOST", "localhost"),
        "port": os.getenv(f"CLICKHOUSE_PORT", "8123"),
        "database": os.getenv(f"CLICKHOUSE_DATABASE",  f"risk_scoring_{network.lower()}"),
        "user": os.getenv(f"CLICKHOUSE_USER", "default"),
        "password": os.getenv(f"CLICKHOUSE_PASSWORD", 'miner'),
        "max_execution_time": os.getenv("CLICKHOUSE_MAX_EXECUTION_TIME", "1800")
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
    logger.info(f"Database {connection_params['database']} created/verified")


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
        raise FileNotFoundError(f"schema {schema} does not exist at {schema_path}")
    
    raw = schema_path.read_text(encoding="utf-8")
    
    statements = list(_split_clickhouse_sql(raw))
    if not statements:
        return
    
    for stmt in statements:
        client.command(stmt)
    
    logger.info(f"Schema {schema} applied successfully")


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
                'output_format_parquet_compression_method': 'zstd',
                'async_insert': 0,
                'wait_for_async_insert': 1,
                'max_execution_time': 300,
                'max_query_size': 100000
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

    def run_migrations(self):
        schemas = [
            "raw_alerts.sql",
            "raw_features.sql",
            "raw_clusters.sql",
            "raw_money_flows.sql",
            "raw_address_labels.sql",
            "alert_scores.sql",
            "alert_rankings.sql",
            "cluster_scores.sql",
            "batch_metadata.sql",
            "trained_models.sql",
        ]
        
        for schema_file in schemas:
            try:
                apply_schema(self.client, schema_file)
            except Exception as e:
                logger.error(f"Error applying schema {schema_file}: {e}")
                raise