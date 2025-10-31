from clickhouse_connect.driver import Client
from typing import List, Dict, Any, Optional
from packages.storage.utils import row_to_dict


def get_latest_date(client: Client) -> Optional[str]:
    query = """
    SELECT processing_date
    FROM batch_metadata
    WHERE status = 'COMPLETED'
    ORDER BY processing_date DESC
    LIMIT 1
    """
    result = client.query(query)
    if not result.result_rows:
        return None
    return str(result.result_rows[0][0])


def get_available_dates(client: Client) -> List[str]:
    query = """
    SELECT DISTINCT processing_date
    FROM batch_metadata
    WHERE status = 'COMPLETED'
    ORDER BY processing_date DESC
    """
    result = client.query(query)
    return [str(row[0]) for row in result.result_rows]


def get_batch_metadata(client: Client, processing_date: str) -> Optional[Dict[str, Any]]:
    query = """
    SELECT *
    FROM batch_metadata
    WHERE processing_date = {date:Date}
    """
    result = client.query(query, parameters={'date': processing_date})
    
    if not result.result_rows:
        return None
    
    return row_to_dict(result.result_rows[0], result.column_names)


def get_alert_scores(client: Client, processing_date: str) -> List[Dict[str, Any]]:
    query = """
    SELECT
        alert_id,
        score
    FROM alert_scores
    WHERE processing_date = {date:Date}
    ORDER BY alert_id
    """
    result = client.query(query, parameters={'date': processing_date})
    
    return [
        {
            'alert_id': row[0],
            'score': float(row[1])
        }
        for row in result.result_rows
    ]


def get_alert_rankings(client: Client, processing_date: str) -> List[Dict[str, Any]]:
    query = """
    SELECT
        r.alert_id,
        r.rank,
        s.score
    FROM alert_rankings r
    LEFT JOIN alert_scores s ON r.alert_id = s.alert_id AND r.processing_date = s.processing_date
    WHERE r.processing_date = {date:Date}
    ORDER BY r.rank
    """
    result = client.query(query, parameters={'date': processing_date})
    
    return [
        {
            'alert_id': row[0],
            'rank': int(row[1]),
            'score': float(row[2]) if row[2] is not None else 0.0
        }
        for row in result.result_rows
    ]


def get_cluster_scores(client: Client, processing_date: str) -> List[Dict[str, Any]]:
    query = """
    SELECT
        cluster_id,
        score
    FROM cluster_scores
    WHERE processing_date = {date:Date}
    ORDER BY cluster_id
    """
    result = client.query(query, parameters={'date': processing_date})
    
    return [
        {
            'cluster_id': row[0],
            'score': float(row[1])
        }
        for row in result.result_rows
    ]


def get_network_statistics(client: Client) -> Dict[str, Any]:
    query = """
    SELECT 
        COUNT(DISTINCT processing_date) as total_dates,
        MIN(processing_date) as first_date,
        MAX(processing_date) as last_date,
        SUM(output_counts_alert_scores) as total_alerts_scored,
        SUM(output_counts_cluster_scores) as total_clusters_scored
    FROM batch_metadata
    WHERE status = 'COMPLETED'
    """
    result = client.query(query)
    
    if not result.result_rows:
        return {
            'total_dates': 0,
            'first_date': None,
            'last_date': None,
            'total_alerts_scored': 0,
            'total_clusters_scored': 0,
            'avg_alerts_per_day': 0
        }
    
    row = result.result_rows[0]
    total_dates = row[0]
    total_alerts = row[3] or 0
    
    return {
        'total_dates': total_dates,
        'first_date': str(row[1]) if row[1] else None,
        'last_date': str(row[2]) if row[2] else None,
        'total_alerts_scored': int(total_alerts),
        'total_clusters_scored': int(row[4] or 0),
        'avg_alerts_per_day': int(total_alerts / total_dates) if total_dates > 0 else 0
    }