#!/usr/bin/env python3
"""
Add positive training labels to raw_address_labels table.

This script creates training labels from high-severity alerts to ensure
the cluster_scorer has both positive and negative examples to learn from.

Usage:
    python scripts/add_training_labels.py --network torus --processing-date 2025-08-01
"""

import argparse
import pandas as pd
from datetime import datetime
from loguru import logger
from packages.storage import ClientFactory, get_connection_params


def create_labels_from_alerts(
    client,
    network: str,
    processing_date: str,
    window_days: int = 195,
    min_confidence: float = 0.6,
    target_positive_count: int = 30
) -> pd.DataFrame:
    
    logger.info(
        f"Creating training labels from high-severity alerts",
        extra={
            "network": network,
            "processing_date": processing_date,
            "min_confidence": min_confidence,
            "target_count": target_positive_count
        }
    )
    
    query = f"""
        SELECT DISTINCT
            '{processing_date}' as processing_date,
            {window_days} as window_days,
            '{network}' as network,
            address,
            CASE 
                WHEN severity = 'critical' THEN 'critical'
                WHEN severity = 'high' THEN 'high'
                ELSE 'medium'
            END as risk_level,
            alert_confidence_score as confidence_score,
            'auto_labeled_from_alerts' as source,
            now() as labeled_at
        FROM raw_alerts
        WHERE processing_date = '{processing_date}'
          AND window_days = {window_days}
          AND network = '{network}'
          AND severity IN ('high', 'critical')
          AND alert_confidence_score >= {min_confidence}
        ORDER BY alert_confidence_score DESC, severity DESC
        LIMIT {target_positive_count}
    """
    
    result = client.query(query)
    
    if not result.result_rows:
        logger.warning("No high-severity alerts found to create labels from")
        return pd.DataFrame()
    
    df = pd.DataFrame(
        result.result_rows,
        columns=[col[0] for col in result.column_names]
    )
    
    logger.success(
        f"Created {len(df)} positive training labels",
        extra={
            "high": (df['risk_level'] == 'high').sum(),
            "critical": (df['risk_level'] == 'critical').sum()
        }
    )
    
    return df


def check_existing_labels(
    client,
    network: str,
    processing_date: str,
    window_days: int = 195
) -> dict:
    
    query = f"""
        SELECT 
            risk_level,
            COUNT(*) as count
        FROM raw_address_labels
        WHERE processing_date = '{processing_date}'
          AND window_days = {window_days}
          AND network = '{network}'
        GROUP BY risk_level
        ORDER BY risk_level
    """
    
    result = client.query(query)
    
    distribution = {}
    for row in result.result_rows:
        distribution[row[0]] = row[1]
    
    total = sum(distribution.values())
    high_critical = distribution.get('high', 0) + distribution.get('critical', 0)
    positive_rate = high_critical / total if total > 0 else 0.0
    
    logger.info(
        "Current label distribution",
        extra={
            "total": total,
            "low": distribution.get('low', 0),
            "medium": distribution.get('medium', 0),
            "high": distribution.get('high', 0),
            "critical": distribution.get('critical', 0),
            "positive_rate": f"{positive_rate:.1%}"
        }
    )
    
    return distribution


def insert_labels(client, labels_df: pd.DataFrame):
    
    if labels_df.empty:
        logger.warning("No labels to insert")
        return
    
    records = labels_df.to_dict('records')
    
    client.execute(
        'INSERT INTO raw_address_labels VALUES',
        records
    )
    
    logger.success(f"Inserted {len(records)} training labels into raw_address_labels")


def main():
    parser = argparse.ArgumentParser(
        description="Add positive training labels for cluster_scorer"
    )
    parser.add_argument(
        '--network',
        required=True,
        help='Network name (e.g., torus)'
    )
    parser.add_argument(
        '--processing-date',
        required=True,
        help='Processing date in YYYY-MM-DD format'
    )
    parser.add_argument(
        '--window-days',
        type=int,
        default=195,
        help='Window days (default: 195)'
    )
    parser.add_argument(
        '--min-confidence',
        type=float,
        default=0.6,
        help='Minimum alert confidence score (default: 0.6)'
    )
    parser.add_argument(
        '--target-count',
        type=int,
        default=30,
        help='Target number of positive labels to add (default: 30)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Skip confirmation prompt'
    )
    
    args = parser.parse_args()
    
    logger.info(
        "Starting label creation",
        extra={
            "network": args.network,
            "processing_date": args.processing_date,
            "window_days": args.window_days
        }
    )
    
    connection_params = get_connection_params(args.network)
    client_factory = ClientFactory(connection_params)
    
    with client_factory.client_context() as client:
        logger.info("Checking current label distribution")
        existing = check_existing_labels(
            client,
            args.network,
            args.processing_date,
            args.window_days
        )
        
        high_critical_count = existing.get('high', 0) + existing.get('critical', 0)
        
        if high_critical_count > 0 and not args.force:
            logger.info(
                f"Found {high_critical_count} existing positive labels. "
                "Use --force to add more anyway."
            )
            response = input("Continue adding more labels? (y/n): ")
            if response.lower() != 'y':
                logger.info("Cancelled by user")
                return
        
        labels_df = create_labels_from_alerts(
            client,
            args.network,
            args.processing_date,
            args.window_days,
            args.min_confidence,
            args.target_count
        )
        
        if labels_df.empty:
            logger.error("No labels created. Check that high-severity alerts exist.")
            return
        
        logger.info(
            "About to insert labels",
            extra={
                "count": len(labels_df),
                "high": (labels_df['risk_level'] == 'high').sum(),
                "critical": (labels_df['risk_level'] == 'critical').sum()
            }
        )
        
        if not args.force:
            response = input("Proceed with insertion? (y/n): ")
            if response.lower() != 'y':
                logger.info("Cancelled by user")
                return
        
        insert_labels(client, labels_df)
        
        logger.info("Checking updated label distribution")
        check_existing_labels(
            client,
            args.network,
            args.processing_date,
            args.window_days
        )
        
        logger.success(
            "Training labels added successfully. "
            "You can now retrain the cluster_scorer model."
        )
        logger.info(
            "Next step: python scripts/train_model.py "
            f"--network {args.network} "
            f"--start-date {args.processing_date} "
            f"--end-date {args.processing_date} "
            f"--window-days {args.window_days} "
            "--model-type cluster_scorer"
        )


if __name__ == '__main__':
    main()