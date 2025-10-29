import argparse
import os
from loguru import logger

from alert_scoring.storage import get_connection_params, create_database, ClientFactory, MigrateSchema


def main():
    parser = argparse.ArgumentParser(description='Initialize ClickHouse database for alert scoring')
    parser.add_argument('--network', default='ethereum', help='Network name (e.g., ethereum, bitcoin, polygon)')
    parser.add_argument('--host', help='ClickHouse host (overrides env)')
    parser.add_argument('--port', type=int, help='ClickHouse port (overrides env)')
    parser.add_argument('--user', help='ClickHouse user (overrides env)')
    parser.add_argument('--password', help='ClickHouse password (overrides env)')
    parser.add_argument('--database', help='Database name (overrides env)')
    
    args = parser.parse_args()
    
    connection_params = get_connection_params(args.network)
    
    if args.host:
        connection_params['host'] = args.host
    if args.port:
        connection_params['port'] = args.port
    if args.user:
        connection_params['user'] = args.user
    if args.password:
        connection_params['password'] = args.password
    if args.database:
        connection_params['database'] = args.database
    
    logger.info(f"Initializing database for network: {args.network}")
    logger.info(f"ClickHouse host: {connection_params['host']}:{connection_params['port']}")
    logger.info(f"Database: {connection_params['database']}")
    
    try:
        logger.info("Creating database...")
        create_database(connection_params)
        
        logger.info("Running migrations...")
        client_factory = ClientFactory(connection_params)
        with client_factory.client_context() as client:
            migrator = MigrateSchema(client)
            migrator.run_migrations()
        
        logger.info(f"Database successfully initialized for {args.network}")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize database: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())