"""Developer entry point for data ingestion with predefined parameters"""
import sys

if __name__ == "__main__":
    sys.argv = [
        'ingest_data.py',
        '--network', 'torus',
        '--processing-date', '2025-08-01',
        '--days', '195'
    ]
    
    from scripts.ingest_data import main
    main()