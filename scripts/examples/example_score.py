"""Developer entry point for risk scoring with predefined parameters"""
import sys

if __name__ == "__main__":
    sys.argv = [
        'score_batch.py',
        '--network', 'torus',
        '--processing-date', '2025-08-01',
        '--window-days', '195'
    ]
    
    from scripts.score_batch import main
    main()