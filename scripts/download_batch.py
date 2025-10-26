import argparse
from pathlib import Path
from datetime import datetime, timedelta
import json
from loguru import logger


def download_batch_from_sot(network: str, processing_date: str, output_dir: Path) -> bool:
    date_dir = output_dir / processing_date
    date_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading batch for {network} on {processing_date}")
    logger.info(f"Output directory: {date_dir}")
    
    logger.warning("SOT download not yet implemented - this is a placeholder")
    logger.info("For testing, manually place Parquet files in input/{processing_date}/")
    logger.info("Required files:")
    logger.info(f"  - {date_dir}/alerts.parquet")
    logger.info(f"  - {date_dir}/features.parquet")
    logger.info(f"  - {date_dir}/clusters.parquet (optional)")
    
    metadata = {
        'network': network,
        'processing_date': processing_date,
        'downloaded_at': datetime.utcnow().isoformat(),
        'source': 'SOT (placeholder)',
        'status': 'manual_upload_required'
    }
    
    with open(date_dir / 'download_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return False


def download_date_range(network: str, start_date: str, end_date: str, 
                        output_dir: Path) -> dict:
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    if start > end:
        raise ValueError(f"Start date {start_date} is after end date {end_date}")
    
    results = {
        'success': [],
        'failed': [],
        'total_days': 0
    }
    
    current = start
    while current <= end:
        processing_date = current.strftime('%Y-%m-%d')
        
        try:
            success = download_batch_from_sot(network, processing_date, output_dir)
            
            if success:
                results['success'].append(processing_date)
            else:
                results['failed'].append(processing_date)
            
            results['total_days'] += 1
            
        except Exception as e:
            logger.error(f"Error downloading {processing_date}: {e}")
            results['failed'].append(processing_date)
        
        current += timedelta(days=1)
    
    logger.info(f"Download summary: {len(results['success'])} successful, {len(results['failed'])} failed")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Download batch data from SOT')
    parser.add_argument('--network', default='ethereum', 
                       choices=['ethereum', 'bitcoin', 'polygon'],
                       help='Blockchain network')
    parser.add_argument('--processing-date', help='Single date to download (YYYY-MM-DD)')
    parser.add_argument('--start-date', help='Start date for range download (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date for range download (YYYY-MM-DD)')
    parser.add_argument('--days', type=int, help='Number of days to download (backward from today)')
    parser.add_argument('--output-dir', default='input', help='Output directory for downloaded data')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.processing_date:
        try:
            datetime.strptime(args.processing_date, '%Y-%m-%d')
        except ValueError:
            logger.error(f"Invalid date format: {args.processing_date}. Expected YYYY-MM-DD")
            return 1
        
        success = download_batch_from_sot(args.network, args.processing_date, output_dir)
        return 0 if success else 1
    
    elif args.start_date and args.end_date:
        try:
            datetime.strptime(args.start_date, '%Y-%m-%d')
            datetime.strptime(args.end_date, '%Y-%m-%d')
        except ValueError as e:
            logger.error(f"Invalid date format: {e}")
            return 1
        
        results = download_date_range(args.network, args.start_date, args.end_date, output_dir)
        
        return 0 if len(results['failed']) == 0 else 1
    
    elif args.days:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days - 1)
        
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        logger.info(f"Downloading last {args.days} days: {start_date_str} to {end_date_str}")
        
        results = download_date_range(args.network, start_date_str, end_date_str, output_dir)
        
        return 0 if len(results['failed']) == 0 else 1
    
    else:
        logger.error("Must specify either --processing-date, --start-date/--end-date, or --days")
        parser.print_help()
        return 1


if __name__ == "__main__":
    exit(main())