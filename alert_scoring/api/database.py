import duckdb
from pathlib import Path
from typing import Optional, List
import pandas as pd
from loguru import logger


class ResultsDatabase:
    def __init__(self, output_dir: str = "output/"):
        self.output_dir = Path(output_dir)
        self.conn = duckdb.connect(":memory:")
        self._register_parquet_files()
    
    def _register_parquet_files(self):
        alert_scores_pattern = str(self.output_dir / "*/alert_scores.parquet")
        alert_rankings_pattern = str(self.output_dir / "*/alert_rankings.parquet")
        cluster_scores_pattern = str(self.output_dir / "*/cluster_scores.parquet")
        
        try:
            self.conn.execute(f"""
                CREATE OR REPLACE VIEW alert_scores AS
                SELECT 
                    *,
                    CAST(regexp_extract(filename, '[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}', 0) AS DATE) as processing_date
                FROM read_parquet('{alert_scores_pattern}', filename=true, union_by_name=true)
            """)
            logger.info(f"Registered alert_scores view from {alert_scores_pattern}")
        except Exception as e:
            logger.warning(f"Could not register alert_scores view: {e}")
        
        try:
            self.conn.execute(f"""
                CREATE OR REPLACE VIEW alert_rankings AS
                SELECT 
                    *,
                    CAST(regexp_extract(filename, '[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}', 0) AS DATE) as processing_date
                FROM read_parquet('{alert_rankings_pattern}', filename=true, union_by_name=true)
            """)
            logger.info(f"Registered alert_rankings view from {alert_rankings_pattern}")
        except Exception as e:
            logger.warning(f"Could not register alert_rankings view: {e}")
        
        try:
            self.conn.execute(f"""
                CREATE OR REPLACE VIEW cluster_scores AS
                SELECT 
                    *,
                    CAST(regexp_extract(filename, '[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}', 0) AS DATE) as processing_date
                FROM read_parquet('{cluster_scores_pattern}', filename=true, union_by_name=true)
            """)
            logger.info(f"Registered cluster_scores view from {cluster_scores_pattern}")
        except Exception as e:
            logger.warning(f"Could not register cluster_scores view: {e}")
    
    def get_alert_scores(self, processing_date: str) -> pd.DataFrame:
        try:
            result = self.conn.execute(
                "SELECT * FROM alert_scores WHERE processing_date = ?",
                [processing_date]
            ).df()
            
            if 'processing_date' in result.columns:
                result = result.drop(columns=['processing_date'])
            
            logger.info(f"Retrieved {len(result)} alert scores for {processing_date}")
            return result
        
        except Exception as e:
            logger.error(f"Error retrieving alert scores for {processing_date}: {e}")
            return pd.DataFrame()
    
    def get_alert_rankings(self, processing_date: str) -> pd.DataFrame:
        try:
            result = self.conn.execute(
                "SELECT * FROM alert_rankings WHERE processing_date = ? ORDER BY rank",
                [processing_date]
            ).df()
            
            if 'processing_date' in result.columns:
                result = result.drop(columns=['processing_date'])
            
            logger.info(f"Retrieved {len(result)} alert rankings for {processing_date}")
            return result
        
        except Exception as e:
            logger.error(f"Error retrieving alert rankings for {processing_date}: {e}")
            return pd.DataFrame()
    
    def get_cluster_scores(self, processing_date: str) -> pd.DataFrame:
        try:
            result = self.conn.execute(
                "SELECT * FROM cluster_scores WHERE processing_date = ?",
                [processing_date]
            ).df()
            
            if 'processing_date' in result.columns:
                result = result.drop(columns=['processing_date'])
            
            logger.info(f"Retrieved {len(result)} cluster scores for {processing_date}")
            return result
        
        except Exception as e:
            logger.error(f"Error retrieving cluster scores for {processing_date}: {e}")
            return pd.DataFrame()
    
    def get_available_dates(self) -> List[str]:
        try:
            result = self.conn.execute(
                "SELECT DISTINCT processing_date FROM alert_scores ORDER BY processing_date DESC"
            ).df()
            
            dates = result['processing_date'].astype(str).tolist()
            logger.info(f"Found {len(dates)} available processing dates")
            return dates
        
        except Exception as e:
            logger.error(f"Error retrieving available dates: {e}")
            return []
    
    def get_date_metadata(self, processing_date: str) -> Optional[dict]:
        metadata_path = self.output_dir / processing_date / "processing_metadata.json"
        
        if not metadata_path.exists():
            logger.warning(f"Metadata file not found: {metadata_path}")
            return None
        
        try:
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            logger.info(f"Retrieved metadata for {processing_date}")
            return metadata
        
        except Exception as e:
            logger.error(f"Error reading metadata for {processing_date}: {e}")
            return None
    
    def get_latest_date(self) -> Optional[str]:
        dates = self.get_available_dates()
        return dates[0] if dates else None
    
    def refresh(self):
        logger.info("Refreshing database views...")
        self._register_parquet_files()
    
    def close(self):
        self.conn.close()
        logger.info("Database connection closed")