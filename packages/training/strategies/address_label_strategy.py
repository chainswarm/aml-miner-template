from typing import Dict
import pandas as pd
from loguru import logger
from .base import LabelStrategy


class AddressLabelStrategy(LabelStrategy):
    
    def __init__(
        self,
        positive_risk_levels: list = None,
        negative_risk_levels: list = None,
        use_confidence_weights: bool = True
    ):
        self.positive_risk_levels = positive_risk_levels or ['high', 'critical']
        self.negative_risk_levels = negative_risk_levels or ['low', 'medium']
        self.use_confidence_weights = use_confidence_weights
    
    def derive_labels(
        self,
        alerts_df: pd.DataFrame,
        data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        
        logger.info("Deriving labels from address_labels (SOT baseline)")
        
        labels_df = data.get('address_labels')
        if labels_df is None or labels_df.empty:
            raise ValueError("No address_labels data available")
        
        label_map = {}
        confidence_map = {}
        
        for _, row in labels_df.iterrows():
            addr = row['address']
            risk = row['risk_level'].lower()
            confidence = row.get('confidence_score', 1.0)
            
            if risk in self.positive_risk_levels:
                label_map[addr] = 1
                confidence_map[addr] = confidence
            elif risk in self.negative_risk_levels:
                label_map[addr] = 0
                confidence_map[addr] = confidence
        
        alerts_df['label'] = alerts_df['address'].map(label_map)
        alerts_df['label_confidence'] = alerts_df['address'].map(confidence_map)
        alerts_df['label_source'] = alerts_df['address'].map(
            lambda x: 'sot_address_labels' if x in label_map else None
        )
        
        num_labeled = alerts_df['label'].notna().sum()
        num_positive = (alerts_df['label'] == 1).sum()
        num_negative = (alerts_df['label'] == 0).sum()
        
        logger.info(
            f"Labeled {num_labeled}/{len(alerts_df)} alerts: "
            f"{num_positive} positive, {num_negative} negative"
        )
        
        return alerts_df
    
    def validate_labels(self, alerts_df: pd.DataFrame) -> bool:
        
        if 'label' not in alerts_df.columns:
            logger.error("No 'label' column found")
            return False
        
        labeled = alerts_df['label'].notna().sum()
        if labeled == 0:
            logger.error("No labeled samples found")
            return False
        
        unique_labels = alerts_df['label'].dropna().unique()
        if not set(unique_labels).issubset({0, 1}):
            logger.error(f"Invalid labels found: {unique_labels}")
            return False
        
        logger.info("Label validation passed")
        return True
    
    def get_label_weights(self, alerts_df: pd.DataFrame) -> pd.Series:
        
        if self.use_confidence_weights and 'label_confidence' in alerts_df.columns:
            weights = alerts_df['label_confidence'].fillna(1.0)
            logger.info(f"Using confidence-based sample weights (range: {weights.min():.2f}-{weights.max():.2f})")
            return weights
        
        return pd.Series(1.0, index=alerts_df.index)