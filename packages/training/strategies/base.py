from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any
import pandas as pd
import numpy as np


class LabelStrategy(ABC):
    
    @abstractmethod
    def derive_labels(
        self,
        alerts_df: pd.DataFrame,
        data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        
        pass
    
    @abstractmethod
    def validate_labels(self, alerts_df: pd.DataFrame) -> bool:
        
        pass
    
    def get_label_weights(self, alerts_df: pd.DataFrame) -> pd.Series:
        
        return pd.Series(1.0, index=alerts_df.index)


class ModelTrainer(ABC):
    
    @abstractmethod
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: pd.Series = None
    ) -> Any:
        
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        
        pass
    
    @abstractmethod
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, float]:
        
        pass
    
    @abstractmethod
    def save(self, path: str):
        
        pass
    
    @abstractmethod
    def load(self, path: str):
        
        pass