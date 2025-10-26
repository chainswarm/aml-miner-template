import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List
import json
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif


class FeatureSelector:
    
    def __init__(self, max_features: int = 100):
        self.max_features = max_features
        self.selected_features: Optional[List[str]] = None
        self.feature_importance: Optional[pd.DataFrame] = None
        self.correlation_matrix: Optional[pd.DataFrame] = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> None:
        logger.info(f"Fitting feature selector on {X.shape[1]} features")
        
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_features) == 0:
            raise ValueError("No numeric features found in input DataFrame")
        
        X_numeric = X[numeric_features]
        
        if y is not None:
            logger.info("Computing feature importance with supervised method")
            importances = self._compute_supervised_importance(X_numeric, y)
        else:
            logger.info("Computing feature importance with unsupervised method")
            importances = self._compute_unsupervised_importance(X_numeric)
        
        self.correlation_matrix = X_numeric.corr().abs()
        
        redundant_features = self._identify_redundant_features(self.correlation_matrix, threshold=0.95)
        
        importances_filtered = importances[~importances['feature'].isin(redundant_features)]
        
        top_features = importances_filtered.nlargest(self.max_features, 'importance')['feature'].tolist()
        
        self.selected_features = top_features
        self.feature_importance = importances_filtered
        
        logger.info(f"Selected {len(self.selected_features)} features out of {len(numeric_features)}")
        logger.info(f"Removed {len(redundant_features)} redundant features")
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.selected_features is None:
            raise ValueError("Feature selector has not been fitted. Call fit() first.")
        
        available_features = [f for f in self.selected_features if f in X.columns]
        
        if len(available_features) < len(self.selected_features):
            missing = set(self.selected_features) - set(available_features)
            logger.warning(f"Missing {len(missing)} features in input DataFrame: {missing}")
        
        X_selected = X[available_features].copy()
        
        logger.info(f"Transformed to {X_selected.shape[1]} selected features")
        return X_selected
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)
    
    def get_feature_importance(self) -> pd.DataFrame:
        if self.feature_importance is None:
            raise ValueError("Feature selector has not been fitted. Call fit() first.")
        
        return self.feature_importance.sort_values('importance', ascending=False).reset_index(drop=True)
    
    def save_selected_features(self, path: Path) -> None:
        if self.selected_features is None:
            raise ValueError("Feature selector has not been fitted. Call fit() first.")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        feature_data = {
            'selected_features': self.selected_features,
            'max_features': self.max_features,
            'feature_importance': self.feature_importance.to_dict('records') if self.feature_importance is not None else []
        }
        
        with open(path, 'w') as f:
            json.dump(feature_data, f, indent=2)
        
        logger.info(f"Saved {len(self.selected_features)} selected features to {path}")
    
    def load_selected_features(self, path: Path) -> None:
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Feature file not found: {path}")
        
        with open(path, 'r') as f:
            feature_data = json.load(f)
        
        self.selected_features = feature_data['selected_features']
        self.max_features = feature_data['max_features']
        
        if feature_data['feature_importance']:
            self.feature_importance = pd.DataFrame(feature_data['feature_importance'])
        
        logger.info(f"Loaded {len(self.selected_features)} selected features from {path}")
    
    def _compute_supervised_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        X_clean = X.fillna(0)
        X_clean = X_clean.replace([np.inf, -np.inf], 0)
        
        rf = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_clean, y)
        
        importances = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        })
        
        return importances
    
    def _compute_unsupervised_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        X_clean = X.fillna(0)
        X_clean = X_clean.replace([np.inf, -np.inf], 0)
        
        variance = X_clean.var()
        
        variance_normalized = (variance - variance.min()) / (variance.max() - variance.min() + 1e-10)
        
        importances = pd.DataFrame({
            'feature': X.columns,
            'importance': variance_normalized.values
        })
        
        return importances
    
    def _identify_redundant_features(
        self,
        corr_matrix: pd.DataFrame,
        threshold: float = 0.95
    ) -> List[str]:
        redundant = set()
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > threshold:
                    redundant.add(corr_matrix.columns[j])
        
        return list(redundant)