# Training Labels Architecture for Miner Template

## Current Situation

### What We Have
- SOT provides unlabeled data (alerts, features, clusters, money flows, address labels)
- SOT uses **unsupervised learning** (Isolation Forest) for anomaly detection
- Template includes supervised learning pipeline expecting `label` or `ground_truth` columns
- No ground truth labels exist in current SOT data

### The Challenge
The miner template is designed as an extensible framework where:
- Multiple miners will build upon this foundation
- Each miner may have different data sources and capabilities
- Some miners may have access to labeled datasets, others may not
- The template should support both supervised and unsupervised approaches

---

## Architectural Options Analysis

### Option 1: Miner-Provided Labeled Datasets

**Description:** Each miner maintains their own labeled datasets and extends the ingestion pipeline to include labels.

#### Pros
- Maximum flexibility for miners with proprietary labeled data
- Enables true supervised learning with validated ground truth
- Miners can compete based on quality of their labeled datasets
- Better model performance for miners with good labels
- Supports different labeling schemas per miner

#### Cons
- Creates inequality between miners (those with labels vs without)
- Requires additional infrastructure for label management
- No shared benchmark dataset across miners
- Harder to validate/compare miner performance objectively
- Initial miners start with zero labeled data (cold start problem)

#### Implementation Requirements
- Add `label` column to `raw_alerts` schema (optional, nullable)
- Create label ingestion pipeline separate from SOT ingestion
- Document labeling guidelines/schema
- Provide example labeled dataset (even if small)
- Support both labeled and unlabeled training modes

---

### Option 2: Hybrid Supervised/Unsupervised Approach

**Description:** Use unsupervised learning (like SOT's Isolation Forest) as baseline, with optional supervised fine-tuning when labels become available.

#### Pros
- Works immediately without labels (using SOT's approach)
- Allows gradual transition to supervised as labels accumulate
- Miners can start competing immediately
- Supports semi-supervised learning techniques
- Can use SOT's anomaly scores as pseudo-labels

#### Cons
- More complex training pipeline (two modes)
- Pseudo-labels may introduce bias
- Harder to explain which approach is being used
- Performance metrics differ between supervised/unsupervised

#### Implementation Requirements
- Modify training pipeline to support both modes
- Use SOT's anomaly scores as initial pseudo-labels
- Create semi-supervised learning techniques (self-training, pseudo-labeling)
- Implement active learning for gradual label collection
- Clear documentation on mode selection

---

### Option 3: Synthetic/Proxy Labels from SOT Data

**Description:** Derive proxy labels from existing SOT data attributes (severity, confidence scores, anomaly scores).

#### Pros
- Works immediately with existing data
- No additional data collection needed
- Enables supervised learning techniques
- Consistent across all miners
- Can be validated/improved over time

#### Cons
- Proxy labels may not represent true ground truth
- Risk of learning artifacts from SOT's own biases
- Limited by SOT's current labeling logic
- May perpetuate existing model limitations
- Harder to improve beyond SOT's baseline

#### Implementation Requirements
- Define proxy label derivation logic (e.g., `severity >= 'high'` = positive)
- Document assumptions and limitations
- Add confidence weighting based on `alert_confidence_score`
- Enable easy swapping when real labels become available
- Clear metrics showing proxy label quality

---

### Option 4: Shared Community-Labeled Dataset

**Description:** Create a shared, community-validated labeled dataset that all miners can use as baseline.

#### Pros
- Level playing field for all miners
- Objective performance comparison
- Community effort improves dataset quality
- Reduces cold start problem
- Enables standardized benchmarking

#### Cons
- Requires coordination and governance
- Initial dataset creation effort
- May not cover all edge cases
- Static dataset may become outdated
- Potential for gaming/overfitting

#### Implementation Requirements
- Create initial seed dataset (even 100-1000 labeled samples)
- Establish labeling guidelines and validation process
- Build community review mechanism
- Version control for dataset updates
- Keep dataset separate from template (as external resource)

---

### Option 5: Empty Template with Multiple Training Strategies

**Description:** Provide template with pluggable training strategies, let each miner choose based on their capabilities.

#### Pros
- Maximum flexibility
- No assumptions about miner capabilities
- Supports innovation in approach
- Clear separation of concerns
- Easy to extend with new strategies

#### Cons
- More complex initial setup
- Requires more documentation
- Miners may be confused about best approach
- Harder to provide default working example
- More code to maintain

#### Implementation Requirements
- Abstract `TrainingStrategy` base class
- Implement multiple strategies:
  - `SupervisedStrategy` (requires labels)
  - `UnsupervisedStrategy` (Isolation Forest)
  - `SemiSupervisedStrategy` (hybrid)
  - `ProxyLabelStrategy` (use severity/confidence)
- Factory pattern for strategy selection
- Clear examples for each strategy

---

## Recommended Hybrid Solution

### Phase 1: Immediate (Start with Proxy Labels)
Use **Option 3** as the default to get started quickly:

```python
# Derive proxy labels from SOT data
def derive_proxy_labels(alerts_df):
    """
    Proxy label strategy:
    - Positive (1): severity in ['high', 'critical'] AND confidence >= 0.7
    - Negative (0): severity in ['low', 'medium'] OR confidence < 0.3
    - Uncertain (ignore): everything else
    """
    positive_mask = (
        (alerts_df['severity'].isin(['high', 'critical'])) &
        (alerts_df['alert_confidence_score'] >= 0.7)
    )
    negative_mask = (
        (alerts_df['severity'].isin(['low', 'medium'])) |
        (alerts_df['alert_confidence_score'] < 0.3)
    )
    
    alerts_df['label'] = -1  # uncertain
    alerts_df.loc[positive_mask, 'label'] = 1
    alerts_df.loc[negative_mask, 'label'] = 0
    
    # Only train on confident samples
    return alerts_df[alerts_df['label'] != -1]
```

**Advantages:**
- Works immediately with existing SOT data
- Provides supervised learning baseline
- Clear, documented limitations
- Easy to replace later

### Phase 2: Enable Miner Extensions (Support Custom Labels)
Implement **Option 1** as extension mechanism:

```sql
-- Add optional label columns to raw_alerts
ALTER TABLE raw_alerts ADD COLUMN IF NOT EXISTS label Int8 DEFAULT NULL;
ALTER TABLE raw_alerts ADD COLUMN IF NOT EXISTS label_source String DEFAULT '';
ALTER TABLE raw_alerts ADD COLUMN IF NOT EXISTS label_confidence Float32 DEFAULT NULL;
ALTER TABLE raw_alerts ADD COLUMN IF NOT EXISTS labeled_at DateTime DEFAULT NULL;
```

**Advantages:**
- Miners can provide their own labels
- Labels tracked with source and confidence
- Backward compatible (NULL = use proxy labels)
- Supports gradual improvement

### Phase 3: Strategy Pattern (Long-term Flexibility)
Implement **Option 5** for maximum flexibility:

```python
class TrainingStrategy(ABC):
    @abstractmethod
    def prepare_labels(self, data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.Series]:
        pass

class ProxyLabelStrategy(TrainingStrategy):
    def prepare_labels(self, data):
        # Use severity + confidence as proxy
        
class SupervisedStrategy(TrainingStrategy):
    def prepare_labels(self, data):
        # Require explicit labels
        
class UnsupervisedStrategy(TrainingStrategy):
    def prepare_labels(self, data):
        # No labels, use Isolation Forest
```

---

## Implementation Roadmap

### Immediate Actions (This Sprint)
1. **Fix Current Error**: Make label column optional in feature builder
2. **Add Proxy Label Logic**: Implement severity/confidence-based labeling
3. **Document Approach**: Clear README explaining proxy label strategy
4. **Add Configuration**: Allow switching strategies via config

### Short-term (Next 2-3 Sprints)
1. **Extend Schema**: Add optional label columns to raw_alerts
2. **Label Ingestion**: Create pipeline for custom miner labels
3. **Validation Tools**: Scripts to validate label quality
4. **Example Dataset**: Create 100-500 hand-labeled samples as example

### Long-term (Future)
1. **Strategy Pattern**: Refactor to support multiple training approaches
2. **Semi-supervised**: Implement pseudo-labeling and self-training
3. **Active Learning**: Tools for intelligent sample selection
4. **Community Dataset**: Coordinate shared labeled dataset if miners agree

---

## Recommendations

### For Template (Now)
Use **Proxy Labels** as default with clear documentation:
- Derive labels from `severity` + `alert_confidence_score`
- Document this as "baseline proxy labels"
- Make it easy to override with real labels
- Show metrics for both proxy and (future) real labels

### For Miners (Future)
Support **Miner-Provided Labels** as extension:
- Add optional label columns to schema
- Provide label ingestion utilities
- Document labeling guidelines
- Enable competition based on label quality + model performance

### For Community (Long-term)
Consider **Shared Dataset** if ecosystem grows:
- Coordinate community labeling effort
- Create validation/review process
- Version control labeled datasets
- Use for benchmarking and comparison

---

## Schema Changes Required

```sql
-- Extend raw_alerts to support optional labels
ALTER TABLE raw_alerts 
ADD COLUMN IF NOT EXISTS label Int8 DEFAULT NULL,
ADD COLUMN IF NOT EXISTS label_source Enum8('proxy' = 1, 'miner' = 2, 'community' = 3, 'validated' = 4) DEFAULT 'proxy',
ADD COLUMN IF NOT EXISTS label_confidence Float32 DEFAULT NULL,
ADD COLUMN IF NOT EXISTS labeled_at DateTime DEFAULT NULL,
ADD COLUMN IF NOT EXISTS labeled_by String DEFAULT '';

-- Index for filtering labeled data
CREATE INDEX IF NOT EXISTS idx_label ON raw_alerts(label) TYPE set(0) GRANULARITY 4;
CREATE INDEX IF NOT EXISTS idx_label_source ON raw_alerts(label_source) TYPE set(0) GRANULARITY 4;
```

---

## Code Changes Required

### 1. Update FeatureExtractor
```python
# packages/training/feature_extraction.py
def _extract_alerts(self, start_date, end_date, window_days):
    query = f"""
        SELECT
            alert_id,
            address,
            processing_date,
            window_days,
            typology_type,
            -- ... other columns ...
            label,  -- Add this
            label_source,
            label_confidence
        FROM raw_alerts
        WHERE processing_date >= '{start_date}'
          AND processing_date <= '{end_date}'
          AND window_days = {window_days}
    """
```

### 2. Update FeatureBuilder
```python
# packages/training/feature_builder.py
def build_training_features(self, data):
    X = data['alerts'].copy()
    
    # Check if labels exist, otherwise derive proxy labels
    if 'label' not in X.columns or X['label'].isna().all():
        logger.warning("No explicit labels found, deriving proxy labels")
        X = self._derive_proxy_labels(X)
    
    y = X['label']
    # ... rest of code ...

def _derive_proxy_labels(self, df):
    """Derive proxy labels from severity and confidence"""
    positive_mask = (
        (df['severity'].isin(['high', 'critical'])) &
        (df['alert_confidence_score'] >= 0.7)
    )
    negative_mask = (
        (df['severity'].isin(['low', 'medium'])) |
        (df['alert_confidence_score'] < 0.3)
    )
    
    df['label'] = -1
    df.loc[positive_mask, 'label'] = 1
    df.loc[negative_mask, 'label'] = 0
    df['label_source'] = 'proxy'
    
    # Filter uncertain samples
    df = df[df['label'] != -1].copy()
    
    logger.info(
        f"Derived proxy labels: {(df['label']==1).sum()} positive, "
        f"{(df['label']==0).sum()} negative"
    )
    
    return df
```

### 3. Add Configuration
```yaml
# alert_scoring/config/model_config.yaml
training:
  label_strategy: "proxy"  # Options: proxy, supervised, unsupervised
  proxy_label_config:
    positive_severity: ["high", "critical"]
    positive_confidence_threshold: 0.7
    negative_severity: ["low", "medium"]
    negative_confidence_threshold: 0.3
```

---

## Decision Matrix

| Criterion | Proxy Labels | Miner Labels | Hybrid | Community | Strategy Pattern |
|-----------|--------------|--------------|--------|-----------|------------------|
| **Immediate Usability** | ✅ High | ❌ Low | ⚠️ Medium | ❌ Low | ⚠️ Medium |
| **Long-term Flexibility** | ❌ Low | ✅ High | ✅ High | ⚠️ Medium | ✅ High |
| **Implementation Complexity** | ✅ Low | ⚠️ Medium | ❌ High | ❌ High | ❌ High |
| **Miner Competition Fairness** | ✅ High | ❌ Low | ⚠️ Medium | ✅ High | ⚠️ Medium |
| **Model Performance Potential** | ⚠️ Medium | ✅ High | ✅ High | ⚠️ Medium | ✅ High |
| **Maintenance Burden** | ✅ Low | ⚠️ Medium | ❌ High | ⚠️ Medium | ❌ High |

**Legend:** ✅ Good | ⚠️ Moderate | ❌ Challenging

---

## Conclusion

**Recommended Approach: Phased Implementation**

1. **Now (Sprint 1):** Implement proxy labels to unblock training pipeline
2. **Soon (Sprint 2-3):** Add optional miner label support for customization  
3. **Later (Sprint 4+):** Implement strategy pattern for maximum flexibility

This balances immediate needs (getting training working) with long-term goals (enabling miner innovation and competition).