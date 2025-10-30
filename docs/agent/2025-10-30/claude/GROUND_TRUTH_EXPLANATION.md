# Ground Truth in T+τ Validation - Explained

**Date**: 2025-10-30  
**Purpose**: Explain what "Ground Truth" means in the context of T+τ validation

---

## What is T+τ?

### Timeline Notation
- **T** = Today (when alert was scored)
- **τ** = Tau = Time delay (typically 7-30 days)
- **T+τ** = T plus tau = Some days after scoring

### Example
```
Day 0 (T):     Miner scores alert_001 = 0.87 (high risk)
Day 7 (T+7):   Ground truth becomes available
Day 14 (T+14): More ground truth accumulates
Day 30 (T+30): Full ground truth picture emerges
```

---

## What is Ground Truth?

**Ground Truth** = Real-world confirmations of whether an alert was actually illicit or benign.

### Why Do We Need It?

At **T+0** (when miner scores alerts), we don't know if predictions are correct:
```
Day 0:
├─ alert_001: score=0.87  ← Miner predicts HIGH RISK
├─ alert_002: score=0.23  ← Miner predicts LOW RISK
└─ alert_003: score=0.65  ← Miner predicts MEDIUM RISK

Question: Are these predictions accurate? 🤷
```

At **T+τ** (days/weeks later), real-world events reveal truth:
```
Day 7-30:
├─ alert_001: SAR filed, confirmed money laundering    ✅ Prediction was correct!
├─ alert_002: No issues found, was legitimate transfer ✅ Prediction was correct!
└─ alert_003: Turned out to be benign                  ❌ Should have been lower score

Now we can measure: How accurate were the predictions?
```

---

## Sources of Ground Truth

### 1. SAR Filings (Suspicious Activity Reports)
**What**: Financial institutions file SARs with regulators when they detect suspicious activity

**Timeline**:
- Day 0: Alert generated
- Day 1-30: Investigation period
- Day 30: SAR filed (or not)

**Example**:
```python
ground_truth = {
  'alert_001': {
    'sar_filed': True,        # Bank filed SAR
    'sar_date': '2025-11-05',
    'confirmed_illicit': True
  }
}
```

### 2. Exchange Labels
**What**: Cryptocurrency exchanges label addresses they've identified as risky

**Timeline**:
- Day 0: Alert generated
- Day 7-14: Exchange investigates
- Day 14+: Address labeled (mixer, scam, etc.)

**Example**:
```python
ground_truth = {
  'alert_002': {
    'exchange_label': 'mixer',
    'labeled_by': 'Binance',
    'label_date': '2025-11-08',
    'confirmed_illicit': True
  }
}
```

### 3. Blockchain Forensics
**What**: Analysis of subsequent transactions reveals illicit patterns

**Timeline**:
- Day 0: Alert generated
- Day 1-30: Monitor address activity
- Day 30+: Pattern confirmed (e.g., funds moved to known mixer)

**Example**:
```python
ground_truth = {
  'alert_003': {
    'flows_to_mixer': True,
    'mixer_address': 'tornado_cash',
    'flow_date': '2025-11-10',
    'confirmed_illicit': True
  }
}
```

### 4. Law Enforcement Actions
**What**: Government seizures, arrests, sanctions

**Timeline**:
- Day 0: Alert generated
- Day 30-90: Investigation
- Day 90+: Official action taken

**Example**:
```python
ground_truth = {
  'alert_004': {
    'sanctioned': True,
    'sanctioning_authority': 'OFAC',
    'sanction_date': '2025-12-01',
    'confirmed_illicit': True
  }
}
```

### 5. Negative Confirmations (Benign)
**What**: Evidence that an alert was a false positive

**Timeline**:
- Day 0: Alert generated
- Day 7-30: No suspicious activity observed
- Day 30: Confirmed benign

**Example**:
```python
ground_truth = {
  'alert_005': {
    'subsequent_activity': 'normal_business',
    'kyc_verified': True,
    'investigation_closed': '2025-11-01',
    'confirmed_illicit': False  # False positive
  }
}
```

---

## Ground Truth Data Structure

### Complete Example

```python
# Ground truth for batch 2025-10-26 (collected by T+30)
ground_truth_2025_10_26 = {
  'alerts': {
    'alert_001': {
      'confirmed_illicit': True,
      'confirmation_type': 'sar_filing',
      'confirmation_date': '2025-11-05',
      'confidence': 0.95
    },
    'alert_002': {
      'confirmed_illicit': False,
      'confirmation_type': 'investigation_closed',
      'confirmation_date': '2025-11-12',
      'confidence': 0.85
    },
    'alert_003': {
      'confirmed_illicit': True,
      'confirmation_type': 'exchange_label',
      'confirmation_date': '2025-11-08',
      'confidence': 0.90
    }
  },
  
  'clusters': {
    'cluster_001': {
      'confirmed_illicit': True,
      'confirmation_type': 'network_analysis',
      'description': 'Coordinated layering scheme confirmed',
      'confidence': 0.92
    }
  }
}
```

### In ClickHouse Format

```sql
CREATE TABLE ground_truth (
    alert_id String,
    processing_date Date,
    confirmed_illicit Bool,           -- The key label!
    confirmation_type String,          -- How we know (SAR, exchange, etc.)
    confirmation_date Date,            -- When we found out
    confidence Float32,                -- How sure we are
    evidence_json String              -- Supporting evidence
)
ENGINE = MergeTree()
ORDER BY (processing_date, alert_id);
```

---

## How Ground Truth is Used in Validation

### Step 1: Miner Scores Alerts (Day 0)
```python
# Day 0: 2025-10-26
miner_scores = {
  'alert_001': 0.87,  # Predicted HIGH RISK
  'alert_002': 0.23,  # Predicted LOW RISK
  'alert_003': 0.65   # Predicted MEDIUM RISK
}
```

### Step 2: Ground Truth Collected (Day 0 → Day 30)
```python
# Days 1-30: Real-world events happen
# Day 30: We have ground truth
ground_truth = {
  'alert_001': True,   # Was actually illicit (SAR filed)
  'alert_002': False,  # Was actually benign
  'alert_003': True    # Was actually illicit (exchange labeled)
}
```

### Step 3: Validation Metrics Computed (Day 30)
```python
from sklearn.metrics import roc_auc_score, average_precision_score

y_true = [1, 0, 1]  # Ground truth
y_pred = [0.87, 0.23, 0.65]  # Miner predictions

# Compute metrics
auc_roc = roc_auc_score(y_true, y_pred)
# Result: 0.875 - Good discrimination!

auc_pr = average_precision_score(y_true, y_pred)
# Result: 0.823 - Good precision-recall!

# Ground truth score (out of 0.5 max)
ground_truth_score = 0.3 * auc_roc + 0.2 * auc_pr
# Result: 0.428 out of 0.5
```

### Step 4: Final Miner Score
```python
# Immediate validation (Day 0)
immediate_score = 0.48  # (integrity + behavior)

# Ground truth validation (Day 30)
ground_truth_score = 0.428

# Final score
final_score = immediate_score + ground_truth_score
# Result: 0.908 out of 1.0

# Miner performed well! 90.8% score
```

---

## Why τ (Time Delay)?

### The Fundamental Problem
**You can't know if a prediction was correct immediately**

```
Day 0:
├─ Miner says: "This address is money laundering (90% sure)"
└─ Reality: We don't know yet! Need time to observe what happens.

Day 30:
├─ If SAR was filed → Prediction was correct ✅
└─ If no issues found → Prediction was wrong ❌
```

### Common τ Values

| τ Value | Use Case | Pros | Cons |
|---------|----------|------|------|
| **7 days** | Fast feedback | Quick iteration | Fewer confirmations |
| **14 days** | Balanced | Good coverage | Medium wait |
| **30 days** | Comprehensive | Most confirmations | Slow feedback |
| **90 days** | Law enforcement | Full picture | Very slow |

### Multiple τ Windows (Recommended)

```python
# Validate at multiple timepoints
validation_windows = {
  'T+7':  {'weight': 0.2, 'tau': 7},   # Early signals
  'T+14': {'weight': 0.3, 'tau': 14},  # Medium-term
  'T+30': {'weight': 0.5, 'tau': 30}   # Long-term
}

# Weighted average of validations
final_ground_truth_score = (
  0.2 * validate_at_tau_7() +
  0.3 * validate_at_tau_14() +
  0.5 * validate_at_tau_30()
)
```

---

## Ground Truth Quality Considerations

### High-Quality Ground Truth
✅ **Multiple independent confirmations**
```python
{
  'alert_001': {
    'sar_filed': True,           # Bank confirmed
    'exchange_labeled': True,     # Binance confirmed
    'flows_to_mixer': True,       # On-chain confirmed
    'confirmed_illicit': True,
    'confidence': 0.98            # High confidence
  }
}
```

### Low-Quality Ground Truth
⚠️ **Single weak signal**
```python
{
  'alert_002': {
    'no_further_activity': True,  # Absence of evidence
    'confirmed_illicit': False,   # Assumed benign
    'confidence': 0.60            # Low confidence
  }
}
```

### Missing Ground Truth
❌ **Not all alerts get confirmed**
```python
# Some alerts never get ground truth:
- No SAR filed (but could still be illicit)
- No subsequent activity (dormant address)
- Investigation ongoing (no conclusion yet)

# These alerts are excluded from validation:
ground_truth = {
  'alert_003': None,  # No ground truth available
  # Excluded from AUC calculation
}
```

---

## Ground Truth in Our System

### Current Implementation

We use **address labels** as proxy ground truth:

```python
# raw_address_labels table contains:
{
  'address': '0xabc...',
  'label': 'mixer',              # Known bad
  'risk_level': 'critical',      # Ground truth label
  'source': 'chainalysis',       # Where it came from
  'confidence_score': 0.95       # How sure we are
}
```

### Label → Ground Truth Mapping

```python
# For training:
def derive_ground_truth(alert, address_labels):
    label = address_labels.get(alert['address'])
    
    if label is None:
        return None  # No ground truth
    
    # Map risk level to binary label
    if label['risk_level'] in ['high', 'critical']:
        return 1  # Illicit
    else:
        return 0  # Benign
```

### Why This Works

**Address labels are retrospective confirmations**:
- Exchanges label addresses after observing illicit behavior
- Chainalysis labels after forensic analysis
- Regulators sanction after investigations

**These are real-world outcomes** that occurred after the original suspicious activity!

---

## T+τ Validation Workflow

### Complete Timeline

```
┌─────────────────────────────────────────────────────────────┐
│ Day 0 (October 26, 2025) - T+0                              │
├─────────────────────────────────────────────────────────────┤
│ 01:00 UTC: Batch 2025-10-26 published                      │
│ 01:05 UTC: Miners score alerts                             │
│            alert_001 → 0.87                                 │
│            alert_002 → 0.23                                 │
│            alert_003 → 0.65                                 │
│                                                             │
│ 01:30 UTC: Validator runs immediate validation             │
│            ├─ Integrity: Pass (0.20 pts)                   │
│            └─ Behavior: Pass (0.28 pts)                    │
│            Immediate Score: 0.48/0.5                        │
└─────────────────────────────────────────────────────────────┘
                         ↓
                    (Wait τ days)
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ Day 7 (November 2, 2025) - T+7                             │
├─────────────────────────────────────────────────────────────┤
│ Early ground truth available:                              │
│ ├─ alert_001: SAR filed → confirmed_illicit=True          │
│ └─ alert_002: No issues → confirmed_illicit=False         │
│                                                             │
│ Validator validates (partial):                             │
│ └─ AUC-ROC: 0.75 (based on 2/3 alerts with GT)           │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ Day 30 (November 25, 2025) - T+30                          │
├─────────────────────────────────────────────────────────────┤
│ Complete ground truth available:                           │
│ ├─ alert_001: SAR filed → confirmed_illicit=True          │
│ ├─ alert_002: Investigation closed → confirmed_illicit=False│
│ └─ alert_003: Exchange labeled mixer → confirmed_illicit=True│
│                                                             │
│ Validator validates (complete):                            │
│ ├─ AUC-ROC: 0.8421                                        │
│ ├─ AUC-PR: 0.7834                                         │
│ └─ Ground Truth Score: 0.42/0.5                           │
│                                                             │
│ Final Miner Score: 0.48 + 0.42 = 0.90                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Ground Truth Collection Methods

### Method 1: SAR Filing Database
```python
# Banks/FIs report to regulators
SAR_DATABASE = {
  'alert_001': {
    'filing_date': '2025-11-05',
    'filing_institution': 'Bank XYZ',
    'activity_type': 'money_laundering',
    'confirmed': True
  }
}
```

### Method 2: Exchange Labels API
```python
# Query major exchanges for address labels
import requests

response = requests.get(
    'https://api.exchange.com/v1/address-labels',
    params={'address': '0xabc123...'}
)

labels = response.json()
# {
#   'label': 'mixer',
#   'confidence': 0.95,
#   'last_updated': '2025-11-08'
# }
```

### Method 3: Blockchain Analysis
```python
# Trace funds forward in time
def trace_forward(address, days=30):
    transactions = get_transactions(address, after=alert_date)
    
    # Check if funds go to known bad actors
    for tx in transactions:
        if tx['to_address'] in KNOWN_MIXERS:
            return {'confirmed_illicit': True, 'reason': 'flows_to_mixer'}
        if tx['to_address'] in SANCTIONED_ADDRESSES:
            return {'confirmed_illicit': True, 'reason': 'sanctioned_destination'}
    
    return {'confirmed_illicit': False, 'reason': 'normal_activity'}
```

### Method 4: Manual Investigation
```python
# Analyst reviews and labels
ANALYST_LABELS = {
  'alert_005': {
    'reviewed_by': 'analyst_smith',
    'review_date': '2025-11-15',
    'conclusion': 'legitimate_business',
    'confirmed_illicit': False
  }
}
```

---

## Ground Truth Data Format

### Alert-Level Ground Truth
```python
# Stored in ClickHouse or similar
ground_truth_alerts = pd.DataFrame([
  {
    'alert_id': 'alert_001',
    'processing_date': '2025-10-26',
    'confirmed_illicit': True,
    'confirmation_type': 'sar_filing',
    'confirmation_date': '2025-11-05',
    'confidence': 0.95,
    'evidence': '{"sar_id": "SAR-2025-12345", "institution": "Bank XYZ"}'
  },
  {
    'alert_id': 'alert_002',
    'processing_date': '2025-10-26',
    'confirmed_illicit': False,
    'confirmation_type': 'investigation_closed',
    'confirmation_date': '2025-11-12',
    'confidence': 0.80,
    'evidence': '{"reason": "legitimate_business_transfer"}'
  }
])
```

### Cluster-Level Ground Truth
```python
ground_truth_clusters = pd.DataFrame([
  {
    'cluster_id': 'cluster_001',
    'processing_date': '2025-10-26',
    'confirmed_illicit': True,
    'confirmation_type': 'network_forensics',
    'description': 'Coordinated layering scheme confirmed',
    'confidence': 0.92
  }
])
```

---

## Validation Calculation

### Step-by-Step Example

**Miner Predictions (Day 0)**:
```python
predictions = {
  'alert_001': 0.87,
  'alert_002': 0.23,
  'alert_003': 0.65,
  'alert_004': 0.92,
  'alert_005': 0.31
}
```

**Ground Truth (Day 30)**:
```python
ground_truth = {
  'alert_001': True,   # Actually illicit
  'alert_002': False,  # Actually benign
  'alert_003': True,   # Actually illicit
  'alert_004': True,   # Actually illicit
  'alert_005': False   # Actually benign
}
```

**Validation Metrics**:
```python
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

y_true = [1, 0, 1, 1, 0]
y_pred = [0.87, 0.23, 0.65, 0.92, 0.31]

# AUC-ROC: How well does the model separate classes?
auc_roc = roc_auc_score(y_true, y_pred)
# Result: 0.9167 - Excellent!

# AUC-PR: Precision-recall tradeoff
auc_pr = average_precision_score(y_true, y_pred)
# Result: 0.8821 - Excellent!

# F1 Score (at best threshold)
y_pred_binary = (y_pred > 0.60).astype(int)  # Best threshold found
f1 = f1_score(y_true, y_pred_binary)
# Result: 0.8571

# Ground truth score (out of 0.5 max)
gt_score = 0.3 * auc_roc + 0.2 * auc_pr
# Result: 0.275 + 0.176 = 0.451
```

---

## Challenges with Ground Truth

### 1. Incomplete Coverage
Not all alerts get ground truth:
```python
# Out of 10,000 alerts:
ground_truth_available = 2,500  # Only 25%
no_ground_truth = 7,500         # 75% unknown

# Validation only uses the 2,500 with GT
# Results may not represent full performance
```

### 2. Time Delay
Ground truth takes time to accumulate:
```python
# Immediate feedback would be better, but:
τ=7:  Only ~30% of ground truth available
τ=14: ~60% of ground truth available
τ=30: ~85% of ground truth available
τ=90: ~95% of ground truth available

# Trade-off: Speed vs Completeness
```

### 3. Label Noise
Ground truth is not perfect:
```python
# Example: SAR was filed, but alert was actually benign
{
  'alert_006': {
    'confirmed_illicit': True,   # SAR filed (label=1)
    'actual_truth': False,       # But was false positive
    'confidence': 0.70           # Medium confidence
  }
}

# Use confidence scores to weight samples
```

### 4. Class Imbalance
Most alerts are benign:
```python
# Typical distribution
ground_truth = {
  'positive_samples': 250,   # Illicit (10%)
  'negative_samples': 2250   # Benign (90%)
}

# Need to handle imbalance in validation:
- Use AUC-PR (better for imbalanced data)
- Weighted F1 score
- Precision@K metrics
```

---

## Ground Truth in Our Training System

### How We Use It

Currently, we use **address labels as ground truth**:

```python
# packages/training/feature_builder.py
def _derive_labels_from_address_labels(self, alerts_df, address_labels_df):
    # Join alerts with address_labels
    merged = alerts_df.merge(
        address_labels_df[['address', 'risk_level', 'confidence_score']],
        on='address',
        how='left'
    )
    
    # Derive binary labels
    def map_risk_to_label(risk_level):
        if risk_level in ['high', 'critical']:
            return 1  # Illicit
        elif risk_level in ['low', 'medium']:
            return 0  # Benign
        else:
            return None  # Unknown
    
    merged['label'] = merged['risk_level'].apply(map_risk_to_label)
    
    # Filter to labeled samples only
    labeled = merged[merged['label'].notna()]
    
    return labeled
```

### Why Address Labels Work as Ground Truth

Address labels represent **retrospective confirmations**:

1. **Exchange labeled mixer** → Exchange investigated and confirmed
2. **Chainalysis labeled scam** → Forensic analysis confirmed
3. **OFAC sanctioned** → Government investigation confirmed
4. **Community labeled** → Multiple sources confirmed

These are **real-world outcomes** that happened after suspicious activity was detected!

---

## Summary

### What is Ground Truth?
Real-world confirmations (T+τ days after scoring) that reveal whether predictions were correct.

### Sources
- SAR filings
- Exchange labels
- Blockchain forensics
- Law enforcement actions
- Manual investigations

### Why T+τ?
Need time for:
- Investigations to complete
- SARs to be filed
- Patterns to emerge
- Evidence to accumulate

### How It's Used
```python
# Validate miner predictions against reality
auc_roc = roc_auc_score(
  y_true=ground_truth,      # What actually happened
  y_pred=miner_predictions  # What miner predicted
)

# Reward miners for accurate predictions
miner_score = immediate_validation + ground_truth_validation
#              (T+0, 0.5 pts)        (T+τ, 0.5 pts)
```

### In Our System
We use **address labels** from SOT as ground truth proxy:
- Labels are retrospective (already confirmed)
- Risk levels map to binary labels
- Confidence scores weight samples
- Enables supervised learning immediately

**Ground truth is the reality check that proves whether ML predictions were accurate!**