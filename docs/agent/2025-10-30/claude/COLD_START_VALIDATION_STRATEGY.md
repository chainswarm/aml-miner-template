# Cold Start Validation Strategy

**Date**: 2025-10-30  
**Purpose**: How to validate miners when NO ground truth exists (subnet cold start)

---

## The Cold Start Problem

### Scenario: Subnet Launch (Day 0)

```
Day 0: Subnet launches
├─ No historical ground truth
├─ No SAR filings available yet
├─ No confirmed outcomes
└─ BUT miners need to be scored to distribute rewards!

Question: How do we validate without ground truth? 🤔
```

### Why This Happens

**At subnet launch:**
- ✅ We have raw alerts from blockchain data
- ✅ We have features and clusters
- ✅ Miners can generate scores
- ❌ We have NO confirmations yet (need to wait τ days)
- ❌ Can't compute AUC-ROC without labels
- ❌ Can't reward miners for accuracy

**This is a critical problem!**

---

## Solution: Immediate Validation Only (0.5 points)

### Validation Score Breakdown

```
┌──────────────────────────────────────────────────────────┐
│           VALIDATION SCORING (TOTAL = 1.0)                │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  IMMEDIATE VALIDATION (Available Day 0)                  │
│  ├─ Tier 1: Integrity (0.2 pts)                         │
│  │  └─ Schema, completeness, latency, determinism       │
│  └─ Tier 2: Behavior (0.3 pts)                          │
│     └─ Gaming detection, pattern traps                  │
│  SUBTOTAL: 0.5 pts ← CAN SCORE IMMEDIATELY               │
│                                                           │
│  GROUND TRUTH VALIDATION (Available T+τ)                 │
│  └─ Tier 3: Accuracy (0.5 pts)                          │
│     └─ AUC-ROC, AUC-PR against real outcomes            │
│  SUBTOTAL: 0.5 pts ← NEED TO WAIT                        │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

### During Cold Start (Months 1-3)

**Miners can earn up to 0.5 points** from immediate validation alone:

```python
# Day 0: No ground truth exists
miner_score = immediate_validation_only()

# Breakdown:
integrity_score = 0.20  # Pass all technical checks
behavior_score = 0.30   # No gaming detected

total_score = 0.50  # Maximum possible during cold start
```

---

## Tier 1: Integrity Validation (0.2 points)

### What We Can Validate Immediately

**No ground truth needed!** These are technical/format checks:

```python
class IntegrityValidator:
    def validate(self, scores_df, alerts_df):
        checks = {}
        
        # 1. Completeness: All alerts scored?
        checks['completeness'] = len(scores_df) == len(alerts_df)
        
        # 2. Schema: Correct columns?
        required = ['alert_id', 'score', 'model_version', 'latency_ms']
        checks['schema'] = all(col in scores_df.columns for col in required)
        
        # 3. Score Range: All in [0, 1]?
        checks['score_range'] = scores_df['score'].between(0, 1).all()
        
        # 4. Latency: Under threshold?
        checks['latency'] = scores_df['latency_ms'].mean() < 100
        
        # 5. Determinism: Same input → same output?
        # (Run same batch twice, compare results)
        checks['determinism'] = True  # Checked via re-scoring
        
        if all(checks.values()):
            return {'passed': True, 'score': 0.20}
        else:
            return {'passed': False, 'score': 0.00}
```

**Examples of violations:**
- ❌ Miner scores only 9,000 of 10,000 alerts (incomplete)
- ❌ Scores contain values like 1.5 or -0.2 (out of range)
- ❌ Average latency is 200ms per alert (too slow)
- ❌ Running same batch twice gives different scores (non-deterministic)

---

## Tier 2: Behavior Validation (0.3 points)

### Pattern Traps (No Ground Truth Needed!)

**Key Insight**: Validator embeds **synthetic alerts** with known expected behaviors.

#### How Pattern Traps Work

**Step 1: Validator creates traps**
```python
# Create synthetic "canary" alerts
pattern_traps = [
  {
    'trap_id': 'trap_constant_high',
    'alert_id': 'alert_TRAP_001',
    'expected_behavior': 'high_score',  # Should score > 0.8
    'reason': 'Extreme volume + mixer destination'
  },
  {
    'trap_id': 'trap_constant_low', 
    'alert_id': 'alert_TRAP_002',
    'expected_behavior': 'low_score',   # Should score < 0.2
    'reason': 'Small amount + legitimate exchange'
  }
]
```

**Step 2: Embed traps in batch**
```python
# Mix traps with real alerts (miners don't know which are traps)
batch_alerts = real_alerts + trap_alerts
# 10,000 real + 100 traps = 10,100 total
```

**Step 3: Check if miner falls for traps**
```python
def validate_behavior(scores, pattern_traps):
    traps_detected = []
    
    for trap in pattern_traps:
        actual_score = scores[trap['alert_id']]['score']
        
        if trap['expected_behavior'] == 'high_score':
            if actual_score < 0.8:
                traps_detected.append(trap['trap_id'])
        
        elif trap['expected_behavior'] == 'low_score':
            if actual_score > 0.2:
                traps_detected.append(trap['trap_id'])
    
    # Penalize for each trap failed
    penalty = len(traps_detected) * 0.05
    
    return {
        'traps_detected': traps_detected,
        'score': max(0.0, 0.30 - penalty)
    }
```

#### Types of Pattern Traps

**1. Obvious High-Risk Traps**
```python
{
  'alert_id': 'alert_TRAP_HIGH_001',
  'volume_usd': 10_000_000,  # Extreme volume
  'to_address': 'KNOWN_MIXER',
  'severity': 'CRITICAL',
  'expected': 'score > 0.9'  # Should definitely catch this
}
```

**2. Obvious Low-Risk Traps**
```python
{
  'alert_id': 'alert_TRAP_LOW_001',
  'volume_usd': 50,  # Tiny amount
  'to_address': 'VERIFIED_EXCHANGE',
  'severity': 'INFO',
  'expected': 'score < 0.1'  # Should definitely ignore this
}
```

**3. Gaming Detection Traps**
```python
# Detect if miner just returns constant scores
if std(all_scores) < 0.001:
    # Miner is gaming by returning same value for everything
    penalty = -0.10
```

**4. Plagiarism Detection**
```python
# Compare miner scores to each other
correlation_matrix = compute_pairwise_correlation(all_miner_scores)

if correlation_with_other_miner > 0.98:
    # Miner is copying another miner's outputs
    penalty = -0.15
```

### Behavior Validation Examples

**Good Miner (0.30 points)**:
```python
scores = {
  'alert_TRAP_HIGH_001': 0.94,  # ✅ Caught high-risk
  'alert_TRAP_LOW_001': 0.08,   # ✅ Ignored low-risk
  'variance': 0.18,             # ✅ Good spread
  'correlation_with_others': 0.65  # ✅ Independent
}
# behavior_score = 0.30
```

**Gaming Miner (0.05 points)**:
```python
scores = {
  'alert_TRAP_HIGH_001': 0.50,  # ❌ Missed obvious high-risk
  'alert_TRAP_LOW_001': 0.50,   # ❌ Scored benign same as risky
  'variance': 0.001,            # ❌ All scores ~0.50 (constant)
  'correlation_with_others': 0.99  # ❌ Copying others
}
# behavior_score = 0.30 - 0.10 - 0.05 - 0.10 = 0.05
```

---

## Cold Start Timeline

### Months 1-3: Immediate Validation Only

```
┌─────────────────────────────────────────────────────────────┐
│  MONTH 1-3: COLD START PERIOD                               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Day 0-90: Building Ground Truth                            │
│  ├─ Miners score alerts (no GT available)                  │
│  ├─ Validators use ONLY immediate validation (0.5 pts)     │
│  ├─ Miners ranked by integrity + behavior                  │
│  └─ Collecting real-world outcomes in background           │
│                                                              │
│  Miner Scores:                                              │
│  ├─ Best miners: 0.45-0.50 (excellent technical quality)  │
│  ├─ Good miners: 0.35-0.45 (solid performance)            │
│  ├─ Gaming miners: 0.00-0.20 (detected and penalized)     │
│  └─ Rewards distributed based on immediate validation only │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Month 4+: Ground Truth Becomes Available

```
┌─────────────────────────────────────────────────────────────┐
│  MONTH 4+: FULL VALIDATION ACTIVE                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Day 90+: Enough historical outcomes accumulated            │
│  ├─ SAR filings for alerts from Days 0-60                  │
│  ├─ Exchange labels accumulated                            │
│  ├─ Blockchain forensics completed                         │
│  └─ Ground truth dataset built                             │
│                                                              │
│  Validation Now Includes:                                    │
│  ├─ Immediate validation (0.5 pts)                         │
│  └─ Ground truth validation (0.5 pts) ← NOW AVAILABLE!     │
│                                                              │
│  Miner Scores:                                              │
│  ├─ Best miners: 0.85-1.00 (technical + accurate)         │
│  ├─ Good miners: 0.65-0.85 (decent accuracy)              │
│  ├─ Poor miners: 0.30-0.50 (passed immediate, failed GT)  │
│  └─ Gaming miners: 0.00-0.20 (caught by behavior checks)  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Bootstrapping Ground Truth

### How to Build Initial Ground Truth Dataset

#### Method 1: Historical Data
**Use past data with known outcomes**

```python
# If you have ANY historical data with confirmations:
historical_ground_truth = {
  # 2024 data with known outcomes
  '2024-01-01': {
    'alert_historical_001': True,  # We know this was illicit
    'alert_historical_002': False  # We know this was benign
  }
}

# Use for:
- Training initial models
- Testing validation framework
- Bootstrapping τ=0 validation
```

#### Method 2: Expert Labeling
**Have domain experts manually label sample**

```python
# Expert labels 1,000 alerts from first batch
expert_labels = {
  'alert_001': {
    'confirmed_illicit': True,
    'labeled_by': 'expert_smith',
    'confidence': 0.85,
    'reasoning': 'Classic layering pattern'
  }
}

# Use immediately for ground truth validation
# Quality: Medium (expert judgment, not real outcomes)
# Coverage: Low (only 1,000 / 10,000 alerts)
```

#### Method 3: Address Labels (Existing Known Bad Actors)
**Use existing labeled addresses**

```python
# We already have this!
address_labels = {
  '0xmixer123': {
    'label': 'tornado_cash',
    'risk_level': 'critical',
    'confirmed_illicit': True  # Known mixer
  },
  '0xexchange456': {
    'label': 'binance_deposit',
    'risk_level': 'low',
    'confirmed_illicit': False  # Known legitimate
  }
}

# Alerts involving these addresses have ground truth:
if alert['address'] in address_labels:
    ground_truth = address_labels[alert['address']]['confirmed_illicit']
```

**This is what we're doing!** Address labels provide immediate ground truth for a subset of alerts.

#### Method 4: Simulated Ground Truth
**Use high-confidence SOT signals as proxy**

```python
# Use SOT's own anomaly scores as proxy labels
proxy_labels = {
  'alert_001': {
    'confirmed_illicit': alerts_df['behavioral_anomaly_score'] > 0.9,
    'confidence': 0.70,  # Lower confidence (proxy, not real)
    'source': 'sot_anomaly_proxy'
  }
}

# During cold start:
# - Better than nothing
# - Allows ground truth validation to work
# - Lower confidence weight
```

---

## Cold Start Validation Strategies

### Strategy 1: Immediate-Only (Months 1-3)

**Score = Integrity + Behavior (max 0.5)**

```python
def validate_during_cold_start(miner_scores, input_alerts, pattern_traps):
    # Run immediate validation only
    integrity = IntegrityValidator().validate(miner_scores, input_alerts)
    behavior = BehaviorValidator().validate(miner_scores, pattern_traps)
    
    # Ground truth component is ZERO during cold start
    ground_truth = {'score': 0.0, 'available': False}
    
    # Final score (out of 0.5, not 1.0)
    final_score = integrity['score'] + behavior['score']
    
    # Normalize to 0-1 range for fair comparison
    normalized_score = final_score / 0.5  # Scale up to use full range
    
    return {
        'final_score': normalized_score,
        'cold_start_mode': True,
        'ground_truth_pending': True
    }
```

**Miner rankings during cold start:**
```
Rank 1: Miner_A (0.48/0.5 = 0.96 normalized)
Rank 2: Miner_B (0.45/0.5 = 0.90 normalized)
Rank 3: Miner_C (0.42/0.5 = 0.84 normalized)
```

### Strategy 2: Partial Ground Truth (Months 2-4)

**Use whatever ground truth is available**

```python
def validate_with_partial_ground_truth(miner_scores, ground_truth_partial):
    # Immediate validation (always available)
    integrity = validate_integrity()  # 0.20
    behavior = validate_behavior()    # 0.30
    
    # Partial ground truth
    coverage = len(ground_truth_partial) / len(all_alerts)
    # coverage might be 15% (only 1,500 / 10,000 alerts have GT)
    
    if coverage > 0.10:  # At least 10% coverage
        # Compute ground truth metrics on available subset
        gt_auc = compute_auc(
            y_true=[ground_truth_partial[aid] for aid in ground_truth_partial],
            y_pred=[miner_scores[aid] for aid in ground_truth_partial]
        )
        
        # Scale GT score by coverage
        gt_score = (0.5 * gt_auc) * coverage
        # Example: AUC=0.85, coverage=0.15 → score = 0.425 * 0.15 = 0.064
    else:
        gt_score = 0.0  # Not enough coverage
    
    final_score = integrity + behavior + gt_score
    
    return {
        'final_score': final_score,
        'ground_truth_coverage': coverage,
        'partial_mode': True
    }
```

**Gradual transition:**
```
Month 1: GT coverage = 0%   → Scores: 0.00-0.50
Month 2: GT coverage = 15%  → Scores: 0.00-0.58
Month 3: GT coverage = 40%  → Scores: 0.00-0.70
Month 4: GT coverage = 70%  → Scores: 0.00-0.85
Month 6: GT coverage = 90%  → Scores: 0.00-1.00 (full validation)
```

### Strategy 3: Consensus-Based Validation (Alternative)

**Use miner agreement as proxy for correctness**

```python
def consensus_validation(all_miner_scores):
    # If multiple miners agree on an alert's risk:
    # - High consensus → Probably correct
    # - Low consensus → Uncertain
    
    consensus_scores = []
    for alert_id in all_alerts:
        scores = [miner[alert_id] for miner in all_miner_scores]
        
        # Measure agreement
        std = np.std(scores)
        mean = np.mean(scores)
        
        if std < 0.1:  # High agreement
            consensus_scores.append({
                'alert_id': alert_id,
                'consensus_score': mean,
                'confidence': 1.0 - std  # Lower std = higher confidence
            })
    
    # Validate each miner against consensus
    for miner in all_miners:
        agreement = correlation(miner.scores, consensus_scores)
        consensus_bonus = 0.2 * agreement  # Up to 0.2 points
```

**Pros:**
- Works without ground truth
- Rewards miners who align with majority
- Detects outliers/gaming

**Cons:**
- Assumes consensus = correct (may not be true)
- Penalizes innovative miners
- Vulnerable to collusion

---

## How to Get SAR Data

### Reality Check: SAR Data is **Very Hard** to Get

#### Challenge 1: Privacy & Confidentiality
**SARs are confidential by law**

```
❌ Banks cannot publicly disclose SAR filings
❌ FinCEN (US) keeps SARs confidential
❌ No public SAR database exists
❌ Sharing SARs is illegal in most jurisdictions
```

#### Challenge 2: Timing
**SARs filed 30-60 days after suspicious activity**

```
Day 0:    Suspicious transaction occurs
Day 1-30: Bank investigates internally
Day 30:   Bank files SAR with FinCEN
Day 60:   FinCEN processes SAR
Day 90:   Outcome known (maybe)
```

### Practical Alternatives to SAR Data

#### Alternative 1: Exchange Labels (RECOMMENDED)

**Major exchanges label risky addresses**

```python
# Exchanges publish risk labels
exchange_labels = {
  'source': 'binance_risk_labels',
  'addresses': {
    '0xabc123': {
      'label': 'high_risk',
      'reason': 'linked_to_mixer',
      'confidence': 0.90
    }
  }
}

# How to get:
# 1. API from major exchanges (if available)
# 2. Published blocklists
# 3. Shared industry databases
```

**Examples:**
- Binance's risk scoring API (if available)
- Coinbase's sanctioned addresses
- Kraken's high-risk labels

#### Alternative 2: Blockchain Forensics

**Use on-chain analysis as ground truth**

```python
# Track what happens AFTER alert
def build_forensic_ground_truth(alert, days=30):
    address = alert['address']
    
    # Get subsequent transactions
    future_txs = get_transactions(
        address=address,
        after=alert['timestamp'],
        days=days
    )
    
    # Check for illicit indicators
    illicit_signals = []
    
    # Signal 1: Flows to known mixers
    if any(tx['to'] in KNOWN_MIXERS for tx in future_txs):
        illicit_signals.append('flows_to_mixer')
    
    # Signal 2: Rapid dispersion (layering)
    unique_destinations = len(set(tx['to'] for tx in future_txs))
    if unique_destinations > 20:
        illicit_signals.append('rapid_dispersion')
    
    # Signal 3: Small sequential amounts (structuring)
    if detect_structuring_pattern(future_txs):
        illicit_signals.append('structuring')
    
    # Derive ground truth
    if len(illicit_signals) >= 2:
        return {'confirmed_illicit': True, 'confidence': 0.80}
    elif len(illicit_signals) == 0:
        return {'confirmed_illicit': False, 'confidence': 0.70}
    else:
        return None  # Uncertain
```

#### Alternative 3: Regulatory Data

**Publicly available regulatory actions**

```python
# OFAC Sanctioned Addresses (Public)
ofac_sanctions = download_ofac_sdn_list()
# Addresses sanctioned → confirmed_illicit = True

# Law Enforcement Actions (Public)
# - FBI seizures (public announcements)
# - DOJ indictments (public records)
# - Chainalysis reports (some public)

ground_truth_from_public_sources = {
  '0xsanctioned123': {
    'confirmed_illicit': True,
    'source': 'OFAC_SDN',
    'sanction_date': '2025-11-01',
    'confidence': 1.0  # Regulatory = highest confidence
  }
}
```

**How to collect:**
- OFAC SDN list (updated weekly): https://sanctionssearch.ofac.treas.gov/
- FBI's Internet Crime Complaint Center (IC3)
- Chainalysis public reports
- News articles about seizures/arrests

#### Alternative 4: Crowdsourced Validation

**Community labels subset of alerts**

```python
# Create labeling platform
crowdsourced_labels = {
  'alert_001': {
    'labeled_by': ['expert_1', 'expert_2', 'expert_3'],
    'votes_illicit': 3,
    'votes_benign': 0,
    'confirmed_illicit': True,
    'confidence': 0.85
  }
}

# Quality control:
- Require multiple labelers
- Track labeler accuracy over time
- Weight by labeler reputation
```

#### Alternative 5: Synthetic Ground Truth

**Generate synthetic test cases**

```python
# Create synthetic alerts with KNOWN ground truth
synthetic_alerts = [
  {
    'alert_id': 'synth_001',
    'pattern': 'tornado_cash_deposit',  # Known illicit
    'ground_truth': True,
    'confidence': 1.0  # We created it, we know the truth
  },
  {
    'alert_id': 'synth_002',
    'pattern': 'exchange_deposit',  # Known benign
    'ground_truth': False,
    'confidence': 1.0
  }
]

# Mix with real alerts (10% synthetic)
# Use for immediate validation of accuracy
```

---

## Recommended Cold Start Strategy

### Phase 1: Launch (Month 1)

**Validation:**
- ✅ Integrity validation (0.2 pts)
- ✅ Behavior validation with pattern traps (0.3 pts)
- ❌ No ground truth validation yet

**Ground Truth Building:**
```python
# Start collecting data for future GT:
future_ground_truth = {
  'alerts_to_track': all_alerts_from_month_1,
  'tracking_started': '2025-11-01',
  'check_again_at': '2025-12-01'  # T+30
}
```

**Miner Scoring:**
```python
# Rank miners by immediate validation only
miner_rankings = sorted_by(integrity_score + behavior_score)
# Max possible: 0.50
```

### Phase 2: Transition (Months 2-3)

**Validation:**
- ✅ Integrity validation (0.2 pts)
- ✅ Behavior validation (0.3 pts)
- 🟡 Partial ground truth (0.0-0.2 pts) ← GROWING

**Ground Truth Sources:**
```python
ground_truth_month_2 = {
  # 1. Address labels (we have this)
  'from_address_labels': 150 alerts,  # ~1.5% coverage
  
  # 2. Blockchain forensics (automated)
  'from_forensics': 300 alerts,  # ~3% coverage
  
  # 3. Public regulatory (automated)
  'from_ofac': 50 alerts,  # ~0.5% coverage
  
  # Total: 500 / 10,000 = 5% coverage
}

# Partial GT validation:
gt_score = 0.5 * auc_roc * coverage
# If AUC=0.85, coverage=0.05 → gt_score = 0.02125
```

**Miner Scoring:**
```python
final_score = 0.20 + 0.28 + 0.02 = 0.50
# Still mostly immediate validation
```

### Phase 3: Maturity (Months 4-6)

**Validation:**
- ✅ Integrity validation (0.2 pts)
- ✅ Behavior validation (0.3 pts)
- ✅ Full ground truth (0.5 pts) ← FULL WEIGHT

**Ground Truth Sources:**
```python
ground_truth_month_4 = {
  # Now have outcomes for Month 1 alerts (T+90)
  'from_sar_filings': 1,200 alerts,      # ~12% (via partnerships)
  'from_exchange_labels': 2,500 alerts,  # ~25%
  'from_forensics': 1,500 alerts,        # ~15%
  'from_address_labels': 800 alerts,     # ~8%
  
  # Total: 6,000 / 10,000 = 60% coverage
}

# Full GT validation:
gt_score = 0.5 * auc_roc  # No coverage penalty (>50%)
```

**Miner Scoring:**
```python
final_score = 0.20 + 0.28 + 0.43 = 0.91
# Now mostly accuracy-driven!
```

---

## Getting SAR Data (Practical Approaches)

### ❌ Not Possible (Directly)
- Access FinCEN SAR database (illegal)
- Get SAR filings from banks (confidential)
- Buy SAR data (doesn't exist commercially)

### ✅ Possible (Indirectly)

#### 1. Partner with Regulated Entities
```
Bank/FI Partnership:
└─ They file SARs internally
└─ They share anonymized outcomes with you
└─ "alert_001 → SAR filed = True"
└─ Confidential data sharing agreement
└─ ONLY outcomes, NOT SAR details
```

#### 2. Use Proxy Indicators
```python
# Indicators that correlate with SAR filings:
sar_proxy_indicators = {
  'high_risk_jurisdiction': 0.3,  # Transfers to high-risk country
  'structuring_pattern': 0.4,     # Classic structuring detected
  'mixer_usage': 0.5,             # Funds through mixer
  'rapid_movement': 0.3,          # Quick in-and-out
  'high_volume': 0.2              # Large amounts
}

# Combined proxy score
proxy_sar_likelihood = sum(indicators_present) / len(indicators)

# Use as ground truth with lower confidence
ground_truth_proxy = {
  'confirmed_illicit': proxy_sar_likelihood > 0.6,
  'confidence': 0.60,  # Lower than real SAR
  'source': 'sar_proxy_model'
}
```

#### 3. Academic/Research Access
```
Research Partnership:
└─ Universities with FinCEN access (rare)
└─ Aggregated/anonymized SAR statistics
└─ Published research papers with labeled datasets
└─ Example: "FinCEN Files" leak (historical)
```

#### 4. Build It Yourself Over Time
```python
# Most practical for cold start:
def build_ground_truth_organically():
    """
    Month 1-3: Use immediate validation only
    Month 4+:  Use accumulated real-world outcomes
    """
    
    # Sources you control:
    ground_truth_sources = {
        # 1. Your own investigations
        'manual_review': manual_label_top_1000_alerts(),
        
        # 2. Blockchain forensics (automated)
        'forensics': track_addresses_forward(30_days),
        
        # 3. Public data (automated)
        'ofac': download_sanctioned_addresses(),
        'exchange_labels': aggregate_exchange_blocklists(),
        
        # 4. Partner data (if available)
        'partner_confirmations': get_partner_confirmations()
    }
    
    # Combine sources
    combined_gt = merge_all_sources(ground_truth_sources)
    
    return combined_gt
```

---

## Cold Start Best Practices

### 1. Start with What You Have

```python
# Immediate ground truth (Day 0):
- ✅ Address labels (we have this in raw_address_labels)
- ✅ OFAC sanctions (public, downloadable)
- ✅ Known mixer addresses (public lists)

# Use these immediately for partial GT validation
initial_coverage = ~5-10% of alerts
```

### 2. Build Ground Truth Pipeline

```python
# Automated ground truth collection
class GroundTruthCollector:
    def collect_for_date(self, processing_date):
        # 1. Blockchain forensics (30 days later)
        forensics_gt = self.trace_addresses_forward(
            date=processing_date,
            days=30
        )
        
        # 2. Download public sources (weekly)
        public_gt = self.download_public_sources()
        
        # 3. Merge with existing labels
        combined = merge(forensics_gt, public_gt, address_labels)
        
        # 4. Store for validation
        save_ground_truth(processing_date, combined)

# Run weekly
schedule.every().week.do(collector.collect_for_date)
```

### 3. Gradual Transition

```python
# Adjust weights as GT accumulates
def compute_validation_weights(ground_truth_coverage):
    if coverage < 0.10:
        # Cold start: Immediate only
        return {
            'integrity': 0.40,   # Double weight
            'behavior': 0.60,    # Double weight
            'ground_truth': 0.0
        }
    elif coverage < 0.50:
        # Transition: Partial GT
        return {
            'integrity': 0.20,
            'behavior': 0.30,
            'ground_truth': 0.50 * coverage  # Scale up
        }
    else:
        # Mature: Full GT
        return {
            'integrity': 0.20,
            'behavior': 0.30,
            'ground_truth': 0.50  # Full weight
        }
```

### 4. Transparency

**Communicate to miners:**

```markdown
## Current Validation Status

- **Ground Truth Coverage**: 8.5% (growing)
- **Validation Weights**:
  - Integrity: 40% (2x normal)
  - Behavior: 60% (2x normal)
  - Ground Truth: 0% (building dataset)
  
- **Expected Full GT**: Month 6 (when coverage > 50%)

Miners are currently ranked by **technical quality** only.
Accuracy-based ranking will begin when ground truth coverage reaches 50%.
```

---

## Summary

### Cold Start Solution

**Without any ground truth:**
1. ✅ Use **Integrity validation** (0.2 pts) - Technical checks
2. ✅ Use **Behavior validation** (0.3 pts) - Pattern traps
3. ❌ Skip **Ground truth validation** (0.0 pts) - Not available yet

**Miners can still be scored and ranked!** (0.0-0.5 range)

### Getting "Ground Truth"

**You CAN'T get:**
- Real SAR filings (confidential)
- Bank investigation results (private)

**You CAN get:**
- ✅ **Address labels** (we have this!)
- ✅ **OFAC sanctions** (public)
- ✅ **Exchange labels** (some public)
- ✅ **Blockchain forensics** (automated on-chain analysis)
- ✅ **Expert labels** (manual review of sample)

### Recommended Approach

**Month 1-3: Immediate validation only**
```python
miner_score = integrity (0.2) + behavior (0.3) = 0.5 max
```

**Month 4+: Add growing ground truth**
```python
miner_score = integrity (0.2) + behavior (0.3) + ground_truth (0.0→0.5)
# As coverage grows: 10% → 30% → 50% → 90%
```

**Ground truth sources to build:**
1. Start with address labels (have it)
2. Add OFAC/sanctions (public, easy)
3. Add blockchain forensics (automated)
4. Add exchange partnerships (harder)
5. Add expert labeling (manual, expensive)

**You don't need perfect ground truth immediately - start with immediate validation and build ground truth over time!**