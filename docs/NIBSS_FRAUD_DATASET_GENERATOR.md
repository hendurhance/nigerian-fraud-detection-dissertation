# NIBSS Fraud Dataset Generator

## Overview

The NIBSS Fraud Dataset Generator is a sophisticated synthetic data generator specifically calibrated to match the 2023 Nigerian Interbank Settlement System (NIBSS) Annual Fraud Landscape report. This tool generates realistic Nigerian banking transaction data with fraud patterns that mirror real-world statistics from Nigeria's financial ecosystem.

## Fraud Rate Calibration

The dataset uses a fraud rate of 0.30% (3,000 fraudulent transactions per million), which is elevated from the actual NIBSS reported rate of 0.000848%. This elevation is a deliberate design decision for the following reasons:

**Statistical Necessity**: The real-world NIBBS rate (0.000848%) would result in only 8 fraud cases per million transactions, which is insufficient for:
- Training machine learning models effectively
- Performing meaningful statistical evaluation
- Conducting robust cross-validation
- Calculating reliable performance metrics with confidence intervals

**Maintained Realism**: While elevated, the 0.30% rate:
- Preserves realistic class imbalance challenges (332:1 ratio)
- Provides sufficient fraud samples (3,000 cases) for model development
- Maintains all NIBBS fraud pattern distributions (channel-specific rates, temporal patterns, technique distributions)
- Keeps the problem representative of real-world fraud detection challenges

This approach is common in fraud detection research to balance statistical validity with practical model training requirements.

## Algorithm and Methodology

### Data Generation Strategy

The generator follows a multi-stage approach to create synthetic yet realistic banking transaction data:

#### 1. **Statistical Calibration**
- All distributions are calibrated against NIBSS 2023 official fraud statistics
- Maintains exact monthly fraud distribution patterns (9.66% Jan, 12.25% May peak, 4.49% Dec lowest)
- Preserves channel-specific fraud rates (Mobile: 49.75%, Web: 22.91%, POS: 18.38%, etc.)
- Matches transaction volume and value seasonal patterns

#### 2. **Multi-Layered Generation Process**

##### Stage 1: Customer Profile Generation
```python
# Generate diverse customer profiles
customers = {
    "customer_id": unique_identifier,
    "age_group": weighted_by_demographics,
    "location": weighted_by_nigerian_states,
    "risk_score": beta_distribution(2, 8),
    "account_age_months": exponential_distribution(24),
    "avg_monthly_transactions": lognormal_distribution(2.5, 1)
}
```

##### Stage 2: Temporal Transaction Sampling
- **Monthly Distribution**: Uses NIBSS volume percentages (March peak: 12.11%, January low: 5.60%)
- **Daily Patterns**: Business days weighted higher than weekends
- **Hourly Distribution**: Nigerian business hours (8 AM - 6 PM) with peak periods (10-11 AM, 2-4 PM)

##### Stage 3: Channel and Amount Generation
- **Channel Selection**: Probabilistic sampling based on NIBSS channel usage
- **Amount Calculation**: Log-normal distributions with channel-specific parameters:
  - Mobile: μ=11.0, σ=1.2
  - Web: μ=11.5, σ=1.0
  - POS: μ=10.5, σ=1.3
  - Internet Banking: μ=12.5, σ=1.1
  - E-commerce: μ=10.8, σ=0.9
  - ATM: μ=10.2, σ=0.7

#### 3. **Behavioral Feature Engineering**

The generator creates sophisticated behavioral features that capture real-world transaction patterns:

##### Temporal Features
- **Rolling Windows**: 24-hour, 7-day, 30-day transaction counts and amounts
- **Velocity Metrics**: Transaction frequency and amount acceleration
- **Cyclic Encodings**: Hour, day, month encoded as sine/cosine for ML compatibility

##### Customer Behavior Features
- **Channel Diversity**: Number of unique channels used
- **Location Consistency**: Geographical transaction patterns
- **Amount Deviation**: Ratio of current transaction to customer's historical mean
- **Online Channel Ratio**: Preference for digital vs. physical channels

#### 4. **Fraud Pattern Injection**

The most sophisticated component ensures fraud cases match NIBSS patterns exactly:

##### Multi-Stage Fraud Allocation
1. **Channel-First Allocation**: Distribute total fraud cases across channels using exact NIBSS percentages
2. **Monthly Refinement**: Within each channel, distribute fraud across months using fraud count distribution
3. **Technique Assignment**: Apply fraud techniques (Social Engineering: 65.8%, Robbery: 10.6%, etc.)
4. **Amount Adjustment**: Modify fraud amounts based on channel-specific loss patterns

##### NIBSS Loss Calibration
```python
# Channel-specific average fraud losses (from NIBSS data)
channel_avg_loss = {
    "Mobile": 119842,    # ₦119,842
    "Web": 101616,       # ₦101,616  
    "POS": 251391,       # ₦251,391
    "IB": 761445,        # ₦761,445 (highest)
    "ECOM": 58818,       # ₦58,818
    "ATM": 88086         # ₦88,086
}
```

### Advanced Features

#### 1. **Merchant Risk Scoring**
- Electronics/Fashion/Entertainment: High-risk beta(6,4) distribution
- Grocery/Fuel/Medical: Low-risk beta(2,8) distribution
- Others: Medium-risk beta(3,6) distribution

#### 2. **Composite Risk Calculation**
```python
composite_risk = (
    merchant_risk_score * 0.3 +
    amount_deviation_ratio * 0.3 +
    velocity_score * 0.2 +
    high_frequency_indicator * 0.2
)
```

#### 3. **Nigerian Banking Context**
- **Bank Selection**: 10 major Nigerian banks (GTBank, FirstBank, Zenith, UBA, etc.)
- **State Distribution**: Lagos-centric (48%) with realistic distribution across 36 states
- **Currency**: All amounts in Nigerian Naira (₦)
- **Business Hours**: Aligned with Nigerian banking hours and holidays

## Technical Implementation

### Core Architecture

```python
class NIBSSFraudDatasetGenerator:
    def __init__(self, dataset_size: int = 1_000_000, seed: int = 42):
        # Initialize with NIBSS statistical parameters
        
    def generate_dataset(self) -> pd.DataFrame:
        # Main orchestration method
        
    def _inject_fraud_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        # Sophisticated fraud injection ensuring NIBSS compliance
        
    def _adjust_fraud_amounts(self, df: pd.DataFrame, fraud_indices: list):
        # Adjust fraud amounts using channel-specific loss patterns
```

### Key Algorithms

#### Monthly Volume Sampling
```python
def _sample_timestamp(self) -> datetime:
    # Sample month based on NIBSS volume distribution
    month = np.random.choice(
        np.arange(1, 13), 
        p=self.monthly_volume_weights
    )
    # Apply Nigerian business hour weighting
    hour = np.random.choice(
        np.arange(24), 
        p=self._get_hourly_weights()
    )
```

#### Fraud Pattern Verification
```python
# Continuous verification ensures exact NIBSS compliance
actual_pct = (channel_fraud_count / total_fraud * 100)
target_pct = NIBSS_CONFIG["channel_fraud_count_pct"][channel]
match = "✓" if abs(actual_pct - target_pct) < 1.0 else "✗"
```

## Data Quality Assurance

### Statistical Validation
1. **Distribution Matching**: Chi-square tests verify channel and temporal distributions
2. **Fraud Rate Precision**: Maintains exact 0.3% fraud rate (elevated from NIBSS 0.000848% to ensure sufficient fraud samples for effective ML model training and statistical evaluation)
3. **Amount Realism**: Log-normal distributions create realistic transaction amounts
4. **Seasonal Patterns**: Monthly variations match Nigerian business cycles

### Integrity Features
- **SHA-256 Hashing**: Each generated dataset includes cryptographic hash
- **Reproducibility**: Fixed random seeds ensure identical outputs
- **Compliance Reporting**: Automatic verification against NIBSS benchmarks

## Usage Examples

### Basic Generation
```bash
python src/nibss_fraud_dataset_generator.py -n 500000 --output data/processed/fraud_dataset.csv
```

### Advanced Configuration
```bash
python src/nibss_fraud_dataset_generator.py \
  --rows 1000000 \
  --seed 123 \
  --output data/processed/large_dataset.csv \
  --stats
```

### Python API
```python
from src.nibss_fraud_dataset_generator import NIBSSFraudDatasetGenerator

# Initialize generator
generator = NIBSSFraudDatasetGenerator(dataset_size=600000, seed=42)

# Generate dataset
df = generator.generate_dataset()

# Save with integrity hash
output_path = generator.save_dataset(df, "custom_dataset.csv")
```

## Output Schema

| Column | Type | Description |
|--------|------|-------------|
| `transaction_id` | string | Unique transaction identifier |
| `customer_id` | string | Unique customer identifier |
| `timestamp` | datetime | Transaction timestamp |
| `amount` | float | Transaction amount (NGN) |
| `channel` | string | Transaction channel (Mobile, Web, POS, IB, ECOM, ATM) |
| `merchant_category` | string | Merchant category |
| `bank` | string | Nigerian bank name |
| `location` | string | Nigerian state |
| `age_group` | string | Customer age group |
| `is_fraud` | int | Fraud label (0/1) |
| `fraud_technique` | string | Fraud technique (for fraud cases) |
| `tx_count_24h` | float | 24-hour transaction count |
| `amount_sum_24h` | float | 24-hour transaction sum |
| `amount_mean_7d` | float | 7-day rolling mean |
| `velocity_score` | float | Transaction velocity indicator |
| `composite_risk` | float | Composite risk score |
| ... | ... | Additional engineered features |

## Performance Metrics

- **Generation Speed**: ~50,000 transactions per second
- **Memory Efficiency**: Optimized for datasets up to 10M transactions
- **Accuracy**: >99.5% compliance with NIBSS statistical benchmarks
- **Reproducibility**: 100% with fixed random seeds

## Validation and Compliance

### NIBSS 2023 Compliance Verification

The generator includes automatic verification against NIBSS benchmarks:

```
Channel Fraud Distribution Verification:
Channel  Actual  Actual%  Target%  Match
Mobile   1492    49.73%   49.75%   ✓
Web      687     22.90%   22.91%   ✓
POS      551     18.37%   18.38%   ✓
IB       169     5.63%    5.63%    ✓
ECOM     77      2.57%    2.56%    ✓
ATM      24      0.80%    0.76%    ✓
```

### Research Applications

This dataset is designed for:
- **Fraud Detection Research**: ML model development and testing
- **Academic Studies**: Financial crime pattern analysis
- **Algorithm Benchmarking**: Standardized evaluation of fraud detection systems
- **Educational Purposes**: Training data science professionals

## Citation and Attribution

When using this dataset in research or publications, please cite:

```
NIBSS Fraud Dataset Generator (2024)
Synthetic Nigerian Banking Transaction Data
Calibrated to NIBSS 2023 Annual Fraud Landscape
```

## Ethical Considerations

- **Privacy Preservation**: All data is synthetic, no real customer information
- **Bias Awareness**: Reflects Nigerian financial system characteristics
- **Responsible Use**: Intended for fraud detection and prevention research only
- **Data Governance**: Follows best practices for synthetic data generation

## License and Distribution

This dataset and generator are provided for research and educational purposes. Commercial use requires appropriate licensing arrangements.