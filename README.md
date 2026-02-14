# Supervised Learning Model for Fraud Risk Scoring in Nigerian Financial Transactions

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**BSc Statistics Final Year Dissertation Implementation**  
*University of Lagos - Academic Year 2024/2025*

## Abstract
Nigeria's electronic payment transactions reached N600 trillion in 2023, representing a 55\% increase from the previous year, yet sophisticated fraud detection mechanisms remain underdeveloped in the Nigerian banking sector. This research develops and evaluates supervised machine learning models for fraud risk scoring using a simulated dataset of 1,000,000 Nigerian financial transactions, calibrated to realistic banking patterns using NIBSS fraud landscape data. Three supervised learning algorithms: Logistic Regression, Random Forest, and XGBoost, were implemented and compared using 38 engineered features encompassing temporal patterns, behavioral analytics, and Nigerian-specific risk indicators. While XGBoost achieved superior statistical performance with F1-Score of 0.854 and 74.6\% recall compared to Random Forest's 0.699 and 53.8\% recall, Random Forest demonstrated marginally higher AUC-ROC of 0.977 [0.966, 0.986] versus XGBoost's 0.973. Cost-sensitive analysis revealed Random Forest's superior economic value, reducing fraud-related costs by 69.1\% compared to XGBoost's 43.7\% through optimal threshold selection. SHAP analysis identified amount-based features as dominant fraud indicators, with Web banking (0.34\%) and Mobile banking (0.33\%) emerging as highest-risk channels, consistent with documented NIBSS patterns. This study demonstrates that ensemble methods provide superior fraud detection capabilities for Nigerian banking contexts when combined with appropriate feature engineering and threshold optimization. The research establishes the first publicly available Nigerian financial fraud detection dataset and production-ready framework, addressing a critical gap in African financial fraud research. The synthetic data approach successfully balances research validity with ethical compliance, providing a replicable methodology for emerging market fraud detection research with immediate implementation value for Nigerian financial institutions

## 🎓 Dissertation Overview

This repository contains the complete implementation of a BSc Statistics dissertation investigating supervised learning approaches for fraud risk scoring in Nigerian financial transactions. The research addresses the critical gap between Nigeria's rapidly growing electronic payment ecosystem (₦600 trillion in 2023) and the sophistication of fraud detection mechanisms employed by Nigerian financial institutions.

### Research Objectives

1. **Primary Objective**: Develop and evaluate supervised machine learning models for fraud risk scoring tailored to Nigerian financial transaction patterns
2. **Secondary Objectives**:
   - Create a comprehensive synthetic dataset calibrated to realistic NIBSS fraud patterns
   - Compare performance of three supervised learning algorithms (Logistic Regression, Random Forest, XGBoost)
   - Implement cost-sensitive optimization for Nigerian banking economics
   - Provide interpretable model insights through SHAP analysis
   - Establish a replicable framework for emerging market fraud detection research

### Academic Contributions

- **First publicly available Nigerian financial fraud detection dataset** with 1,000,000 synthetic transactions
- **Novel feature engineering approach** combining temporal patterns, behavioral analytics, and Nigerian-specific risk indicators (38 features total)
- **Comprehensive comparative analysis** of supervised learning algorithms in the Nigerian banking context
- **Cost-sensitive optimization framework** demonstrating 69.1% reduction in fraud-related costs
- **Production-ready implementation** with immediate deployment potential for Nigerian financial institutions

## 🔬 Research Methodology

### Dataset Development
- **Scale**: 1,000,000 synthetic Nigerian financial transactions
- **Calibration Source**: NIBSS 2023 Annual Fraud Landscape data
- **Fraud Rate**: 0.30% (elevated from NIBSS reported 0.000848% to ensure sufficient fraud samples for model training)
- **Feature Engineering**: 38 features across temporal, behavioral, and risk categories
- **Ethical Compliance**: Synthetic data approach ensuring privacy and regulatory compliance

### Model Implementation
- **Algorithms**: Logistic Regression (baseline), Random Forest (ensemble), XGBoost (gradient boosting)
- **Evaluation Framework**: Stratified cross-validation with bootstrap confidence intervals
- **Performance Metrics**: AUC-ROC, Precision, Recall, F1-Score, Cost-Adjusted Metrics
- **Interpretability**: SHAP (SHapley Additive exPlanations) analysis

### Statistical Analysis
- **Bootstrap Validation**: 100 iterations for confidence interval estimation
- **Significance Testing**: Paired statistical tests with Bonferroni correction
- **Cost-Sensitive Optimization**: Threshold selection based on Nigerian banking cost structures

## 🚀 Implementation Setup

### Prerequisites
- Python 3.8+ with scientific computing libraries
- Jupyter Lab/Notebook environment
- Minimum 8GB RAM (16GB recommended for full dataset processing)
- Git for repository management

### Installation and Setup

```bash
# Clone the dissertation repository
git clone https://github.com/hendurhance/nigerian-fraud-detection-dissertation.git
cd nigerian-fraud-detection-dissertation

# Create isolated environment for reproducibility
python -m venv dissertation_env
source dissertation_env/bin/activate  # On Windows: dissertation_env\Scripts\activate

# Install research dependencies
pip install -r requirements.txt
```

### Reproduce Dissertation Results

```bash
# Step 1: Generate the research dataset (1M transactions)
python src/nibss_fraud_dataset_generator.py -n 1000000 --output data/processed/research_dataset.csv --stats

# Step 2: Launch Jupyter environment for analysis
jupyter lab

# Step 3: Execute dissertation chapters in sequence:
# Chapter 3: Data Collection and Preprocessing
# └── notebooks/01_data_preprocessing_and_pipeline_setup.ipynb

# Chapter 4: Model Development and Implementation  
# └── notebooks/02_model_optimization.ipynb

# Chapter 5: Results and Evaluation
# └── notebooks/03_model_evaluation.ipynb

# Chapter 6: Model Interpretability and Analysis
# └── notebooks/04_feature_importance_analysis.ipynb

# Chapter 7: Cost-Benefit Analysis
# └── notebooks/05_cost_optimization_analysis.ipynb
```

## 📁 Dissertation Repository Structure

```
nigerian-fraud-detection-dissertation/
├── README.md                          # Dissertation implementation guide
├── requirements.txt                   # Research environment dependencies
├── .gitignore                        # Version control configuration
├── LICENSE                           # Academic use license
├── CONTRIBUTING.md                   # Contribution guidelines
│
├── config/                           # Research configuration
│   └── config.yaml                   # Experimental parameters and settings
│
├── src/                              # Core implementation modules
│   ├── nibss_fraud_dataset_generator.py  # NIBSS-calibrated data generator
│   ├── preprocessing.py              # Chapter 3: Data preprocessing pipeline
│   ├── fraud_analysis.py            # Chapter 4-6: ML model implementations
│   ├── cleaning.py                  # Data quality assurance utilities
│   ├── channel_fraud_analysis.py    # Nigerian banking channel analysis
│   ├── temporal_fraud_analysis.py   # Time-series fraud pattern analysis
│   └── correlation_multicollinearity_analysis.py  # Statistical validation
│
├── notebooks/                        # Dissertation chapter implementations
│   ├── 01_data_preprocessing_and_pipeline_setup.ipynb     # Chapter 3
│   ├── 02_model_optimization.ipynb                        # Chapter 4
│   ├── 03_model_evaluation.ipynb                          # Chapter 5
│   ├── 04_feature_importance_analysis.ipynb               # Chapter 6
│   └── 05_cost_optimization_analysis.ipynb                # Chapter 7
│
├── data/                             # Research datasets
│   ├── raw/                         # Original NIBSS-calibrated data
│   ├── processed/                   # Preprocessed research datasets
│   ├── external/                    # NIBSS reference statistics
│   └── cleaning/                    # Data quality analysis outputs
│
├── models/                           # Trained model artifacts
│   ├── best_model_random_forest.pkl # Optimal Random Forest model
│   ├── best_model_xgboost.pkl      # Optimal XGBoost model
│   ├── best_model_logistic_regression.pkl  # Baseline Logistic model
│   └── evaluation_results.pkl      # Comprehensive performance metrics
│
├── results/                          # Dissertation findings
│   ├── performance_metrics.csv     # Chapter 5: Model comparison results
│   ├── feature_importance.csv      # Chapter 6: SHAP analysis results
│   └── cost_analysis.csv          # Chapter 7: Economic impact analysis
│
├── docs/                            # Academic documentation
│   ├── NIBSS_FRAUD_DATASET_GENERATOR.md  # Data generation methodology
│   ├── PROJECT_RESTRUCTURE_SUMMARY.md    # Implementation documentation
│   ├── images/                      # Dissertation figures and visualizations
│   │   ├── figure_4_6_roc_curves.png          # ROC curve comparisons
│   │   ├── figure_4_9_shap_importance.png     # Feature importance plots
│   │   ├── cost_threshold_curves.png          # Cost optimization analysis
│   │   └── fraud_analysis_charts/             # Channel and temporal analysis
│   └── reports/                     # Generated analysis reports
│
├── tests/                           # Unit tests for reproducibility
└── scripts/                         # Automation and utility scripts
    └── setup_environment.sh        # Research environment setup
```

## 📊 Research Dataset Specifications

### 📥 **Dataset Available on Kaggle**
**🔗 [Download the NIBSS Fraud Dataset](https://www.kaggle.com/datasets/hendurhance/nibsss-fraud-dataset)**

The complete 1,000,000 transaction dataset with comprehensive documentation is freely available on Kaggle for research and educational use.

### NIBSS 2023 Calibrated Statistics (Ground Truth)

| Metric | Value | Source | Implementation |
|--------|-------|--------|----------------|
| **Transaction Volume** | 1,000,000 | Research Scale | Synthetic Generation |
| **Fraud Rate** | 0.30% (3,000 fraud cases) | Elevated from NIBSS 0.000848% | Statistical Calibration for ML Training |
| **Top Fraud Channel** | Mobile (49.75%) | NIBSS 2023 | Channel Distribution |
| **Peak Fraud Month** | May (12.25%) | NIBSS 2023 | Temporal Modeling |
| **Average Fraud Loss** | ₦384,959 | NIBSS 2023 | Amount Distribution |
| **Primary Fraud Technique** | Social Engineering (65.8%) | NIBSS 2023 | Fraud Categorization |

### Fraud Rate Calibration Rationale

The dataset has a fraud rate of 0.30% (3,000 fraudulent transactions), which creates a class imbalance ratio of approximately 332:1. Based on the report from NIBSS, the actual fraud rate is about 0.000848% [NIBSS(2024)].

**Why elevate the fraud rate?**
The real-world rate would mean that the dataset will only have 8 fraud cases per million transactions, which is not sufficient to train models effectively or perform statistical evaluation. The 0.30% rate keeps the class imbalance problem realistic while providing enough fraud samples for model development. This approach:
- Maintains realistic class imbalance challenges (332:1 ratio)
- Provides sufficient fraud samples (3,000 cases) for robust model training
- Enables meaningful statistical evaluation and performance metrics
- Preserves all NIBSS fraud pattern distributions (channel, temporal, technique)
- Allows for effective cross-validation and bootstrap analysis

### Feature Engineering Framework (38 Features Total)

**Temporal Features (12)**
- Cyclic time encodings (hour, day, month, quarter)
- Rolling transaction counts (24h, 7d, 30d windows)
- Velocity metrics and frequency patterns
- Nigerian business hour indicators

**Behavioral Features (15)**
- Amount deviation ratios and z-scores
- Transaction velocity and acceleration
- Channel diversity and switching patterns
- Location consistency indicators
- Customer transaction history patterns

**Risk Assessment Features (11)**
- Composite risk scores
- Merchant category risk classifications
- Channel-specific risk indicators
- Anomaly detection scores
- Cross-channel risk correlation metrics

## 🏆 Dissertation Results Summary

### Primary Research Findings

| Algorithm | AUC-ROC [95% CI] | Precision [95% CI] | Recall [95% CI] | F1-Score [95% CI] | Cost Reduction |
|-----------|------------------|-------------------|-----------------|-------------------|----------------|
| **XGBoost** | **0.973** [0.963, 0.981] | **1.000** [1.000, 1.000] | **0.746** [0.703, 0.786] | **0.854** [0.827, 0.880] | **43.7%** |
| **Random Forest** | **0.977** [0.966, 0.986] | **1.000** [1.000, 1.000] | **0.538** [0.493, 0.584] | **0.699** [0.662, 0.739] | **69.1%** |
| **Logistic Regression** | 0.799 [0.777, 0.824] | 0.007 [0.007, 0.008] | 0.699 [0.654, 0.741] | 0.015 [0.013, 0.016] | 1.9% |

*Statistical significance validated through bootstrap analysis (n=100)*

### Key Research Contributions

1. **XGBoost Statistical Superiority, Random Forest Economic Advantage**: XGBoost achieved superior statistical performance with highest F1-Score (0.854) and Recall (0.746), while Random Forest delivered optimal economic value with perfect precision (1.000), highest AUC-ROC (0.977), and 69.1% cost reduction
2. **Channel Risk Analysis**: Web banking (0.34%) and Mobile banking (0.33%) identified as highest-risk channels
3. **Feature Importance**: Amount-based features dominate fraud detection (SHAP analysis)
4. **Economic Impact**: Random Forest reduces fraud costs by 69.1% through optimal threshold selection
5. **Nigerian Context**: First comprehensive ML framework calibrated to Nigerian banking patterns

## 💰 Economic Impact Analysis (Chapter 4 Findings)

### Cost-Sensitive Optimization Results

| Model | Optimal Threshold | Cost Reduction | Business Value |
|-------|------------------|----------------|----------------|
| **Random Forest** | 0.03 | **69.1%** | ₦12.3B annually* |
| **XGBoost** | 0.00 | 43.7% | ₦7.8B annually* |
| **Logistic Regression** | 0.55 | 1.9% | ₦0.3B annually* |

*Based on 2023 NIBSS fraud statistics (₦17.8B total losses)*

## 🔍 Key Research Insights

### Nigerian Fraud Pattern Analysis
1. **Channel Risk Hierarchy**: Web (0.34%) > Mobile (0.33%) > POS (0.18%) > Internet Banking (0.15%)
2. **Temporal Distribution**: May peak (12.25%), December trough (4.49%), business hour concentration
3. **Geographic Concentration**: Lagos State dominance (48% of cases), Northern states higher value
4. **Fraud Technique Distribution**: Social Engineering (65.8%), Technical exploitation (22.3%), Physical access (11.9%)

### Permutation Feature Importance (Top 10 by AUC Decrease)

**Logistic Regression:**
1. **Amount vs Mean Ratio** (0.385) - Primary behavioral deviation indicator
2. **Amount Sum 24H** (0.102) - Short-term spending aggregation
3. **Transaction Count 24H** (0.027) - Daily transaction frequency
4. **Amount Mean Total** (0.016) - Historical spending baseline
5. **Amount Std 7D** (0.013) - Weekly spending variability

**Random Forest:**
1. **Amount vs Mean Ratio** (0.121) - Dominant fraud predictor
2. **Amount Sum 24H** (0.012) - Secondary behavioral pattern
3. **Velocity Score** (0.003) - Transaction velocity indicator
4. **Composite Risk** (0.003) - Multi-factor risk assessment
5. **Remaining features** (<0.001) - Minimal individual impact

**XGBoost:**
1. **Amount vs Mean Ratio** (0.280) - Critical fraud indicator
2. **Velocity Score** (0.036) - Behavioral anomaly detection
3. **Amount Sum 24H/Transaction Count 24H/Amount** (0.001) - Minimal impact
4. **Remaining features** (0.000) - No measurable importance

**Key Insights:**
- **Amount vs Mean Ratio** emerges as the most critical feature across all models
- **Logistic Regression** shows broader feature dependency compared to ensemble methods
- **XGBoost** demonstrates high feature selectivity with only top 2 features showing substantial importance
- **Temporal and categorical features** show negligible discriminative power in this dataset

### Statistical Validation
- **Bootstrap Confidence Intervals**: 100 iterations ensuring 95% statistical confidence
- **Cross-validation**: 5-fold stratified validation maintaining fraud distribution
- **Significance Testing**: Paired t-tests with Bonferroni correction (α = 0.0167)
- **Effect Size**: Cohen's d > 0.8 for Random Forest vs. Logistic Regression comparison

## 🔧 Research Configuration

### Experimental Parameters (`config/config.yaml`)

```yaml
# Research dataset specifications
dataset:
  size: 1000000                    # 1M transactions for statistical power
  fraud_rate: 0.003               # 0.30% (elevated from NIBSS 0.000848% for ML training)
  seed: 42                        # Reproducibility seed
  validation_split: 0.2           # 80-20 train-test split

# Model hyperparameter search spaces
models:
  cross_validation_folds: 5       # Stratified k-fold CV
  bootstrap_iterations: 1000      # Statistical validation
  random_seed: 42                 # Deterministic results
  
  # Random Forest optimization grid
  random_forest:
    n_estimators: [100, 200, 300]
    max_depth: [10, 20, None]
    min_samples_split: [2, 5, 10]
  
  # XGBoost optimization grid  
  xgboost:
    n_estimators: [100, 200, 300]
    max_depth: [3, 6, 10]
    learning_rate: [0.01, 0.1, 0.2]

# Nigerian banking cost structure (Chapter 7)
costs:
  false_positive: 1900           # Manual review + friction
  false_negative: 464959         # Average fraud loss
  optimization_metric: "total_cost"  # Business objective
```

## 📚 Academic Documentation

### Core Research Documentation
- **[NIBSS Dataset Generator](docs/NIBSS_FRAUD_DATASET_GENERATOR.md)**: Synthetic data methodology and validation
- **[Analysis Reports](docs/reports/)**: Comprehensive fraud pattern analysis outputs

### Dissertation Chapter Mapping
1. **Chapter 4.4** → `notebooks/01_data_preprocessing_and_pipeline_setup.ipynb`
2. **Chapter 4.5** → `notebooks/02_model_optimization.ipynb` 
3. **Chapter 4.6 & 4.7** → `notebooks/03_model_evaluation.ipynb`
4. **Chapter 4.8** → `notebooks/04_feature_importance_analysis.ipynb`
5. **Chapter 4.9** → `notebooks/05_cost_optimization_analysis.ipynb`

### Research Artifacts
- **Model Artifacts**: `models/` - Trained models and evaluation results
- **Visualizations**: `docs/images/` - All dissertation figures and charts
- **Statistical Results**: `results/` - Performance metrics and analysis outputs


## 📄 Academic License and Usage

This research implementation is licensed under the Creative Commons Attribution 4.0 International License for academic and educational purposes. See the [LICENSE](LICENSE) file for complete details.

### Citation Requirements
When using this research implementation, please cite:

```bibtex
@misc{nigerian_fraud_detection_2024,
  title={Supervised Learning Model for Fraud Risk Scoring in Nigerian Financial Transactions},
  author={Owolabi, Josiah Endurance},
  year={2025},
  institution={[University of Lagos]},
  type={BSc Statistics Dissertation},
  url={https://github.com/hendurhance/nigerian-fraud-detection-dissertation}
}
```

## 🙏 Research Acknowledgments

### Data Sources and Validation
- **Nigerian Interbank Settlement System (NIBSS)**: 2023 Annual Fraud Landscape statistical data for synthetic data calibration
- **Central Bank of Nigeria (CBN)**: Regulatory framework and banking industry guidelines
- **Nigerian Banking Industry**: Domain expertise consultation and pattern validation

## 👨‍🎓 Author Information

- **Student**: JOSIAH ENDURANCE OWOLABI
- **Student ID**: *******
- **Program**: BSc Statistics
- **Institution**: University of Lagos
- **Academic Year**: 2024/2025

- **Dissertation Supervisor**: DR. IDOWU G. A. (Ph.D.) 
- **Department**: Statistics
- **Institution**: Lagos State University

## 📈 Research Impact and Future Work

### Immediate Applications
- Implementation framework ready for Nigerian banking institutions
- Synthetic data methodology applicable to other emerging markets
- Cost-sensitive optimization techniques for financial fraud detection

### Future Research Directions
1. **Real-time Implementation**: Streaming fraud detection with online learning
2. **Cross-Country Analysis**: Comparative study across African financial markets
3. **Deep Learning Extensions**: Neural network approaches for pattern recognition
4. **Regulatory Compliance**: Integration with Nigerian banking regulations and compliance frameworks

## 🔄 Implementation Version History

- **v1.0.0** (Academic Milestone): Initial dissertation implementation with NIBSS 2023 calibration
- **v1.1.0** (Research Enhancement): Advanced feature engineering and SHAP interpretability analysis  
- **v1.2.0** (Economic Analysis): Cost-sensitive optimization and business impact quantification
- **v1.3.0** (Dissertation Final): Complete academic implementation with statistical validation

---

## ⚠️ Important Academic Disclaimers

### Research Ethics and Data Privacy
- **Synthetic Data Only**: No real customer financial data used or exposed in this research
- **Privacy Compliance**: All synthetic data generated follows ethical AI research guidelines  
- **NIBSS Compliance**: Statistical patterns derived from publicly available NIBSS reports only
- **Academic Use**: This implementation is designed for educational and research purposes

### Research Limitations
- **Synthetic Data Constraints**: While statistically calibrated, synthetic data may not capture all real-world fraud complexities
- **Temporal Scope**: Research based on 2023 NIBSS statistics; fraud patterns evolve continuously
- **Geographic Scope**: Focused specifically on Nigerian banking context; generalization requires validation
- **Implementation Environment**: Academic research environment; production deployment requires additional security and compliance measures

### Reproducibility Statement
This research prioritizes reproducibility through:
- Fixed random seeds for deterministic results
- Comprehensive documentation of all experimental parameters  
- Statistical validation through bootstrap confidence intervals
- Open-source implementation with detailed methodology documentation

**Last Updated**: 2025-09-07
**Research Status**: BSc Dissertation - Final Implementation