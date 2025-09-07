#!/usr/bin/env python3
"""
Feature Correlation and Multicollinearity Analysis for Nigerian Fraud Dataset

This script generates:
1. Complete correlation matrix with all features and fraud status
2. Multicollinearity assessment using VIF
3. Comprehensive 4-panel visualization
4. Feature importance ranking with proper correlation calculations

Author: Your Name
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import pointbiserialr, pearsonr
import warnings
warnings.filterwarnings('ignore')

class CorrelationMulticollinearityAnalyzer:
    """Analyze feature correlations and multicollinearity in fraud detection dataset."""
    
    def __init__(self, dataset_path):
        """Initialize with dataset path."""
        self.dataset_path = dataset_path
        self.df = None
        self.feature_df = None
        self.all_features_df = None
        self.correlation_results = {}
        self.vif_results = {}
        self.full_correlation_matrix = None
        
    def load_and_prepare_data(self):
        """Load dataset and prepare ALL features for analysis."""
        print("Loading and preparing dataset...")
        self.df = pd.read_csv(self.dataset_path)
        print(f"Dataset shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        
        # Prepare ALL features for correlation analysis
        self.prepare_all_features()
        
    def prepare_all_features(self):
        """Prepare ALL features including categorical encoded ones for analysis."""
        print("\nPreparing ALL features for analysis...")
        
        # Start with ALL numeric features (don't exclude anything yet)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove only non-predictive identifiers
        exclude_columns = ['transaction_id', 'customer_id']  # Keep everything else
        
        # Create feature dataframe with all numeric features
        feature_df = self.df[numeric_cols].copy()
        
        # Remove only the excluded columns
        for col in exclude_columns:
            if col in feature_df.columns:
                feature_df = feature_df.drop(col, axis=1)
        
        # Encode ALL categorical features
        categorical_columns = ['channel', 'merchant_category', 'bank', 'location', 'age_group']
        
        print(f"Encoding categorical features: {categorical_columns}")
        for col in categorical_columns:
            if col in self.df.columns:
                le = LabelEncoder()
                feature_df[f'{col}_encoded'] = le.fit_transform(self.df[col].astype(str))
                print(f"  - Encoded {col}: {len(self.df[col].unique())} unique values")
        
        # Store fraud target separately
        if 'is_fraud' in feature_df.columns:
            self.fraud_target = feature_df['is_fraud'].copy()
        else:
            self.fraud_target = self.df['is_fraud'].copy()
        
        # Create two versions:
        # 1. All features including is_fraud for correlation matrix
        self.all_features_df = feature_df.copy()
        
        # 2. Features without is_fraud for VIF calculation
        if 'is_fraud' in feature_df.columns:
            self.feature_df = feature_df.drop('is_fraud', axis=1)
        else:
            self.feature_df = feature_df.copy()
        
        self.feature_names = list(self.feature_df.columns)
        
        print(f"\nFeature preparation complete:")
        print(f"  - Total features for analysis: {len(self.feature_names)}")
        print(f"  - Numeric features: {len([c for c in self.feature_names if not c.endswith('_encoded')])}")
        print(f"  - Encoded categorical features: {len([c for c in self.feature_names if c.endswith('_encoded')])}")
        print(f"  - Including fraud target for correlation matrix: {len(self.all_features_df.columns)} total columns")
        
        # Display all features
        print(f"\nAll features included:")
        for i, feature in enumerate(self.feature_names, 1):
            print(f"  {i:2d}. {feature}")
        
    def calculate_correlations_with_fraud(self):
        """Calculate correlations between ALL features and fraud status."""
        print(f"\nCalculating correlations with fraud status for {len(self.feature_names)} features...")
        
        correlations = {}
        
        for i, feature in enumerate(self.feature_names, 1):
            if i % 10 == 0:
                print(f"  Processing feature {i}/{len(self.feature_names)}: {feature}")
            
            try:
                # Calculate point-biserial correlation (for binary target vs continuous feature)
                feature_data = self.feature_df[feature].dropna()
                fraud_data = self.fraud_target[feature_data.index]
                
                # Remove any remaining NaN values
                valid_indices = ~(feature_data.isna() | fraud_data.isna())
                if valid_indices.sum() < 10:  # Need at least 10 valid observations
                    print(f"    Warning: Insufficient valid data for {feature}")
                    continue
                
                feature_clean = feature_data[valid_indices]
                fraud_clean = fraud_data[valid_indices]
                
                # Calculate correlation
                if len(feature_clean.unique()) > 1:  # Feature has variance
                    corr_coef, p_value = pointbiserialr(fraud_clean, feature_clean)
                else:
                    corr_coef, p_value = 0.0, 1.0
                
                # Handle NaN correlations
                if np.isnan(corr_coef):
                    corr_coef = 0.0
                
                # Categorize correlation strength
                abs_corr = abs(corr_coef)
                if abs_corr >= 0.7:
                    strength = "very strong"
                elif abs_corr >= 0.5:
                    strength = "strong"
                elif abs_corr >= 0.3:
                    strength = "moderate"
                elif abs_corr >= 0.1:
                    strength = "weak"
                else:
                    strength = "negligible"
                
                correlations[feature] = {
                    'correlation': corr_coef,
                    'abs_correlation': abs_corr,
                    'p_value': p_value,
                    'strength': strength,
                    'valid_observations': len(feature_clean)
                }
                
            except Exception as e:
                print(f"    Error calculating correlation for {feature}: {e}")
                correlations[feature] = {
                    'correlation': 0.0,
                    'abs_correlation': 0.0,
                    'p_value': 1.0,
                    'strength': 'error',
                    'valid_observations': 0
                }
        
        # Sort by absolute correlation
        self.correlation_results = dict(sorted(correlations.items(), 
                                             key=lambda x: x[1]['abs_correlation'], 
                                             reverse=True))
        
        # Print comprehensive results
        print(f"\n{'='*80}")
        print(f"CORRELATION ANALYSIS RESULTS")
        print(f"{'='*80}")
        
        # Summary by strength
        strength_counts = {}
        for feature, stats in self.correlation_results.items():
            strength = stats['strength']
            strength_counts[strength] = strength_counts.get(strength, 0) + 1
        
        print(f"\nCorrelation Strength Distribution:")
        for strength in ['very strong', 'strong', 'moderate', 'weak', 'negligible', 'error']:
            count = strength_counts.get(strength, 0)
            if count > 0:
                print(f"  {strength:12}: {count:3d} features")
        
        # Top correlations
        print(f"\nTop 15 correlations with fraud status:")
        print(f"{'Rank':<4} {'Feature':<30} {'Correlation':<12} {'Abs Corr':<10} {'Strength':<12} {'P-value'}")
        print(f"{'-'*80}")
        
        for i, (feature, stats) in enumerate(list(self.correlation_results.items())[:15], 1):
            print(f"{i:3d}. {feature:<30} {stats['correlation']:8.4f}   {stats['abs_correlation']:8.4f}   {stats['strength']:<12} {stats['p_value']:.2e}")
        
        return self.correlation_results
    
    def calculate_full_correlation_matrix(self):
        """Calculate complete correlation matrix between ALL features including fraud."""
        print(f"\nCalculating full correlation matrix for all {len(self.all_features_df.columns)} features...")
        
        # Calculate correlation matrix for ALL features including is_fraud
        self.full_correlation_matrix = self.all_features_df.corr()
        
        print(f"Full correlation matrix shape: {self.full_correlation_matrix.shape}")
        
        # Find highly correlated feature pairs (excluding fraud correlations)
        high_corr_pairs = []
        feature_cols = [col for col in self.full_correlation_matrix.columns if col != 'is_fraud']
        
        for i in range(len(feature_cols)):
            for j in range(i+1, len(feature_cols)):
                corr_val = self.full_correlation_matrix.loc[feature_cols[i], feature_cols[j]]
                if abs(corr_val) > 0.7:  # High correlation threshold
                    high_corr_pairs.append({
                        'feature1': feature_cols[i],
                        'feature2': feature_cols[j],
                        'correlation': corr_val
                    })
        
        print(f"Found {len(high_corr_pairs)} highly correlated feature pairs (|r| > 0.7)")
        if high_corr_pairs:
            print("Top 10 highly correlated pairs:")
            for i, pair in enumerate(sorted(high_corr_pairs, key=lambda x: abs(x['correlation']), reverse=True)[:10], 1):
                print(f"  {i:2d}. {pair['feature1']:<25} <-> {pair['feature2']:<25}: r = {pair['correlation']:6.3f}")
        
        return self.full_correlation_matrix
    
    def calculate_vif(self):
        """Calculate Variance Inflation Factors for multicollinearity assessment."""
        print(f"\nCalculating Variance Inflation Factors (VIF) for {len(self.feature_df.columns)} features...")
        
        # Prepare data for VIF calculation
        X = self.feature_df.copy()
        
        # Remove any features with zero variance
        zero_var_features = []
        for col in X.columns:
            if X[col].var() == 0 or X[col].nunique() <= 1:
                zero_var_features.append(col)
        
        if zero_var_features:
            print(f"Removing {len(zero_var_features)} zero-variance features: {zero_var_features}")
            X = X.drop(columns=zero_var_features)
        
        # Remove features with any infinite or missing values
        inf_features = []
        for col in X.columns:
            if np.isinf(X[col]).any() or X[col].isna().any():
                inf_features.append(col)
        
        if inf_features:
            print(f"Removing {len(inf_features)} features with inf/missing values: {inf_features}")
            X = X.drop(columns=inf_features)
        
        print(f"Computing VIF for {len(X.columns)} clean features...")
        
        # Calculate VIF for each feature
        vif_data = []
        feature_names = X.columns.tolist()
        
        for i, feature in enumerate(feature_names):
            if (i + 1) % 5 == 0:
                print(f"  Processing VIF {i+1}/{len(feature_names)}: {feature}")
            
            try:
                vif_value = variance_inflation_factor(X.values, i)
                
                # Handle infinite VIF values
                if np.isinf(vif_value) or np.isnan(vif_value):
                    vif_value = 999.99
                
                # Categorize VIF level
                if vif_value >= 10:
                    vif_level = "high"
                elif vif_value >= 5:
                    vif_level = "moderate"
                elif vif_value >= 3:
                    vif_level = "elevated"
                else:
                    vif_level = "acceptable"
                
                vif_data.append({
                    'feature': feature,
                    'vif': vif_value,
                    'vif_level': vif_level
                })
                
            except Exception as e:
                print(f"    Warning: Could not calculate VIF for {feature}: {e}")
                vif_data.append({
                    'feature': feature,
                    'vif': 999.99,
                    'vif_level': "error"
                })
        
        # Sort by VIF value
        vif_data.sort(key=lambda x: x['vif'], reverse=True)
        self.vif_results = vif_data
        
        # Summary statistics
        valid_vifs = [x['vif'] for x in vif_data if x['vif'] < 999]
        if valid_vifs:
            max_vif = max(valid_vifs)
            mean_vif = np.mean(valid_vifs)
            problematic_features = [x for x in vif_data if x['vif'] >= 10]
            
            print(f"\nVIF Summary:")
            print(f"  Maximum VIF: {max_vif:.2f}")
            print(f"  Mean VIF: {mean_vif:.2f}")
            print(f"  Features with VIF ≥ 10: {len(problematic_features)}")
            print(f"  Features with VIF ≥ 5: {len([x for x in vif_data if x['vif'] >= 5])}")
            print(f"  Features with VIF ≥ 3: {len([x for x in vif_data if x['vif'] >= 3])}")
            print(f"  Features with VIF < 3: {len([x for x in vif_data if x['vif'] < 3])}")
            
            # Show most problematic features
            print(f"\nTop 15 features with highest VIF:")
            print(f"{'Rank':<4} {'Feature':<30} {'VIF':<10} {'Level'}")
            print(f"{'-'*60}")
            for i, vif_info in enumerate(vif_data[:15], 1):
                if vif_info['vif'] < 999:
                    print(f"{i:3d}. {vif_info['feature']:<30} {vif_info['vif']:8.2f}  {vif_info['vif_level']}")
        
        return self.vif_results
    
    def create_correlation_visualizations(self):
        """Create comprehensive correlation and multicollinearity visualizations."""
        print("\nCreating correlation and multicollinearity visualizations...")
        
        # Create 2x2 subplot figure with better spacing
        fig = plt.figure(figsize=(24, 20))
        
        # Create subplots with custom spacing
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25, 
                             left=0.08, right=0.95, top=0.92, bottom=0.08)
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
        
        fig.suptitle('Feature Correlation and Multicollinearity Analysis\nNigerian Banking Fraud Detection Dataset', 
                     fontsize=18, fontweight='bold', y=0.97)
        
        # Plot 1: Complete Feature Correlation Matrix
        print("  Creating correlation matrix heatmap...")
        
        # Select top correlated features for visualization (include all important ones)
        important_features = [
            'tx_count_24h', 'amount_vs_mean_ratio', 'velocity_score', 'amount', 
            'composite_risk', 'merchant_risk_score', 'amount_sum_24h',
            'channel_encoded', 'merchant_category_encoded', 'amount_log',
            'hour', 'is_weekend', 'is_peak_hour', 'is_fraud'
        ]
        
        # Add any missing high-correlation features
        top_features = list(self.correlation_results.keys())[:20]
        for feature in top_features:
            if feature not in important_features and feature in self.full_correlation_matrix.columns:
                important_features.append(feature)
        
        # Ensure we have features that exist in our data
        available_features = [f for f in important_features if f in self.full_correlation_matrix.columns]
        
        if len(available_features) < 15:
            # Add more features to reach at least 15
            for feature in self.full_correlation_matrix.columns:
                if feature not in available_features and len(available_features) < 25:
                    available_features.append(feature)
        
        print(f"    Visualizing {len(available_features)} key features in correlation matrix")
        
        corr_subset = self.full_correlation_matrix.loc[available_features, available_features]
        
        # Create correlation heatmap with better readability
        mask = np.triu(np.ones_like(corr_subset, dtype=bool), k=1)
        
        sns.heatmap(corr_subset, mask=mask, annot=False, cmap='RdBu_r', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8}, 
                   ax=ax1, vmin=-1, vmax=1)
        
        ax1.set_title('Feature Correlation Matrix\n(Blue-White-Red Diverging Scheme)', 
                     fontsize=14, fontweight='bold', pad=20)
        
        # Fix x-axis labels - better rotation and positioning
        ax1.set_xticklabels([f.replace('_', ' ').title() for f in available_features], 
                           rotation=45, ha='right', fontsize=9)
        ax1.set_yticklabels([f.replace('_', ' ').title() for f in available_features], 
                           rotation=0, ha='right', fontsize=9)
        
        # Adjust subplot to prevent label cutoff
        plt.setp(ax1.get_xticklabels(), ha='right', rotation_mode='anchor')
        
        # Plot 2: Top Correlations with Fraud Status
        print("  Creating top correlations bar chart...")
        
        top_n = 20
        top_corr_features = list(self.correlation_results.keys())[:top_n]
        top_corr_values = [self.correlation_results[f]['correlation'] for f in top_corr_features]
        
        # Color-code by correlation strength
        colors = []
        for v in top_corr_values:
            abs_v = abs(v)
            if abs_v >= 0.3:
                colors.append('darkred')
            elif abs_v >= 0.1:
                colors.append('orange')
            elif abs_v >= 0.05:
                colors.append('lightcoral')
            else:
                colors.append('lightblue')
        
        y_pos = range(len(top_corr_features))
        bars = ax2.barh(y_pos, top_corr_values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([f.replace('_', ' ').title()[:25] for f in top_corr_features], fontsize=10)
        ax2.set_xlabel('Correlation with Fraud Status', fontsize=12)
        ax2.set_title('Top Correlations with Fraud Status\n(Red: Strong |r|≥0.3, Orange: Moderate |r|≥0.1)', 
                     fontsize=14, fontweight='bold', pad=20)
        
        # Add reference lines
        ax2.axvline(x=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
        ax2.axvline(x=0.1, color='orange', linestyle='--', alpha=0.7, linewidth=2, label='Moderate (0.1)')
        ax2.axvline(x=0.3, color='darkred', linestyle='--', alpha=0.7, linewidth=2, label='Strong (0.3)')
        ax2.axvline(x=-0.1, color='orange', linestyle='--', alpha=0.7, linewidth=2)
        ax2.axvline(x=-0.3, color='darkred', linestyle='--', alpha=0.7, linewidth=2)
        ax2.grid(axis='x', alpha=0.3)
        ax2.legend(fontsize=10)
        
        # Add correlation values as text
        for i, (bar, val) in enumerate(zip(bars, top_corr_values)):
            if abs(val) >= 0.01:  # Only show values >= 0.01
                x_pos = val + (0.015 if val >= 0 else -0.015)
                ax2.text(x_pos, bar.get_y() + bar.get_height()/2, 
                        f'{val:.3f}', ha='left' if val >= 0 else 'right', va='center', 
                        fontsize=9, fontweight='bold')
        
        # Plot 3: Distribution of Correlation Strengths
        print("  Creating correlation strength distribution...")
        
        corr_strengths = [self.correlation_results[f]['strength'] for f in self.correlation_results if self.correlation_results[f]['strength'] != 'error']
        strength_counts = pd.Series(corr_strengths).value_counts()
        
        strength_order = ['very strong', 'strong', 'moderate', 'weak', 'negligible']
        strength_counts = strength_counts.reindex(strength_order, fill_value=0)
        
        colors_strength = ['darkred', 'red', 'orange', 'lightcoral', 'lightgray']
        bars3 = ax3.bar(strength_counts.index, strength_counts.values, 
                       color=colors_strength, alpha=0.8, edgecolor='black', linewidth=1)
        
        ax3.set_title('Distribution of Correlation Strengths\nwith Fraud Status', 
                     fontsize=14, fontweight='bold', pad=20)
        ax3.set_ylabel('Number of Features', fontsize=12)
        ax3.set_xlabel('Correlation Strength Category', fontsize=12)
        ax3.tick_params(axis='x', rotation=45, labelsize=11)
        ax3.grid(axis='y', alpha=0.3)
        
        # Add count labels on bars
        for bar, count in zip(bars3, strength_counts.values):
            if count > 0:
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(strength_counts.values)*0.01,
                        str(count), ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Plot 4: Feature Importance by Correlation Strength (Lollipop Chart)
        print("  Creating feature importance lollipop chart...")
        
        abs_corr_values = [self.correlation_results[f]['abs_correlation'] for f in top_corr_features]
        
        # Create lollipop chart
        y_pos = range(len(top_corr_features))
        
        # Color points based on correlation strength
        point_colors = []
        for v in abs_corr_values:
            if v >= 0.3:
                point_colors.append('darkred')
            elif v >= 0.1:
                point_colors.append('orange')
            elif v >= 0.05:
                point_colors.append('lightcoral')
            else:
                point_colors.append('lightblue')
        
        # Create lollipops
        for i, (y, x) in enumerate(zip(y_pos, abs_corr_values)):
            ax4.plot([0, x], [y, y], color='gray', linestyle='-', alpha=0.6, linewidth=2)
            ax4.scatter(x, y, s=100, alpha=0.9, c=point_colors[i], 
                       edgecolors='black', linewidth=1, zorder=5)
        
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels([f.replace('_', ' ').title()[:25] for f in top_corr_features], fontsize=10)
        ax4.set_xlabel('Absolute Correlation with Fraud', fontsize=12)
        ax4.set_title('Feature Importance by Correlation Strength', 
                     fontsize=14, fontweight='bold', pad=20)
        
        # Add reference lines
        ax4.axvline(x=0.1, color='orange', linestyle='--', alpha=0.7, linewidth=2, label='Moderate (0.1)')
        ax4.axvline(x=0.3, color='darkred', linestyle='--', alpha=0.7, linewidth=2, label='Strong (0.3)')
        ax4.grid(axis='x', alpha=0.3)
        ax4.legend(fontsize=10)
        
        # Add correlation values as text
        for i, (y, val) in enumerate(zip(y_pos, abs_corr_values)):
            if val >= 0.01:  # Only show values >= 0.01
                ax4.text(val + 0.008, y, f'{val:.3f}', ha='left', va='center', 
                        fontsize=9, fontweight='bold')
        
        plt.savefig('correlation_multicollinearity_analysis.png', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print("✓ Correlation analysis visualization saved as: correlation_multicollinearity_analysis.png")
        
        return fig
    
    def create_vif_visualization(self):
        """Create VIF (multicollinearity) visualization with better spacing."""
        print("\nCreating VIF visualization...")
        
        # Prepare VIF data
        vif_data_clean = [x for x in self.vif_results if x['vif'] < 999]
        
        if not vif_data_clean:
            print("No valid VIF values to plot")
            return None
        
        vif_features = [x['feature'] for x in vif_data_clean]
        vif_values = [x['vif'] for x in vif_data_clean]
        
        # Color-code by VIF level
        colors = []
        for vif in vif_values:
            if vif >= 10:
                colors.append('red')
            elif vif >= 5:
                colors.append('orange')  
            elif vif >= 3:
                colors.append('yellow')
            else:
                colors.append('green')
        
        # Create VIF plot with better spacing
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        
        y_pos = range(len(vif_features))
        bars = ax.barh(y_pos, vif_values, color=colors, alpha=0.8, 
                      edgecolor='black', linewidth=0.8)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f.replace('_', ' ').title() for f in vif_features], fontsize=11)
        ax.set_xlabel('Variance Inflation Factor (VIF)', fontsize=14)
        ax.set_title('Multicollinearity Assessment: Variance Inflation Factors\n' + 
                    '(Green: VIF<3, Yellow: 3≤VIF<5, Orange: 5≤VIF<10, Red: VIF≥10)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Add reference lines with better visibility
        ax.axvline(x=3, color='orange', linestyle='--', alpha=0.8, linewidth=2, 
                  label='Moderate Threshold (3.0)')
        ax.axvline(x=5, color='red', linestyle='--', alpha=0.8, linewidth=2, 
                  label='Problematic Threshold (5.0)')
        ax.axvline(x=10, color='darkred', linestyle='--', alpha=0.8, linewidth=2, 
                  label='Severe Threshold (10.0)')
        
        # Add VIF values as text
        for i, (bar, vif) in enumerate(zip(bars, vif_values)):
            x_pos = vif + max(vif_values) * 0.01
            ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                   f'{vif:.1f}', ha='left', va='center', 
                   fontsize=10, fontweight='bold')
        
        ax.legend(fontsize=12, loc='lower right')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('vif_multicollinearity_analysis.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print("✓ VIF analysis visualization saved as: vif_multicollinearity_analysis.png")
        
        return fig
    
    def generate_summary_report(self):
        """Generate comprehensive summary report."""
        print(f"\n" + "="*100)
        print("FEATURE CORRELATION AND MULTICOLLINEARITY SUMMARY")
        print("="*100)
        
        # Correlation summary
        print(f"\nCorrelation with Fraud Status Summary:")
        print("-"*60)
        
        valid_corr_results = {k: v for k, v in self.correlation_results.items() if v['strength'] != 'error'}
        
        very_strong_corr = [f for f, stats in valid_corr_results.items() if stats['abs_correlation'] >= 0.7]
        strong_corr = [f for f, stats in valid_corr_results.items() if 0.5 <= stats['abs_correlation'] < 0.7]
        moderate_corr = [f for f, stats in valid_corr_results.items() if 0.3 <= stats['abs_correlation'] < 0.5]
        weak_corr = [f for f, stats in valid_corr_results.items() if 0.1 <= stats['abs_correlation'] < 0.3]
        negligible_corr = [f for f, stats in valid_corr_results.items() if stats['abs_correlation'] < 0.1]
        
        print(f"Very strong correlations (|r| ≥ 0.7): {len(very_strong_corr)}")
        print(f"Strong correlations (0.5 ≤ |r| < 0.7): {len(strong_corr)}")
        print(f"Moderate correlations (0.3 ≤ |r| < 0.5): {len(moderate_corr)}")
        print(f"Weak correlations (0.1 ≤ |r| < 0.3): {len(weak_corr)}")
        print(f"Negligible correlations (|r| < 0.1): {len(negligible_corr)}")
        
        print(f"\nTop 10 Features Correlated with Fraud:")
        for i, (feature, stats) in enumerate(list(valid_corr_results.items())[:10], 1):
            print(f"{i:2d}. {feature:<35}: r = {stats['correlation']:7.4f} ({stats['strength']:<12}) p = {stats['p_value']:.2e}")
        
        # Identify key behavioral vs traditional features
        behavioral_features = [f for f in valid_corr_results.keys() if any(keyword in f.lower() for keyword in 
                             ['velocity', 'count', 'ratio', 'deviation', 'risk'])]
        traditional_features = [f for f in valid_corr_results.keys() if any(keyword in f.lower() for keyword in 
                              ['amount', 'hour', 'channel', 'merchant', 'bank'])]
        
        if behavioral_features:
            print(f"\nBehavioral Features Performance:")
            behavioral_corrs = [valid_corr_results[f]['abs_correlation'] for f in behavioral_features[:5]]
            print(f"  Average |correlation|: {np.mean(behavioral_corrs):.4f}")
            
        if traditional_features:
            print(f"\nTraditional Features Performance:")
            traditional_corrs = [valid_corr_results[f]['abs_correlation'] for f in traditional_features[:5]]
            print(f"  Average |correlation|: {np.mean(traditional_corrs):.4f}")
        
        # VIF summary
        if self.vif_results:
            print(f"\nMulticollinearity (VIF) Summary:")
            print("-"*50)
            
            valid_vifs = [x for x in self.vif_results if x['vif'] < 999]
            if valid_vifs:
                high_vif = [x for x in valid_vifs if x['vif'] >= 10]
                moderate_vif = [x for x in valid_vifs if 5 <= x['vif'] < 10]
                elevated_vif = [x for x in valid_vifs if 3 <= x['vif'] < 5]
                acceptable_vif = [x for x in valid_vifs if x['vif'] < 3]
                
                max_vif = max([x['vif'] for x in valid_vifs])
                mean_vif = np.mean([x['vif'] for x in valid_vifs])
                
                print(f"Maximum VIF: {max_vif:.2f}")
                print(f"Mean VIF: {mean_vif:.2f}")
                print(f"High multicollinearity (VIF ≥ 10): {len(high_vif)} features")
                print(f"Moderate multicollinearity (5 ≤ VIF < 10): {len(moderate_vif)} features")
                print(f"Elevated multicollinearity (3 ≤ VIF < 5): {len(elevated_vif)} features")
                print(f"Acceptable multicollinearity (VIF < 3): {len(acceptable_vif)} features")
                
                if high_vif:
                    print(f"\nFeatures with High Multicollinearity (VIF ≥ 10):")
                    for i, vif_info in enumerate(high_vif[:10], 1):
                        print(f"  {i:2d}. {vif_info['feature']:<35}: VIF = {vif_info['vif']:7.2f}")
                
                if moderate_vif:
                    print(f"\nFeatures with Moderate Multicollinearity (5 ≤ VIF < 10):")
                    for i, vif_info in enumerate(moderate_vif[:5], 1):
                        print(f"  {i:2d}. {vif_info['feature']:<35}: VIF = {vif_info['vif']:7.2f}")
        
        # Key Insights
        print(f"\nKey Insights:")
        print("-"*30)
        
        # Best discriminators
        top_features = list(valid_corr_results.items())[:3]
        if top_features:
            print(f"• Top discriminative features:")
            for i, (feature, stats) in enumerate(top_features, 1):
                pct_change = stats['correlation'] * 100
                direction = "increases" if stats['correlation'] > 0 else "decreases"
                print(f"  {i}. {feature}: {abs(pct_change):.1f}% correlation - fraud {direction} with higher values")
        
        # Behavioral vs Traditional
        if behavioral_features and traditional_features:
            behavioral_avg = np.mean([valid_corr_results[f]['abs_correlation'] for f in behavioral_features[:5]])
            traditional_avg = np.mean([valid_corr_results[f]['abs_correlation'] for f in traditional_features[:5]])
            
            if behavioral_avg > traditional_avg:
                ratio = behavioral_avg / traditional_avg
                print(f"• Behavioral features show {ratio:.1f}x stronger correlation than traditional features")
            else:
                ratio = traditional_avg / behavioral_avg
                print(f"• Traditional features show {ratio:.1f}x stronger correlation than behavioral features")
        
        # Recommendations
        print(f"\nRecommendations:")
        print("-"*25)
        
        if strong_corr or moderate_corr:
            total_strong_moderate = len(strong_corr) + len(moderate_corr)
            print(f"• Focus modeling on {total_strong_moderate} features with strong/moderate correlations")
            print(f"• Priority features: {', '.join(list(valid_corr_results.keys())[:5])}")
        else:
            print(f"• Consider non-linear methods - limited linear correlations found")
            print(f"• Ensemble methods may capture complex feature interactions")
        
        if self.vif_results:
            high_vif_count = len([x for x in self.vif_results if x['vif'] >= 10])
            moderate_vif_count = len([x for x in self.vif_results if 5 <= x['vif'] < 10])
            
            if high_vif_count > 0:
                print(f"• Apply regularization (Ridge/Lasso) to address {high_vif_count} high-VIF features")
                print(f"• Consider feature selection to remove redundant variables")
            elif moderate_vif_count > 0:
                print(f"• Monitor {moderate_vif_count} moderate-VIF features during model training")
            else:
                print(f"• Multicollinearity levels are acceptable for most ML algorithms")
        
        # Feature engineering suggestions
        top_corr_types = {}
        for feature, stats in list(valid_corr_results.items())[:10]:
            if 'count' in feature.lower() or 'velocity' in feature.lower():
                top_corr_types['behavioral'] = top_corr_types.get('behavioral', 0) + 1
            elif 'amount' in feature.lower():
                top_corr_types['amount'] = top_corr_types.get('amount', 0) + 1
            elif 'time' in feature.lower() or 'hour' in feature.lower():
                top_corr_types['temporal'] = top_corr_types.get('temporal', 0) + 1
        
        if top_corr_types:
            dominant_type = max(top_corr_types, key=top_corr_types.get)
            print(f"• {dominant_type.title()} features dominate top correlations - expand feature engineering in this area")
    
    def run_complete_analysis(self):
        """Run the complete correlation and multicollinearity analysis."""
        print("Starting Feature Correlation and Multicollinearity Analysis...")
        print("="*80)
        
        # Load and prepare data
        self.load_and_prepare_data()
        
        # Calculate correlations
        self.calculate_correlations_with_fraud()
        self.calculate_full_correlation_matrix()
        
        # Calculate VIF
        self.calculate_vif()
        
        # Create visualizations
        corr_fig = self.create_correlation_visualizations()
        vif_fig = self.create_vif_visualization()
        
        # Generate summary
        self.generate_summary_report()
        
        print(f"\n" + "="*80)
        print("✅ CORRELATION AND MULTICOLLINEARITY ANALYSIS COMPLETED!")
        print("="*80)
        print(f"Files generated:")
        print(f"  • correlation_multicollinearity_analysis.png - Correlation analysis (4-panel)")
        print(f"  • vif_multicollinearity_analysis.png - VIF analysis")
        print(f"  • Analysis covers {len(self.feature_names)} features")
        print(f"  • Correlation matrix includes {len(self.all_features_df.columns)} total columns")
        
        return {
            'correlation_results': self.correlation_results,
            'correlation_matrix': self.full_correlation_matrix,
            'vif_results': self.vif_results,
            'correlation_fig': corr_fig,
            'vif_fig': vif_fig,
            'feature_count': len(self.feature_names),
            'total_columns': len(self.all_features_df.columns)
        }


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Feature correlation and multicollinearity analysis')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the fraud dataset CSV file')
    
    args = parser.parse_args()
    
    # Run analysis
    analyzer = CorrelationMulticollinearityAnalyzer(args.dataset)
    results = analyzer.run_complete_analysis()
    
    return results


if __name__ == "__main__":
    # Example usage
    try:
        results = main()
    except Exception as e:
        print(f"Error during execution: {e}")
        print("\nUsage example:")
        print("python correlation_multicollinearity_analysis.py --dataset nibss_fraud_dataset.csv")
        
        # Alternative: Run with hardcoded dataset path
        print("\nRunning with default dataset path...")
        analyzer = CorrelationMulticollinearityAnalyzer("nibss_fraud_dataset.csv")
        results = analyzer.run_complete_analysis()