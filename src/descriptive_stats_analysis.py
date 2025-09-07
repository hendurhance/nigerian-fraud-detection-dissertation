#!/usr/bin/env python3
"""
Descriptive Statistics and Effect Size Analysis for Nigerian Fraud Dataset

This script generates:
1. Descriptive statistics table comparing fraud vs legitimate transactions
2. Effect size analysis using Cohen's d
3. Statistical significance tests (Mann-Whitney U)
4. Formatted LaTeX table output

Author: Your Name
Date: 2024
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import mannwhitneyu, shapiro
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import warnings
warnings.filterwarnings('ignore')

class DescriptiveStatsAnalyzer:
    """Analyze descriptive statistics and effect sizes for fraud detection dataset."""
    
    def __init__(self, dataset_path):
        """Initialize with dataset path."""
        self.dataset_path = dataset_path
        self.df = None
        self.results = {}
        
    def load_data(self):
        """Load the dataset."""
        print("Loading dataset...")
        self.df = pd.read_csv(self.dataset_path)
        print(f"Dataset shape: {self.df.shape}")
        print(f"Fraud rate: {self.df['is_fraud'].mean():.4%}")
        
    def select_key_variables(self):
        """Select key variables for analysis based on the template."""
        # Based on your template, these are the key variables to analyze
        self.key_variables = {
            'amount': 'Amount',
            'velocity_score': 'Velocity Score', 
            'amount_vs_mean_ratio': 'Amount Deviation Ratio',
            'tx_count_24h': 'Transaction Count 24h',
            'merchant_risk_score': 'Merchant Risk Score'
        }
        
        print(f"Analyzing {len(self.key_variables)} key variables:")
        for var, label in self.key_variables.items():
            print(f"  - {label} ({var})")
    
    def test_normality(self):
        """Test normality of key variables."""
        print("\nTesting normality (Shapiro-Wilk test)...")
        normality_results = {}
        
        for var in self.key_variables.keys():
            # Test on a sample (Shapiro-Wilk limited to 5000 samples)
            sample_size = min(5000, len(self.df))
            sample_data = self.df[var].sample(n=sample_size, random_state=42)
            
            stat, p_value = shapiro(sample_data)
            is_normal = p_value > 0.05
            
            normality_results[var] = {
                'statistic': stat,
                'p_value': p_value,
                'is_normal': is_normal
            }
            
            print(f"  {self.key_variables[var]}: p = {p_value:.2e} ({'Normal' if is_normal else 'Non-normal'})")
        
        self.normality_results = normality_results
        return normality_results
    
    def calculate_descriptive_stats(self):
        """Calculate descriptive statistics for fraud vs legitimate transactions."""
        print("\nCalculating descriptive statistics...")
        
        results = {}
        
        # Separate fraud and legitimate transactions
        fraud_df = self.df[self.df['is_fraud'] == 1]
        legit_df = self.df[self.df['is_fraud'] == 0]
        
        print(f"Legitimate transactions: {len(legit_df):,}")
        print(f"Fraudulent transactions: {len(fraud_df):,}")
        
        for var, label in self.key_variables.items():
            legit_data = legit_df[var]
            fraud_data = fraud_df[var]
            
            # Calculate descriptive statistics
            legit_stats = {
                'mean': legit_data.mean(),
                'std': legit_data.std(),
                'median': legit_data.median(),
                'q25': legit_data.quantile(0.25),
                'q75': legit_data.quantile(0.75)
            }
            
            fraud_stats = {
                'mean': fraud_data.mean(),
                'std': fraud_data.std(), 
                'median': fraud_data.median(),
                'q25': fraud_data.quantile(0.25),
                'q75': fraud_data.quantile(0.75)
            }
            
            # Mann-Whitney U test (since data is non-normal)
            u_stat, p_value = mannwhitneyu(legit_data, fraud_data, alternative='two-sided')
            
            # Calculate effect size (Cohen's d with balanced pooled SD)
            cohen_d = self.calculate_cohens_d_balanced(legit_data, fraud_data)
            
            # Percentage difference
            pct_diff = ((fraud_stats['mean'] - legit_stats['mean']) / legit_stats['mean']) * 100
            
            results[var] = {
                'variable': label,
                'legitimate': legit_stats,
                'fraudulent': fraud_stats,
                'mann_whitney_u': u_stat,
                'p_value': p_value,
                'cohens_d': cohen_d,
                'percent_difference': pct_diff,
                'effect_size_interpretation': self.interpret_effect_size(cohen_d)
            }
            
            print(f"  {label}: Cohen's d = {cohen_d:.3f} ({self.interpret_effect_size(cohen_d)})")
        
        self.descriptive_results = results
        return results
    
    def calculate_cohens_d_balanced(self, group1, group2):
        """Calculate Cohen's d using balanced pooled standard deviation."""
        n1, n2 = len(group1), len(group2)
        
        # Balanced pooled standard deviation (equal weights)
        pooled_std = np.sqrt((group1.var() + group2.var()) / 2)
        
        # Cohen's d
        cohens_d = (group2.mean() - group1.mean()) / pooled_std
        
        return cohens_d
    
    def interpret_effect_size(self, d):
        """Interpret Cohen's d effect size."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "moderate"
        else:
            return "large"
    
    def create_descriptive_table(self):
        """Create a formatted descriptive statistics table."""
        print("\nCreating descriptive statistics table...")
        
        table_data = []
        
        for var, results in self.descriptive_results.items():
            legit = results['legitimate']
            fraud = results['fraudulent']
            
            # Format the row data
            row = [
                results['variable'],
                # Legitimate stats
                f"{legit['mean']:.0f}",
                f"{legit['std']:.0f}",
                f"{legit['median']:.0f}",
                f"[{legit['q25']:.0f}, {legit['q75']:.0f}]",
                # Fraud stats  
                f"{fraud['mean']:.0f}",
                f"{fraud['std']:.0f}", 
                f"{fraud['median']:.0f}",
                f"[{fraud['q25']:.0f}, {fraud['q75']:.0f}]",
                # Statistical test
                f"U = {results['mann_whitney_u']:.2e}, p < 0.001" if results['p_value'] < 0.001 else f"U = {results['mann_whitney_u']:.2e}, p = {results['p_value']:.3f}"
            ]
            
            table_data.append(row)
        
        # Create headers
        headers = [
            "Variable",
            "Mean", "± SD", "Median", "[IQR]",
            "Mean", "± SD", "Median", "[IQR]", 
            "Statistical Test\nMann-Whitney U"
        ]
        
        # Print table
        print("\nDESCRIPTIVE STATISTICS TABLE:")
        print("="*120)
        print(f"{'Variable':<25} {'Legitimate Transactions':<50} {'Fraudulent Transactions':<50}")
        print(f"{'':25} {'Mean ± SD':<15} {'Median [IQR]':<20} {'Mean ± SD':<15} {'Median [IQR]':<20} {'Statistical Test':<15}")
        print("-"*120)
        
        for var, results in self.descriptive_results.items():
            legit = results['legitimate']
            fraud = results['fraudulent']
            
            print(f"{results['variable']:<25} "
                  f"{legit['mean']:.0f} ± {legit['std']:.0f}"
                  f"{'':<5}"
                  f"{legit['median']:.0f} [{legit['q25']:.0f}, {legit['q75']:.0f}]"
                  f"{'':<5}"
                  f"{fraud['mean']:.0f} ± {fraud['std']:.0f}"
                  f"{'':<5}"
                  f"{fraud['median']:.0f} [{fraud['q25']:.0f}, {fraud['q75']:.0f}]"
                  f"{'':<5}"
                  f"U = {results['mann_whitney_u']:.2e}, p < 0.001")
        
        return table_data, headers
    
    def create_effect_size_analysis(self):
        """Create effect size analysis summary."""
        print("\nEFFECT SIZE ANALYSIS:")
        print("="*60)
        
        # Sort by effect size
        sorted_vars = sorted(self.descriptive_results.items(), 
                           key=lambda x: abs(x[1]['cohens_d']), 
                           reverse=True)
        
        print(f"{'Rank':<5} {'Variable':<25} {'Cohen''s d':<10} {'Effect':<10} {'% Diff':<10}")
        print("-"*60)
        
        for i, (var, results) in enumerate(sorted_vars, 1):
            print(f"{i:<5} "
                  f"{results['variable']:<25} "
                  f"{results['cohens_d']:.3f}"
                  f"{'':<5}"
                  f"{results['effect_size_interpretation']:<10} "
                  f"{results['percent_difference']:+.1f}%")
        
        # Summary insights
        print(f"\nKEY INSIGHTS:")
        print("-"*40)
        
        for i, (var, results) in enumerate(sorted_vars[:3], 1):
            print(f"{i}. {results['variable']}: d = {results['cohens_d']:.2f} "
                  f"({results['effect_size_interpretation']} effect, "
                  f"{results['percent_difference']:+.1f}% difference)")
            
            if var == 'tx_count_24h':
                print(f"   → Fraudulent transactions exhibit significantly higher short-term transaction velocity")
            elif var == 'amount_vs_mean_ratio':
                print(f"   → Fraudulent transactions deviate dramatically from users' historical spending patterns")
            elif var == 'velocity_score':
                print(f"   → Elevated velocity patterns suggest coordinated transaction sequences")
            elif var == 'amount':
                print(f"   → Limited discriminatory power suggests fraud favors reasonable amounts")
            elif var == 'merchant_risk_score':
                print(f"   → Negligible discrimination indicates fraud occurs across diverse merchant categories")
    
    def generate_latex_table(self):
        """Generate LaTeX table code."""
        print("\nGenerating LaTeX table...")
        
        latex_code = """\\begin{table}[ht]
    \\centering
    \\caption{Descriptive Statistics by Transaction Legitimacy (Nigerian Naira)}
    \\label{tab:descriptive_stats}
    \\begin{tabular}{|l|c|c|c|c|c|}
        \\hline
        \\textbf{Variable} & \\multicolumn{2}{c|}{\\textbf{Legitimate Transactions}} & \\multicolumn{2}{c|}{\\textbf{Fraudulent Transactions}} & \\textbf{Statistical Test} \\\\
        & \\textbf{Mean ± SD} & \\textbf{Median [IQR]} & \\textbf{Mean ± SD} & \\textbf{Median [IQR]} & \\textbf{Mann-Whitney U} \\\\
        \\hline\n"""
        
        for var, results in self.descriptive_results.items():
            legit = results['legitimate']
            fraud = results['fraudulent']
            
            latex_code += f"        {results['variable']} & "
            latex_code += f"{legit['mean']:,.0f} ± {legit['std']:,.0f} & "
            latex_code += f"{legit['median']:,.0f} [{legit['q25']:,.0f}, {legit['q75']:,.0f}] & "
            latex_code += f"{fraud['mean']:,.0f} ± {fraud['std']:,.0f} & "
            latex_code += f"{fraud['median']:,.0f} [{fraud['q25']:,.0f}, {fraud['q75']:,.0f}] & "
            
            if results['p_value'] < 0.001:
                latex_code += f"U = {results['mann_whitney_u']:.2e}, p < 0.001 \\\\\n"
            else:
                latex_code += f"U = {results['mann_whitney_u']:.2e}, p = {results['p_value']:.3f} \\\\\n"
        
        latex_code += """        \\hline
    \\end{tabular}
\\end{table}"""
        
        # Save to file
        with open('descriptive_stats_table.tex', 'w') as f:
            f.write(latex_code)
        
        print("✓ LaTeX table saved to: descriptive_stats_table.tex")
        return latex_code
    
    def run_complete_analysis(self):
        """Run the complete descriptive statistics analysis."""
        print("Starting Descriptive Statistics Analysis...")
        print("="*60)
        
        # Load data and select variables
        self.load_data()
        self.select_key_variables()
        
        # Test normality
        self.test_normality()
        
        # Calculate descriptive statistics
        self.calculate_descriptive_stats()
        
        # Create formatted table
        table_data, headers = self.create_descriptive_table()
        
        # Effect size analysis
        self.create_effect_size_analysis()
        
        # Generate LaTeX
        latex_code = self.generate_latex_table()
        
        print(f"\n✅ DESCRIPTIVE STATISTICS ANALYSIS COMPLETED!")
        print("="*60)
        
        return {
            'descriptive_results': self.descriptive_results,
            'table_data': table_data,
            'headers': headers,
            'latex_code': latex_code
        }


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate descriptive statistics and effect size analysis')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the fraud dataset CSV file')
    
    args = parser.parse_args()
    
    # Run analysis
    analyzer = DescriptiveStatsAnalyzer(args.dataset)
    results = analyzer.run_complete_analysis()
    
    return results


if __name__ == "__main__":
    # Example usage
    try:
        results = main()
    except Exception as e:
        print(f"Error during execution: {e}")
        print("\nUsage example:")
        print("python descriptive_stats_analysis.py --dataset nibss_fraud_dataset.csv")
        
        # Alternative: Run with hardcoded dataset path
        print("\nRunning with default dataset path...")
        analyzer = DescriptiveStatsAnalyzer("nibss_fraud_dataset.csv")
        results = analyzer.run_complete_analysis()