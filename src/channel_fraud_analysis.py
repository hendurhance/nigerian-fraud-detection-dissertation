#!/usr/bin/env python3
"""
Channel-Specific Fraud Analysis for Nigerian Banking Dataset

This script generates:
1. Chi-square analysis of channel-fraud associations
2. Fraud rates by channel with confidence intervals
3. Comprehensive channel visualization (4-panel plot)
4. Statistical significance testing

Author: Your Name
Date: 2024
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.proportion import proportion_confint
import warnings
warnings.filterwarnings('ignore')

class ChannelFraudAnalyzer:
    """Analyze channel-specific fraud patterns in Nigerian banking dataset."""
    
    def __init__(self, dataset_path):
        """Initialize with dataset path."""
        self.dataset_path = dataset_path
        self.df = None
        self.channel_stats = {}
        
    def load_data(self):
        """Load the dataset."""
        print("Loading dataset...")
        self.df = pd.read_csv(self.dataset_path)
        print(f"Dataset shape: {self.df.shape}")
        print(f"Overall fraud rate: {self.df['is_fraud'].mean():.4%}")
        print(f"Channels found: {sorted(self.df['channel'].unique())}")
        
    def calculate_channel_statistics(self):
        """Calculate fraud statistics by channel."""
        print("\nCalculating channel-specific fraud statistics...")
        
        channel_stats = {}
        
        for channel in self.df['channel'].unique():
            channel_data = self.df[self.df['channel'] == channel]
            
            total_transactions = len(channel_data)
            fraud_transactions = channel_data['is_fraud'].sum()
            fraud_rate = fraud_transactions / total_transactions
            
            # Calculate 95% confidence interval for fraud rate
            ci_lower, ci_upper = proportion_confint(fraud_transactions, total_transactions, alpha=0.05)
            
            # Volume percentage
            volume_pct = (total_transactions / len(self.df)) * 100
            
            channel_stats[channel] = {
                'total_transactions': total_transactions,
                'fraud_transactions': fraud_transactions,
                'fraud_rate': fraud_rate,
                'fraud_rate_pct': fraud_rate * 100,
                'ci_lower': ci_lower * 100,
                'ci_upper': ci_upper * 100,
                'volume_pct': volume_pct
            }
            
            print(f"  {channel:<12}: {fraud_rate:.3%} fraud rate ({fraud_transactions:,}/{total_transactions:,} transactions)")
        
        self.channel_stats = channel_stats
        return channel_stats
    
    def chi_square_analysis(self):
        """Perform chi-square test for channel-fraud association."""
        print("\nPerforming chi-square analysis...")
        
        # Create contingency table
        contingency_table = pd.crosstab(self.df['channel'], self.df['is_fraud'])
        
        # Chi-square test
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        # Cramér's V (effect size)
        n = contingency_table.sum().sum()
        cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
        
        print(f"Chi-square test results:")
        print(f"  χ² = {chi2:.2f}")
        print(f"  df = {dof}")
        print(f"  p-value = {p_value:.2e}")
        print(f"  Cramér's V = {cramers_v:.3f}")
        
        # Interpretation
        if p_value < 0.001:
            significance = "highly significant (p < 0.001)"
        elif p_value < 0.01:
            significance = "very significant (p < 0.01)"
        elif p_value < 0.05:
            significance = "significant (p < 0.05)"
        else:
            significance = "not significant (p ≥ 0.05)"
        
        print(f"  Interpretation: {significance}")
        
        self.chi_square_results = {
            'chi2': chi2,
            'p_value': p_value,
            'dof': dof,
            'cramers_v': cramers_v,
            'contingency_table': contingency_table,
            'significance': significance
        }
        
        return self.chi_square_results
    
    def print_channel_rankings(self):
        """Print channel vulnerability rankings."""
        print(f"\nNIGERIAN BANKING CHANNEL FRAUD RATES:")
        print("="*50)
        
        # Sort channels by fraud rate
        sorted_channels = sorted(self.channel_stats.items(), 
                               key=lambda x: x[1]['fraud_rate'], 
                               reverse=True)
        
        for i, (channel, stats) in enumerate(sorted_channels, 1):
            print(f"{i}. {channel}: {stats['fraud_rate']:.2%} fraud rate "
                  f"(95% CI [{stats['ci_lower']:.2f}%, {stats['ci_upper']:.2f}%])")
        
        # Identify high-risk vs low-risk channels
        high_risk = [ch for ch, st in sorted_channels[:3]]
        low_risk = [ch for ch, st in sorted_channels[-3:]]
        
        print(f"\nHigh-risk channels: {', '.join(high_risk)}")
        print(f"Low-risk channels: {', '.join(low_risk)}")
    
    def create_channel_visualizations(self):
        """Create comprehensive channel fraud analysis visualization."""
        print("\nCreating channel fraud visualizations...")
        
        # Prepare data for plotting
        channels = list(self.channel_stats.keys())
        fraud_rates = [self.channel_stats[ch]['fraud_rate_pct'] for ch in channels]
        volumes = [self.channel_stats[ch]['total_transactions'] for ch in channels]
        volume_pcts = [self.channel_stats[ch]['volume_pct'] for ch in channels]
        ci_lower = [self.channel_stats[ch]['ci_lower'] for ch in channels]
        ci_upper = [self.channel_stats[ch]['ci_upper'] for ch in channels]
        
        # Create 2x2 subplot figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Channel-Specific Fraud Analysis for Nigerian Banking Dataset', fontsize=16, fontweight='bold')
        
        # Color palette
        colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(channels)))
        
        # Plot 1: Transaction Volume by Channel with Fraud Rates
        bars1 = ax1.bar(channels, volumes, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax1.set_title('Transaction Volume by Channel with Fraud Rates\n(Error bars show 95% CI)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Total Transactions', fontsize=11)
        ax1.set_xlabel('Transaction Channel', fontsize=11)
        
        # Add fraud rate labels on bars
        for i, (bar, rate, ci_l, ci_u) in enumerate(zip(bars1, fraud_rates, ci_lower, ci_upper)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(volumes)*0.01,
                    f'{rate:.2f}%\n[{ci_l:.2f}%, {ci_u:.2f}%]',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: Fraud Rates by Channel (with confidence intervals)
        bars2 = ax2.bar(channels, fraud_rates, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Add error bars
        errors = [[fr - ci_l for fr, ci_l in zip(fraud_rates, ci_lower)],
                 [ci_u - fr for fr, ci_u in zip(fraud_rates, ci_upper)]]
        ax2.errorbar(channels, fraud_rates, yerr=errors, fmt='none', color='black', capsize=5, capthick=2)
        
        ax2.set_title('Fraud Rates by Channel\n(with 95% Confidence Intervals)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Fraud Rate (%)', fontsize=11)
        ax2.set_xlabel('Transaction Channel', fontsize=11)
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(axis='y', alpha=0.3)
        
        # Add fraud rate labels
        for bar, rate in zip(bars2, fraud_rates):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(fraud_rates)*0.02,
                    f'{rate:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Plot 3: Transaction Volume vs Fraud Rate Scatter
        scatter = ax3.scatter(volumes, fraud_rates, c=fraud_rates, s=[v/1000 for v in volumes], 
                            cmap='Reds', alpha=0.7, edgecolors='black', linewidth=1)
        
        # Add channel labels
        for i, channel in enumerate(channels):
            ax3.annotate(channel, (volumes[i], fraud_rates[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')
        
        ax3.set_title('Transaction Volume vs Fraud Rate by Channel', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Transaction Volume', fontsize=11)
        ax3.set_ylabel('Fraud Rate (%)', fontsize=11)
        ax3.grid(alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('Fraud Rate (%)', rotation=270, labelpad=15)
        
        # Plot 4: Risk-Volume Matrix
        # Bubble chart: x=volume_pct, y=fraud_rate, bubble_size=total_volume
        bubble_sizes = [v/5000 for v in volumes]  # Scale bubble sizes
        
        colors_mapped = [plt.cm.Reds(rate/max(fraud_rates)) for rate in fraud_rates]
        
        for i, channel in enumerate(channels):
            ax4.scatter(volume_pcts[i], fraud_rates[i], s=bubble_sizes[i], 
                       c=[colors_mapped[i]], alpha=0.7, edgecolors='black', linewidth=1,
                       label=channel)
            
            # Add channel labels
            ax4.annotate(channel, (volume_pcts[i], fraud_rates[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')
        
        ax4.set_title('Risk-Volume Matrix by Channel\n(Bubble size = Transaction volume)', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Volume Percentage (%)', fontsize=11)
        ax4.set_ylabel('Fraud Rate (%)', fontsize=11)
        ax4.grid(alpha=0.3)
        
        # Add reference lines
        avg_fraud_rate = np.mean(fraud_rates)
        avg_volume_pct = np.mean(volume_pcts)
        ax4.axhline(y=avg_fraud_rate, color='gray', linestyle='--', alpha=0.5, label='Avg Fraud Rate')
        ax4.axvline(x=avg_volume_pct, color='gray', linestyle='--', alpha=0.5, label='Avg Volume %')
        
        plt.tight_layout()
        plt.savefig('channel_fraud_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ Channel fraud analysis visualization saved as: channel_fraud_analysis.png")
        
        return fig
    
    def generate_summary_report(self):
        """Generate summary report for channel analysis."""
        print(f"\n" + "="*70)
        print("CHANNEL-SPECIFIC FRAUD ANALYSIS SUMMARY")
        print("="*70)
        
        # Chi-square results
        chi2_res = self.chi_square_results
        print(f"\nStatistical Association Test:")
        p_value_str = ' < 0.001' if chi2_res['p_value'] < 0.001 else f" = {chi2_res['p_value']:.3f}"
        print(f"Chi-square: χ²({chi2_res['dof']}) = {chi2_res['chi2']:.2f}, p{p_value_str}")
        print(f"Cramér's V: {chi2_res['cramers_v']:.3f} (effect size)")
        print(f"Result: {chi2_res['significance']}")
        
        # Channel rankings
        print(f"\nChannel Vulnerability Ranking (High to Low Risk):")
        sorted_channels = sorted(self.channel_stats.items(), 
                               key=lambda x: x[1]['fraud_rate'], 
                               reverse=True)
        
        for i, (channel, stats) in enumerate(sorted_channels, 1):
            risk_level = "HIGH" if stats['fraud_rate'] > 0.003 else "MODERATE" if stats['fraud_rate'] > 0.002 else "LOW"
            print(f"{i}. {channel:<12}: {stats['fraud_rate']:.3%} [{stats['ci_lower']:.2f}%-{stats['ci_upper']:.2f}%] ({risk_level} RISK)")
        
        # Key insights
        print(f"\nKey Insights:")
        highest_risk = sorted_channels[0]
        lowest_risk = sorted_channels[-1]
        
        print(f"• Highest risk channel: {highest_risk[0]} ({highest_risk[1]['fraud_rate']:.3%})")
        print(f"• Lowest risk channel: {lowest_risk[0]} ({lowest_risk[1]['fraud_rate']:.3%})")
        
        # Digital vs Traditional channels
        digital_channels = ['Mobile', 'Web', 'ECOM']
        traditional_channels = ['POS', 'ATM', 'Branch']
        
        digital_rates = [self.channel_stats[ch]['fraud_rate'] for ch in digital_channels if ch in self.channel_stats]
        traditional_rates = [self.channel_stats[ch]['fraud_rate'] for ch in traditional_channels if ch in self.channel_stats]
        
        if digital_rates and traditional_rates:
            avg_digital = np.mean(digital_rates)
            avg_traditional = np.mean(traditional_rates)
            print(f"• Digital channels avg fraud rate: {avg_digital:.3%}")
            print(f"• Traditional channels avg fraud rate: {avg_traditional:.3%}")
            print(f"• Digital vs Traditional risk ratio: {avg_digital/avg_traditional:.1f}x higher")
    
    def run_complete_analysis(self):
        """Run the complete channel fraud analysis."""
        print("Starting Channel-Specific Fraud Analysis...")
        print("="*60)
        
        # Load data
        self.load_data()
        
        # Calculate channel statistics
        self.calculate_channel_statistics()
        
        # Chi-square analysis
        self.chi_square_analysis()
        
        # Print rankings
        self.print_channel_rankings()
        
        # Create visualizations
        fig = self.create_channel_visualizations()
        
        # Generate summary
        self.generate_summary_report()
        
        print(f"\n✅ CHANNEL FRAUD ANALYSIS COMPLETED!")
        print("="*60)
        
        return {
            'channel_stats': self.channel_stats,
            'chi_square_results': self.chi_square_results,
            'visualization': fig
        }


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Channel-specific fraud analysis for Nigerian banking dataset')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the fraud dataset CSV file')
    
    args = parser.parse_args()
    
    # Run analysis
    analyzer = ChannelFraudAnalyzer(args.dataset)
    results = analyzer.run_complete_analysis()
    
    return results


if __name__ == "__main__":
    # Example usage
    try:
        results = main()
    except Exception as e:
        print(f"Error during execution: {e}")
        print("\nUsage example:")
        print("python channel_fraud_analysis.py --dataset nibss_fraud_dataset.csv")
        
        # Alternative: Run with hardcoded dataset path
        print("\nRunning with default dataset path...")
        analyzer = ChannelFraudAnalyzer("nibss_fraud_dataset.csv")
        results = analyzer.run_complete_analysis()