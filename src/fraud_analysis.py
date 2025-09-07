# fraud_only_analysis.py
"""
Comprehensive analysis script focusing exclusively on fraudulent transactions
from the NIBSS fraud dataset. Provides detailed insights into fraud patterns,
amounts, techniques, and temporal distributions.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_fraud_data(csv_file):
    """Load dataset and extract only fraudulent transactions."""
    print(f"Loading dataset from: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Extract only fraud cases
    fraud_df = df[df['is_fraud'] == 1].copy()
    
    # Add derived columns for analysis
    month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                   7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    fraud_df['month_name'] = fraud_df['month'].map(month_names)
    
    # Add hour categories
    fraud_df['hour_category'] = fraud_df['hour'].apply(lambda x: 
        'Night (00-06)' if 0 <= x <= 6 else
        'Morning (07-12)' if 7 <= x <= 12 else
        'Afternoon (13-18)' if 13 <= x <= 18 else
        'Evening (19-23)')
    
    # Add amount categories
    fraud_df['amount_category'] = pd.cut(fraud_df['amount'], 
                                        bins=[0, 10000, 50000, 200000, 1000000, float('inf')],
                                        labels=['Small (≤₦10K)', 'Medium (₦10K-50K)', 
                                               'Large (₦50K-200K)', 'Very Large (₦200K-1M)', 
                                               'Extreme (>₦1M)'])
    
    print(f"Total transactions in dataset: {len(df):,}")
    print(f"Fraudulent transactions: {len(fraud_df):,}")
    print(f"Fraud rate: {len(fraud_df)/len(df)*100:.4f}%")
    
    return fraud_df, df

def create_fraud_summary_stats(fraud_df):
    """Generate comprehensive fraud summary statistics."""
    print(f"\n{'='*80}")
    print("FRAUD ANALYSIS SUMMARY STATISTICS")
    print(f"{'='*80}")
    
    # Basic stats
    print(f"\n📊 BASIC FRAUD STATISTICS:")
    print(f"Total Fraud Cases: {len(fraud_df):,}")
    print(f"Total Fraud Amount: ₦{fraud_df['amount'].sum():,.2f}")
    print(f"Average Fraud Amount: ₦{fraud_df['amount'].mean():,.2f}")
    print(f"Median Fraud Amount: ₦{fraud_df['amount'].median():,.2f}")
    print(f"Standard Deviation: ₦{fraud_df['amount'].std():,.2f}")
    print(f"Min Fraud Amount: ₦{fraud_df['amount'].min():,.2f}")
    print(f"Max Fraud Amount: ₦{fraud_df['amount'].max():,.2f}")
    
    # Channel analysis
    print(f"\n🏦 FRAUD BY CHANNEL:")
    channel_analysis = fraud_df.groupby('channel').agg({
        'amount': ['count', 'sum', 'mean', 'median']
    }).round(2)
    channel_analysis.columns = ['Count', 'Total_Amount', 'Avg_Amount', 'Median_Amount']
    channel_analysis['Percentage'] = (channel_analysis['Count'] / len(fraud_df) * 100).round(2)
    channel_analysis = channel_analysis.sort_values('Count', ascending=False)
    
    for channel, row in channel_analysis.iterrows():
        print(f"  {channel:>12}: {int(row['Count']):>4} cases ({row['Percentage']:>5.1f}%) | "
              f"₦{row['Total_Amount']:>12,.0f} total | ₦{row['Avg_Amount']:>9,.0f} avg")
    
    # Monthly analysis
    print(f"\n📅 FRAUD BY MONTH:")
    monthly_analysis = fraud_df.groupby(['month', 'month_name']).agg({
        'amount': ['count', 'sum', 'mean']
    }).round(2)
    monthly_analysis.columns = ['Count', 'Total_Amount', 'Avg_Amount']
    monthly_analysis['Percentage'] = (monthly_analysis['Count'] / len(fraud_df) * 100).round(2)
    monthly_analysis = monthly_analysis.reset_index().sort_values('month')
    
    for _, row in monthly_analysis.iterrows():
        print(f"  {row['month_name']:>3}: {int(row['Count']):>4} cases ({row['Percentage']:>5.1f}%) | "
              f"₦{row['Total_Amount']:>12,.0f} total | ₦{row['Avg_Amount']:>9,.0f} avg")
    
    # Technique analysis (if available)
    if 'fraud_technique' in fraud_df.columns and fraud_df['fraud_technique'].notna().any():
        print(f"\n🎯 FRAUD BY TECHNIQUE:")
        technique_analysis = fraud_df[fraud_df['fraud_technique'].notna()].groupby('fraud_technique').agg({
            'amount': ['count', 'sum', 'mean']
        }).round(2)
        technique_analysis.columns = ['Count', 'Total_Amount', 'Avg_Amount']
        technique_analysis['Percentage'] = (technique_analysis['Count'] / len(fraud_df[fraud_df['fraud_technique'].notna()]) * 100).round(2)
        technique_analysis = technique_analysis.sort_values('Count', ascending=False)
        
        for technique, row in technique_analysis.iterrows():
            print(f"  {technique:>20}: {int(row['Count']):>4} cases ({row['Percentage']:>5.1f}%) | "
                  f"₦{row['Avg_Amount']:>9,.0f} avg")
    
    # Time-based analysis
    print(f"\n⏰ FRAUD BY TIME OF DAY:")
    hour_analysis = fraud_df.groupby('hour_category').agg({
        'amount': ['count', 'sum', 'mean']
    }).round(2)
    hour_analysis.columns = ['Count', 'Total_Amount', 'Avg_Amount']
    hour_analysis['Percentage'] = (hour_analysis['Count'] / len(fraud_df) * 100).round(2)
    hour_analysis = hour_analysis.sort_values('Count', ascending=False)
    
    for hour_cat, row in hour_analysis.iterrows():
        print(f"  {hour_cat:>20}: {int(row['Count']):>4} cases ({row['Percentage']:>5.1f}%) | "
              f"₦{row['Avg_Amount']:>9,.0f} avg")
    
    # Amount categories
    print(f"\n💰 FRAUD BY AMOUNT CATEGORY:")
    amount_analysis = fraud_df.groupby('amount_category').agg({
        'amount': ['count', 'sum', 'mean']
    }).round(2)
    amount_analysis.columns = ['Count', 'Total_Amount', 'Avg_Amount']
    amount_analysis['Percentage'] = (amount_analysis['Count'] / len(fraud_df) * 100).round(2)
    
    for amount_cat, row in amount_analysis.iterrows():
        print(f"  {amount_cat:>20}: {int(row['Count']):>4} cases ({row['Percentage']:>5.1f}%) | "
              f"₦{row['Total_Amount']:>12,.0f} total")
    
    return channel_analysis, monthly_analysis

def create_fraud_visualizations(fraud_df, output_dir='fraud_analysis_charts'):
    """Create comprehensive fraud visualization charts."""
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Set up color schemes
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']
    
    # 1. Fraud Distribution by Channel
    plt.figure(figsize=(14, 8))
    
    # Count and amount by channel
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Chart 1: Fraud count by channel
    channel_counts = fraud_df['channel'].value_counts()
    bars1 = ax1.bar(channel_counts.index, channel_counts.values, color=colors[:len(channel_counts)])
    ax1.set_title('Fraud Cases by Channel', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Channel', fontweight='bold')
    ax1.set_ylabel('Number of Fraud Cases', fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # Chart 2: Average fraud amount by channel
    channel_avg = fraud_df.groupby('channel')['amount'].mean().sort_values(ascending=False)
    bars2 = ax2.bar(channel_avg.index, channel_avg.values, color=colors[:len(channel_avg)])
    ax2.set_title('Average Fraud Amount by Channel', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Channel', fontweight='bold')
    ax2.set_ylabel('Average Amount (₦)', fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₦{x:,.0f}'))
    
    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'₦{height:,.0f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fraud_by_channel.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Monthly Fraud Patterns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Monthly fraud count
    monthly_counts = fraud_df.groupby(['month', 'month_name']).size().reset_index(name='count')
    monthly_counts = monthly_counts.sort_values('month')
    
    bars1 = ax1.bar(monthly_counts['month_name'], monthly_counts['count'], 
                    color='#FF6B6B', alpha=0.8)
    ax1.set_title('Fraud Cases by Month', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Month', fontweight='bold')
    ax1.set_ylabel('Number of Fraud Cases', fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # Monthly fraud amount
    monthly_amounts = fraud_df.groupby(['month', 'month_name'])['amount'].sum().reset_index()
    monthly_amounts = monthly_amounts.sort_values('month')
    
    bars2 = ax2.bar(monthly_amounts['month_name'], monthly_amounts['amount'], 
                    color='#4ECDC4', alpha=0.8)
    ax2.set_title('Total Fraud Amount by Month', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Month', fontweight='bold')
    ax2.set_ylabel('Total Amount (₦)', fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₦{x/1e6:.1f}M'))
    
    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'₦{height/1e6:.1f}M', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fraud_by_month.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Fraud Amount Distribution
    plt.figure(figsize=(14, 10))
    
    # Create subplot for amount analysis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Histogram of fraud amounts
    ax1.hist(fraud_df['amount'], bins=50, color='#45B7D1', alpha=0.7, edgecolor='black')
    ax1.set_title('Distribution of Fraud Amounts', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Fraud Amount (₦)', fontweight='bold')
    ax1.set_ylabel('Frequency', fontweight='bold')
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₦{x/1e6:.1f}M'))
    
    # Box plot by channel
    fraud_df.boxplot(column='amount', by='channel', ax=ax2)
    ax2.set_title('Fraud Amount Distribution by Channel', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Channel', fontweight='bold')
    ax2.set_ylabel('Fraud Amount (₦)', fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₦{x/1e3:.0f}K'))
    
    # Amount categories pie chart
    amount_counts = fraud_df['amount_category'].value_counts()
    ax3.pie(amount_counts.values, labels=amount_counts.index, autopct='%1.1f%%', 
            colors=colors[:len(amount_counts)])
    ax3.set_title('Fraud Cases by Amount Category', fontsize=14, fontweight='bold')
    
    # Time of day analysis
    hour_counts = fraud_df['hour_category'].value_counts()
    bars4 = ax4.bar(hour_counts.index, hour_counts.values, color=colors[:len(hour_counts)])
    ax4.set_title('Fraud Cases by Time of Day', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Time Period', fontweight='bold')
    ax4.set_ylabel('Number of Fraud Cases', fontweight='bold')
    ax4.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fraud_amount_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Fraud Technique Analysis (if available)
    if 'fraud_technique' in fraud_df.columns and fraud_df['fraud_technique'].notna().any():
        plt.figure(figsize=(14, 8))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Technique count
        technique_counts = fraud_df[fraud_df['fraud_technique'].notna()]['fraud_technique'].value_counts()
        bars1 = ax1.bar(range(len(technique_counts)), technique_counts.values, 
                        color=colors[:len(technique_counts)])
        ax1.set_title('Fraud Cases by Technique', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Fraud Technique', fontweight='bold')
        ax1.set_ylabel('Number of Cases', fontweight='bold')
        ax1.set_xticks(range(len(technique_counts)))
        ax1.set_xticklabels(technique_counts.index, rotation=45, ha='right')
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        # Technique average amounts
        technique_avg = fraud_df[fraud_df['fraud_technique'].notna()].groupby('fraud_technique')['amount'].mean().sort_values(ascending=False)
        bars2 = ax2.bar(range(len(technique_avg)), technique_avg.values, 
                        color=colors[:len(technique_avg)])
        ax2.set_title('Average Fraud Amount by Technique', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Fraud Technique', fontweight='bold')
        ax2.set_ylabel('Average Amount (₦)', fontweight='bold')
        ax2.set_xticks(range(len(technique_avg)))
        ax2.set_xticklabels(technique_avg.index, rotation=45, ha='right')
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₦{x:,.0f}'))
        
        # Add value labels
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'₦{height:,.0f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/fraud_by_technique.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✓ All fraud analysis charts saved to: {output_dir}/")

def create_fraud_correlation_analysis(fraud_df, output_dir='fraud_analysis_charts'):
    """Create correlation and advanced analysis charts."""
    
    # Select numeric columns for correlation
    numeric_cols = ['amount', 'hour', 'day_of_week', 'month', 'tx_count_24h', 
                   'amount_sum_24h', 'amount_mean_7d', 'velocity_score', 
                   'merchant_risk_score', 'composite_risk']
    
    # Filter to only existing columns
    available_cols = [col for col in numeric_cols if col in fraud_df.columns]
    
    if len(available_cols) > 3:
        plt.figure(figsize=(12, 10))
        
        # Create correlation matrix
        corr_matrix = fraud_df[available_cols].corr()
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
                   square=True, fmt='.2f', cbar_kws={"shrink": .5})
        
        plt.title('Fraud Data Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/fraud_correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Correlation analysis saved to: {output_dir}/fraud_correlation_matrix.png")

def generate_fraud_report(fraud_df, total_df, output_file='fraud_analysis_report.txt'):
    """Generate a comprehensive text report of fraud analysis."""
    
    with open(output_file, 'w') as f:
        f.write("NIBSS FRAUD ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Executive Summary
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Transactions Analyzed: {len(total_df):,}\n")
        f.write(f"Fraudulent Transactions: {len(fraud_df):,}\n")
        f.write(f"Overall Fraud Rate: {len(fraud_df)/len(total_df)*100:.4f}%\n")
        f.write(f"Total Fraud Loss: ₦{fraud_df['amount'].sum():,.2f}\n")
        f.write(f"Average Fraud Amount: ₦{fraud_df['amount'].mean():,.2f}\n")
        f.write(f"Median Fraud Amount: ₦{fraud_df['amount'].median():,.2f}\n\n")
        
        # High-risk patterns
        f.write("HIGH-RISK PATTERNS IDENTIFIED\n")
        f.write("-" * 40 + "\n")
        
        # Top risk channels
        channel_avg = fraud_df.groupby('channel')['amount'].agg(['count', 'mean', 'sum']).sort_values('mean', ascending=False)
        f.write(f"Highest Average Loss Channel: {channel_avg.index[0]} (₦{channel_avg.iloc[0]['mean']:,.2f})\n")
        f.write(f"Most Frequent Fraud Channel: {channel_avg.sort_values('count', ascending=False).index[0]} ({channel_avg.sort_values('count', ascending=False).iloc[0]['count']} cases)\n")
        
        # Peak months
        monthly_counts = fraud_df.groupby('month').size().sort_values(ascending=False)
        month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                      7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
        f.write(f"Peak Fraud Month: {month_names[monthly_counts.index[0]]} ({monthly_counts.iloc[0]} cases)\n")
        
        # Time patterns
        hour_counts = fraud_df.groupby('hour').size().sort_values(ascending=False)
        f.write(f"Peak Fraud Hour: {hour_counts.index[0]}:00 ({hour_counts.iloc[0]} cases)\n\n")
        
        # Recommendations
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 40 + "\n")
        f.write("1. Enhance monitoring for high-value transactions, especially in channels with highest average losses\n")
        f.write("2. Implement additional verification steps during peak fraud hours and months\n")
        f.write("3. Focus fraud prevention efforts on the most frequently targeted channels\n")
        f.write("4. Consider dynamic risk scoring based on transaction patterns identified\n")
        f.write("5. Increase customer education about fraud prevention, particularly for high-risk channels\n")
    
    print(f"✓ Comprehensive fraud report saved to: {output_file}")

def main():
    """Main function to run complete fraud analysis."""
    if len(sys.argv) != 2:
        print("Usage: python fraud_only_analysis.py <csv_file>")
        print("Example: python fraud_only_analysis.py nibss_fraud_dataset.csv")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    # Check if file exists
    if not Path(csv_file).exists():
        print(f"Error: File '{csv_file}' not found!")
        sys.exit(1)
    
    try:
        # Load and analyze fraud data
        fraud_df, total_df = load_fraud_data(csv_file)
        
        if len(fraud_df) == 0:
            print("No fraudulent transactions found in the dataset!")
            return
        
        # Generate summary statistics
        channel_analysis, monthly_analysis = create_fraud_summary_stats(fraud_df)
        
        # Create visualizations
        print(f"\n📊 Creating fraud analysis visualizations...")
        create_fraud_visualizations(fraud_df)
        
        # Create correlation analysis
        print(f"📈 Creating correlation analysis...")
        create_fraud_correlation_analysis(fraud_df)
        
        # Generate comprehensive report
        print(f"📝 Generating fraud analysis report...")
        generate_fraud_report(fraud_df, total_df)
        
        print(f"\n{'='*80}")
        print("FRAUD ANALYSIS COMPLETE!")
        print(f"{'='*80}")
        print("Generated Files:")
        print("  📊 fraud_analysis_charts/ - Directory with all visualization charts")
        print("  📝 fraud_analysis_report.txt - Comprehensive text report")
        print("\nAnalysis covers:")
        print("  • Fraud distribution by channel, month, time, and amount")
        print("  • Statistical summaries and correlations")
        print("  • High-risk pattern identification")
        print("  • Actionable recommendations")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()