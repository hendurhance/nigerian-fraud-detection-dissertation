# simple_nibss_visualization.py
"""
Simple script to visualize NIBSS fraud dataset.
Just run: python simple_nibss_visualization.py your_dataset.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

def create_analysis(csv_file):
    """Create the 4-chart NIBSS analysis figure."""
    
    # Load data
    print(f"Loading data from: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # Define peak months and colors
    peak_months = [3, 11, 12]  # March, November, December
    regular_color = "#028DC8"  # Blue
    peak_color = "#C11111"     # Red
    
    # Month names mapping
    month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                   7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    df['month_name'] = df['month'].map(month_names)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Chart A1: Channel Distribution
    ax1 = axes[0, 0]
    channel_counts = df['channel'].value_counts().sort_values(ascending=False)
    bars1 = ax1.bar(channel_counts.index, channel_counts.values, 
                    color=regular_color, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax1.set_title('Channel Distribution (Number of Transactions)', fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel('Transaction Channel', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Transactions', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{int(height):,}', ha='center', va='bottom', fontweight='bold')
    
    # Chart A2: Fraud Rate by Channel
    ax2 = axes[0, 1]
    channel_fraud = df.groupby('channel').agg({'is_fraud': ['sum', 'count']})
    channel_fraud.columns = ['fraud_count', 'total_count']
    channel_fraud['fraud_rate'] = (channel_fraud['fraud_count'] / channel_fraud['total_count'] * 100)
    channel_fraud = channel_fraud.sort_values('fraud_rate', ascending=False)
    
    bars2 = ax2.bar(channel_fraud.index, channel_fraud['fraud_rate'],
                    color=regular_color, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax2.set_title('Fraud Rate by Channel (% of Transactions)', fontsize=14, fontweight='bold', pad=20)
    ax2.set_xlabel('Transaction Channel', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Fraud Rate (%)', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{height:.3f}%', ha='center', va='bottom', fontweight='bold')
    
    # Chart B1: Monthly Transaction Distribution
    ax3 = axes[1, 0]
    monthly_counts = df.groupby(['month', 'month_name']).size().reset_index(name='count')
    monthly_counts = monthly_counts.sort_values('month')
    
    # Color bars based on peak months
    colors = [peak_color if month in peak_months else regular_color 
              for month in monthly_counts['month']]
    
    bars3 = ax3.bar(monthly_counts['month_name'], monthly_counts['count'],
                    color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax3.set_title('Monthly Transaction Distribution (Number of Transactions)', 
                  fontsize=14, fontweight='bold', pad=20)
    ax3.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Number of Transactions', fontsize=12, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{int(height):,}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Chart B2: Monthly Fraud Rate Distribution
    ax4 = axes[1, 1]
    monthly_fraud = df.groupby(['month', 'month_name']).agg({'is_fraud': ['sum', 'count']})
    monthly_fraud.columns = ['fraud_count', 'total_count']
    monthly_fraud['fraud_rate'] = (monthly_fraud['fraud_count'] / monthly_fraud['total_count'] * 100)
    monthly_fraud = monthly_fraud.reset_index().sort_values('month')
    
    # Color bars based on peak months
    colors = [peak_color if month in peak_months else regular_color 
              for month in monthly_fraud['month']]
    
    bars4 = ax4.bar(monthly_fraud['month_name'], monthly_fraud['fraud_rate'],
                    color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax4.set_title('Monthly Fraud Rate Distribution (% of Transactions)', 
                  fontsize=14, fontweight='bold', pad=20)
    ax4.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Fraud Rate (%)', fontsize=12, fontweight='bold')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{height:.3f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Add legends for monthly charts
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=peak_color, alpha=0.8, label='NIBSS Peak Months (Mar, Nov, Dec)'),
        Patch(facecolor=regular_color, alpha=0.8, label='Regular Months')
    ]
    ax3.legend(handles=legend_elements, loc='upper right', fontsize=10)
    ax4.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save and display
    output_file = 'nibss_fraud_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Analysis saved to: {output_file}")
    
    # Print summary
    print(f"\\nDataset Summary:")
    print(f"Total Transactions: {len(df):,}")
    print(f"Fraudulent Transactions: {df['is_fraud'].sum():,}")
    print(f"Overall Fraud Rate: {df['is_fraud'].mean()*100:.4f}%")
    
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python simple_nibss_visualization.py <csv_file>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    create_analysis(csv_file)