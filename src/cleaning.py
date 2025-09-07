import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def analyze_outliers_iqr(df, column, multiplier=1.5):
    """
    Analyze outliers using IQR method
    
    Parameters:
    df: DataFrame
    column: Column name to analyze
    multiplier: IQR multiplier (1.5 for standard, 3.0 for extreme outliers)
    
    Returns:
    dict with outlier statistics and boundaries
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    outliers_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
    n_outliers = outliers_mask.sum()
    
    return {
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'n_outliers': n_outliers,
        'outliers_mask': outliers_mask,
        'outlier_percentage': (n_outliers / len(df)) * 100
    }

def clean_dataset_enhanced(input_file, output_file):
    """
    Enhanced cleaning of the simulated Nigerian fraud dataset with outlier detection
    """
    print("=== ENHANCED DATASET CLEANING ===\n")
    
    # Load the dataset
    print("Loading dataset...")
    df = pd.read_csv(input_file)
    print(f"Original dataset shape: {df.shape}")
    print(f"Original fraud cases: {df['is_fraud'].sum()}")
    print(f"Original fraud rate: {df['is_fraud'].mean():.4f}")
    
    # Store original statistics
    original_stats = {
        'total_transactions': len(df),
        'fraud_cases': df['is_fraud'].sum(),
        'fraud_rate': df['is_fraud'].mean()
    }
    
    print("\n" + "="*50)
    print("STEP 1: BASIC CLEANING")
    print("="*50)
    
    # 1. Handle missing values
    print("1.1 Handling missing values...")
    missing_before = df.isnull().sum().sum()
    
    # Fill missing fraud_technique with 'None' for legitimate transactions
    df['fraud_technique'] = df['fraud_technique'].fillna('None')
    
    # Ensure fraud cases have techniques (set default if missing)
    df.loc[(df['is_fraud'] == 1) & (df['fraud_technique'] == 'None'), 'fraud_technique'] = 'Unknown_Fraud'
    
    missing_after = df.isnull().sum().sum()
    print(f"   Missing values: {missing_before} → {missing_after}")
    
    # 2. Remove duplicate rows
    print("1.2 Checking for duplicates...")
    duplicates_before = df.duplicated().sum()
    df = df.drop_duplicates()
    duplicates_after = df.duplicated().sum()
    print(f"   Duplicates removed: {duplicates_before}")
    print(f"   Shape after duplicate removal: {df.shape}")
    
    # 3. Handle zero and negative amounts
    print("1.3 Handling invalid amounts...")
    zero_amounts = (df['amount'] == 0).sum()
    negative_amounts = (df['amount'] < 0).sum()
    
    print(f"   Zero amounts found: {zero_amounts}")
    print(f"   Negative amounts found: {negative_amounts}")
    
    # Remove zero and negative amounts
    df_before_amount_filter = df.copy()
    df = df[(df['amount'] > 0)]
    
    amount_filter_stats = {
        'transactions_before': len(df_before_amount_filter),
        'transactions_after': len(df),
        'removed': len(df_before_amount_filter) - len(df)
    }
    
    print(f"   Transactions after removing invalid amounts: {len(df)}")
    print(f"   Transactions removed: {amount_filter_stats['removed']}")
    
    print("\n" + "="*50)
    print("STEP 2: OUTLIER ANALYSIS")
    print("="*50)
    
    # Analyze numerical columns for outliers
    numerical_columns = ['amount']  # Add other numerical columns as needed
    outlier_stats = {}
    
    for col in numerical_columns:
        print(f"\n2.1 Analyzing outliers in '{col}'...")
        
        # Calculate quartiles and basic stats
        stats_dict = {
            'count': df[col].count(),
            'mean': df[col].mean(),
            'std': df[col].std(),
            'min': df[col].min(),
            'Q1': df[col].quantile(0.25),
            'median': df[col].median(),
            'Q3': df[col].quantile(0.75),
            'max': df[col].max()
        }
        
        print(f"   Basic Statistics for {col}:")
        print(f"   Count: {stats_dict['count']:,}")
        print(f"   Mean: ₦{stats_dict['mean']:,.2f}")
        print(f"   Std: ₦{stats_dict['std']:,.2f}")
        print(f"   Min: ₦{stats_dict['min']:,.2f}")
        print(f"   Q1 (25%): ₦{stats_dict['Q1']:,.2f}")
        print(f"   Median (50%): ₦{stats_dict['median']:,.2f}")
        print(f"   Q3 (75%): ₦{stats_dict['Q3']:,.2f}")
        print(f"   Max: ₦{stats_dict['max']:,.2f}")
        
        # Standard outlier detection (1.5 * IQR)
        outlier_analysis_standard = analyze_outliers_iqr(df, col, multiplier=1.5)
        
        # Extreme outlier detection (3.0 * IQR) - as mentioned in the document
        outlier_analysis_extreme = analyze_outliers_iqr(df, col, multiplier=3.0)
        
        print(f"\n   IQR Analysis:")
        print(f"   IQR = Q3 - Q1 = ₦{outlier_analysis_standard['IQR']:,.2f}")
        print(f"   Standard bounds (1.5 × IQR): [₦{outlier_analysis_standard['lower_bound']:,.2f}, ₦{outlier_analysis_standard['upper_bound']:,.2f}]")
        print(f"   Extreme bounds (3.0 × IQR): [₦{outlier_analysis_extreme['lower_bound']:,.2f}, ₦{outlier_analysis_extreme['upper_bound']:,.2f}]")
        
        print(f"\n   Outlier Detection Results:")
        print(f"   Standard outliers (1.5 × IQR): {outlier_analysis_standard['n_outliers']:,} ({outlier_analysis_standard['outlier_percentage']:.2f}%)")
        print(f"   Extreme outliers (3.0 × IQR): {outlier_analysis_extreme['n_outliers']:,} ({outlier_analysis_extreme['outlier_percentage']:.2f}%)")
        
        outlier_stats[col] = {
            'basic_stats': stats_dict,
            'standard_outliers': outlier_analysis_standard,
            'extreme_outliers': outlier_analysis_extreme
        }
    
    print("\n" + "="*50)
    print("STEP 3: OUTLIER REMOVAL DECISION")
    print("="*50)
    
    # Following the document's approach: investigate extreme outliers (3.0 × IQR)
    print("3.1 Investigating extreme outliers (Q3 + 3 × IQR)...")
    
    amount_outliers = outlier_stats['amount']['extreme_outliers']
    extreme_outliers_mask = amount_outliers['outliers_mask']
    
    # Analyze extreme outliers
    if amount_outliers['n_outliers'] > 0:
        outlier_transactions = df[extreme_outliers_mask]
        
        print(f"   Found {amount_outliers['n_outliers']:,} extreme outliers")
        print(f"   Outlier amount range: ₦{outlier_transactions['amount'].min():,.2f} - ₦{outlier_transactions['amount'].max():,.2f}")
        print(f"   Fraud cases among outliers: {outlier_transactions['is_fraud'].sum()}")
        print(f"   Fraud rate among outliers: {outlier_transactions['is_fraud'].mean():.4f}")
        
        # Based on document: "Domain expertise consultation confirmed these represent 
        # legitimate high-value transactions common in Nigerian corporate banking, 
        # real estate, and international trade"
        
        # For demonstration, we'll follow the document's decision to RETAIN these outliers
        print(f"\n   DECISION: RETAINING extreme outliers")
        print(f"   Rationale: Following document methodology - extreme values likely represent")
        print(f"   legitimate high-value transactions (corporate banking, real estate, international trade)")
        
        df_final = df.copy()  # Keep all transactions including outliers
        
    else:
        print("   No extreme outliers found")
        df_final = df.copy()
    
    print("\n" + "="*50)
    print("STEP 4: FINAL DATASET STATISTICS")
    print("="*50)
    
    final_stats = {
        'total_transactions': len(df_final),
        'fraud_cases': df_final['is_fraud'].sum(),
        'fraud_rate': df_final['is_fraud'].mean(),
        'amount_stats': {
            'min': df_final['amount'].min(),
            'max': df_final['amount'].max(),
            'mean': df_final['amount'].mean(),
            'median': df_final['amount'].median()
        }
    }
    
    print("4.1 Processing Summary:")
    print(f"   Original transactions: {original_stats['total_transactions']:,}")
    print(f"   After duplicate removal: {len(df_before_amount_filter):,}")
    print(f"   After removing zero/negative amounts: {amount_filter_stats['transactions_after']:,}")
    print(f"   After outlier analysis: {final_stats['total_transactions']:,}")
    print(f"   Total transactions removed: {original_stats['total_transactions'] - final_stats['total_transactions']:,}")
    
    print(f"\n4.2 Fraud Statistics:")
    print(f"   Original fraud cases: {original_stats['fraud_cases']:,}")
    print(f"   Final fraud cases: {final_stats['fraud_cases']:,}")
    print(f"   Original fraud rate: {original_stats['fraud_rate']:.4f}")
    print(f"   Final fraud rate: {final_stats['fraud_rate']:.4f}")
    
    print(f"\n4.3 Amount Statistics:")
    print(f"   Min amount: ₦{final_stats['amount_stats']['min']:,.2f}")
    print(f"   Max amount: ₦{final_stats['amount_stats']['max']:,.2f}")
    print(f"   Mean amount: ₦{final_stats['amount_stats']['mean']:,.2f}")
    print(f"   Median amount: ₦{final_stats['amount_stats']['median']:,.2f}")
    
    # Validate final dataset
    print(f"\n4.4 Final Validation:")
    final_missing = df_final.isnull().sum().sum()
    final_duplicates = df_final.duplicated().sum()
    final_negative = (df_final['amount'] <= 0).sum()
    
    print(f"   Missing values: {final_missing}")
    print(f"   Duplicates: {final_duplicates}")
    print(f"   Zero/negative amounts: {final_negative}")
    print(f"   Final dataset shape: {df_final.shape}")
    
    # Save cleaned dataset
    df_final.to_csv(output_file, index=False)
    print(f"\n✓ Enhanced cleaned dataset saved to: {output_file}")
    
    # Return both the cleaned dataset and statistics
    return df_final, {
        'original_stats': original_stats,
        'amount_filter_stats': amount_filter_stats,
        'outlier_stats': outlier_stats,
        'final_stats': final_stats
    }

def create_outlier_visualization(df, outlier_stats, output_prefix='outlier_analysis'):
    """
    Create visualizations for outlier analysis
    """
    print(f"\n4.5 Creating outlier visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Nigerian Fraud Dataset - Outlier Analysis', fontsize=16, fontweight='bold')
    
    # 1. Box plot
    axes[0, 0].boxplot(df['amount'])
    axes[0, 0].set_title('Amount Distribution - Box Plot')
    axes[0, 0].set_ylabel('Amount (₦)')
    axes[0, 0].tick_params(axis='y', rotation=45)
    
    # 2. Histogram with outlier boundaries
    axes[0, 1].hist(df['amount'], bins=50, alpha=0.7, edgecolor='black')
    
    # Add outlier boundaries
    extreme_bounds = outlier_stats['amount']['extreme_outliers']
    axes[0, 1].axvline(extreme_bounds['upper_bound'], color='red', linestyle='--', 
                       label=f'Extreme Upper Bound (₦{extreme_bounds["upper_bound"]:,.0f})')
    axes[0, 1].axvline(extreme_bounds['Q3'], color='orange', linestyle='--', 
                       label=f'Q3 (₦{extreme_bounds["Q3"]:,.0f})')
    
    axes[0, 1].set_title('Amount Distribution - Histogram')
    axes[0, 1].set_xlabel('Amount (₦)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    
    # 3. Log-scale histogram
    axes[1, 0].hist(np.log10(df['amount']), bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('Amount Distribution - Log Scale')
    axes[1, 0].set_xlabel('Log10(Amount)')
    axes[1, 0].set_ylabel('Frequency')
    
    # 4. Fraud vs Amount scatter
    fraud_amounts = df[df['is_fraud'] == 1]['amount']
    legit_amounts = df[df['is_fraud'] == 0]['amount']
    
    axes[1, 1].scatter(range(len(legit_amounts)), legit_amounts, alpha=0.1, s=1, label='Legitimate', color='blue')
    axes[1, 1].scatter(range(len(fraud_amounts)), fraud_amounts, alpha=0.8, s=2, label='Fraud', color='red')
    axes[1, 1].set_title('Transaction Amounts by Fraud Status')
    axes[1, 1].set_xlabel('Transaction Index')
    axes[1, 1].set_ylabel('Amount (₦)')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_visualization.png', dpi=300, bbox_inches='tight')
    print(f"   Visualization saved as: {output_prefix}_visualization.png")
    plt.close()

def main():
    """
    Main function to run the enhanced cleaning process
    """
    # Specify input and output files
    input_file = 'nibss_fraud_dataset.csv'  # Change this to your input file name
    output_file = 'nibss_fraud_dataset_cleaned.csv'
    
    try:
        # Run the enhanced cleaning process
        cleaned_df, cleaning_stats = clean_dataset_enhanced(input_file, output_file)
        
        # Create visualizations
        create_outlier_visualization(cleaned_df, cleaning_stats['outlier_stats'])
        
        print(f"\n" + "="*60)
        print("✅ ENHANCED CLEANING COMPLETED!")
        print("="*60)
        print("Summary:")
        print(f"• Original transactions: {cleaning_stats['original_stats']['total_transactions']:,}")
        print(f"• Final transactions: {cleaning_stats['final_stats']['total_transactions']:,}")
        print(f"• Fraud rate maintained: {cleaning_stats['final_stats']['fraud_rate']:.4f}")
        print(f"• High-value outliers retained for authenticity")
        print("="*60)
        
    except FileNotFoundError:
        print(f"❌ Error: Input file '{input_file}' not found!")
        print("Please ensure the dataset file exists in the current directory.")
        
    except Exception as e:
        print(f"❌ Error during enhanced cleaning: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()