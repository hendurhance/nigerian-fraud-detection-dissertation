import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_simulated_data_quality(df):
    """
    Comprehensive analysis of simulated dataset quality
    """
    print("=== SIMULATED DATASET QUALITY ANALYSIS ===\n")
    
    # Basic dataset info
    print(f"Dataset Shape: {df.shape}")
    print(f"Total Transactions: {len(df):,}")
    print(f"Fraud Cases: {df['is_fraud'].sum():,}")
    print(f"Fraud Rate: {df['is_fraud'].mean():.4%}")
    print(f"Class Imbalance Ratio: {(1-df['is_fraud'].mean())/df['is_fraud'].mean():.0f}:1")
    print("\n" + "="*50 + "\n")
    
    # 1. Missing Values Check
    print("1. MISSING VALUES ANALYSIS")
    missing_values = df.isnull().sum()
    print("Missing values per column:")
    if missing_values.sum() == 0:
        print("✓ No missing values found (expected for simulated data)")
    else:
        print(missing_values[missing_values > 0])
    print()
    
    # 2. Duplicate Analysis
    print("2. DUPLICATE ANALYSIS")
    total_duplicates = df.duplicated().sum()
    transaction_id_duplicates = df['transaction_id'].duplicated().sum()
    print(f"Total duplicate rows: {total_duplicates}")
    print(f"Duplicate transaction IDs: {transaction_id_duplicates}")
    if total_duplicates == 0 and transaction_id_duplicates == 0:
        print("✓ No duplicates found (expected for simulated data)")
    print()
    
    # 3. Transaction Amount Analysis
    print("3. TRANSACTION AMOUNT ANALYSIS")
    print(f"Amount range: ₦{df['amount'].min():,.2f} to ₦{df['amount'].max():,.2f}")
    print(f"Mean amount: ₦{df['amount'].mean():,.2f}")
    print(f"Median amount: ₦{df['amount'].median():,.2f}")
    
    # Check for negative amounts
    negative_amounts = (df['amount'] < 0).sum()
    print(f"Negative amounts: {negative_amounts}")
    if negative_amounts == 0:
        print("✓ No negative amounts found (expected for simulated data)")
    
    # Check for zero amounts
    zero_amounts = (df['amount'] == 0).sum()
    print(f"Zero amounts: {zero_amounts}")
    
    # Outlier analysis using IQR
    Q1 = df['amount'].quantile(0.25)
    Q3 = df['amount'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR
    
    outliers = df[(df['amount'] < lower_bound) | (df['amount'] > upper_bound)]
    print(f"Statistical outliers (Q3 + 3*IQR): {len(outliers):,} ({len(outliers)/len(df):.2%})")
    
    if len(outliers) > 0:
        print(f"Outlier range: ₦{outliers['amount'].min():,.2f} to ₦{outliers['amount'].max():,.2f}")
        print("Outlier distribution by channel:")
        print(outliers['channel'].value_counts())
    print()
    
    # 4. Categorical Variables Analysis
    print("4. CATEGORICAL VARIABLES ANALYSIS")
    categorical_cols = ['channel', 'merchant_category', 'bank', 'location', 'age_group']
    
    for col in categorical_cols:
        if col in df.columns:
            unique_values = df[col].nunique()
            print(f"{col}: {unique_values} unique values")
            print(f"  Top 3: {df[col].value_counts().head(3).to_dict()}")
    print()
    
    # 5. Timestamp Analysis
    print("5. TIMESTAMP ANALYSIS")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Time span: {(df['timestamp'].max() - df['timestamp'].min()).days} days")
    
    # Check for future dates
    current_date = pd.Timestamp.now()
    future_dates = (df['timestamp'] > current_date).sum()
    print(f"Future dates: {future_dates}")
    print()
    
    # 6. Feature Consistency Check
    print("6. FEATURE CONSISTENCY ANALYSIS")
    
    # Check if derived features are consistent
    inconsistencies = []
    
    # Check weekend flag consistency
    df['calculated_weekend'] = df['day_of_week'].isin([5, 6])  # Saturday=5, Sunday=6
    weekend_inconsistency = (df['is_weekend'] != df['calculated_weekend']).sum()
    if weekend_inconsistency > 0:
        inconsistencies.append(f"Weekend flag: {weekend_inconsistency} inconsistencies")
    
    # Check hour consistency
    df['calculated_hour'] = df['timestamp'].dt.hour
    hour_inconsistency = (df['hour'] != df['calculated_hour']).sum()
    if hour_inconsistency > 0:
        inconsistencies.append(f"Hour extraction: {hour_inconsistency} inconsistencies")
    
    if not inconsistencies:
        print("✓ All derived features are consistent")
    else:
        for issue in inconsistencies:
            print(f"⚠ {issue}")
    print()
    
    # 7. Fraud Distribution Analysis
    print("7. FRAUD DISTRIBUTION ANALYSIS")
    fraud_by_channel = df.groupby('channel')['is_fraud'].agg(['count', 'sum', 'mean'])
    fraud_by_channel['fraud_rate'] = fraud_by_channel['mean'] * 100
    print("Fraud rate by channel:")
    print(fraud_by_channel[['count', 'sum', 'fraud_rate']].round(3))
    print()
    
    return {
        'total_transactions': len(df),
        'fraud_cases': df['is_fraud'].sum(),
        'fraud_rate': df['is_fraud'].mean(),
        'missing_values': missing_values.sum(),
        'duplicates': total_duplicates,
        'negative_amounts': negative_amounts,
        'outliers': len(outliers),
        'inconsistencies': len(inconsistencies)
    }

def plot_data_quality_summary(df):
    """
    Create visualizations for data quality assessment
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Amount distribution
    axes[0,0].hist(df['amount'], bins=50, alpha=0.7, edgecolor='black')
    axes[0,0].set_title('Transaction Amount Distribution')
    axes[0,0].set_xlabel('Amount (₦)')
    axes[0,0].set_ylabel('Frequency')
    
    # Fraud rate by channel
    fraud_by_channel = df.groupby('channel')['is_fraud'].mean() * 100
    axes[0,1].bar(fraud_by_channel.index, fraud_by_channel.values)
    axes[0,1].set_title('Fraud Rate by Channel')
    axes[0,1].set_xlabel('Channel')
    axes[0,1].set_ylabel('Fraud Rate (%)')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Monthly transaction volume
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    monthly_volume = df.groupby(df['timestamp'].dt.month).size()
    axes[1,0].plot(monthly_volume.index, monthly_volume.values, marker='o')
    axes[1,0].set_title('Monthly Transaction Volume')
    axes[1,0].set_xlabel('Month')
    axes[1,0].set_ylabel('Transaction Count')
    
    # Amount by fraud status
    axes[1,1].boxplot([df[df['is_fraud']==0]['amount'], df[df['is_fraud']==1]['amount']], 
                     labels=['Legitimate', 'Fraud'])
    axes[1,1].set_title('Amount Distribution by Fraud Status')
    axes[1,1].set_ylabel('Amount (₦)')
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to run the analysis
    """
    # Load your dataset here - replace 'your_dataset.csv' with your actual file path
    try:
        # Option 1: If your CSV file is in the same directory
        df = pd.read_csv('nibss_fraud_dataset.csv')
        
        # Option 2: If you want to specify the full path
        # df = pd.read_csv('/path/to/your/dataset.csv')
        
        # Option 3: If you want to prompt user for file path
        # file_path = input("Enter the path to your CSV file: ")
        # df = pd.read_csv(file_path)
        
        print("Dataset loaded successfully!")
        print(f"Shape: {df.shape}")
        print("\nFirst few rows:")
        print(df.head())
        
        # Run the analysis
        quality_report = analyze_simulated_data_quality(df)
        
        # Create visualizations
        plot_data_quality_summary(df)
        
        # Print summary report
        print("\n=== SUMMARY REPORT ===")
        for key, value in quality_report.items():
            print(f"{key.replace('_', ' ').title()}: {value:,}")
            
    except FileNotFoundError:
        print("Error: Dataset file not found!")
        print("Please ensure your CSV file is in the correct location.")
        print("Current expected filename: 'nigerian_fraud_dataset.csv'")
        print("\nTo use a different filename, modify the file path in the script.")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please check your file path and ensure the CSV is properly formatted.")

if __name__ == "__main__":
    main()