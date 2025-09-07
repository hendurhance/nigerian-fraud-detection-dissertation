#!/usr/bin/env python3
"""
Fast Class Imbalance Strategy Testing using Stratified Sampling

This version uses a stratified sample to reduce computation time while maintaining
statistical validity for strategy comparison.

Author: Your Name
Date: 2024
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import time
import warnings
warnings.filterwarnings('ignore')

class FastClassImbalanceEvaluator:
    """Fast evaluation using stratified sampling for large datasets."""
    
    def __init__(self, dataset_path, sample_size=100000, random_state=42):
        """
        Initialize the fast evaluator.
        
        Args:
            dataset_path (str): Path to the CSV dataset
            sample_size (int): Size of stratified sample to use
            random_state (int): Random state for reproducibility
        """
        self.dataset_path = dataset_path
        self.sample_size = sample_size
        self.random_state = random_state
        self.df = None
        self.df_sample = None
        self.X = None
        self.y = None
        self.feature_columns = None
        self.results = {}
        
    def load_and_sample_data(self):
        """Load dataset and create stratified sample."""
        print("Loading full dataset...")
        self.df = pd.read_csv(self.dataset_path)
        
        original_fraud_rate = self.df['is_fraud'].mean()
        original_fraud_count = self.df['is_fraud'].sum()
        
        print(f"Full dataset shape: {self.df.shape}")
        print(f"Original fraud rate: {original_fraud_rate:.4%}")
        print(f"Original fraud cases: {original_fraud_count:,}")
        
        # Create stratified sample
        print(f"\nCreating stratified sample of {self.sample_size:,} transactions...")
        
        fraud_cases = self.df[self.df['is_fraud'] == 1]
        legit_cases = self.df[self.df['is_fraud'] == 0]
        
        # Calculate sample sizes to maintain fraud rate
        target_fraud_count = int(self.sample_size * original_fraud_rate)
        target_legit_count = self.sample_size - target_fraud_count
        
        # Ensure we don't sample more than available
        actual_fraud_count = min(target_fraud_count, len(fraud_cases))
        actual_legit_count = min(target_legit_count, len(legit_cases))
        
        # Sample with replacement if needed for fraud cases (usually very few)
        if actual_fraud_count < len(fraud_cases):
            sampled_fraud = fraud_cases.sample(n=actual_fraud_count, random_state=self.random_state)
        else:
            sampled_fraud = fraud_cases.sample(n=actual_fraud_count, replace=True, random_state=self.random_state)
            
        sampled_legit = legit_cases.sample(n=actual_legit_count, random_state=self.random_state)
        
        # Combine samples
        self.df_sample = pd.concat([sampled_fraud, sampled_legit], ignore_index=True)
        self.df_sample = self.df_sample.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        
        sample_fraud_rate = self.df_sample['is_fraud'].mean()
        sample_fraud_count = self.df_sample['is_fraud'].sum()
        
        print(f"Sample dataset shape: {self.df_sample.shape}")
        print(f"Sample fraud rate: {sample_fraud_rate:.4%}")
        print(f"Sample fraud cases: {sample_fraud_count:,}")
        print(f"Fraud rate preservation: {abs(original_fraud_rate - sample_fraud_rate) < 0.001}")
        
        # Prepare features
        self._prepare_features()
        
        # Split into features and target
        self.y = self.df_sample['is_fraud'].values
        self.X = self.df_sample[self.feature_columns].values
        
        print(f"Features selected: {len(self.feature_columns)}")
        print(f"Class distribution: {Counter(self.y)}")
        
    def _prepare_features(self):
        """Prepare features for modeling."""
        # Exclude non-predictive columns
        exclude_columns = [
            'transaction_id', 'customer_id', 'timestamp', 'is_fraud', 'fraud_technique'
        ]
        
        # Select numeric features and encode categorical ones
        feature_columns = []
        categorical_columns = ['channel', 'merchant_category', 'bank', 'location', 'age_group']
        
        # Add numeric features
        numeric_features = self.df_sample.select_dtypes(include=[np.number]).columns.tolist()
        numeric_features = [col for col in numeric_features if col not in exclude_columns]
        feature_columns.extend(numeric_features)
        
        # Encode categorical features
        for col in categorical_columns:
            if col in self.df_sample.columns:
                le = LabelEncoder()
                self.df_sample[f'{col}_encoded'] = le.fit_transform(self.df_sample[col].astype(str))
                feature_columns.append(f'{col}_encoded')
        
        self.feature_columns = feature_columns
        
    def evaluate_strategies_quickly(self):
        """Evaluate all strategies with minimal computation."""
        print("\n" + "="*60)
        print("FAST CLASS IMBALANCE STRATEGY EVALUATION")
        print("="*60)
        
        # Split data once
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=self.random_state, stratify=self.y
        )
        
        strategies_to_test = {
            'Baseline': {'type': 'baseline'},
            'Class_Weight_Balanced': {'type': 'class_weight', 'weight': 'balanced'},
            'Class_Weight_Manual': {'type': 'class_weight', 'weight': {0: 1, 1: Counter(y_train)[0]/Counter(y_train)[1]}},
            'SMOTE_1_5': {'type': 'smote', 'ratio': 0.2},
            'SMOTE_1_3': {'type': 'smote', 'ratio': 0.33},
            'Undersample_1_5': {'type': 'undersample', 'ratio': 0.2},
            'Undersample_1_3': {'type': 'undersample', 'ratio': 0.33}
        }
        
        results = {}
        
        for strategy_name, config in strategies_to_test.items():
            print(f"\nTesting: {strategy_name}")
            start_time = time.time()
            
            # Prepare training data based on strategy
            if config['type'] == 'baseline':
                X_train_proc, y_train_proc = X_train, y_train
                model_params = {}
                
            elif config['type'] == 'class_weight':
                X_train_proc, y_train_proc = X_train, y_train
                model_params = {'class_weight': config['weight']}
                
            elif config['type'] == 'smote':
                smote = SMOTE(sampling_strategy=config['ratio'], random_state=self.random_state)
                X_train_proc, y_train_proc = smote.fit_resample(X_train, y_train)
                model_params = {}
                
            elif config['type'] == 'undersample':
                undersampler = RandomUnderSampler(sampling_strategy=config['ratio'], random_state=self.random_state)
                X_train_proc, y_train_proc = undersampler.fit_resample(X_train, y_train)
                model_params = {}
            
            # Train RandomForest (faster and usually better for fraud detection)
            model = RandomForestClassifier(
                random_state=self.random_state, 
                n_estimators=50,  # Reduced for speed
                **model_params
            )
            
            model.fit(X_train_proc, y_train_proc)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)
            
            elapsed_time = time.time() - start_time
            
            results[strategy_name] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc': auc,
                'training_samples': len(X_train_proc),
                'fraud_samples': sum(y_train_proc),
                'time_seconds': elapsed_time,
                'config': config
            }
            
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  AUC: {auc:.4f}")
            print(f"  Training samples: {len(X_train_proc):,}")
            print(f"  Time: {elapsed_time:.2f}s")
        
        self.results = results
        return results
    
    def cross_validate_top_strategies(self, top_n=3):
        """Cross-validate the top N strategies."""
        print(f"\n" + "="*60)
        print(f"CROSS-VALIDATION OF TOP {top_n} STRATEGIES")
        print("="*60)
        
        # Sort strategies by F1 score
        sorted_strategies = sorted(
            self.results.items(), 
            key=lambda x: x[1]['f1_score'], 
            reverse=True
        )[:top_n]
        
        cv_results = {}
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)  # Reduced folds for speed
        
        for strategy_name, _ in sorted_strategies:
            print(f"\nCross-validating: {strategy_name}")
            config = self.results[strategy_name]['config']
            
            f1_scores = []
            auc_scores = []
            
            for train_idx, val_idx in cv.split(self.X, self.y):
                X_train_cv, X_val_cv = self.X[train_idx], self.X[val_idx]
                y_train_cv, y_val_cv = self.y[train_idx], self.y[val_idx]
                
                # Apply strategy
                if config['type'] == 'baseline':
                    X_train_proc, y_train_proc = X_train_cv, y_train_cv
                    model_params = {}
                    
                elif config['type'] == 'class_weight':
                    X_train_proc, y_train_proc = X_train_cv, y_train_cv
                    model_params = {'class_weight': config['weight']}
                    
                elif config['type'] == 'smote':
                    smote = SMOTE(sampling_strategy=config['ratio'], random_state=self.random_state)
                    X_train_proc, y_train_proc = smote.fit_resample(X_train_cv, y_train_cv)
                    model_params = {}
                    
                elif config['type'] == 'undersample':
                    undersampler = RandomUnderSampler(sampling_strategy=config['ratio'], random_state=self.random_state)
                    X_train_proc, y_train_proc = undersampler.fit_resample(X_train_cv, y_train_cv)
                    model_params = {}
                
                # Train and evaluate
                model = RandomForestClassifier(
                    random_state=self.random_state, 
                    n_estimators=50,
                    **model_params
                )
                model.fit(X_train_proc, y_train_proc)
                
                y_pred = model.predict(X_val_cv)
                y_prob = model.predict_proba(X_val_cv)[:, 1]
                
                f1_scores.append(f1_score(y_val_cv, y_pred))
                auc_scores.append(roc_auc_score(y_val_cv, y_prob))
            
            cv_results[strategy_name] = {
                'f1_mean': np.mean(f1_scores),
                'f1_std': np.std(f1_scores),
                'auc_mean': np.mean(auc_scores),
                'auc_std': np.std(auc_scores)
            }
            
            print(f"  F1-Score: {np.mean(f1_scores):.4f} (+/- {np.std(f1_scores) * 2:.4f})")
            print(f"  AUC: {np.mean(auc_scores):.4f} (+/- {np.std(auc_scores) * 2:.4f})")
        
        return cv_results
    
    def generate_summary_report(self):
        """Generate summary report."""
        print("\n" + "="*80)
        print("FAST CLASS IMBALANCE STRATEGY SUMMARY")
        print("="*80)
        
        # Create summary table
        summary_data = []
        
        for strategy_name, metrics in self.results.items():
            summary_data.append({
                'Strategy': strategy_name,
                'Training_Samples': f"{metrics['training_samples']:,}",
                'Fraud_Samples': f"{metrics['fraud_samples']:,}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1_Score': f"{metrics['f1_score']:.4f}",
                'AUC': f"{metrics['auc']:.4f}",
                'Time_Seconds': f"{metrics['time_seconds']:.2f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Sort by F1 score
        summary_df['F1_Float'] = summary_df['F1_Score'].astype(float)
        summary_df = summary_df.sort_values('F1_Float', ascending=False)
        
        print("\nRESULTS TABLE (Sorted by F1-Score):")
        print("-" * 110)
        print(f"{'Strategy':<20} {'Train_Samples':<12} {'Fraud_Samples':<12} {'Precision':<10} {'Recall':<10} {'F1_Score':<10} {'AUC':<10} {'Time(s)':<8}")
        print("-" * 110)
        
        for _, row in summary_df.iterrows():
            print(f"{row['Strategy']:<20} {row['Training_Samples']:<12} {row['Fraud_Samples']:<12} {row['Precision']:<10} {row['Recall']:<10} {row['F1_Score']:<10} {row['AUC']:<10} {row['Time_Seconds']:<8}")
        
        # Best strategy
        best_strategy = summary_df.iloc[0]
        print(f"\n🏆 BEST STRATEGY: {best_strategy['Strategy']}")
        print(f"   F1-Score: {best_strategy['F1_Score']}")
        print(f"   AUC: {best_strategy['AUC']}")
        print(f"   Training Time: {best_strategy['Time_Seconds']}s")
        
        # Save results
        summary_df.drop('F1_Float', axis=1).to_csv('fast_class_imbalance_results.csv', index=False)
        print(f"\n✓ Results saved to: fast_class_imbalance_results.csv")
        
        return summary_df
    
    def run_complete_evaluation(self):
        """Run the complete fast evaluation."""
        total_start_time = time.time()
        
        print("Starting Fast Class Imbalance Strategy Evaluation...")
        print("="*60)
        
        # Load and sample data
        self.load_and_sample_data()
        
        # Evaluate strategies
        self.evaluate_strategies_quickly()
        
        # Cross-validate top strategies
        cv_results = self.cross_validate_top_strategies()
        
        # Generate summary
        summary_df = self.generate_summary_report()
        
        total_time = time.time() - total_start_time
        
        print(f"\n✅ FAST EVALUATION COMPLETED!")
        print(f"Total Time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        print("="*60)
        
        return summary_df, cv_results


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fast class imbalance strategy evaluation')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the fraud dataset CSV file')
    parser.add_argument('--sample_size', type=int, default=100000, help='Size of stratified sample (default: 100,000)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Initialize and run evaluator
    evaluator = FastClassImbalanceEvaluator(
        args.dataset, 
        sample_size=args.sample_size,
        random_state=args.seed
    )
    results_df, cv_results = evaluator.run_complete_evaluation()
    
    return results_df, cv_results


if __name__ == "__main__":
    # Example usage
    try:
        results, cv_results = main()
    except Exception as e:
        print(f"Error during execution: {e}")
        print("\nUsage example:")
        print("python fast_class_imbalance_testing.py --dataset nibss_fraud_dataset.csv --sample_size 100000")
        
        # Alternative: Run with hardcoded dataset path for testing
        print("\nRunning with default settings...")
        evaluator = FastClassImbalanceEvaluator("nibss_fraud_dataset.csv", sample_size=100000)
        results, cv_results = evaluator.run_complete_evaluation()