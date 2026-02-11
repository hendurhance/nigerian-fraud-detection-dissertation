# nibss_fraud_dataset_generator.py
"""
Synthetic Nigerian financial transaction generator calibrated to NIBSS 2023 Annual Fraud Landscape.

This refactored version follows the exact NIBSS patterns for:
- Monthly transaction volume and value distributions
- Monthly fraud count and loss distributions  
- Channel-specific fraud patterns
- Accurate fraud prevalence and loss amounts

Run `python nibss_fraud_dataset_generator.py --help` for CLI usage.
"""

from __future__ import annotations

import argparse
import calendar
import hashlib
import random
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# NIBSS 2023 Calibrated Configuration
# ──────────────────────────────────────────────────────────────────────────────

NIBSS_CONFIG = {
    "fraud_rate": 0.003,  # 0.30% overall fraud ratio
    
    # Monthly transaction volume percentages (NIBSS 2023)
    "monthly_volume_pct": {
        1: 5.60,   # Jan
        2: 8.15,   # Feb 
        3: 12.11,  # Mar (peak volume)
        4: 8.42,   # Apr
        5: 7.99,   # May
        6: 7.55,   # Jun
        7: 7.69,   # Jul
        8: 7.82,   # Aug
        9: 7.71,   # Sep
        10: 8.36,  # Oct
        11: 8.57,  # Nov
        12: 10.02  # Dec
    },
    
    # Monthly transaction value percentages (NIBSS 2023)
    "monthly_value_pct": {
        1: 6.46,   # Jan
        2: 6.13,   # Feb
        3: 8.05,   # Mar
        4: 6.88,   # Apr
        5: 7.65,   # May
        6: 7.55,   # Jun
        7: 7.90,   # Jul
        8: 8.48,   # Aug
        9: 8.50,   # Sep
        10: 9.83,  # Oct
        11: 10.60, # Nov
        12: 11.98  # Dec (peak value)
    },
    
    # Monthly fraud count percentages (NIBSS 2023)
    "monthly_fraud_count_pct": {
        1: 9.66,   # Jan
        2: 9.93,   # Feb
        3: 9.23,   # Mar
        4: 8.61,   # Apr
        5: 12.25,  # May (peak fraud count)
        6: 6.53,   # Jun
        7: 9.05,   # Jul
        8: 8.58,   # Aug
        9: 7.85,   # Sep
        10: 7.17,  # Oct
        11: 6.66,  # Nov
        12: 4.49   # Dec (lowest fraud)
    },
    
    # Monthly fraud loss value percentages (NIBSS 2023)
    "monthly_fraud_loss_pct": {
        1: 15.44,  # Jan (highest loss)
        2: 7.61,   # Feb
        3: 5.38,   # Mar
        4: 6.26,   # Apr
        5: 8.73,   # May
        6: 4.25,   # Jun
        7: 6.57,   # Jul
        8: 7.77,   # Aug
        9: 7.18,   # Sep
        10: 21.18, # Oct (second highest loss)
        11: 5.79,  # Nov
        12: 3.83   # Dec
    },
    
    # Channel distribution (transaction volume)
    "channels": {
        "Mobile": 0.45,  # 45%
        "Web": 0.20,     # 20%
        "POS": 0.18,     # 18%
        "IB": 0.10,      # 10% (Internet Banking)
        "ECOM": 0.05,    # 5% (E-commerce)
        "ATM": 0.02      # 2%
    },
    
    # Channel fraud count percentages (NIBSS 2023)
    "channel_fraud_count_pct": {
        "Mobile": 49.75,  # 49.75%
        "Web": 22.91,     # 22.91%
        "POS": 18.38,     # 18.38%
        "IB": 5.63,       # 5.63%
        "ECOM": 2.56,     # 2.56%
        "ATM": 0.76       # 0.76%
    },
    
    # Channel fraud loss value percentages (NIBSS 2023)
    "channel_fraud_loss_pct": {
        "Mobile": 34.21,  # 34.21%
        "Web": 13.37,     # 13.37%
        "POS": 26.54,     # 26.54%
        "IB": 24.63,      # 24.63%
        "ECOM": 0.87,     # 0.87%
        "ATM": 0.38       # 0.38%
    },
    
    # Derived average loss per fraud case by channel (calculated from NIBSS data)
    "channel_avg_loss": {
        "Mobile": 119842,   # ₦119,842
        "Web": 101616,      # ₦101,616
        "POS": 251391,      # ₦251,391
        "IB": 761445,       # ₦761,445
        "ECOM": 58818,      # ₦58,818
        "ATM": 88086        # ₦88,086
    },
    
    # Demographics (from original NIBSS data)
    "age_band": {
        "<20": 0.02,   # 2%
        "20-29": 0.28, # 28%
        "30-39": 0.28, # 28%
        "40+": 0.42    # 42%
    },
    
    "states": {
        "Lagos": 0.48,   # 48%
        "Abuja": 0.05,   # 5%
        "Rivers": 0.04,  # 4%
        "Ogun": 0.04,    # 4%
        "Oyo": 0.03,     # 3%
        "Other": 0.36    # 36%
    },
    
    # Fraud techniques (from original NIBSS data)
    "fraud_technique": {
        "SOCIAL_ENGINEERING": 0.658,  # 65.8%
        "ROBBERY": 0.106,             # 10.6%
        "CARD_THEFT": 0.071,          # 7.1%
        "PIN_COMPROMISE": 0.055,      # 5.5%
        "PHISHING": 0.047,            # 4.7%
        "OTHER": 0.063                # 6.3%
    }
}

# Nigerian banks and merchant categories
NIGERIAN_BANKS = [
    "GTBank", "FirstBank", "Zenith", "UBA", "Access", "Fidelity",
    "Sterling", "FCMB", "Wema", "Union"
]

MERCHANT_CATEGORIES = [
    "Retail", "Grocery", "Fuel", "ATM_Withdrawal", "Transfer",
    "Bill_Payment", "Airtime", "Restaurant", "Transport", "Medical",
    "Education", "Entertainment", "Fashion", "Electronics"
]

# ──────────────────────────────────────────────────────────────────────────────
# Main Generator Class
# ──────────────────────────────────────────────────────────────────────────────

class NIBSSFraudDatasetGenerator:
    """NIBSS 2023 calibrated synthetic fraud dataset generator."""
    
    def __init__(self, dataset_size: int = 1_000_000, seed: int = 42):
        self.dataset_size = int(dataset_size)
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Load NIBSS configuration
        self.config = NIBSS_CONFIG
        self.fraud_rate = self.config["fraud_rate"]
        self.fraud_count = int(self.dataset_size * self.fraud_rate)
        self.legit_count = self.dataset_size - self.fraud_count
        
        print(f"Generating {self.dataset_size:,} transactions with {self.fraud_count:,} fraudulent ({self.fraud_rate:.3%})")
        
        # Normalize distributions
        self._prepare_distributions()
        
        # Nigerian business hours
        self.peak_hours = [10, 11, 14, 15, 16]
        self._hourly_weights = self._build_hourly_weights()
    
    def _prepare_distributions(self):
        """Prepare and normalize all probability distributions."""
        
        # Channel distributions
        self.channels = list(self.config["channels"].keys())
        self.channel_weights = np.array(list(self.config["channels"].values()))
        self.channel_weights /= self.channel_weights.sum()
        
        # Monthly distributions (properly normalized to probabilities)
        monthly_volume_raw = np.array([
            self.config["monthly_volume_pct"][i] for i in range(1, 13)
        ])
        self.monthly_volume_weights = monthly_volume_raw / monthly_volume_raw.sum()
        
        monthly_value_raw = np.array([
            self.config["monthly_value_pct"][i] for i in range(1, 13)
        ])
        self.monthly_value_weights = monthly_value_raw / monthly_value_raw.sum()
        
        monthly_fraud_count_raw = np.array([
            self.config["monthly_fraud_count_pct"][i] for i in range(1, 13)
        ])
        self.monthly_fraud_count_weights = monthly_fraud_count_raw / monthly_fraud_count_raw.sum()
        
        monthly_fraud_loss_raw = np.array([
            self.config["monthly_fraud_loss_pct"][i] for i in range(1, 13)
        ])
        self.monthly_fraud_loss_weights = monthly_fraud_loss_raw / monthly_fraud_loss_raw.sum()
        
        # Demographic distributions
        self.age_bands = list(self.config["age_band"].keys())
        self.age_weights = list(self.config["age_band"].values())
        
        self.states = list(self.config["states"].keys())
        self.state_weights = list(self.config["states"].values())
        
        # Fraud technique distribution
        self.techniques = list(self.config["fraud_technique"].keys())
        self.technique_weights = list(self.config["fraud_technique"].values())
        
        # Channel fraud distributions (properly normalized)
        channel_fraud_count_raw = np.array([
            self.config["channel_fraud_count_pct"][channel] for channel in self.channels
        ])
        self.channel_fraud_count_weights = channel_fraud_count_raw / channel_fraud_count_raw.sum()
        
        channel_fraud_loss_raw = np.array([
            self.config["channel_fraud_loss_pct"][channel] for channel in self.channels
        ])
        self.channel_fraud_loss_weights = channel_fraud_loss_raw / channel_fraud_loss_raw.sum()
    
    def generate_dataset(self) -> pd.DataFrame:
        """Generate the complete synthetic dataset."""
        print("Generating synthetic NIBSS-calibrated dataset...")
        
        # Generate customer profiles
        customers = self._generate_customer_profiles()
        print(f"• Generated {len(customers):,} customer profiles")
        
        # Generate transactions
        records = []
        for i in range(self.dataset_size):
            if i and i % 50_000 == 0:
                print(f"  – Generated {i:,} transactions")
            
            customer = random.choice(customers)
            timestamp = self._sample_timestamp()
            transaction = self._generate_base_transaction(customer, timestamp)
            records.append(transaction)
        
        # Convert to DataFrame and add features
        df = pd.DataFrame.from_records(records)
        df = self._add_behavioral_features(df)
        df = self._inject_fraud_patterns(df)
        df = self._add_final_features(df)
        
        return df
    
    def _generate_customer_profiles(self) -> list:
        """Generate customer profiles."""
        customer_count = max(5_000, self.dataset_size // 100)
        customers = []
        
        for _ in range(customer_count):
            customers.append({
                "customer_id": f"CUST_{uuid.uuid4().hex[:8].upper()}",
                "age_group": np.random.choice(self.age_bands, p=self.age_weights),
                "location": np.random.choice(self.states, p=self.state_weights),
            })
        
        return customers
    
    def _sample_timestamp(self) -> datetime:
        """Sample timestamp using NIBSS volume distribution."""
        year = 2023
        
        # Sample month based on transaction volume distribution
        month = np.random.choice(np.arange(1, 13), p=self.monthly_volume_weights)
        day = random.randint(1, calendar.monthrange(year, month)[1])
        hour = np.random.choice(np.arange(24), p=self._hourly_weights)
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        
        return datetime(year, month, day, hour, minute, second)
    
    def _build_hourly_weights(self) -> np.ndarray:
        """Build hourly transaction weights (Nigerian business hours)."""
        weights = np.ones(24) * 0.02  # Base weight for off-hours
        weights[8:18] = 0.08          # Business hours
        weights[self.peak_hours] = 0.12  # Peak hours
        return weights / weights.sum()
    
    def _generate_base_transaction(self, customer: dict, timestamp: datetime) -> dict:
        """Generate base transaction record."""
        
        # Sample channel
        channel = np.random.choice(self.channels, p=self.channel_weights)
        
        # Generate amount based on monthly value distribution
        month_value_factor = self.monthly_value_weights[timestamp.month - 1] / self.monthly_volume_weights[timestamp.month - 1]
        
        # Base amount calculation with channel-specific adjustments
        if channel == "Mobile":
            base_amount = np.random.lognormal(11.0, 1.2)
        elif channel == "Web":
            base_amount = np.random.lognormal(11.5, 1.0)
        elif channel == "POS":
            base_amount = np.random.lognormal(10.5, 1.3)
        elif channel == "IB":
            base_amount = np.random.lognormal(12.5, 1.1)
        elif channel == "ECOM":
            base_amount = np.random.lognormal(10.8, 0.9)
        else:  # ATM
            base_amount = np.random.lognormal(10.2, 0.7)
        
        # Apply monthly value factor
        amount = base_amount * month_value_factor * np.random.uniform(0.8, 1.2)
        amount = max(100, min(amount, 10_000_000))  # Clamp to reasonable range
        
        return {
            "transaction_id": f"TXN_{uuid.uuid4().hex[:12].upper()}",
            "customer_id": customer["customer_id"],
            "timestamp": timestamp,
            "amount": round(amount, 2),
            "channel": channel,
            "merchant_category": random.choice(MERCHANT_CATEGORIES),
            "bank": random.choice(NIGERIAN_BANKS),
            "location": customer["location"],
            "age_group": customer["age_group"],
            "hour": timestamp.hour,
            "day_of_week": timestamp.weekday(),
            "month": timestamp.month,
            "is_weekend": timestamp.weekday() >= 5,
            "is_peak_hour": timestamp.hour in self.peak_hours,
        }
    
    def _add_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add behavioral features based on transaction patterns."""
        df = df.sort_values(["customer_id", "timestamp"]).reset_index(drop=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        print("• Computing behavioral features...")
        
        # Initialize behavioral columns
        df["tx_count_24h"] = 1.0
        df["amount_sum_24h"] = df["amount"]
        df["amount_mean_7d"] = df["amount"]
        df["amount_std_7d"] = 0.0
        
        # Calculate rolling features for each customer
        for customer_id, group in df.groupby("customer_id"):
            if len(group) <= 1:
                continue
                
            indices = group.index
            timestamps = group['timestamp'].values
            amounts = group['amount'].values
            
            for i, idx in enumerate(indices):
                current_time = timestamps[i]
                
                # 24-hour window
                time_24h_ago = current_time - pd.Timedelta(hours=24)
                mask_24h = (timestamps <= current_time) & (timestamps >= time_24h_ago)
                
                # 7-day window
                time_7d_ago = current_time - pd.Timedelta(days=7)
                mask_7d = (timestamps <= current_time) & (timestamps >= time_7d_ago)
                
                if mask_24h.any():
                    df.loc[idx, "tx_count_24h"] = float(mask_24h.sum())
                    df.loc[idx, "amount_sum_24h"] = float(amounts[mask_24h].sum())
                
                if mask_7d.any() and mask_7d.sum() > 1:
                    amounts_7d = amounts[mask_7d]
                    df.loc[idx, "amount_mean_7d"] = float(amounts_7d.mean())
                    df.loc[idx, "amount_std_7d"] = float(amounts_7d.std()) if len(amounts_7d) > 1 else 0.0
        
        # Customer-level aggregations
        customer_stats = df.groupby("customer_id").agg(
            tx_count_total=("amount", "size"),
            amount_mean_total=("amount", "mean"),
            amount_std_total=("amount", "std"),
            channel_diversity=("channel", "nunique"),
            location_diversity=("location", "nunique"),
        ).fillna(0)
        
        df = df.merge(customer_stats, on="customer_id", how="left")
        
        # Derived ratios
        df["online_channel_ratio"] = (
            df.groupby("customer_id")["channel"]
            .transform(lambda x: x.isin(["Mobile", "Web", "ECOM"]).mean())
        )
        
        return df
    
    def _inject_fraud_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inject fraud patterns following NIBSS distributions EXACTLY."""
        print("• Injecting fraud patterns...")
        
        df["is_fraud"] = 0
        df["fraud_technique"] = None
        
        # Step 1: First, ensure we get the exact channel distribution for ALL fraud cases
        # Calculate target fraud counts by channel (based on channel_fraud_count_pct)
        fraud_by_channel = {}
        for channel in self.channels:
            channel_fraud_weight = self.config["channel_fraud_count_pct"][channel] / 100.0
            channel_fraud_count = int(self.fraud_count * channel_fraud_weight)
            fraud_by_channel[channel] = channel_fraud_count
        
        # Adjust to ensure total equals fraud_count exactly
        total_assigned = sum(fraud_by_channel.values())
        if total_assigned != self.fraud_count:
            # Adjust the largest channel to match exactly
            largest_channel = max(fraud_by_channel.keys(), key=lambda x: fraud_by_channel[x])
            fraud_by_channel[largest_channel] += (self.fraud_count - total_assigned)
        
        print("  Target fraud distribution by channel:")
        for channel, count in fraud_by_channel.items():
            pct = (count / self.fraud_count * 100) if self.fraud_count > 0 else 0
            target_pct = self.config["channel_fraud_count_pct"][channel]
            print(f"    {channel:<8}: {count:4d} ({pct:5.2f}% vs {target_pct:5.2f}% target)")
        
        # Step 2: For each channel, select fraud cases respecting monthly distribution
        all_fraud_indices = []
        
        for channel in self.channels:
            target_channel_fraud_count = fraud_by_channel[channel]
            if target_channel_fraud_count <= 0:
                continue
                
            # Get all transactions for this channel
            channel_transactions = df[df['channel'] == channel]
            if len(channel_transactions) == 0:
                print(f"    Warning: No transactions found for channel {channel}")
                continue
            
            # For this channel, distribute fraud across months using monthly_fraud_count_weights
            channel_fraud_by_month = {}
            for i, month in enumerate(range(1, 13)):
                month_fraud_weight = self.monthly_fraud_count_weights[i]
                month_fraud_count = int(target_channel_fraud_count * month_fraud_weight)
                channel_fraud_by_month[month] = month_fraud_count
            
            # Adjust to ensure channel total equals target exactly
            channel_total_assigned = sum(channel_fraud_by_month.values())
            if channel_total_assigned != target_channel_fraud_count:
                # Find month with most fraud for this channel and adjust
                largest_month = max(channel_fraud_by_month.keys(), key=lambda x: channel_fraud_by_month[x])
                channel_fraud_by_month[largest_month] += (target_channel_fraud_count - channel_total_assigned)
            
            # Step 3: Select fraud cases for this channel, month by month
            channel_fraud_indices = []
            
            for month in range(1, 13):
                target_month_fraud = channel_fraud_by_month[month]
                if target_month_fraud <= 0:
                    continue
                    
                # Get transactions for this channel and month
                month_channel_transactions = channel_transactions[channel_transactions['month'] == month]
                if len(month_channel_transactions) == 0:
                    continue
                
                # Sample fraud cases for this channel-month combination
                available_count = len(month_channel_transactions)
                actual_fraud_count = min(target_month_fraud, available_count)
                
                if actual_fraud_count > 0:
                    sampled_indices = np.random.choice(
                        month_channel_transactions.index,
                        size=actual_fraud_count,
                        replace=False
                    ).tolist()
                    channel_fraud_indices.extend(sampled_indices)
            
            all_fraud_indices.extend(channel_fraud_indices)
            print(f"    {channel:<8}: Selected {len(channel_fraud_indices):4d} fraud cases")
        
        # Step 4: Handle any shortfall by random sampling from remaining transactions
        if len(all_fraud_indices) < self.fraud_count:
            remaining_indices = [idx for idx in df.index if idx not in all_fraud_indices]
            shortfall = self.fraud_count - len(all_fraud_indices)
            if shortfall > 0 and len(remaining_indices) > 0:
                additional_count = min(shortfall, len(remaining_indices))
                additional_fraud = np.random.choice(remaining_indices, size=additional_count, replace=False)
                all_fraud_indices.extend(additional_fraud.tolist())
                print(f"    Added {additional_count} random fraud cases to meet target")
        
        # Step 5: If we have too many, randomly remove excess
        if len(all_fraud_indices) > self.fraud_count:
            all_fraud_indices = np.random.choice(all_fraud_indices, size=self.fraud_count, replace=False).tolist()
        
        # Step 6: Mark fraud cases
        df.loc[all_fraud_indices, "is_fraud"] = 1
        
        # Step 7: Assign fraud techniques
        for idx in all_fraud_indices:
            technique = np.random.choice(self.techniques, p=self.technique_weights)
            df.loc[idx, "fraud_technique"] = technique
        
        # Step 8: Adjust fraud amounts based on NIBSS loss patterns
        self._adjust_fraud_amounts(df, all_fraud_indices)
        
        # Verification: Print actual distributions
        print("  Verification - Monthly fraud distribution:")
        actual_monthly_fraud = df[df['is_fraud'] == 1].groupby('month').size()
        for month in range(1, 13):
            actual_count = actual_monthly_fraud.get(month, 0)
            actual_pct = (actual_count / self.fraud_count * 100) if self.fraud_count > 0 else 0
            target_pct = self.config["monthly_fraud_count_pct"][month]
            print(f"    Month {month:2d}: {actual_count:3d} ({actual_pct:5.2f}% vs {target_pct:5.2f}%)")
        
        print("  Verification - Channel fraud distribution:")
        actual_channel_fraud = df[df['is_fraud'] == 1]['channel'].value_counts()
        for channel in self.channels:
            actual_count = actual_channel_fraud.get(channel, 0)
            actual_pct = (actual_count / self.fraud_count * 100) if self.fraud_count > 0 else 0
            target_pct = self.config["channel_fraud_count_pct"][channel]
            match = "✓" if abs(actual_pct - target_pct) < 1.0 else "✗"
            print(f"    {channel:<8}: {actual_count:4d} ({actual_pct:5.2f}% vs {target_pct:5.2f}%) {match}")
        
        return df

    def _adjust_fraud_amounts(self, df: pd.DataFrame, fraud_indices: list):
        """Adjust fraud amounts to match NIBSS loss distribution."""
        
        for idx in fraud_indices:
            channel = df.loc[idx, "channel"]
            month = df.loc[idx, "month"]
            technique = df.loc[idx, "fraud_technique"]
            
            # Get target average loss for this channel
            target_avg_loss = self.config["channel_avg_loss"][channel]
            
            # Apply monthly loss pattern adjustment
            month_idx = month - 1  # Convert to 0-based index
            month_loss_factor = (
                self.monthly_fraud_loss_weights[month_idx] / 
                self.monthly_fraud_count_weights[month_idx]
            )
            
            # Generate fraud amount around target with some variance
            fraud_amount = np.random.lognormal(
                np.log(target_avg_loss * month_loss_factor), 
                0.8
            )
            
            # Apply technique-specific adjustments
            if technique in ["SOCIAL_ENGINEERING", "PIN_COMPROMISE"]:
                fraud_amount *= np.random.uniform(1.2, 2.5)
            elif technique == "ROBBERY":
                fraud_amount *= np.random.uniform(0.8, 1.5)
            elif technique == "PHISHING":
                fraud_amount *= np.random.uniform(0.6, 1.2)
            
            # Clamp to reasonable range
            fraud_amount = max(1000, min(fraud_amount, 50_000_000))
            df.loc[idx, "amount"] = round(fraud_amount, 2)
    
    def _add_final_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add final engineered features."""
        print("• Adding final features...")
        
        # Cyclic time encodings
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        
        # Amount-based features (computed after fraud injection so amounts are final)
        df["amount_vs_mean_ratio"] = df["amount"] / (df["amount_mean_total"] + 1)
        df["amount_log"] = np.log1p(df["amount"])
        df["amount_rounded"] = (df["amount"] % 1000 == 0).astype(int)
        
        # Risk scores
        df["velocity_score"] = (
            df["tx_count_24h"] * df["amount_sum_24h"] / 
            (df["amount_mean_total"] + 1)
        )
        
        # Merchant risk scores
        merchant_risk_map = {}
        for category in df["merchant_category"].unique():
            if category in {"Electronics", "Fashion", "Entertainment"}:
                risk = np.random.beta(6, 4)
            elif category in {"Grocery", "Fuel", "Medical"}:
                risk = np.random.beta(2, 8)
            else:
                risk = np.random.beta(3, 6)
            merchant_risk_map[category] = risk
        
        df["merchant_risk_score"] = df["merchant_category"].map(merchant_risk_map)
        
        # Composite risk score
        df["composite_risk"] = (
            df["merchant_risk_score"] * 0.3 +
            df["amount_vs_mean_ratio"].clip(0, 10) / 10 * 0.3 +
            df["velocity_score"].clip(0, 100) / 100 * 0.2 +
            (df["tx_count_24h"] > 5).astype(int) * 0.2
        )
        
        return df
    
    def save_dataset(self, df: pd.DataFrame, output_path: str | Path = "nibss_fraud_dataset.csv") -> Path:
        """Save dataset to CSV with integrity hash."""
        output_path = Path(output_path)
        df.to_csv(output_path, index=False)
        
        # Calculate and display hash
        sha256 = hashlib.sha256(output_path.read_bytes()).hexdigest()
        print(f"✓ Dataset saved to: {output_path}")
        print(f"  SHA-256: {sha256}")
        
        return output_path


# ──────────────────────────────────────────────────────────────────────────────
# CLI Interface
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate NIBSS 2023-calibrated synthetic Nigerian fraud dataset"
    )
    parser.add_argument(
        "-n", "--rows", type=int, default=600_000,
        help="Number of transactions to generate (default: 600,000)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "-o", "--output", type=str, default="nibss_fraud_dataset.csv",
        help="Output CSV filename (default: nibss_fraud_dataset.csv)"
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Display detailed dataset statistics after generation"
    )
    
    args = parser.parse_args()
    
    # Generate dataset
    generator = NIBSSFraudDatasetGenerator(
        dataset_size=args.rows,
        seed=args.seed
    )
    
    df = generator.generate_dataset()
    output_path = generator.save_dataset(df, args.output)
    
    # Display summary statistics
    print(f"\n{'='*60}")
    print("DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"Total transactions: {len(df):,}")
    print(f"Fraudulent transactions: {df['is_fraud'].sum():,}")
    print(f"Fraud rate: {df['is_fraud'].mean():.4%}")
    print(f"Total features: {len(df.columns)}")
    
    if args.stats:
        print(f"\n{'Channel Distribution:':<25}")
        channel_dist = df['channel'].value_counts(normalize=True).sort_index()
        for channel, pct in channel_dist.items():
            print(f"  {channel:<15} {pct:>8.2%}")
        
        print(f"\n{'Monthly Distribution:':<25}")
        monthly_dist = df['month'].value_counts(normalize=True).sort_index()
        for month, pct in monthly_dist.items():
            print(f"  Month {month:>2d}       {pct:>8.2%}")
        
        print(f"\n{'Fraud by Channel:':<25}")
        fraud_by_channel = df[df['is_fraud'] == 1]['channel'].value_counts(normalize=True).sort_index()
        for channel, pct in fraud_by_channel.items():
            print(f"  {channel:<15} {pct:>8.2%}")
        
        print(f"\n{'='*60}")
        print("NIBSS COMPLIANCE VERIFICATION")
        print(f"{'='*60}")
        
        # Monthly fraud distribution verification
        print(f"\n{'Monthly Fraud Distribution Verification:'}")
        fraud_monthly = df[df['is_fraud'] == 1]['month'].value_counts().sort_index()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        target_monthly_pct = [9.66, 9.93, 9.23, 8.61, 12.25, 6.53, 9.05, 8.58, 7.85, 7.17, 6.66, 4.49]
        
        print(f"{'Month':<5} {'Actual':<8} {'Target':<8} {'Actual%':<8} {'Target%':<8} {'Match':<8}")
        print("-" * 50)
        for i, month in enumerate(range(1, 13)):
            actual_count = fraud_monthly.get(month, 0)
            actual_pct = (actual_count / df['is_fraud'].sum() * 100) if df['is_fraud'].sum() > 0 else 0
            target_pct = target_monthly_pct[i]
            match = "✓" if abs(actual_pct - target_pct) < 1.0 else "✗"
            print(f"{month_names[i]:<5} {actual_count:<8} {'-':<8} {actual_pct:<8.2f} {target_pct:<8.2f} {match:<8}")
        
        # Channel fraud distribution verification
        print(f"\n{'Channel Fraud Distribution Verification:'}")
        fraud_channel = df[df['is_fraud'] == 1]['channel'].value_counts()
        target_channel_pct = {"Mobile": 49.75, "Web": 22.91, "POS": 18.38, "IB": 5.63, "ECOM": 2.56, "ATM": 0.76}
        
        print(f"{'Channel':<8} {'Actual':<8} {'Actual%':<8} {'Target%':<8} {'Match':<8}")
        print("-" * 45)
        for channel in df['channel'].unique():
            actual_count = fraud_channel.get(channel, 0)
            actual_pct = (actual_count / df['is_fraud'].sum() * 100) if df['is_fraud'].sum() > 0 else 0
            target_pct = target_channel_pct.get(channel, 0)
            match = "✓" if abs(actual_pct - target_pct) < 2.0 else "✗"
            print(f"{channel:<8} {actual_count:<8} {actual_pct:<8.2f} {target_pct:<8.2f} {match:<8}")


if __name__ == "__main__":
    main()