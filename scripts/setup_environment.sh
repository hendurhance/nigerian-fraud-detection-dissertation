#!/bin/bash

# NIBSS Fraud Detection System - Environment Setup Script
# This script sets up the development environment for the project

set -e  # Exit on any error

echo "🚀 Setting up NIBSS Fraud Detection System environment..."

# Check Python version
python_version=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.8"

if (( $(echo "$python_version >= $required_version" | bc -l) )); then
    echo "✅ Python $python_version detected (>= $required_version required)"
else
    echo "❌ Python $required_version or higher is required. Found: $python_version"
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
if [ -d "venv" ]; then
    echo "⚠️  Virtual environment already exists. Removing..."
    rm -rf venv
fi

python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📋 Installing Python dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p data/{raw,processed,external}
mkdir -p models
mkdir -p results
mkdir -p logs
mkdir -p tests

# Generate sample dataset
echo "🎲 Generating sample dataset (10K transactions)..."
python src/nibss_fraud_dataset_generator.py -n 10000 --output data/processed/sample_dataset.csv --seed 42

# Run basic tests
echo "🧪 Running basic tests..."
python -c "
import pandas as pd
import numpy as np
from src.nibss_fraud_dataset_generator import NIBSSFraudDatasetGenerator

print('Testing dataset generator...')
generator = NIBSSFraudDatasetGenerator(dataset_size=1000, seed=42)
df = generator.generate_dataset()
print(f'Generated {len(df)} transactions with {df[\"is_fraud\"].sum()} fraud cases')
print(f'Fraud rate: {df[\"is_fraud\"].mean():.4%}')
print('✅ Generator test passed!')
"

# Check if Jupyter is available
if command -v jupyter &> /dev/null; then
    echo "📊 Jupyter is available. You can run: jupyter lab"
else
    echo "📝 To use notebooks, install Jupyter: pip install jupyterlab"
fi

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Generate full dataset: python src/nibss_fraud_dataset_generator.py -n 600000 --output data/processed/nibss_fraud_dataset.csv"
echo "3. Launch Jupyter Lab: jupyter lab"
echo "4. Open and run notebooks in the notebooks/ directory"
echo ""
echo "For more information, see README.md"