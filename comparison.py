import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from typed_tpsa_prediction import create_plot
import re

# Define paths
save_path = 'tpsa_saved_data'
comparison_file = os.path.join(save_path, 'tpsa_comparison_results.csv')
plot_output = os.path.join(save_path, 'tpsa_comparison_plot.png')
metrics_output = os.path.join(save_path, 'tpsa_comparison_metrics.csv')

# Load dataset
if not os.path.exists(comparison_file):
    print(f"File not found: {comparison_file}")
    exit()

comparison_df = pd.read_csv(comparison_file)

# Ensure required columns exist
required_cols = {'Known_TPSA', 'Predicted_TPSA'}
if not required_cols.issubset(comparison_df.columns):
    print(f"Error: Missing required columns in {comparison_file}. Expected {required_cols}")
    exit()

# Function to extract the first floating-point number from a string
def extract_numeric(value):
    if isinstance(value, str):
        match = re.search(r"[-+]?\d*\.\d+|\d+", value)  # Extract first number
        return float(match.group()) if match else np.nan
    return value  # If already a number, return as-is

# Apply extraction and conversion
comparison_df['Known_TPSA'] = comparison_df['Known_TPSA'].apply(extract_numeric)
comparison_df['Predicted_TPSA'] = comparison_df['Predicted_TPSA'].apply(extract_numeric)

# Drop rows where conversion failed (NaN values)
comparison_df = comparison_df.dropna(subset=['Known_TPSA', 'Predicted_TPSA'])

# Rename Known_TPSA to Calculated_TPSA
comparison_df.rename(columns={'Known_TPSA': 'Calculated_TPSA'}, inplace=True)

# Remove extreme outliers
outlier_threshold = 1000  # Setting an arbitrary threshold for sanity check
comparison_df = comparison_df[comparison_df['Predicted_TPSA'] < outlier_threshold]

# Extract numeric arrays
calculated_tpsa = comparison_df['Calculated_TPSA'].values
predicted_tpsa = comparison_df['Predicted_TPSA'].values

# Calculate RMSE, MAE, and median error
errors = np.abs(predicted_tpsa - calculated_tpsa)
rmse = np.sqrt(mean_squared_error(calculated_tpsa, predicted_tpsa))
median_error = np.median(predicted_tpsa - calculated_tpsa)
mae = np.mean(errors)

# Save metrics
metrics_df = pd.DataFrame([{
    'RMSE': rmse,
    'MAE': mae,
    'Median Error': median_error
}])
metrics_df.to_csv(metrics_output, index=False)
print(f"Metrics saved to {metrics_output}")

# Generate and save the plot
create_plot(
    x='Calculated_TPSA',
    y='Predicted_TPSA',
    marker='o',
    file_name='tpsa_comparison_plot.png',
    predictions=comparison_df,
    path='results',
    model_name="DrugAssist7B"
)

print(f"Plot saved to {plot_output}")

# Print metrics to console
print(metrics_df.to_string(index=False))