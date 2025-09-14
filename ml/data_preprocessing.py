import pandas as pd
import numpy as np

# Load the dataset
csv_path = 'D:\\User\\Documents\\water borne disease\\ml\\northeast_india_cgwb_water_quality.csv'
df = pd.read_csv(csv_path)

# Verify expected columns
expected_columns = ['state', 'district', 'season', 'temperature', 'pH', 'conductivity', 'TDS', 
                   'turbidity', 'alkalinity', 'hardness', 'chloride', 'fluoride', 'nitrate', 
                   'sulphate', 'iron', 'arsenic', 'BOD', 'dissolved_oxygen', 'coliform_total', 
                   'coliform_fecal']
missing_cols = [col for col in expected_columns if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns in dataset: {missing_cols}. Available columns: {list(df.columns)}")

# Stricter outbreak risk calculation (require multiple critical exceedances)
def calculate_outbreak_risk(row):
    """
    Calculate binary outbreak risk (1 = high risk, 0 = low risk).
    High risk if at least 3 critical thresholds are exceeded.
    """
    critical_exceedances = 0
    thresholds = [
        (row['coliform_total'] > 10, 1),  # Stricter: >10/100 mL
        (row['coliform_fecal'] > 0, 1),  # Any fecal coliform is critical
        (row['turbidity'] > 5, 1),       # Stricter: >5 NTU
        (row['nitrate'] > 20, 1),        # Stricter: >20 mg/L
        (row['dissolved_oxygen'] < 4, 1),# Stricter: <4 mg/L
        (row['BOD'] > 10, 1),           # Stricter: >10 mg/L
        (row['pH'] < 6.0 or row['pH'] > 9.0, 1),
        (row['temperature'] > 30, 1),    # Stricter: >30°C
        (row['iron'] > 2.0, 1),         # Stricter: >2 mg/L
        (row['arsenic'] > 0.05, 1),     # Stricter: >0.05 mg/L
        (row['fluoride'] > 4.0, 1),
        (row['sulphate'] > 400, 1),     # Stricter: >400 mg/L
        (row['chloride'] > 400, 1),     # Stricter: >400 mg/L
        (row['TDS'] > 1500, 1)          # Stricter: >1500 mg/L
    ]
    critical_exceedances = sum(weight for condition, weight in thresholds if condition)
    return 1 if critical_exceedances >= 3 else 0

# Continuous risk score
def calculate_risk_score(row):
    """
    Calculate a continuous risk score (0-1) based on weighted thresholds.
    """
    thresholds = [
        (row['coliform_total'] > 10, 0.3),  # High weight for coliform
        (row['coliform_fecal'] > 0, 0.3),
        (row['turbidity'] > 5, 0.1),
        (row['nitrate'] > 20, 0.1),
        (row['dissolved_oxygen'] < 4, 0.05),
        (row['BOD'] > 10, 0.05),
        (row['pH'] < 6.0 or row['pH'] > 9.0, 0.05),
        (row['temperature'] > 30, 0.05),
        (row['iron'] > 2.0, 0.03),
        (row['arsenic'] > 0.05, 0.03),
        (row['fluoride'] > 4.0, 0.03),
        (row['sulphate'] > 400, 0.02),
        (row['chloride'] > 400, 0.02),
        (row['TDS'] > 1500, 0.02)
    ]
    score = sum(weight for condition, weight in thresholds if condition)
    return min(score, 1.0)

# Apply functions
df['outbreak_risk'] = df.apply(calculate_outbreak_risk, axis=1)
df['risk_score'] = df.apply(calculate_risk_score, axis=1)

# Handle missing values
numerical_cols = df.select_dtypes(include=[np.number]).columns
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

# Save updated DataFrame
updated_csv_path = 'D:\\User\\Documents\\water borne disease\\ml\\updated_dataset_with_risk.csv'
df.to_csv(updated_csv_path, index=False)

# Check class distribution
print(f"Updated dataset saved to: {updated_csv_path}")
print("\nOutbreak Risk Distribution:")
print(df['outbreak_risk'].value_counts(normalize=True))
print("\nSample of updated DataFrame:")
print(df.head())