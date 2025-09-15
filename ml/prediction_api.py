import pandas as pd
import joblib

# Load the saved model
model_path = 'D:\\User\\Documents\\water borne disease\\ml\\water_quality_model.joblib'
model = joblib.load(model_path)
print(f"Model loaded from: {model_path}")

# Load the dataset to get valid categorical values
df = pd.read_csv('D:\\User\\Documents\\water borne disease\\ml\\updated_dataset_with_risk.csv')

# Define features
features = ['state', 'district', 'season', 'temperature', 'pH', 'conductivity', 'TDS', 
            'turbidity', 'alkalinity', 'hardness', 'chloride', 'fluoride', 'nitrate', 
            'sulphate', 'iron', 'arsenic', 'BOD', 'dissolved_oxygen', 'coliform_fecal']

# User input with suggestions
print("\nEnter new data for prediction:")
print(f"Valid states: {list(df['state'].unique())}")
print(f"Valid districts: {list(df['district'].unique())}")
print(f"Valid seasons: {list(df['season'].unique())}")
new_data = {}
for col in features:
    if col in ['state', 'district', 'season']:
        while True:
            value = input(f"Enter {col} (text, choose from {list(df[col].unique())}): ")
            if value in df[col].unique():
                break
            print(f"Invalid {col}. Choose from {list(df[col].unique())}")
    else:
        while True:
            try:
                value = float(input(f"Enter {col} (number, e.g., 7.5 for pH): "))
                break
            except ValueError:
                print(f"Please enter a valid number for {col}")
    new_data[col] = [value]

new_data_df = pd.DataFrame(new_data)
prediction = model.predict(new_data_df)
print(f"\nPredicted Outbreak Risk: {prediction[0]} (1 = High Risk, 0 = Low Risk)")