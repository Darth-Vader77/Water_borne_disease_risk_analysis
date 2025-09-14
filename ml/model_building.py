import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Load the updated dataset
df = pd.read_csv('D:\\User\\Documents\\water borne disease\\ml\\updated_dataset_with_risk.csv')

# Define features and target
features = ['state', 'district', 'season', 'temperature', 'pH', 'conductivity', 'TDS', 
            'turbidity', 'alkalinity', 'hardness', 'chloride', 'fluoride', 'nitrate', 
            'sulphate', 'iron', 'arsenic', 'BOD', 'dissolved_oxygen', 'coliform_fecal']
target = 'outbreak_risk'

# Verify columns
missing_cols = [col for col in features + [target] if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns in dataset: {missing_cols}. Available columns: {list(df.columns)}")

X = df[features]
y = df[target]

# Define categorical and numerical columns
categorical_cols = ['state', 'district', 'season']
numerical_cols = ['temperature', 'pH', 'conductivity', 'TDS', 'turbidity', 'alkalinity', 
                 'hardness', 'chloride', 'fluoride', 'nitrate', 'sulphate', 'iron', 
                 'arsenic', 'BOD', 'dissolved_oxygen', 'coliform_fecal']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ])

# Create the model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Cross-Validation Accuracy Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (± {cv_scores.std():.4f})")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = model.named_steps['classifier'].feature_importances_
feature_names = (numerical_cols + 
                 model.named_steps['preprocessor']
                 .named_transformers_['cat']
                 .get_feature_names_out(categorical_cols).tolist())

# Print feature importance
print("\nFeature Importance:")
for name, importance in zip(feature_names, feature_importance):
    print(f"{name}: {importance:.4f}")

# User input with suggestions
print("\nEnter new data for prediction:")
print(f"Valid states: {list(df['state'].unique())}")
print(f"Valid districts: {list(df['district'].unique())}")
print(f"Valid seasons: {list(df['season'].unique())}")
new_data = {}
for col in features:
    if col in categorical_cols:
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