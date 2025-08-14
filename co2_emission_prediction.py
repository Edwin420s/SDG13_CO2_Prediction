# CO2 Emissions Prediction for Climate Action (SDG 13)
# Machine Learning Model using Linear Regression
# Dataset: World Bank CO2 Emissions Data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ======================
# 1. Data Loading
# ======================

# Load dataset from World Bank (alternative: Kaggle/UN datasets)
try:
    # Try loading from online source
    url = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"
    data = pd.read_csv(url)
except:
    # Fallback to local file
    print("Online loading failed. Using local dataset.")
    data = pd.read_csv("owid-co2-data.csv")

# ======================
# 2. Data Preprocessing
# ======================

# Select relevant features and target
features = ['year', 'gdp', 'population', 'energy_use', 'cement_co2', 'coal_co2']
target = 'co2'

# Filter for countries with sufficient data
df = data[['country', 'year', target] + features].dropna()

# Let user select a country
print(f"Available countries: {df['country'].unique()[:10]}... (Total: {len(df['country'].unique())})")
selected_country = "Kenya"  # Change this to any country

country_df = df[df['country'] == selected_country].copy()

# Feature engineering: Add GDP per capita
country_df['gdp_per_capita'] = country_df['gdp'] / country_df['population']

# Split into features (X) and target (y)
X = country_df[features]
y = country_df[target]

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ======================
# 3. Model Training
# ======================

# Initialize models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

# Train and evaluate each model
results = {}
for name, model in models.items():
    # Training
    model.fit(X_train_scaled, y_train)
    
    # Prediction
    y_pred = model.predict(X_test_scaled)
    
    # Evaluation
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {
        'model': model,
        'mae': mae,
        'r2': r2,
        'predictions': y_pred
    }

# ======================
# 4. Results & Visualization
# ======================

# Print evaluation metrics
print("\nModel Performance Comparison:")
print("="*50)
for name, metrics in results.items():
    print(f"{name}:")
    print(f"- MAE: {metrics['mae']:.2f}")
    print(f"- R² Score: {metrics['r2']:.2f}")
    print("-"*30)

# Plot actual vs predicted for best model
best_model_name = max(results, key=lambda x: results[x]['r2'])
best_model = results[best_model_name]

plt.figure(figsize=(12, 6))
sns.scatterplot(x=y_test, y=best_model['predictions'])
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.xlabel('Actual CO2 Emissions (kt)')
plt.ylabel('Predicted CO2 Emissions (kt)')
plt.title(f'Actual vs Predicted CO2 Emissions in {selected_country}\n({best_model_name}, R²={best_model["r2"]:.2f})')
plt.grid(True)
plt.savefig('co2_prediction_results.png')  # Save for report
plt.show()

# Feature importance (for tree-based models)
if hasattr(best_model['model'], 'feature_importances_'):
    importance = best_model['model'].feature_importances_
    feat_importance = pd.DataFrame({
        'Feature': features,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 5))
    sns.barplot(x='Importance', y='Feature', data=feat_importance)
    plt.title('Feature Importance for CO2 Emissions Prediction')
    plt.tight_layout()
    plt.show()

# ======================
# 5. Ethical Considerations
# ======================

print("\nEthical Considerations:")
print("="*50)
print("1. Data Bias: Developing countries may have less complete data")
print("2. Fairness: Model should be tested across different regions")
print("3. Sustainability Impact: Should be used to inform equitable climate policies")
print("4. Transparency: Model decisions should be explainable to policymakers")

# ======================
# 6. Future Improvements
# ======================

print("\nFuture Enhancements:")
print("="*50)
print("- Incorporate real-time data via APIs")
print("- Build a web dashboard for policymakers")
print("- Include more environmental factors (deforestation, etc.)")
print("- Compare across multiple countries")

# Save model for deployment
import joblib
joblib.dump(best_model['model'], 'co2_emissions_model.pkl')
print("\nBest model saved as 'co2_emissions_model.pkl'")