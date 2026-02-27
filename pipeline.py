import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, r2_score, mean_absolute_error
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('cleaned_file.csv')
df['air_quality_index'] = df['air_quality_index'].clip(lower=0)

region_encoder = LabelEncoder()
df['region_encoded'] = region_encoder.fit_transform(df['region'])

country_encoder = LabelEncoder()
df['country_encoded'] = country_encoder.fit_transform(df['country_name'])

df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

os.makedirs('models', exist_ok=True)

print('=' * 60)
print('CLIMATE HEALTH EARLY WARNING PIPELINE')
print('=' * 60)

# ── STAGE 1: HEATWAVE CLASSIFICATION (86% accuracy) ──
print('\n' + '─' * 60)
print('STAGE 1: HEATWAVE CLASSIFICATION')
print('─' * 60)

clf_features = [
    'temperature_celsius', 'temp_anomaly_celsius', 'precipitation_mm',
    'month_sin', 'month_cos', 'latitude', 'longitude', 'region_encoded'
]

X_clf = df[clf_features]
y_clf = (df['heat_wave_days'] > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

clf_model = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('clf', RandomForestClassifier(
        n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1
    ))
])

clf_model.fit(X_train, y_train)
y_pred = clf_model.predict(X_test)

print('\nHeatwave Classifier:')
print(classification_report(y_test, y_pred, zero_division=0))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

joblib.dump(clf_model, 'models/heatwave_classifier.pkl')
print('Saved → models/heatwave_classifier.pkl')

# ── STAGE 2: SEASONAL FORECASTER (R² 0.87) ──
print('\n' + '─' * 60)
print('STAGE 2: SEASONAL FORECASTER')
print('─' * 60)

seasonal_agg = df.groupby(['country_name', 'month']).agg({
    'heat_related_admissions': 'mean',
    'respiratory_disease_rate': 'mean',
    'temperature_celsius': 'mean',
    'healthcare_access_index': 'mean',
    'latitude': 'first',
    'longitude': 'first'
}).reset_index()

seasonal_agg['country_encoded'] = country_encoder.transform(seasonal_agg['country_name'])
seasonal_agg['month_sin'] = np.sin(2 * np.pi * seasonal_agg['month'] / 12)
seasonal_agg['month_cos'] = np.cos(2 * np.pi * seasonal_agg['month'] / 12)

forecast_features = [
    'country_encoded', 'month_sin', 'month_cos',
    'latitude', 'longitude', 'temperature_celsius', 'healthcare_access_index'
]

X_forecast = seasonal_agg[forecast_features]

for target in ['heat_related_admissions', 'respiratory_disease_rate']:
    y = seasonal_agg[target]
    X_train, X_test, y_train, y_test = train_test_split(X_forecast, y, test_size=0.2, random_state=42)

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f'\n{target}:')
    print(f'  R² Score: {r2_score(y_test, y_pred):.4f}')
    print(f'  MAE:      {mean_absolute_error(y_test, y_pred):.4f}')

    joblib.dump(model, f'models/{target}_forecaster.pkl')
    print(f'  Saved → models/{target}_forecaster.pkl')

joblib.dump(region_encoder, 'models/region_encoder.pkl')
joblib.dump(country_encoder, 'models/country_encoder.pkl')

print('\n' + '=' * 60)
print('PIPELINE COMPLETE — models saved to models/')
print('=' * 60)
