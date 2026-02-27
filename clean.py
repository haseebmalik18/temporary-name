import pandas as pd

# load your dataset
df = pd.read_csv("global_climate_health_impact_tracker_2015_2025.csv")

# drop columns
df = df.drop(columns=[
    'income_level',
    'pm25_ugm3',
    'week',
    'mental_health_index','food_security_index'
])

# save cleaned file
df.to_csv("cleaned_file.csv", index=False)

print("✅ Columns dropped and new file saved")