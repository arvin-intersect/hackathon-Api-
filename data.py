import pandas as pd
import numpy as np
from datetime import datetime
import json

np.random.seed(42)
n = 10000

age = np.random.randint(18, 66, n)
stress_probs = [0.15, 0.55, 0.30]
stress_level_numeric = np.random.choice([0, 1, 2], n, p=stress_probs)

technology_hours = np.zeros(n)
social_media_hours = np.zeros(n)
gaming_hours = np.zeros(n)
screen_time_hours = np.zeros(n)
sleep_hours = np.zeros(n)
physical_activity_hours = np.zeros(n)

for i in range(n):
    stress = stress_level_numeric[i]
    if stress == 0:
        technology_hours[i] = np.random.uniform(2, 5)
        social_media_hours[i] = np.random.uniform(0.5, 2)
        gaming_hours[i] = np.random.uniform(0, 1.5)
        sleep_hours[i] = np.random.uniform(7.5, 9)
        physical_activity_hours[i] = np.random.uniform(3, 8)
    elif stress == 1:
        technology_hours[i] = np.random.uniform(4.5, 7.5)
        social_media_hours[i] = np.random.uniform(1.8, 4)
        gaming_hours[i] = np.random.uniform(1, 3)
        sleep_hours[i] = np.random.uniform(6, 7.5)
        physical_activity_hours[i] = np.random.uniform(1, 3.5)
    else:
        technology_hours[i] = np.random.uniform(7, 12)
        social_media_hours[i] = np.random.uniform(3.5, 8)
        gaming_hours[i] = np.random.uniform(2.5, 5)
        sleep_hours[i] = np.random.uniform(4, 6.5)
        physical_activity_hours[i] = np.random.uniform(0, 1.5)

screen_time_hours = technology_hours + social_media_hours + gaming_hours
screen_time_hours = np.clip(screen_time_hours + np.random.uniform(-0.5, 0.5, n), 1, 15)

technology_hours += np.random.normal(0, 0.3, n)
social_media_hours += np.random.normal(0, 0.2, n)
gaming_hours += np.random.normal(0, 0.15, n)
sleep_hours += np.random.normal(0, 0.2, n)
physical_activity_hours += np.random.normal(0, 0.2, n)

technology_hours = np.clip(technology_hours, 1, 12)
social_media_hours = np.clip(social_media_hours, 0, 8)
gaming_hours = np.clip(gaming_hours, 0, 5)
sleep_hours = np.clip(sleep_hours, 4, 9)
physical_activity_hours = np.clip(physical_activity_hours, 0, 10)
screen_time_hours = np.clip(screen_time_hours, 1, 15)

stress_level_text = np.where(stress_level_numeric == 0, 'Low',
                     np.where(stress_level_numeric == 1, 'Medium', 'High'))

mental_health_status = np.where(
    stress_level_numeric == 0,
    np.random.choice(['Excellent', 'Good', 'Fair', 'Poor'], n, p=[0.70, 0.25, 0.04, 0.01]),
    np.where(
        stress_level_numeric == 1,
        np.random.choice(['Excellent', 'Good', 'Fair', 'Poor'], n, p=[0.15, 0.50, 0.30, 0.05]),
        np.random.choice(['Excellent', 'Good', 'Fair', 'Poor'], n, p=[0.02, 0.08, 0.35, 0.55])
    )
)

support_systems = np.where(
    stress_level_numeric == 0,
    np.random.choice(['Yes', 'No'], n, p=[0.85, 0.15]),
    np.where(
        stress_level_numeric == 1,
        np.random.choice(['Yes', 'No'], n, p=[0.50, 0.50]),
        np.random.choice(['Yes', 'No'], n, p=[0.20, 0.80])
    )
)

work_environment = np.where(
    stress_level_numeric == 0,
    np.random.choice(['Positive', 'Neutral', 'Negative'], n, p=[0.75, 0.20, 0.05]),
    np.where(
        stress_level_numeric == 1,
        np.random.choice(['Positive', 'Neutral', 'Negative'], n, p=[0.25, 0.50, 0.25]),
        np.random.choice(['Positive', 'Neutral', 'Negative'], n, p=[0.05, 0.20, 0.75])
    )
)

online_support = np.where(
    stress_level_numeric == 2,
    np.random.choice(['Yes', 'No'], n, p=[0.60, 0.40]),
    np.random.choice(['Yes', 'No'], n, p=[0.35, 0.65])
)

gender = np.random.choice(['Male', 'Female', 'Other'], n, p=[0.48, 0.48, 0.04])
user_ids = [f'USER-{str(i+1).zfill(5)}' for i in range(n)]

df = pd.DataFrame({
    'User_ID': user_ids,
    'Age': age.astype(int),
    'Gender': gender,
    'Technology_Usage_Hours': np.round(technology_hours, 2),
    'Social_Media_Usage_Hours': np.round(social_media_hours, 2),
    'Gaming_Hours': np.round(gaming_hours, 2),
    'Screen_Time_Hours': np.round(screen_time_hours, 2),
    'Mental_Health_Status': mental_health_status,
    'Stress_Level': stress_level_text,
    'Sleep_Hours': np.round(sleep_hours, 2),
    'Physical_Activity_Hours': np.round(physical_activity_hours, 2),
    'Support_Systems_Access': support_systems,
    'Work_Environment_Impact': work_environment,
    'Online_Support_Usage': online_support
})

df.to_csv('stress_prediction_dataset_90percent.csv', index=False)

metadata = {
    "dataset_name": "Stress Prediction Dataset (90% Accuracy)",
    "generation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "total_records": int(len(df)),
    "features": {
        "target": "Stress_Level",
        "predictors": [
            "Technology_Usage_Hours",
            "Social_Media_Usage_Hours",
            "Gaming_Hours",
            "Screen_Time_Hours",
            "Sleep_Hours",
            "Physical_Activity_Hours"
        ],
        "additional": [
            "Age", "Gender", "Mental_Health_Status",
            "Support_Systems_Access", "Work_Environment_Impact",
            "Online_Support_Usage"
        ]
    },
    "stress_distribution": {
        "Low": int((df['Stress_Level'] == 'Low').sum()),
        "Medium": int((df['Stress_Level'] == 'Medium').sum()),
        "High": int((df['Stress_Level'] == 'High').sum())
    }
}

with open('dataset_metadata_90percent.json', 'w') as f:
    json.dump(metadata, f, indent=2)
