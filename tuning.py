import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from datetime import datetime
import os

sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
os.makedirs('visualizations', exist_ok=True)

df = pd.read_csv('data.csv')
stress_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
df['Stress_Level_Numeric'] = df['Stress_Level'].map(stress_mapping)

features = [
    'Technology_Usage_Hours',
    'Social_Media_Usage_Hours',
    'Gaming_Hours',
    'Screen_Time_Hours',
    'Sleep_Hours',
    'Physical_Activity_Hours'
]

X = df[features]
y = df['Stress_Level_Numeric']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ))
])
model_pipeline.fit(X_train, y_train)

y_pred = model_pipeline.predict(X_test)
y_pred_proba = model_pipeline.predict_proba(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High'], output_dict=True)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Low', 'Medium', 'High'],
            yticklabels=['Low', 'Medium', 'High'],
            cbar_kws={'label': 'Count'},
            annot_kws={'size': 14, 'weight': 'bold'})
plt.tight_layout()
plt.savefig('visualizations/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

feature_importance = pd.DataFrame({
    'Feature': [f.replace('_', ' ') for f in features],
    'Importance': model_pipeline.named_steps['classifier'].feature_importances_
}).sort_values('Importance', ascending=True)

plt.figure(figsize=(12, 8))
bars = plt.barh(feature_importance['Feature'], feature_importance['Importance'],
                color='steelblue', edgecolor='navy', linewidth=2)
for bar, val in zip(bars, feature_importance['Importance']):
    plt.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center', fontweight='bold', fontsize=11)
plt.tight_layout()
plt.savefig('visualizations/feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

corr_data = df[features].copy()
corr_data['Stress_Level'] = df['Stress_Level_Numeric']
correlation_matrix = corr_data.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0, square=True,
            linewidths=2, cbar_kws={'label': 'Correlation'}, annot_kws={'size': 10, 'weight': 'bold'},
            vmin=-1, vmax=1)
plt.tight_layout()
plt.savefig('visualizations/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

correlations = []
for feature in features:
    corr = df[feature].corr(df['Stress_Level_Numeric'])
    correlations.append({'Feature': feature.replace('_', ' '), 'Correlation': corr})
corr_df = pd.DataFrame(correlations).sort_values('Correlation', key=abs, ascending=True)

plt.figure(figsize=(12, 8))
colors_bars = ['#e74c3c' if x > 0 else '#2ecc71' for x in corr_df['Correlation']]
bars = plt.barh(corr_df['Feature'], corr_df['Correlation'], color=colors_bars, edgecolor='black', linewidth=2)
for bar, val in zip(bars, corr_df['Correlation']):
    x_pos = val + (0.02 if val > 0 else -0.02)
    ha = 'left' if val > 0 else 'right'
    plt.text(x_pos, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center', ha=ha, fontweight='bold', fontsize=11)
plt.tight_layout()
plt.savefig('visualizations/correlation_bars.png', dpi=300, bbox_inches='tight')
plt.close()

stress_counts = df['Stress_Level'].value_counts()
colors_map = {'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c'}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
ax1.pie(stress_counts.values, labels=stress_counts.index, autopct='%1.1f%%',
        colors=[colors_map[level] for level in stress_counts.index],
        startangle=90, explode=[0.05, 0.05, 0.05])
bars = ax2.bar(stress_counts.index, stress_counts.values, color=[colors_map[level] for level in stress_counts.index],
               edgecolor='black', linewidth=2)
plt.tight_layout()
plt.savefig('visualizations/stress_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(2, 3, figsize=(20, 14))
axes = axes.ravel()
colors = {0: '#2ecc71', 1: '#f39c12', 2: '#e74c3c'}
labels_map = {0: 'Low', 1: 'Medium', 2: 'High'}
for idx, feature in enumerate(features):
    ax = axes[idx]
    for stress_val in [0, 1, 2]:
        mask = df['Stress_Level_Numeric'] == stress_val
        ax.scatter(df[feature][mask], df['Stress_Level_Numeric'][mask], alpha=0.4, s=50,
                   c=colors[stress_val], label=labels_map[stress_val], edgecolors='black', linewidth=0.5)
    z = np.polyfit(df[feature], df['Stress_Level_Numeric'], 2)
    p = np.poly1d(z)
    x_trend = np.linspace(df[feature].min(), df[feature].max(), 100)
    ax.plot(x_trend, p(x_trend), "k--", linewidth=2, alpha=0.7)
    corr = df[feature].corr(df['Stress_Level_Numeric'])
    ax.text(0.05, 0.95, f'Corr: {corr:.3f}', transform=ax.transAxes, fontsize=11, fontweight='bold',
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
plt.tight_layout()
plt.savefig('visualizations/feature_vs_stress.png', dpi=300, bbox_inches='tight')
plt.close()

classes = ['Low', 'Medium', 'High']
metrics = ['precision', 'recall', 'f1-score']
data = np.array([[report[cls][metric] for metric in metrics] for cls in classes])
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
ax.set_xticks(np.arange(len(metrics)))
ax.set_yticks(np.arange(len(classes)))
ax.set_xticklabels(metrics)
ax.set_yticklabels(classes)
for i in range(len(classes)):
    for j in range(len(metrics)):
        ax.text(j, i, f'{data[i, j]:.3f}', ha="center", va="center", color="black", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('visualizations/classification_report.png', dpi=300, bbox_inches='tight')
plt.close()

joblib.dump(model_pipeline, 'stress_model.pkl')

config = {
    "features": features,
    "target": "Stress_Level",
    "stress_mapping": stress_mapping,
    "reverse_mapping": {v: k for k, v in stress_mapping.items()},
    "accuracy": float(accuracy),
    "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "correlations": {row['Feature']: float(row['Correlation']) for _, row in corr_df.iterrows()},
    "feature_importance": {features[i]: float(model_pipeline.named_steps['classifier'].feature_importances_[i]) for i in range(len(features))}
}

with open('config.json', 'w') as f:
    json.dump(config, f, indent=2)
with open('model_statistics.json', 'w') as f:
    json.dump(config, f, indent=2)
