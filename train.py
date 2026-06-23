import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# 1. Load your custom Kerala dataset
df = pd.read_csv('data\data\kerala_spots.csv')

# 2. Encode the text 'vibe' column into numbers
vibe_encoder = LabelEncoder()
df['vibe_encoded'] = vibe_encoder.fit_transform(df['vibe'])

# 3. Define Features (X) and Target Destination (y)
X = df[['budget', 'days', 'vibe_encoded']]
y = df['destination']

# 4. Train a simple Decision Tree Classifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# 5. Save the model, encoder, and data mappings
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/kerala_model.joblib')
joblib.dump(vibe_encoder, 'models/vibe_encoder.joblib')

# Save mapping configurations for the Streamlit UI dropdown
config = {
    'vibes': vibe_encoder.classes_.tolist(),
    'min_budget': int(df['budget'].min()),
    'max_budget': int(df['budget'].max()),
    'max_days': int(df['days'].max())
}
with open('models/config.json', 'w') as f:
    json.dump(config, f)

print("Hurray! Your custom Kerala Travel AI has been trained and saved offline.")