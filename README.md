"# AI_KERALA_TRAVEL_APP" 
# 🌴 God's Own Country AI Travel Planner

An offline, hyper-localized **Machine Learning application** that acts as an intelligent travel guide for Kerala. The system bypasses external cloud APIs (like Gemini or OpenAI) by training a custom predictive classifier on an independent dataset of 100 landmark locations across Kerala. 

By analyzing user-defined boundaries for budget, trip duration, and specific travel vibes, the engine predicts and recommends the ideal vacation destination.

---

## 🚀 Key Features
* **100% Standalone Brain:** Completely self-trained model that operates fully offline with zero external platform dependencies.
* **Hyper-Localized Dataset:** Uses a custom independent dataset (`kerala_spots.csv`) packed with 100 distinct destination profiles spanning Beaches, Backwaters, Hill Stations, Wildlife, Historical, Pilgrimage, and Adventure vibes.
* **Dynamic Range Mapping:** Streamlit UI sliders automatically read data boundaries from your dataset, avoiding hardcoded front-end limits.
* **Intelligent Encoding:** Integrates Scikit-learn feature preprocessing (`LabelEncoder`) to cleanly translate textual travel vibes into numerical array shapes acceptable by the model pipeline.

---

## 🛠️ Architecture & Tech Stack

* **Frontend UI:** Streamlit
* **Data Processing & Engineering:** Pandas, Numpy
* **Machine Learning Framework:** Scikit-Learn (`DecisionTreeClassifier`, `LabelEncoder`)
* **Model Serialization:** Joblib
* **Data Interchange:** JSON

```text
Project_Structure/
│
├── data/
│   └── kerala_spots.csv      # Independent dataset (100 rows)
├── models/
│   ├── kerala_model.joblib   # Trained ML Classifier Brain
│   ├── vibe_encoder.joblib   # Serialized string categorical encoder
│   └── config.json           # Extracted UI slider threshold constraints
├── requirements.txt          # Library dependencies
├── train.py                  # ML Pipeline training script
└── app.py                    # Production Streamlit UI
