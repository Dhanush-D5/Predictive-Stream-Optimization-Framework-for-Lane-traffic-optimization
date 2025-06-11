import pandas as pd
import numpy as np
from flask import Flask, request, render_template
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

app = Flask(__name__)

# --- Load dataset ---
df = pd.read_csv("simulated_traffic_data_v2.csv")

# --- Preprocess dataset ---
def preprocess_data(df):
    df = shuffle(df, random_state=42).reset_index(drop=True)

    label_encoders = {}
    categorical_cols = ['Weather', 'Signal Status', 'Priority Flow', 'Incident', 'Lane ID']
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    target_encoder = LabelEncoder()
    df['Congestion Level'] = target_encoder.fit_transform(df['Congestion Level'])

    feature_cols = ['Lane ID', 'Vehicle Count', 'Avg Speed', 'Load (%)', 'Incident', 'Weather', 'Signal Status', 'Priority Flow']
    X = df[feature_cols]
    y = df['Congestion Level']

    return X, y, feature_cols, categorical_cols, label_encoders, target_encoder

# Preprocess and split the data
X, y, feature_cols, categorical_cols, label_encoders, target_encoder = preprocess_data(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cv_scores = cross_val_score(model, X, y, cv=5)
cv_accuracy = round(cv_scores.mean() * 100, 2)

# --- Flask web app ---
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    dropdown_values = {
        'Incident': label_encoders['Incident'].classes_,
        'Weather': label_encoders['Weather'].classes_,
        'Signal Status': label_encoders['Signal Status'].classes_,
        'Priority Flow': label_encoders['Priority Flow'].classes_,
        'Lane ID': label_encoders['Lane ID'].classes_
    }

    if request.method == 'POST':
        try:
            vehicle_count = int(request.form['vehicle_count'])
            avg_speed = float(request.form['avg_speed'])
            load = float(request.form['load'])

            if vehicle_count < 0 or avg_speed < 0 or load < 0:
                raise ValueError("Negative values are not allowed.")

            user_input = {
                'Lane ID': request.form['lane_id'],
                'Vehicle Count': vehicle_count,
                'Avg Speed': avg_speed,
                'Load (%)': load,
                'Incident': request.form['incident'],
                'Weather': request.form['weather'],
                'Signal Status': request.form['signal_status'],
                'Priority Flow': request.form['priority_flow']
            }

            input_df = pd.DataFrame([user_input])
            for col in categorical_cols:
                input_df[col] = label_encoders[col].transform(input_df[col])

            prediction_code = model.predict(input_df[feature_cols])[0]
            prediction = target_encoder.inverse_transform([prediction_code])[0]

        except ValueError as e:
            prediction = f"Invalid input: {str(e)}"

    return render_template('index.html', prediction=prediction, dropdown_values=dropdown_values, accuracy=cv_accuracy)

if __name__ == '__main__':
    app.run(debug=True)
