import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import time
from collections import deque
import datetime # For more realistic time handling

# --- Project Configuration ---
PROJECT_NAME = "Intelligent Traffic Flow Optimizer (ITFO)"
TRAFFIC_STATES = {
    0: "Free Flow",
    1: "Moderate Flow",
    2: "Heavy Congestion",
    3: "Emergency Vehicle Approaching"
}

# Heuristic decision parameters based on predicted state
# These represent recommended green light durations in seconds for an intersection
# with three main lanes/directions (Lane A, B, C)
DECISION_PARAMETERS = {
    "Free Flow": {"lane_A_green_sec": 60, "lane_B_green_sec": 30, "lane_C_green_sec": 30},
    "Moderate Flow": {"lane_A_green_sec": 45, "lane_B_green_sec": 45, "lane_C_green_sec": 20},
    "Heavy Congestion": {"lane_A_green_sec": 30, "lane_B_green_sec": 60, "lane_C_green_sec": 15}, # Prioritize B
    "Emergency Vehicle Approaching": {"lane_A_green_sec": 5, "lane_B_green_sec": 5, "lane_C_green_sec": 90} # Clear path for C (e.g., emergency lane)
}

# --- 1. Data Acquisition & Simulation Layer (ITFO-DataSim) ---
def generate_simulated_traffic_data(num_samples=5000):
    """
    Simulates historical multi-dimensional operational signals for traffic.
    Includes features and a target 'traffic_state'.
    """
    print(f"[{PROJECT_NAME}] Generating {num_samples} simulated historical traffic data samples...")
    data = []
    for _ in range(num_samples):
        # Time of Day (0-23)
        hour = np.random.randint(0, 24)
        # Assuming peak hours are 7-9 AM and 4-7 PM
        is_peak_hour = 1 if (7 <= hour <= 9 or 16 <= hour <= 19) else 0

        # Weather (0: Clear, 1: Light Rain, 2: Heavy Rain/Fog)
        weather = np.random.randint(0, 3)

        # Vehicle counts, queue lengths, and speeds for three hypothetical lanes
        # Lane A: Often busiest (e.g., main straight road)
        vehicles_lane_A = np.random.randint(50, 300)
        queue_length_A = np.random.randint(0, vehicles_lane_A // 8)
        avg_speed_A = np.random.randint(10, 60) # km/h

        # Lane B: Moderately busy (e.g., major turn lane)
        vehicles_lane_B = np.random.randint(30, 200)
        queue_length_B = np.random.randint(0, vehicles_lane_B // 7)
        avg_speed_B = np.random.randint(10, 55)

        # Lane C: Less busy / specific purpose (e.g., minor turn, dedicated bus lane)
        vehicles_lane_C = np.random.randint(10, 100)
        queue_length_C = np.random.randint(0, vehicles_lane_C // 5)
        avg_speed_C = np.random.randint(15, 70)

        # Emergency Vehicle Flag (1% chance of presence)
        emergency_vehicle = 1 if np.random.rand() < 0.01 else 0

        # --- Simplified Logic for Generating 'traffic_state' (Target Label) ---
        # This is how the ground truth for training data is created.
        traffic_state = 0 # Default: Free Flow

        if emergency_vehicle == 1:
            traffic_state = 3 # Overrides all other states for immediate priority
        elif (vehicles_lane_A > 200 and queue_length_A > 30 and avg_speed_A < 20) or \
             (vehicles_lane_B > 150 and queue_length_B > 25 and avg_speed_B < 15):
            traffic_state = 2 # Heavy Congestion
        elif (vehicles_lane_A > 100 or vehicles_lane_B > 80 or (is_peak_hour == 1 and (vehicles_lane_A > 80 or vehicles_lane_B > 60))):
            traffic_state = 1 # Moderate Flow

        # Adjust for weather impact (example)
        if weather == 2 and traffic_state < 2: # Heavy rain/fog makes moderate congestion worse
            traffic_state = 2

        data.append([
            hour, is_peak_hour, weather,
            vehicles_lane_A, queue_length_A, avg_speed_A,
            vehicles_lane_B, queue_length_B, avg_speed_B,
            vehicles_lane_C, queue_length_C, avg_speed_C,
            emergency_vehicle,
            traffic_state # This is our target variable for ML training
        ])

    columns = [
        'hour', 'is_peak_hour', 'weather',
        'vehicles_lane_A', 'queue_length_A', 'avg_speed_A',
        'vehicles_lane_B', 'queue_length_B', 'avg_speed_B',
        'vehicles_lane_C', 'queue_length_C', 'avg_speed_C',
        'emergency_vehicle',
        'traffic_state'
    ]
    return pd.DataFrame(data, columns=columns)

# --- 2. Data Preprocessing & Feature Engineering Layer (ITFO-PreProc) ---
def prepare_features_and_labels(df):
    """
    Prepares the DataFrame into features (X) and labels (y) for ML.
    """
    print(f"[{PROJECT_NAME}] Preparing features and labels...")
    X = df.drop('traffic_state', axis=1)
    y = df['traffic_state']
    return X, y

# --- 3. Machine Learning Model (ITFO-ML Core) ---
def train_traffic_state_predictor(X_train, y_train):
    """
    Trains the RandomForestClassifier model.
    """
    print(f"[{PROJECT_NAME}] Training the Traffic State Predictor (RandomForestClassifier)...")
    # Using class_weight='balanced' to handle potential class imbalance in traffic states
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    print(f"[{PROJECT_NAME}] Model training complete.")
    return model

def predict_current_traffic_state(model, current_data_point):
    """
    Performs real-time inference using the trained ML model.
    """
    # Ensure the input data point is a DataFrame with correct columns for prediction
    if not isinstance(current_data_point, pd.DataFrame):
        current_data_point = pd.DataFrame([current_data_point])

    predicted_class_id = model.predict(current_data_point)[0]
    prediction_probabilities = model.predict_proba(current_data_point)[0]
    return predicted_class_id, prediction_probabilities

# --- 4. Decision Engine & Action Layer (ITFO-Decision) ---
def make_traffic_decisions(predicted_state_id):
    """
    Uses the predicted traffic state to determine optimal traffic light actions.
    """
    predicted_state_name = TRAFFIC_STATES.get(predicted_state_id, "Unknown State")
    print(f"\n[{PROJECT_NAME}] Predicted Traffic State: {predicted_state_name}")

    if predicted_state_name in DECISION_PARAMETERS:
        decisions = DECISION_PARAMETERS[predicted_state_name]
        print(f"[{PROJECT_NAME}] Recommended Green Light Durations (seconds):")
        for lane, duration in decisions.items():
            print(f"  - {lane.replace('_', ' ').title()}: {duration}s")
        return decisions
    else:
        print(f"[{PROJECT_NAME}] No specific decision parameters for state '{predicted_state_name}'. Using default (Moderate Flow).")
        return DECISION_PARAMETERS["Moderate Flow"]

# --- Utility: Simulate Real-time Data Feed ---
def get_current_sensor_readings():
    """
    Simulates receiving real-time data from various sensors.
    """
    now = datetime.datetime.now()
    hour = now.hour
    is_peak_hour = 1 if (7 <= hour <= 9 or 16 <= hour <= 19) else 0 # Assuming Mangaluru peak hours

    # Simulating sensor fluctuations for each reading
    weather = np.random.randint(0, 3) # Random for simulation, could be from a weather API

    vehicles_lane_A = np.random.randint(50, 250) + np.random.randint(-20, 20)
    queue_length_A = np.random.randint(0, vehicles_lane_A // 6)
    avg_speed_A = np.random.randint(10, 50) + np.random.randint(-5, 5)

    vehicles_lane_B = np.random.randint(30, 180) + np.random.randint(-15, 15)
    queue_length_B = np.random.randint(0, vehicles_lane_B // 5)
    avg_speed_B = np.random.randint(10, 45) + np.random.randint(-5, 5)

    vehicles_lane_C = np.random.randint(10, 90) + np.random.randint(-10, 10)
    queue_length_C = np.random.randint(0, vehicles_lane_C // 4)
    avg_speed_C = np.random.randint(15, 60) + np.random.randint(-5, 5)

    # Real-time Emergency Vehicle detection (very low probability)
    emergency_vehicle = 1 if np.random.rand() < 0.002 else 0

    # Ensure values are within reasonable bounds after random fluctuations
    vehicles_lane_A = max(0, vehicles_lane_A)
    queue_length_A = max(0, queue_length_A)
    avg_speed_A = max(1, avg_speed_A) # Speed should not be zero
    # ... apply for other lanes similarly

    current_readings = {
        'hour': hour,
        'is_peak_hour': is_peak_hour,
        'weather': weather,
        'vehicles_lane_A': vehicles_lane_A,
        'queue_length_A': queue_length_A,
        'avg_speed_A': avg_speed_A,
        'vehicles_lane_B': vehicles_lane_B,
        'queue_length_B': queue_length_B,
        'avg_speed_B': avg_speed_B,
        'vehicles_lane_C': vehicles_lane_C,
        'queue_length_C': queue_length_C,
        'avg_speed_C': avg_speed_C,
        'emergency_vehicle': emergency_vehicle
    }
    return current_readings

# --- 5. Feedback & Continuous Learning Loop (ITFO-Feedback - Conceptual) ---
# In a real system, this would be a separate service or scheduled job.
# We'll simulate its data collection aspect.
feedback_buffer = deque(maxlen=1000) # Store recent data for potential retraining

def collect_feedback(input_features, predicted_state, actual_performance_metrics):
    """
    Collects data for the continuous learning loop.
    'actual_performance_metrics' would come from downstream monitoring.
    """
    # For simulation, we'll just log. In reality, you'd store to a database/data lake.
    feedback_data = {
        **input_features,
        'predicted_state': predicted_state,
        'actual_performance': actual_performance_metrics # e.g., {'avg_delay': 10, 'throughput': 150}
    }
    feedback_buffer.append(feedback_data)
    # print(f"[{PROJECT_NAME}] Feedback collected. Buffer size: {len(feedback_buffer)}")

def periodic_retraining_trigger(current_model, min_feedback_samples=500):
    """
    A conceptual function to trigger retraining based on collected feedback.
    In production, this would be a scheduled task (e.g., daily at 2 AM).
    """
    if len(feedback_buffer) >= min_feedback_samples:
        print(f"\n[{PROJECT_NAME}] Initiating periodic retraining with {len(feedback_buffer)} new samples...")
        # Convert buffer to DataFrame
        new_training_data = pd.DataFrame(list(feedback_buffer))
        # Here, you'd need to map 'actual_performance' back to 'traffic_state'
        # or use a more sophisticated RL approach. For simplicity, we'll assume
        # you have a way to derive the correct 'actual_traffic_state' from performance.
        # For this example, we'll just simulate it being a part of the original data gen logic.
        # In a real system, 'actual_traffic_state' would be inferred from real observations.
        # This is where the challenge of "ground truth" in real-time systems lies.

        # For this illustration, let's assume we can get a "corrected" traffic state
        # from the feedback, or simply retrain on all historical + new data.
        # A more robust feedback loop for supervised learning would involve:
        # 1. Manually labeling some feedback data for high-confidence ground truth.
        # 2. Using rule-based systems to infer 'actual_state' from observed metrics.
        # 3. Anomaly detection to identify samples where prediction was far off.

        # For simplicity, we'll just clear the buffer and indicate retraining would happen.
        # A real implementation would append this data to the main historical dataset
        # and retrain the model.
        # For this example, let's just use a fresh dataset from generation as if it were augmented.
        global traffic_model # Access the global model
        print(f"[{PROJECT_NAME}] Simulating retraining with fresh data generation...")
        updated_historical_data = generate_simulated_traffic_data(num_samples=5000) # Re-generate for demo
        X_updated, y_updated = prepare_features_and_labels(updated_historical_data)
        X_train_updated, _, y_train_updated, _ = train_test_split(X_updated, y_updated, test_size=0.2, random_state=42, stratify=y_updated)
        traffic_model = train_traffic_state_predictor(X_train_updated, y_train_updated)
        feedback_buffer.clear()
        print(f"[{PROJECT_NAME}] Retraining complete. Feedback buffer cleared.")
    else:
        print(f"[{PROJECT_NAME}] Not enough feedback samples ({len(feedback_buffer)}/{min_feedback_samples}) for retraining.")


# --- Main Project Execution Workflow ---
if __name__ == "__main__":
    print(f"--- Starting {PROJECT_NAME} ---")

    # --- Phase 1: Initial Model Training ---
    print("\n--- Phase 1: Initial Model Training ---")
    historical_data = generate_simulated_traffic_data(num_samples=8000) # More data for initial training
    X, y = prepare_features_and_labels(historical_data)

    # Split data for training and testing the initial model performance
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    traffic_model = train_traffic_state_predictor(X_train, y_train)

    # Evaluate the initial model
    print("\n--- Initial Model Evaluation on Test Set ---")
    y_pred = traffic_model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=[TRAFFIC_STATES[i] for i in sorted(TRAFFIC_STATES.keys())]))
    print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    # --- Phase 2: Real-time Operation Simulation ---
    print("\n--- Phase 2: Simulating Real-time Operation and Decision Making ---")
    num_simulation_cycles = 15 # Number of times we simulate a real-time decision
    retraining_check_interval = 5 # Check for retraining every X cycles

    for i in range(num_simulation_cycles):
        print(f"\n===== Simulation Cycle {i+1}/{num_simulation_cycles} =====")
        print(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # ITFO-DataSim: Get current readings
        current_readings_dict = get_current_sensor_readings()
        current_readings_df = pd.DataFrame([current_readings_dict]) # Convert to DataFrame for model input
        print("\nCurrent Traffic Sensor Readings:")
        print(current_readings_df.to_string(index=False))

        # ITFO-ML Core: Predict traffic state
        predicted_state_id, prediction_probabilities = predict_current_traffic_state(traffic_model, current_readings_df)
        print(f"Predicted State Probabilities:")
        for state_id, prob in zip(sorted(TRAFFIC_STATES.keys()), prediction_probabilities):
            print(f"  - {TRAFFIC_STATES[state_id]}: {prob:.2f}")

        # ITFO-Decision: Make traffic decisions
        recommended_actions = make_traffic_decisions(predicted_state_id)

        # Simulate applying decisions and observing performance (for feedback)
        # In a real system, you'd measure actual traffic performance here.
        # For this demo, 'actual_performance_metrics' is a placeholder.
        actual_performance_metrics = {
            'avg_delay_sec': np.random.randint(5, 60), # Example simulated metric
            'throughput_veh_per_min': np.random.randint(50, 300) # Example simulated metric
        }
        print(f"[{PROJECT_NAME}] Simulated Performance: {actual_performance_metrics}")

        # ITFO-Feedback: Collect data for continuous learning
        collect_feedback(current_readings_dict, predicted_state_id, actual_performance_metrics)

        # Periodically check for retraining
        if (i + 1) % retraining_check_interval == 0:
            periodic_retraining_trigger(traffic_model, min_feedback_samples=10) # Lower threshold for demo

        time.sleep(3) # Simulate real-time delay between cycles

    print("\n--- Simulation Complete ---")
    print(f"Final feedback buffer size: {len(feedback_buffer)}")
    print("This project demonstrates the core ML-driven traffic optimization loop.")
    print("For production, aspects like data pipelines, robust deployment, and advanced RL would be considered.")