import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from dash import Dash, html, dcc, Output, Input

# Load your dataset
data = pd.read_csv('simbox_modeling_anonymized.csv')  # Adjust the file name accordingly

# Feature Engineering
data.fillna(0, inplace=True)
data['call_ratio'] = data['incoming_call'] / (data['outgoing_call'] + 1e-6)  # Avoid division by zero
data['total_call_duration'] = data['incoming_call_duration'] + data['outgoing_call_duration']
data['average_call_duration'] = np.where(data['incoming_call'] + data['outgoing_call'] > 0,
                                          data['total_call_duration'] / (data['incoming_call'] + data['outgoing_call']),
                                          0)  # Avoid division by zero
data['night_call_ratio'] = data['night_call'] / (data['incoming_call'] + 1e-6)  # Avoid division by zero
data['international_call_ratio'] = data['international_call'] / (data['incoming_call'] + data['outgoing_call'] + 1e-6)  # Avoid division by zero

# Prepare your dataset for supervised learning
X = data.drop(columns=['category'])  # Drop the target variable
y = data['category'].map({'F': 1, 'N': 0})  # Assuming 'F' = 1 (fraud) and 'N' = 0 (normal)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Prepare for Isolation Forest
scaler = StandardScaler()
X_unsupervised = data.drop(columns=['category'])  # Drop the target variable for unsupervised learning
X_unsupervised_scaled = scaler.fit_transform(X_unsupervised)

# Train Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42)  # Adjust contamination based on your needs
y_pred_iso = iso_forest.fit_predict(X_unsupervised_scaled)

# Convert predictions to normal/anomaly labels
y_pred_iso = np.where(y_pred_iso == -1, 1, 0)  # Convert to 1 for anomalies and 0 for normal

# Set up the Dash app
app = Dash(__name__)

# Layout for the Dash app
app.layout = html.Div([
    # Header
    html.Header([
        html.H1("Telecom Fraud Detection Dashboard", style={"margin": "0", "padding": "20px", "text-align": "center"}),
    ], style={"backgroundColor": "#007bff", "color": "white"}),  # Blue header

    # Button to generate random call
    html.Button("Generate Random Call", id='generate-call', n_clicks=0, style={"margin": "20px auto", "display": "block"}),

    # Output Boxes
    html.Div(id='call-output', style={"display": "flex", "justifyContent": "space-around", "margin": "20px"}),
    
    html.Div([
        html.Div(id='random-call-status', className='output-box', style={"flex": "1", "margin": "0 10px"}),
        html.Div(id='anomaly-status', className='output-box', style={"flex": "1", "margin": "0 10px"})
    ], style={"display": "flex"}),  # Flexbox for horizontal layout

    # Graphs
    html.Div([
        dcc.Graph(id='rf-graph', className='graph', style={"flex": "1"}),
        dcc.Graph(id='anomaly-graph', className='graph', style={"flex": "1"})
    ], style={"display": "flex", "margin": "20px"}),

    dcc.Interval(id='interval-component', interval=5000, n_intervals=0),  # Increase interval to slow down updates

    # Footer
    html.Footer([
        html.P("© 2024 Telecom Fraud Detection", style={"text-align": "center", "padding": "10px"})
    ], style={"backgroundColor": "#007bff", "color": "white", "position": "fixed", "bottom": "0", "width": "100%", "text-align": "center"})  # Fixed footer
])

# Function to generate random call data
def generate_random_call(is_anomaly=False):
    if is_anomaly:
        # Generate a call that is more likely to be an anomaly
        return {
            'incoming_call': random.randint(0, 10),
            'incoming_call_duration': random.randint(3000, 10000),  # Longer duration
            'outgoing_call': random.randint(0, 10),
            'outgoing_call_duration': random.randint(3000, 10000),  # Longer duration
            'night_call': random.randint(10, 50),  # More night calls
            'international_call': random.randint(5, 10),  # More international calls
            'cell_sites': random.randint(1, 5),
            'Num_IMSI': random.randint(10, 20),  # More IMSIs
            'Num_Cell_sites': random.randint(10, 20),  # More cell sites
            'Chrono': random.randint(1, 10),
            'CS_Ratio': random.uniform(1, 10),  # Extremely high ratio
            'IO_ratio': random.uniform(1, 10),
            'international_call_duration': random.randint(3000, 10000)
        }
    else:
        # Generate a normal call
        return {
            'incoming_call': random.randint(0, 10),
            'incoming_call_duration': random.randint(0, 300),
            'outgoing_call': random.randint(0, 10),
            'outgoing_call_duration': random.randint(0, 300),
            'night_call': random.randint(0, 10),
            'international_call': random.randint(0, 10),
            'cell_sites': random.randint(1, 5),
            'Num_IMSI': random.randint(1, 10),
            'Num_Cell_sites': random.randint(1, 10),
            'Chrono': random.randint(1, 10),
            'CS_Ratio': random.uniform(0, 1),
            'IO_ratio': random.uniform(0, 1),
            'international_call_duration': random.randint(0, 300)
        }

@app.callback(
    [Output('random-call-status', 'children'),
     Output('anomaly-status', 'children'),
     Output('rf-graph', 'figure'),
     Output('anomaly-graph', 'figure')],
    [Input('generate-call', 'n_clicks'),
     Input('interval-component', 'n_intervals')]
)
def update_graphs(n_clicks, n_intervals):
    # Randomly decide if we are generating an anomaly
    is_anomaly = random.choice([True, False])  # 50% chance to be an anomaly
    random_call = generate_random_call(is_anomaly)
    random_call_df = pd.DataFrame([random_call])

    # Ensure the columns match the trained model's feature set
    random_call_df = random_call_df.reindex(columns=X.columns, fill_value=0)

    # Predict with Random Forest
    rf_prediction = rf.predict(random_call_df)[0]
    rf_prediction_text = 'Fraudulent' if rf_prediction == 1 else 'Non-Fraudulent'

    # Scale the random call for Isolation Forest
    random_call_scaled = scaler.transform(random_call_df)
    anomaly_prediction = iso_forest.predict(random_call_scaled)[0]
    anomaly_prediction_text = 'Anomaly' if anomaly_prediction == -1 else 'Normal'

    # Update graphs
    rf_fig = go.Figure()
    rf_fig.add_trace(go.Indicator(
        mode="number+gauge+delta",
        value=rf_prediction,
        title={"text": "Random Forest Prediction"},
        gauge={"axis": {"range": [0, 1]}, "steps": [{"range": [0, 0.5], "color": "red"}, {"range": [0.5, 1], "color": "green"}]},
        domain={'x': [0, 1], 'y': [0, 1]}
    ))

    # Anomaly Detection
    anomaly_fig = go.Figure()
    anomaly_fig.add_trace(go.Indicator(
        mode="number+gauge+delta",
        value=1 if anomaly_prediction == -1 else 0,
        title={"text": "Anomaly Detection"},
        gauge={"axis": {"range": [-1, 1]}, "steps": [{"range": [-1, 0], "color": "green"}, {"range": [0, 1], "color": "red"}]},
        domain={'x': [0, 1], 'y': [0, 1]}
    ))

    return f"Random Call is: {rf_prediction_text}", f"Anomaly Status: {anomaly_prediction_text}", rf_fig, anomaly_fig

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True, port=7475)
