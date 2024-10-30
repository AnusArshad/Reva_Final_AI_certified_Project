Telecom Fraud Detection Dashboard
Overview
The Telecom Fraud Detection Dashboard is a web-based application designed to detect potential telecom fraud. Built with machine learning models, this app uses Random Forest for fraud classification and Isolation Forest for anomaly detection. The dashboard is created using Plotly Dash, allowing users to interact with real-time predictions and visualize fraud risk in a clear, concise way.

Features
Random Call Simulation: Generates random call data to simulate normal and potentially fraudulent calls.
Fraud Classification: Uses a supervised Random Forest model to predict if a call is fraudulent.
Anomaly Detection: Uses an unsupervised Isolation Forest model to detect anomalies, which may indicate fraud.
Interactive Dashboard: Includes an easy-to-use interface with buttons, graphs, and gauges to display prediction results.
Dynamic Visuals: Shows real-time fraud and anomaly predictions with gauges for each model’s output.
Getting Started
Prerequisites
Ensure you have Python 3.7 or above installed. Required libraries include:

pandas
numpy
scikit-learn
plotly
dash

Project Structure
app.py: Main application script that includes data preprocessing, model training, and the Dash app layout.
requirements.txt: File listing all required packages.
data/: Folder to store datasets like simbox_modeling_anonymized.csv (update file path if located elsewhere).
Key Functionality
Data Preparation
Feature Engineering: Creates additional features such as call_ratio, average_call_duration, and international_call_ratio for better model performance.
Data Splitting: Splits data into training and testing sets.
Models
Random Forest Classifier:
A supervised model for binary classification (fraud vs. normal calls).
Isolation Forest:
An unsupervised model for identifying anomalies that may indicate fraud.
Interactive Dashboard
Generate Random Call: Button to simulate random call data, allowing users to see the model predictions.
Real-time Graphs:
Random Forest Prediction: Gauge showing if the call is likely fraudulent.
Anomaly Detection: Gauge indicating whether the call is considered an anomaly.
