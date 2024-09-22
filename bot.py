import yfinance as yf
import pandas as pd
import numpy as np
import plotly.io as pio
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import tempfile
import webbrowser

# Function to fetch stock data from Yahoo Finance
def fetch_data(ticker, period='1y'):
    data = yf.download(ticker, period=period, interval='1h')
    return data

# Function to analyze and generate signals using different ML algorithms
def analyze_data(data):
    # Calculate SMAs
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()

    # Create features and target
    data['Target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)
    data = data.dropna(subset=['SMA_20', 'SMA_50', 'Target'])
    features = data[['SMA_20', 'SMA_50']]
    targets = data['Target']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.3, random_state=42)

    # Initialize dictionary to store results
    results = {}

    # KNN Model
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    knn_predictions = knn.predict(X_test)
    knn_accuracy = accuracy_score(y_test, knn_predictions)
    results['KNN'] = {'predictions': knn_predictions, 'accuracy': knn_accuracy}

    # Random Forest Model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_predictions = rf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_predictions)
    results['Random Forest'] = {'predictions': rf_predictions, 'accuracy': rf_accuracy}

    # SVM Model
    svm = SVC(kernel='linear')
    svm.fit(X_train, y_train)
    svm_predictions = svm.predict(X_test)
    svm_accuracy = accuracy_score(y_test, svm_predictions)
    results['SVM'] = {'predictions': svm_predictions, 'accuracy': svm_accuracy}

    initial_investment = 10000

    for model in results.keys():
        data.loc[:, f'{model}_Signal'] = np.nan  # Create model-specific Signal column
        data.loc[X_test.index, f'{model}_Signal'] = results[model]['predictions']

        data[f'{model}_Position'] = data[f'{model}_Signal'].shift()
        data[f'{model}_Position'].fillna(0, inplace=True)

        # Calculate returns
        data[f'{model}_Strategy_Return'] = data[f'{model}_Position'] * data['Close'].pct_change()
        data[f'{model}_Cumulative_Strategy_Return'] = (data[f'{model}_Strategy_Return'] + 1).cumprod()
        data[f'{model}_Portfolio_Value'] = initial_investment * data[f'{model}_Cumulative_Strategy_Return']

    return data, results

# Function to plot data in a new window
def plot_data(data, ticker):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], mode='lines', name='20-Day SMA', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], mode='lines', name='50-Day SMA', line=dict(color='green')))

    for model in ['KNN', 'Random Forest', 'SVM']:
        fig.add_trace(go.Scatter(x=data.index, y=data[f'{model}_Portfolio_Value'], mode='lines', name=f'{model} Portfolio Value', line=dict(dash='dash')))

    fig.update_layout(title=f'{ticker.upper()} Stock Analysis with KNN, RF, SVM',
                      xaxis_title='Date',
                      yaxis_title='Price/Portfolio Value',
                      xaxis_rangeslider_visible=False)

    # Save the Plotly graph as an HTML file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as temp_file:
        temp_file.close()
        pio.write_html(fig, file=temp_file.name, auto_open=False)
        webbrowser.open(temp_file.name)

    print("Portfolio final values:")
    for model in ['KNN', 'Random Forest', 'SVM']:
        print(f"{model}: ${data[f'{model}_Portfolio_Value'].iloc[-1]:.2f}")

# Example usage
ticker = 'AAPL'  # Replace with your chosen ticker
data = fetch_data(ticker)
data, model_results = analyze_data(data)

for model in model_results.keys():
    print(f"{model} Accuracy: {model_results[model]['accuracy']:.2f}")

plot_data(data, ticker)
