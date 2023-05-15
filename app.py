import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from datetime import datetime, timedelta

# Load the trained LSTM model
model = load_model('/content/drive/MyDrive/3RD YEAR 2ND SEM/Emerging TECH. 2/Datasets/dengue_lstm_model.h5')

# Load the dataset and preprocess it
data = pd.read_csv('/content/drive/MyDrive/3RD YEAR 2ND SEM/Emerging TECH. 2/Datasets/denguecases.csv', index_col = 'Date', parse_dates = True)
whole_data = data.drop('Region', axis = 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(whole_data.values)

# Define a function to make predictions
def make_predictions(start_date, n_months):
    # Convert the start_date to datetime format
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    
    # Create a list of datetime objects for the next n_months
    date_list = [start_date + timedelta(days=30*i) for i in range(n_months)]
    
    # Create a DataFrame to store the predictions
    predictions = pd.DataFrame({'Date': date_list})
    
    # Predict the number of dengue cases for each month
    for i in range(n_months):
        # Get the scaled input data for the last 12 months
        x = scaled_data[-12:]
        # Reshape the input data into (samples, time steps, features)
        x = np.reshape(x, (1, x.shape[0], x.shape[1]))
        # Make the prediction for the next month
        y_pred = model.predict(x)
        # Inverse transform the scaled prediction
        y_pred = scaler.inverse_transform(y_pred)[0][0]
        # Add the prediction to the DataFrame
        predictions.loc[i, 'Dengue_Cases'] = y_pred
        # Add the next month to the input data for the next prediction
        next_month = scaled_data[-1, 1:]
        next_month = np.append(next_month, y_pred)
        scaled_data = np.vstack((scaled_data, next_month))
    
    # Set the Date column as the index
    predictions.set_index('Date', inplace=True)
    
    return predictions

# Define the Streamlit app
st.title('Dengue Cases Prediction')
st.write('This app predicts the number of dengue cases in the Philippines for the next 12 months.')
start_date = st.date_input('Enter the start date for the prediction:', value=datetime(2023,1,1), min_value=datetime(2010, 1, 1), max_value=datetime(2023, 5, 15))
n_months = 12
if st.button('Make Prediction'):
    predictions = make_predictions(start_date, n_months)
    st.write(predictions)
