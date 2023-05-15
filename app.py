import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

# Load the model
model = load_model('dengue_lstm_model.h5')

# Load the dataset
data = pd.read_csv('denguecases.csv', index_col='Date', parse_dates=True)

# Compute the sum of dengue cases per region
sum_by_region = data.groupby('Region')['Dengue_Cases'].sum()

# Compute the total number of dengue cases per day
total_cases = data.drop('Region', axis=1).sum(axis=1)

# Preprocess the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(total_cases.values.reshape(-1, 1))

# Define a function to predict the next 12 months of dengue cases
def predict_next_12_months(model, data, scaler):
    last_date = data.index[-1]
    last_value = data[-1]
    next_dates = pd.date_range(start=last_date, periods=13, freq='MS')[1:]
    predictions = []
    for i in range(12):
        x = np.array(last_value).reshape(1, 1, 1)
        x_scaled = scaler.transform(x)
        yhat_scaled = model.predict(x_scaled)
        yhat = scaler.inverse_transform(yhat_scaled)[0][0]
        predictions.append(yhat)
        last_value = yhat_scaled
    return next_dates, predictions

# Predict the next 12 months of dengue cases
next_dates, predictions = predict_next_12_months(model, total_cases, scaler)

# Add the predictions to the dataset
next_data = pd.Series(predictions, index=next_dates)
data = pd.concat([data, next_data], axis=0)

# Plot the data
fig = px.line(data, x=data.index, y='Dengue_Cases', title='Dengue Cases in the Philippines')
fig.update_xaxes(title='Date')
fig.update_yaxes(title='Dengue Cases')
st.plotly_chart(fig)

# Show the predicted values for the next 12 months
st.write('Predicted Dengue Cases for the next 12 months:')
for i in range(len(next_dates)):
    st.write(f'{next_dates[i].strftime("%b %Y")}: {int(predictions[i]):,}')
