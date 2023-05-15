import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import plotly.graph_objs as go


# Load the LSTM model
model = load_model('dengue_lstm_model.h5')

# Load the Dengue dataset
data = pd.read_csv('denguecases.csv', index_col='Date', parse_dates=True)

# Compute the sum of dengue cases per region
sum_by_region = data.groupby('Region')['Dengue_Cases'].sum()


# set up the Streamlit app
st.set_page_config(page_title=" Dengue Cases Data in the Philippines (2008 - 2016)", page_icon="🦟")
st.title("Dengue Cases Prediction")
st.write("This app predicts the Dengue Cases in the Philippines based on historical data.")

st.subheader("Dengue Cases Per Region: ")
st.write(sum_by_region)


# Preprocess the data

whole_data = data.drop('Region', axis = 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(whole_data.values)

# Split the data into input and output variables
X_test = whole_data[-12]
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# Make predictions for the next 12 months
y_pred = model.predict(X_test)

# Create a DataFrame of the predicted values with the dates as the index
dates = pd.date_range(start=data.index[-1], periods=12, freq='MS')
pred_df = pd.DataFrame(y_pred.round(0), index=dates, columns=['Predicted Dengue Cases'])


# Getting the data with the highest cases 
highest = whole_data[whole_data.Dengue_Cases == max(whole_data.Dengue_Cases)]
st.subheader("Highest Dengue Cases (2010 - 2020)")
highest

# Getting the data with the lowest cases 
# Getting the data with the highest cases 
lowest = whole_data[whole_data.Dengue_Cases == min(whole_data.Dengue_Cases)]
st.subheader("Lowest Dengue Cases (2010 - 2020)")
lowest

# Show the predicted Dengue cases for the next 12 months
st.write('The predicted number of Dengue cases for the next 12 months are:')
st.write(pred_df)

# Plot the predicted Dengue cases for the next 12 months
# generate a list of y-axis tick values
y_ticks = list(range(1000, 9000, 1000))

# create the plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data['Value'], name='Actual'))
fig.add_trace(go.Scatter(x=pred_df.index, y=pred_df['Predicted Dengue Cases'], name='Predicted'))
fig.update_layout(title='Dengue Cases Prediction', xaxis_title='Date', yaxis_title='Dengue Cases', 
                  yaxis=dict(tickvals=y_ticks))
st.plotly_chart(fig)
