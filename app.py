import streamlit as st

import pandas as pd

import numpy as np

from sklearn.linear_model import LinearRegression

# Title

st.title("Weather Prediction App")

st.write("Predict Temperature based on historical data.")

# Sample Dataset

data = {

    'Day': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],

    'Temperature': [30, 31, 32, 33, 34, 35, 36, 35, 37, 38]

}

df = pd.DataFrame(data)

# Display Dataset

st.subheader("Historical Temperature Data")

st.dataframe(df)

# Train Model

X = df[['Day']]

y = df['Temperature']

model = LinearRegression()

model.fit(X, y)

# User Input

st.subheader("Predict Future Temperature")

future_day = st.number_input(

    "Enter Future Day",

    min_value=1,

    value=11,

    step=1

)

# Prediction

if st.button("Predict"):

    prediction = model.predict([[future_day]])[0]

    st.success(

        f"Predicted Temperature on Day {future_day}: {prediction:.2f} °C"

    )

# Visualization

st.subheader("Temperature Trend")

chart_data = pd.DataFrame({

    'Day': df['Day'],

    'Temperature': df['Temperature']

})

st.line_chart(chart_data.set_index('Day'))

