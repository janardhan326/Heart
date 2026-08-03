import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.title("📈 Sales Prediction App")
st.write("Predict Sales based on Advertising Budget")

data = {
    "Advertising_Budget": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    "Sales": [15, 22, 28, 35, 42, 48, 55, 63, 70, 78]
}

df = pd.DataFrame(data)

st.subheader("Sales Dataset")
st.dataframe(df)


X = df[['Advertising_Budget']]
y = df['Sales']

model = LinearRegression()
model.fit(X, y)


st.subheader("Sales Prediction")

budget = st.number_input(
    "Enter Advertising Budget (in thousands)",
    min_value=1,
    value=50,
    step=1
)


if st.button("Predict Sales"):

    input_data = pd.DataFrame({
        "Advertising_Budget": [budget]
    })

    prediction = model.predict(input_data)[0]

    st.success(
        f"Predicted Sales: {prediction:.2f} Units"
    )
st.subheader("Sales Trend")
st.line_chart(df.set_index("Advertising_Budget"))
