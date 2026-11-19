
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

artifacts = joblib.load("models/house_price_model.pkl")

model = artifacts["model"]
scaler = artifacts["scaler"]
features = artifacts["features"]

st.title("🏠 CALIFORNIA HOUSE PRICE PREDICTOR")

st.write(
    "Enter the housing details below to estimate the median "
    "house value using our trained Machine Learning model."
)

st.divider()

st.subheader("House Details")

med_inc = st.number_input(
    "Median Income (MedInc)",
    min_value=0.0,
    value=3.5,
    step=0.1
)

house_age = st.number_input(
    "House Age",
    min_value=0.0,
    value=25.0,
    step=1.0
)

ave_rooms = st.number_input(
    "Average Rooms",
    min_value=0.0,
    value=5.0,
    step=0.1
)

ave_bedrms = st.number_input(
    "Average Bedrooms",
    min_value=0.0,
    value=1.0,
    step=0.1
)

population = st.number_input(
    "Population",
    min_value=0.0,
    value=1000.0,
    step=10.0
)

ave_occup = st.number_input(
    "Average Occupancy",
    min_value=0.0,
    value=3.0,
    step=0.1
)

latitude = st.number_input(
    "Latitude",
    min_value=32.0,
    max_value=42.0,
    value=36.0,
    step=0.01
)

longitude = st.number_input(
    "Longitude",
    min_value=-125.0,
    max_value=-114.0,
    value=-120.0,
    step=0.01
)

if st.button("Predict House Price", type="primary"):

    input_data = pd.DataFrame(
        [[
            med_inc,
            house_age,
            ave_rooms,
            ave_bedrms,
            population,
            ave_occup,
            latitude,
            longitude
        ]],
        columns=features
    )

    scaled_input = scaler.transform(input_data)

    prediction = model.predict(scaled_input)[0]

  
    predicted_price = prediction * 100000

    st.success("Prediction generated successfully!")

    st.metric(
        "Estimated Median House Value",
        f"${predicted_price:,.2f}"
    )

    st.caption(
        "This estimate uses the California Housing dataset. "
        "It represents a district-level median house value estimate, "
        "not an individual property's market valuation."
    )
